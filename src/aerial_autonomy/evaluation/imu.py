"""Export seeded IMU observations and separate plant truth for two A2 scenarios."""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from aerial_autonomy.config import ImuExperimentConfig, load_imu_experiment
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.logging_setup import configure_logging
from aerial_autonomy.sensors.imu import SimulatedImu, ideal_imu
from aerial_autonomy.simulation.dynamics import VehicleState, advance_state, world_acceleration
from aerial_autonomy.simulation.rotation import quaternion_to_euler
from aerial_autonomy.visualization.imu_plot import save_imu_plot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImuScenarioTrace:
    name: str
    time_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    quaternion_bw: np.ndarray
    body_rate_rad_s: np.ndarray
    acceleration_world_m_s2: np.ndarray
    specific_force_body_m_s2: np.ndarray
    thrust_n: np.ndarray
    torque_nm: np.ndarray
    desired_position_m: np.ndarray
    desired_euler_rad: np.ndarray
    sample_indices: np.ndarray
    measurement_time_s: np.ndarray
    measured_accel_m_s2: np.ndarray
    measured_gyro_rad_s: np.ndarray


def simulate_imu_scenario(config: ImuExperimentConfig, name: str) -> ImuScenarioTrace:
    """Run A1 dynamics/control; sample the IMU without changing the plant step."""
    if name not in ("hover", "lateral_waypoint"):
        raise ValueError("scenario must be 'hover' or 'lateral_waypoint'")
    base = config.base
    dt = base.demo.step_s
    steps = round(base.demo.duration_s / dt)
    time_s = np.arange(steps + 1) * dt
    position = np.zeros((steps + 1, 3))
    velocity = np.zeros_like(position)
    quaternion = np.zeros((steps + 1, 4))
    body_rate = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    specific_force = np.zeros_like(position)
    thrust = np.zeros(steps + 1)
    torque = np.zeros_like(position)
    desired_position = np.zeros_like(position)
    desired_euler = np.zeros_like(position)
    sample_indices: list[int] = []
    measurement_time: list[float] = []
    measured_accel: list[np.ndarray] = []
    measured_gyro: list[np.ndarray] = []

    state = VehicleState.at_rest()
    controller = CascadedHoverController(base.vehicle, base.controller)
    sensor = SimulatedImu(config.imu, dt)
    for index, t in enumerate(time_s):
        if name == "lateral_waypoint" and t >= config.maneuver.switch_time_s:
            target = config.maneuver.target_position_m
            yaw_rad = np.deg2rad(config.maneuver.target_yaw_deg)
        else:
            target = base.demo.target_position_m
            yaw_rad = np.deg2rad(base.demo.target_yaw_deg)
        control = controller.command(state, target, yaw_rad)
        wrench = control.wrench

        position[index] = state.position_m
        velocity[index] = state.velocity_m_s
        quaternion[index] = state.quaternion_bw
        body_rate[index] = state.body_rate_rad_s
        acceleration[index] = world_acceleration(state, wrench, base.vehicle)
        specific_force[index], _ = ideal_imu(state, wrench, base.vehicle)
        thrust[index] = wrench.thrust_n
        torque[index] = wrench.torque_nm
        desired_position[index] = target
        desired_euler[index] = control.desired_euler_rad
        if sensor.should_sample(index):
            reading = sensor.sample(index, state, wrench, base.vehicle)
            sample_indices.append(index)
            measurement_time.append(reading.timestamp_s)
            measured_accel.append(reading.accel_m_s2)
            measured_gyro.append(reading.gyro_rad_s)
        if index < steps:
            state = advance_state(state, wrench, base.vehicle, dt)

    return ImuScenarioTrace(
        name=name,
        time_s=time_s,
        position_m=position,
        velocity_m_s=velocity,
        quaternion_bw=quaternion,
        body_rate_rad_s=body_rate,
        acceleration_world_m_s2=acceleration,
        specific_force_body_m_s2=specific_force,
        thrust_n=thrust,
        torque_nm=torque,
        desired_position_m=desired_position,
        desired_euler_rad=desired_euler,
        sample_indices=np.asarray(sample_indices, dtype=int),
        measurement_time_s=np.asarray(measurement_time),
        measured_accel_m_s2=np.asarray(measured_accel),
        measured_gyro_rad_s=np.asarray(measured_gyro),
    )


def summarize_scenario(trace: ImuScenarioTrace, config: ImuExperimentConfig) -> dict:
    """Small set of inspectable physical and sensor diagnostics."""
    sampled_force = trace.specific_force_body_m_s2[trace.sample_indices]
    sampled_rate = trace.body_rate_rad_s[trace.sample_indices]
    accel_residual = trace.measured_accel_m_s2 - sampled_force
    gyro_residual = trace.measured_gyro_rad_s - sampled_rate
    euler_rad = np.array([quaternion_to_euler(q) for q in trace.quaternion_bw])
    target = trace.desired_position_m[-1]
    steady_mask = trace.time_s >= config.base.demo.duration_s - 2.0
    return {
        "scenario": trace.name,
        "plant_steps": int(len(trace.time_s) - 1),
        "imu_samples": int(len(trace.measurement_time_s)),
        "final_position_m": trace.position_m[-1].tolist(),
        "final_target_m": target.tolist(),
        "final_position_error_m": float(np.linalg.norm(trace.position_m[-1] - target)),
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(euler_rad[:, 0])))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(euler_rad[:, 1])))),
        "max_body_rate_norm_rad_s": float(np.max(np.linalg.norm(trace.body_rate_rad_s, axis=1))),
        "max_abs_world_lateral_acceleration_m_s2": float(np.max(np.abs(trace.acceleration_world_m_s2[:, :2]))),
        "steady_last_2s_mean_true_specific_force_body_m_s2": trace.specific_force_body_m_s2[steady_mask].mean(axis=0).tolist(),
        "steady_last_2s_mean_true_body_rate_rad_s": trace.body_rate_rad_s[steady_mask].mean(axis=0).tolist(),
        "mean_accel_measurement_minus_truth_m_s2": accel_residual.mean(axis=0).tolist(),
        "std_accel_measurement_minus_truth_m_s2": accel_residual.std(axis=0).tolist(),
        "mean_gyro_measurement_minus_truth_rad_s": gyro_residual.mean(axis=0).tolist(),
        "std_gyro_measurement_minus_truth_rad_s": gyro_residual.std(axis=0).tolist(),
    }


def _save_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header),
               comments="", fmt="%.12g")


def export_scenario(trace: ImuScenarioTrace, config: ImuExperimentConfig, directory: Path) -> dict:
    """Write sensor-only CSV separately from evaluation-only truth and metrics."""
    directory.mkdir(parents=True, exist_ok=True)
    axes = ("x", "y", "z")
    measurement_header = ["timestamp_s"] + [f"accel_{axis}_m_s2" for axis in axes] + [
        f"gyro_{axis}_rad_s" for axis in axes
    ]
    _save_csv(directory / "measurements.csv", measurement_header, [
        trace.measurement_time_s, trace.measured_accel_m_s2, trace.measured_gyro_rad_s
    ])

    truth_header = ["timestamp_s"]
    truth_columns: list[np.ndarray] = [trace.time_s]
    for prefix, units, values in (
        ("position_world", "m", trace.position_m),
        ("velocity_world", "m_s", trace.velocity_m_s),
        ("body_rate", "rad_s", trace.body_rate_rad_s),
        ("acceleration_world", "m_s2", trace.acceleration_world_m_s2),
        ("specific_force_body", "m_s2", trace.specific_force_body_m_s2),
        ("torque_body", "nm", trace.torque_nm),
        ("desired_position_world", "m", trace.desired_position_m),
    ):
        truth_header.extend(f"{prefix}_{axis}_{units}" for axis in axes)
        truth_columns.append(values)
    truth_header.extend(("desired_roll_rad", "desired_pitch_rad", "desired_yaw_rad"))
    truth_columns.append(trace.desired_euler_rad)
    truth_header.extend(("quaternion_bw_w", "quaternion_bw_x", "quaternion_bw_y", "quaternion_bw_z", "thrust_n"))
    truth_columns.extend((trace.quaternion_bw, trace.thrust_n))
    _save_csv(directory / "truth.csv", truth_header, truth_columns)

    metadata = {
        "schema_version": 1,
        "scenario": trace.name,
        "estimator_visible": "measurements.csv: timestamp, body accelerometer specific force, body gyro rate",
        "evaluation_only": "truth.csv and metrics.json: plant state, true acceleration, true specific force, true body rate, commands",
        "frames": {"world": "ENU", "body": "FLU", "quaternion_bw": "scalar-first Hamilton, body to world"},
        "sensor_units": {"accelerometer": "m/s^2 specific force", "gyroscope": "rad/s body rate"},
        "noise_model": "Independent zero-mean Gaussian white noise; standard deviations are per sample",
        "imu": {
            "sample_hz": config.imu.sample_hz,
            "sample_period_s": 1.0 / config.imu.sample_hz,
            "accel_bias_m_s2": config.imu.accel_bias_m_s2.tolist(),
            "gyro_bias_rad_s": config.imu.gyro_bias_rad_s.tolist(),
            "accel_noise_std_m_s2": config.imu.accel_noise_std_m_s2.tolist(),
            "gyro_noise_std_rad_s": config.imu.gyro_noise_std_rad_s.tolist(),
            "seed": config.imu.seed,
        },
        "plant_step_s": config.base.demo.step_s,
        "sample_stride_plant_steps": int(round((1.0 / config.imu.sample_hz) / config.base.demo.step_s)),
        "vehicle": {
            "mass_kg": config.base.vehicle.mass_kg,
            "inertia_kg_m2": config.base.vehicle.inertia_kg_m2.tolist(),
            "gravity_m_s2": config.base.vehicle.gravity_m_s2,
            "linear_drag_n_s_m": config.base.vehicle.linear_drag_n_s_m,
        },
        "waypoint": {
            "switch_time_s": config.maneuver.switch_time_s if trace.name == "lateral_waypoint" else None,
            "target_position_m": trace.desired_position_m[-1].tolist(),
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics = summarize_scenario(trace, config)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_imu_plot(trace, directory / "comparison.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/imu.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu"))
    args = parser.parse_args()
    configure_logging()
    config = load_imu_experiment(args.config)
    for name in ("hover", "lateral_waypoint"):
        trace = simulate_imu_scenario(config, name)
        metrics = export_scenario(trace, config, args.output_dir / name)
        LOGGER.info("%s: %d IMU samples; final position error %.4f m; max pitch %.2f deg; max body rate %.3f rad/s",
                    name, metrics["imu_samples"], metrics["final_position_error_m"],
                    metrics["max_abs_pitch_deg"], metrics["max_body_rate_norm_rad_s"])
        LOGGER.info("Saved %s", (args.output_dir / name).resolve())


if __name__ == "__main__":
    main()
