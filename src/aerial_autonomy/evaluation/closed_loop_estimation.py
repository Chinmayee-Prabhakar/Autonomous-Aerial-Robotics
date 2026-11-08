"""Causal, paired IMU/attitude-feedback experiments on the existing quadrotor.

The plant owns truth. Estimators see only timestamped IMU readings. Controller
position, velocity, and body rate always come from truth; only its attitude
field changes among the declared feedback modes.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
from time import perf_counter
import tomllib
from typing import Callable

import numpy as np

from aerial_autonomy.config import ImuConfig, ImuExperimentConfig, load_imu_experiment
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.estimation.madgwick import MadgwickConfig, MadgwickImu
from aerial_autonomy.estimation.quaternion_ukf import QuaternionUkf, QuaternionUkfConfig
from aerial_autonomy.evaluation.madgwick import geodesic_error_deg, load_experiment_config
from aerial_autonomy.evaluation.ukf import load_config as load_ukf_config
from aerial_autonomy.sensors.imu import SimulatedImu, ideal_imu
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, normalize_quaternion, quaternion_to_euler, quaternion_to_matrix,
)


MODES = ("truth", "madgwick", "ukf")


@dataclass(frozen=True)
class Waypoint:
    time_s: float
    position_m: np.ndarray
    yaw_rad: float


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class FailureLimits:
    max_position_error_m: float
    max_abs_position_m: float
    max_tilt_deg: float
    max_body_rate_rad_s: float
    settle_final_error_m: float
    settle_mean_error_m: float
    settle_window_s: float

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("failure and settlement limits must be finite and positive")
        if self.max_tilt_deg >= 180:
            raise ValueError("max_tilt_deg must be less than 180")


@dataclass(frozen=True)
class SensorCondition:
    name: str
    bias_scale: float
    noise_scale: float

    def __post_init__(self) -> None:
        if not self.name or any(not np.isfinite(x) or x < 0
                                for x in (self.bias_scale, self.noise_scale)):
            raise ValueError("sensor condition needs a name and nonnegative finite scales")


@dataclass(frozen=True)
class ClosedLoopConfig:
    imu_experiment: ImuExperimentConfig
    madgwick: MadgwickConfig
    ukf: QuaternionUkfConfig
    controller_hz: float
    modes: tuple[str, ...]
    nominal_seed: int
    scenarios: tuple[Scenario, ...]
    limits: FailureLimits
    sensitivity_seeds: tuple[int, ...]
    sensitivity_scenarios: tuple[str, ...]
    conditions: tuple[SensorCondition, ...]


@dataclass(frozen=True)
class TrialResult:
    metrics: dict
    metadata: dict
    sensor: dict[str, np.ndarray]
    estimator: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]


def load_config(path: Path) -> ClosedLoopConfig:
    """Read M5 scheduling and limits; reuse unchanged M2/M3/M4 parameters."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    inputs = raw["inputs"]
    imu = load_imu_experiment(Path(inputs["imu_config"]))
    madgwick, _ = load_experiment_config(Path(inputs["madgwick_config"]))
    ukf, _ = load_ukf_config(Path(inputs["ukf_config"]))
    if not (np.array_equal(madgwick.initial_quaternion_bw, ukf.initial_quaternion_bw)
            and np.array_equal(ukf.initial_quaternion_bw, VehicleState.at_rest().quaternion_bw)):
        raise ValueError("primary M5 arms require the same declared, level initial quaternion")
    experiment = raw["experiment"]
    hz = float(experiment["controller_hz"])
    if not np.isfinite(hz) or hz <= 0 or not np.isclose(
            hz * imu.base.demo.step_s, 1.0, rtol=0, atol=1e-12):
        raise ValueError("controller_hz must equal the fixed plant integration rate")
    modes = tuple(experiment["modes"])
    if modes != MODES:
        raise ValueError("primary modes must be truth, madgwick, ukf in that order")
    scenarios = []
    for item in raw["scenarios"]:
        duration = float(item["duration_s"])
        if (not np.isfinite(duration) or duration <= 0 or not np.isclose(
                duration / imu.base.demo.step_s,
                round(duration / imu.base.demo.step_s), rtol=0, atol=1e-9)):
            raise ValueError("scenario duration must contain whole plant steps")
        waypoints = []
        for row in item["waypoints"]:
            values = np.asarray(row, dtype=float)
            if values.shape != (5,) or not np.all(np.isfinite(values)):
                raise ValueError("waypoint must contain finite [time,x,y,z,yaw_deg]")
            waypoints.append(Waypoint(float(values[0]), values[1:4],
                                      float(np.deg2rad(values[4]))))
        times = [point.time_s for point in waypoints]
        if not times or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])) or times[-1] >= duration:
            raise ValueError("waypoints must start at zero and increase within the horizon")
        scenarios.append(Scenario(str(item["name"]), duration, tuple(waypoints)))
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names) or not {"hover", "lateral_waypoint", "coupled_waypoints"} <= set(names):
        raise ValueError("three unique M5 scenario names are required")
    limits = FailureLimits(**{key: float(value) for key, value in raw["failure"].items()})
    sweep = raw["sensitivity"]
    seeds = tuple(int(value) for value in sweep["seeds"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("sensitivity needs at least five unique nonnegative seeds")
    sensitivity_scenarios = tuple(sweep["scenarios"])
    if not sensitivity_scenarios or any(name not in names for name in sensitivity_scenarios):
        raise ValueError("sensitivity scenarios must name configured scenarios")
    conditions = tuple(SensorCondition(**item) for item in sweep["conditions"])
    if [condition.name for condition in conditions] != [
            "ideal", "noise_only", "bias_only", "nominal", "stressed"]:
        raise ValueError("M5 sensitivity conditions must retain the declared five presets")
    seed = experiment["nominal_seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("nominal_seed must be a nonnegative integer")
    return ClosedLoopConfig(imu, madgwick, ukf, hz, modes, seed, tuple(scenarios),
                            limits, seeds, sensitivity_scenarios, conditions)


def target_at(scenario: Scenario, time_s: float) -> Waypoint:
    """Right-continuous zero-order-hold reference, including switch instants."""
    return next(point for point in reversed(scenario.waypoints) if time_s + 1e-12 >= point.time_s)


def feedback_state(truth: VehicleState, attitude_bw: np.ndarray) -> VehicleState:
    """Copy truth translation and body rate; inject only the selected attitude."""
    q = np.asarray(attitude_bw, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("invalid_estimator_quaternion")
    try:
        q = normalize_quaternion(q)
    except ValueError as error:
        raise ValueError("invalid_estimator_quaternion") from error
    return VehicleState(truth.position_m.copy(), truth.velocity_m_s.copy(), q,
                        truth.body_rate_rad_s.copy())


def _finite_state(state: VehicleState) -> bool:
    return all(np.all(np.isfinite(value)) for value in (
        state.position_m, state.velocity_m_s, state.quaternion_bw,
        state.body_rate_rad_s))


def _tilt_deg(quaternion_bw: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(
        quaternion_to_matrix(quaternion_bw)[2, 2], -1.0, 1.0))))


def _limit_reason(state: VehicleState, target_m: np.ndarray,
                  limits: FailureLimits) -> str | None:
    if np.linalg.norm(state.position_m - target_m) > limits.max_position_error_m:
        return "position_tracking_limit"
    if np.max(np.abs(state.position_m)) > limits.max_abs_position_m:
        return "workspace_boundary"
    if _tilt_deg(state.quaternion_bw) > limits.max_tilt_deg:
        return "tilt_limit"
    if np.linalg.norm(state.body_rate_rad_s) > limits.max_body_rate_rad_s:
        return "body_rate_limit"
    return None


def scaled_imu(config: ImuConfig, condition: SensorCondition, seed: int) -> ImuConfig:
    """Pair independent per-sample innovation draws by seed across feedback arms."""
    return replace(config, seed=seed,
                   accel_bias_m_s2=config.accel_bias_m_s2 * condition.bias_scale,
                   gyro_bias_rad_s=config.gyro_bias_rad_s * condition.bias_scale,
                   accel_noise_std_m_s2=config.accel_noise_std_m_s2 * condition.noise_scale,
                   gyro_noise_std_rad_s=config.gyro_noise_std_rad_s * condition.noise_scale)


def _arrays(rows: dict[str, list]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in rows.items()}


def _json_ready(value):
    """Convert configuration dataclasses and NumPy vectors to JSON primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _metrics(evaluation: dict[str, np.ndarray], estimator: dict[str, np.ndarray],
             sensor: dict[str, np.ndarray], scenario: Scenario, mode: str,
             condition: SensorCondition, seed: int, reason: str | None,
             failure_time_s: float | None, estimator_runtime_s: float,
             plant_step_s: float, limits: FailureLimits) -> dict:
    time_s = evaluation["timestamp_s"]
    if not len(time_s):
        unavailable = ("position_rmse_m", "altitude_rmse_m", "max_position_error_m",
                       "final_position_error_m", "attitude_geodesic_rmse_deg",
                       "roll_rmse_deg", "pitch_rmse_deg", "relative_yaw_final_error_deg",
                       "true_attitude_tracking_rmse_deg", "max_tilt_deg",
                       "max_body_rate_rad_s", "max_thrust_n", "max_torque_norm_nm")
        return {
            "scenario": scenario.name, "mode": mode, "condition": condition.name, "seed": seed,
            "success": False, "failure_reason": reason, "failure_time_s": failure_time_s,
            "completed_horizon": False,
            "metric_window": "no controller updates; RMSE unavailable",
            "observed_duration_s": 0.0, "control_updates": 0,
            "imu_updates": int(len(sensor["timestamp_s"])),
            **{name: None for name in unavailable},
            "thrust_saturation_updates": 0, "torque_saturation_updates": 0,
            "desired_tilt_saturation_updates": 0,
            "thrust_saturation_duration_s": 0.0,
            "torque_saturation_duration_s": 0.0,
            "desired_tilt_saturation_duration_s": 0.0,
            "estimator_runtime_s": float(estimator_runtime_s),
            "estimator_update_status_counts": {},
            "nonfinite_incidents": int(reason in ("nonfinite_plant_state", "invalid_estimator_quaternion")),
        }
    complete = len(time_s) > 0 and np.isclose(time_s[-1], scenario.duration_s, atol=1e-9)
    position_error = evaluation["position_error_m"]
    if complete and reason is None:
        final_error = float(position_error[-1])
        steady = position_error[time_s >= scenario.duration_s - limits.settle_window_s]
        if final_error > limits.settle_final_error_m or float(np.mean(steady)) > limits.settle_mean_error_m:
            reason, failure_time_s = "target_not_settled", scenario.duration_s
    tilt_errors = evaluation["roll_pitch_estimation_error_deg"]
    relative_yaw = evaluation["relative_yaw_estimation_error_deg"]
    return {
        "scenario": scenario.name, "mode": mode, "condition": condition.name, "seed": seed,
        "success": reason is None, "failure_reason": reason,
        "failure_time_s": failure_time_s, "completed_horizon": bool(complete),
        "metric_window": "all available samples; compare RMSE across arms only when completed_horizon is true",
        "observed_duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "control_updates": int(len(time_s)), "imu_updates": int(len(sensor["timestamp_s"])),
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "altitude_rmse_m": float(np.sqrt(np.mean((evaluation["position_m"][:, 2]
                                                   - evaluation["desired_position_m"][:, 2]) ** 2))),
        "max_position_error_m": float(np.max(position_error)),
        "final_position_error_m": float(position_error[-1]),
        "attitude_geodesic_rmse_deg": float(np.sqrt(np.mean(
            evaluation["attitude_estimation_geodesic_error_deg"] ** 2))),
        "roll_rmse_deg": float(np.sqrt(np.mean(tilt_errors[:, 0] ** 2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(tilt_errors[:, 1] ** 2))),
        "relative_yaw_final_error_deg": float(relative_yaw[-1]),
        "true_attitude_tracking_rmse_deg": float(np.sqrt(np.mean(
            evaluation["true_attitude_tracking_error_deg"] ** 2))),
        "max_tilt_deg": float(np.max(evaluation["true_tilt_deg"])),
        "max_body_rate_rad_s": float(np.max(np.linalg.norm(evaluation["body_rate_rad_s"], axis=1))),
        "max_thrust_n": float(np.max(evaluation["applied_thrust_n"])),
        "max_torque_norm_nm": float(np.max(np.linalg.norm(evaluation["applied_torque_nm"], axis=1))),
        "thrust_saturation_updates": int(np.count_nonzero(evaluation["thrust_saturated"])),
        "torque_saturation_updates": int(np.count_nonzero(evaluation["torque_saturated"])),
        "desired_tilt_saturation_updates": int(np.count_nonzero(evaluation["desired_tilt_saturated"])),
        "thrust_saturation_duration_s": float(np.count_nonzero(evaluation["thrust_saturated"]) * plant_step_s),
        "torque_saturation_duration_s": float(np.count_nonzero(evaluation["torque_saturated"]) * plant_step_s),
        "desired_tilt_saturation_duration_s": float(np.count_nonzero(evaluation["desired_tilt_saturated"]) * plant_step_s),
        "estimator_runtime_s": float(estimator_runtime_s),
        "estimator_update_status_counts": {str(status): int(np.count_nonzero(estimator["update_status"] == status))
                                            for status in np.unique(estimator["update_status"])},
        "nonfinite_incidents": int(reason in ("nonfinite_plant_state", "invalid_estimator_quaternion")),
    }


def run_trial(config: ClosedLoopConfig, scenario: Scenario, mode: str,
              condition: SensorCondition, seed: int,
              estimator_factory: Callable[[str], object] | None = None) -> TrialResult:
    """At t_i: sample prior-wrench IMU, update estimator, command, then integrate.

    The reading at t_i reflects the state at t_i and the command held during
    [t_(i-1), t_i). At t=0 the declared pre-run wrench is zero. The first
    estimator reading initializes time and orientation. Controller runs at each
    0.01 s plant step; a 50 Hz estimate is held at intervening steps.
    """
    if mode not in MODES or seed < 0:
        raise ValueError("unsupported feedback mode or seed")
    base = config.imu_experiment.base
    dt = base.demo.step_s
    steps = round(scenario.duration_s / dt)
    sensor_config = scaled_imu(config.imu_experiment.imu, condition, seed)
    sensor = SimulatedImu(sensor_config, dt)
    controller = CascadedHoverController(base.vehicle, base.controller)
    estimator = (estimator_factory(mode) if estimator_factory else
                 MadgwickImu(config.madgwick) if mode == "madgwick" else
                 QuaternionUkf(config.ukf) if mode == "ukf" else None)
    state = VehicleState.at_rest()
    estimate = (state.quaternion_bw.copy() if mode == "truth" else
                config.madgwick.initial_quaternion_bw.copy() if mode == "madgwick" else
                config.ukf.initial_quaternion_bw.copy())
    prior_wrench = Wrench(0.0, np.zeros(3))
    control_rows: dict[str, list] = {name: [] for name in (
        "timestamp_s", "position_m", "velocity_m_s", "true_quaternion_bw",
        "body_rate_rad_s", "desired_position_m", "desired_euler_rad",
        "true_attitude_tracking_error_deg", "attitude_estimation_geodesic_error_deg",
        "position_error_m", "true_tilt_deg",
        "applied_thrust_n", "applied_torque_nm", "thrust_saturated",
        "torque_saturated", "desired_tilt_saturated")}
    estimate_rows: dict[str, list] = {name: [] for name in (
        "timestamp_s", "quaternion_bw", "euler_deg",
        "measurement_sampled", "update_status", "covariance_rad2",
        "innovation_norm", "innovation_condition", "covariance_condition",
        "covariance_min_eigenvalue_rad2", "numerical_corrections")}
    sensor_rows: dict[str, list] = {name: [] for name in (
        "timestamp_s", "accel_m_s2", "gyro_rad_s")}
    sensor_truth_rows: dict[str, list] = {name: [] for name in (
        "ideal_specific_force_body_m_s2", "ideal_body_rate_rad_s",
        "accel_noise_innovation_m_s2", "gyro_noise_innovation_rad_s",
        "source_thrust_n", "source_torque_nm")}
    runtime_s = 0.0
    reason: str | None = None
    failure_time_s: float | None = None
    last_status = "truth" if mode == "truth" else "initial_hold"
    covariance = np.full((3, 3), np.nan)
    innovation = float("nan")
    innovation_condition = float("nan")
    covariance_condition = float("nan")
    covariance_min_eigenvalue = float("nan")
    numerical_corrections = 0
    for index in range(steps + 1):
        time_s = index * dt
        if not _finite_state(state):
            reason, failure_time_s = "nonfinite_plant_state", time_s
            break
        target = target_at(scenario, time_s)
        sampled = sensor.should_sample(index)
        if sampled:
            ideal_force, ideal_rate = ideal_imu(state, prior_wrench, base.vehicle)
            reading = sensor.sample(index, state, prior_wrench, base.vehicle)
            sensor_rows["timestamp_s"].append(reading.timestamp_s)
            sensor_rows["accel_m_s2"].append(reading.accel_m_s2.copy())
            sensor_rows["gyro_rad_s"].append(reading.gyro_rad_s.copy())
            sensor_truth_rows["ideal_specific_force_body_m_s2"].append(ideal_force)
            sensor_truth_rows["ideal_body_rate_rad_s"].append(ideal_rate)
            sensor_truth_rows["accel_noise_innovation_m_s2"].append(
                reading.accel_m_s2 - ideal_force - sensor_config.accel_bias_m_s2)
            sensor_truth_rows["gyro_noise_innovation_rad_s"].append(
                reading.gyro_rad_s - ideal_rate - sensor_config.gyro_bias_rad_s)
            sensor_truth_rows["source_thrust_n"].append(prior_wrench.thrust_n)
            sensor_truth_rows["source_torque_nm"].append(prior_wrench.torque_nm.copy())
            if estimator is not None:
                start = perf_counter()
                try:
                    update = estimator.update(reading)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    reason, failure_time_s = "estimator_update_error:" + type(error).__name__, time_s
                    break
                runtime_s += perf_counter() - start
                estimate = np.asarray(update.quaternion_bw, dtype=float)
                last_status = update.accel_status
                if mode == "ukf":
                    covariance = update.covariance_rad2.copy()
                    innovation = update.innovation_norm
                    innovation_condition = update.innovation_condition
                    covariance_condition = update.covariance_condition
                    covariance_min_eigenvalue = update.covariance_min_eigenvalue_rad2
                    numerical_corrections = update.numerical_corrections
            else:
                last_status = "truth"
        else:
            last_status = "held" if mode != "truth" else "truth"
        if mode == "truth":
            estimate = state.quaternion_bw.copy()
        try:
            feedback = state if mode == "truth" else feedback_state(state, estimate)
            control = controller.command(feedback, target.position_m, target.yaw_rad)
        except ValueError as error:
            reason, failure_time_s = str(error), time_s
            break
        desired_q = euler_to_quaternion(*control.desired_euler_rad)
        attitude_error = geodesic_error_deg(estimate[None, :], state.quaternion_bw[None, :])[0]
        true_tracking_error = geodesic_error_deg(desired_q[None, :], state.quaternion_bw[None, :])[0]
        tilt = _tilt_deg(state.quaternion_bw)
        position_error = float(np.linalg.norm(state.position_m - target.position_m))
        wrench = control.wrench
        thrust_sat = (wrench.thrust_n <= 1e-12 or
                      wrench.thrust_n >= base.controller.max_total_thrust_n - 1e-12)
        torque_sat = bool(np.any(np.abs(wrench.torque_nm)
                                 >= base.controller.max_torque_nm - 1e-12))
        desired_tilt_sat = (_tilt_deg(desired_q) >= base.controller.max_tilt_deg - 1e-8)
        for name, value in (
            ("timestamp_s", time_s), ("position_m", state.position_m.copy()),
            ("velocity_m_s", state.velocity_m_s.copy()),
            ("true_quaternion_bw", state.quaternion_bw.copy()),
            ("body_rate_rad_s", state.body_rate_rad_s.copy()),
            ("desired_position_m", target.position_m.copy()),
            ("desired_euler_rad", control.desired_euler_rad.copy()),
            ("true_attitude_tracking_error_deg", true_tracking_error),
            ("attitude_estimation_geodesic_error_deg", attitude_error),
            ("position_error_m", position_error), ("true_tilt_deg", tilt),
            ("applied_thrust_n", wrench.thrust_n),
            ("applied_torque_nm", wrench.torque_nm.copy()),
            ("thrust_saturated", thrust_sat), ("torque_saturated", torque_sat),
            ("desired_tilt_saturated", desired_tilt_sat)):
            control_rows[name].append(value)
        for name, value in (
            ("timestamp_s", time_s), ("quaternion_bw", feedback.quaternion_bw.copy()),
            ("euler_deg", np.rad2deg(quaternion_to_euler(feedback.quaternion_bw))),
            ("measurement_sampled", sampled),
            ("update_status", last_status), ("covariance_rad2", covariance.copy()),
            ("innovation_norm", innovation), ("innovation_condition", innovation_condition),
            ("covariance_condition", covariance_condition),
            ("covariance_min_eigenvalue_rad2", covariance_min_eigenvalue),
            ("numerical_corrections", numerical_corrections)):
            estimate_rows[name].append(value)
        reason = _limit_reason(state, target.position_m, config.limits)
        if reason is not None:
            failure_time_s = time_s
            break
        if index < steps:
            try:
                state = advance_state(state, wrench, base.vehicle, dt)
            except (ValueError, FloatingPointError) as error:
                reason, failure_time_s = "plant_update_error:" + type(error).__name__, time_s + dt
                break
            prior_wrench = wrench
    sensor_data = _arrays(sensor_rows)
    estimator_data = _arrays(estimate_rows)
    evaluation = _arrays(control_rows)
    evaluation.update(_arrays(sensor_truth_rows))
    if len(evaluation["timestamp_s"]):
        true_euler = np.rad2deg(np.array([quaternion_to_euler(q)
                                          for q in evaluation["true_quaternion_bw"]]))
        estimated_euler = estimator_data["euler_deg"]
        tilt_errors = (estimated_euler[:, :2] - true_euler[:, :2] + 180) % 360 - 180
        relative_yaw = (np.rad2deg(np.unwrap(np.deg2rad(estimated_euler[:, 2])))
                        - np.rad2deg(np.unwrap(np.deg2rad(true_euler[:, 2]))))
        relative_yaw -= relative_yaw[0]
        evaluation["true_euler_deg"] = true_euler
        evaluation["roll_pitch_estimation_error_deg"] = tilt_errors
        evaluation["relative_yaw_estimation_error_deg"] = relative_yaw
    else:
        evaluation["true_euler_deg"] = np.empty((0, 3))
        evaluation["roll_pitch_estimation_error_deg"] = np.empty((0, 2))
        evaluation["relative_yaw_estimation_error_deg"] = np.empty(0)
    metrics = _metrics(evaluation, estimator_data, sensor_data, scenario, mode,
                       condition, seed, reason, failure_time_s, runtime_s, dt,
                       config.limits)
    metadata = {
        "scenario": scenario.name, "mode": mode, "condition": condition.name,
        "seed": seed, "duration_requested_s": scenario.duration_s,
        "plant_step_s": dt, "controller_hz": config.controller_hz,
        "imu_hz": sensor_config.sample_hz,
        "timing": "sample at t_i from true state and wrench held on preceding interval; update estimator; command; integrate [t_i,t_(i+1)); t=0 pre-run wrench is zero",
        "feedback": "truth position, truth translational velocity, selected attitude, truth body angular rate",
        "initial_attitude": "declared level identity for all arms; no later truth resets",
        "initial_physical_state": "position, velocity and body rate zero; body-to-world quaternion identity",
        "sensor_only": "sensor.npz contains timestamp, measured body specific force and body rate",
        "estimator_only": "estimator.npz contains sensor-derived attitude, covariance and update diagnostics; no truth-scored error",
        "evaluation_only": "evaluation.npz contains true state, truth-scored errors, desired reference, applied wrench and paired innovations",
        "noise_model": "independent Gaussian standard deviations per IMU sample; same seed and draw order across arms",
        "bias_scale": condition.bias_scale, "noise_scale": condition.noise_scale,
        "vehicle_config": _json_ready(asdict(base.vehicle)),
        "controller_config": _json_ready(asdict(base.controller)),
        "madgwick_config": _json_ready(asdict(config.madgwick)),
        "ukf_config": _json_ready(asdict(config.ukf)),
        "accel_bias_m_s2": sensor_config.accel_bias_m_s2.tolist(),
        "gyro_bias_rad_s": sensor_config.gyro_bias_rad_s.tolist(),
        "accel_noise_std_m_s2": sensor_config.accel_noise_std_m_s2.tolist(),
        "gyro_noise_std_rad_s": sensor_config.gyro_noise_std_rad_s.tolist(),
        "waypoints": [{"time_s": wp.time_s, "position_m": wp.position_m.tolist(),
                       "yaw_rad": wp.yaw_rad} for wp in scenario.waypoints],
        "failure_limits": vars(config.limits),
    }
    return TrialResult(metrics, metadata, sensor_data, estimator_data, evaluation)


def save_trial(trial: TrialResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "sensor.npz", **trial.sensor)
    np.savez_compressed(directory / "estimator.npz", **trial.estimator)
    np.savez_compressed(directory / "evaluation.npz", **trial.evaluation)
    (directory / "metrics.json").write_text(json.dumps(trial.metrics, indent=2) + "\n", encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(trial.metadata, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_sensitivity(trials: list[dict]) -> list[dict]:
    """Keep failed/terminated trials visible; compare RMSE only for full horizons."""
    groups = sorted({(row["scenario"], row["condition"], row["mode"]) for row in trials})
    result = []
    for scenario, condition, mode in groups:
        members = [row for row in trials if (row["scenario"], row["condition"], row["mode"])
                   == (scenario, condition, mode)]
        complete = [row for row in members if row["completed_horizon"]]
        row = {"scenario": scenario, "condition": condition, "mode": mode,
               "trials": len(members), "successes": sum(item["success"] for item in members),
               "failures": sum(not item["success"] for item in members),
               "early_terminated": sum(not item["completed_horizon"] for item in members),
               "full_horizon_trials": len(complete)}
        for field in ("attitude_geodesic_rmse_deg", "position_rmse_m",
                      "max_position_error_m", "max_thrust_n", "max_torque_norm_nm"):
            values = [item[field] for item in complete]
            row[field + "_full_horizon_mean"] = float(np.mean(values)) if values else None
            row[field + "_full_horizon_std"] = float(np.std(values)) if values else None
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/closed_loop_estimation.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/closed_loop_estimation"))
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nominal = next(item for item in config.conditions if item.name == "nominal")
    main_trials: dict[str, dict[str, TrialResult]] = {}
    main_rows = []
    for scenario in config.scenarios:
        main_trials[scenario.name] = {}
        for mode in config.modes:
            result = run_trial(config, scenario, mode, nominal, config.nominal_seed)
            main_trials[scenario.name][mode] = result
            save_trial(result, args.output_dir / "primary" / scenario.name / mode)
            main_rows.append(result.metrics)
            print(f"primary {scenario.name} {mode}: position RMSE {result.metrics['position_rmse_m']:.4f} m; attitude RMSE {result.metrics['attitude_geodesic_rmse_deg']:.3f} deg; success {result.metrics['success']}", flush=True)
    from aerial_autonomy.visualization.closed_loop_plot import (
        plot_aggregate, plot_error_relationship, plot_scenario,
    )
    for scenario in config.scenarios:
        plot_scenario(args.output_dir / "primary" / scenario.name,
                      scenario, main_trials[scenario.name])
    _csv(args.output_dir / "primary_summary.csv", main_rows)
    (args.output_dir / "primary_summary.json").write_text(
        json.dumps(main_rows, indent=2) + "\n", encoding="utf-8")
    scenario_by_name = {scenario.name: scenario for scenario in config.scenarios}
    sensitivity_rows = []
    for scenario_name in config.sensitivity_scenarios:
        scenario = scenario_by_name[scenario_name]
        for condition in config.conditions:
            for seed in config.sensitivity_seeds:
                for mode in config.modes:
                    result = run_trial(config, scenario, mode, condition, seed)
                    save_trial(result, args.output_dir / "sensitivity" / scenario.name /
                               condition.name / f"seed_{seed}" / mode)
                    sensitivity_rows.append(result.metrics)
            print(f"sensitivity {scenario_name} {condition.name}: {len(config.sensitivity_seeds) * len(config.modes)} paired runs", flush=True)
    aggregate = summarize_sensitivity(sensitivity_rows)
    _csv(args.output_dir / "sensitivity_trials.csv", sensitivity_rows)
    _csv(args.output_dir / "sensitivity_aggregate.csv", aggregate)
    (args.output_dir / "sensitivity_trials.json").write_text(
        json.dumps(sensitivity_rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "sensitivity_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    plot_aggregate(args.output_dir / "sensitivity_aggregate.png", aggregate)
    plot_error_relationship(args.output_dir / "attitude_vs_tracking.png", sensitivity_rows)
    print(f"Saved {args.output_dir.resolve()}; primary={len(main_rows)} sensitivity={len(sensitivity_rows)}", flush=True)


if __name__ == "__main__":
    main()
