"""Shared-input M4 quaternion UKF, Madgwick, and gyro-only evaluation."""

import argparse
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from time import perf_counter
import tomllib

import numpy as np

from aerial_autonomy.config import load_imu_experiment
from aerial_autonomy.estimation.quaternion_ukf import QuaternionUkf, QuaternionUkfConfig
from aerial_autonomy.evaluation.imu import simulate_imu_scenario
from aerial_autonomy.evaluation.madgwick import (
    FilterRun, evaluate_run, load_aligned_truth, load_experiment_config,
    load_measurements, run_filter,
)
from aerial_autonomy.logging_setup import configure_logging
from aerial_autonomy.sensors.imu import ImuMeasurement, specific_force_body
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, integrate_quaternion, quaternion_to_euler,
    quaternion_to_matrix,
)
from aerial_autonomy.visualization.attitude_plot import save_attitude_plot
from aerial_autonomy.visualization.ukf_plot import (
    save_covariance_plot, save_sensitivity_plot,
)


LOGGER = logging.getLogger(__name__)
Q_FIELDS = ("quaternion_bw_w", "quaternion_bw_x", "quaternion_bw_y", "quaternion_bw_z")


@dataclass(frozen=True)
class UkfTrace:
    filter_run: FilterRun
    covariance_rad2: np.ndarray
    heading_std_deg: np.ndarray
    minimum_eigenvalue_rad2: np.ndarray
    covariance_symmetry_residual: np.ndarray
    covariance_condition: np.ndarray
    innovation_norm: np.ndarray
    innovation_condition: np.ndarray
    mean_iterations: np.ndarray
    sigma_max_norm_deviation: np.ndarray
    numerical_corrections: np.ndarray
    max_correction_norm_rad2: np.ndarray


def run_ukf(readings: list[ImuMeasurement], config: QuaternionUkfConfig) -> UkfTrace:
    """Consume only IMU records and declared parameters; no truth argument."""
    if not readings:
        raise ValueError("At least one IMU record is required")
    estimator = QuaternionUkf(config)
    updates = []
    start = perf_counter()
    for reading in readings:
        updates.append(estimator.update(reading))
    runtime = perf_counter() - start
    quaternions = np.array([item.quaternion_bw for item in updates])
    covariance = np.array([item.covariance_rad2 for item in updates])
    status = tuple(item.accel_status for item in updates)
    filter_run = FilterRun(
        time_s=np.array([item.timestamp_s for item in updates]),
        quaternion_bw=quaternions,
        euler_deg=np.rad2deg(np.array([quaternion_to_euler(q) for q in quaternions])),
        accel_norm_m_s2=np.array([np.linalg.norm(item.accel_m_s2) for item in readings]),
        gyro_norm_rad_s=np.array([np.linalg.norm(item.gyro_rad_s) for item in readings]),
        correction_used=np.array([state == "valid" for state in status]),
        accel_status=status,
        runtime_s=runtime,
    )
    heading_std = []
    for q, P in zip(quaternions, covariance):
        local_gravity_axis = quaternion_to_matrix(q).T @ [0, 0, 1]
        heading_std.append(np.rad2deg(np.sqrt(max(
            local_gravity_axis @ P @ local_gravity_axis, 0.0))))
    return UkfTrace(
        filter_run, covariance, np.asarray(heading_std),
        np.array([item.covariance_min_eigenvalue_rad2 for item in updates]),
        np.array([item.covariance_symmetry_residual for item in updates]),
        np.array([item.covariance_condition for item in updates]),
        np.array([item.innovation_norm for item in updates]),
        np.array([item.innovation_condition for item in updates]),
        np.array([item.mean_iterations for item in updates]),
        np.array([item.sigma_max_norm_deviation for item in updates]),
        np.array([item.numerical_corrections for item in updates]),
        np.array([item.max_correction_norm_rad2 for item in updates]),
    )


def score_ukf(trace: UkfTrace, truth_q: np.ndarray) -> tuple[dict, dict]:
    metrics, series = evaluate_run(trace.filter_run, truth_q)
    valid_innovation = trace.innovation_condition[np.isfinite(trace.innovation_condition)]
    metrics.update({
        "covariance_units": "rad^2 for a body-local rotation vector",
        "covariance_min_eigenvalue_rad2": float(np.min(trace.minimum_eigenvalue_rad2)),
        "covariance_max_symmetry_residual": float(np.max(trace.covariance_symmetry_residual)),
        "covariance_max_condition": float(np.max(trace.covariance_condition)),
        "sigma_max_quaternion_norm_deviation": float(np.max(trace.sigma_max_norm_deviation)),
        "quaternion_mean_max_iterations": int(np.max(trace.mean_iterations)),
        "innovation_max_norm": float(np.max(trace.innovation_norm)),
        "innovation_covariance_max_condition": float(np.max(valid_innovation)) if len(valid_innovation) else None,
        "accelerometer_updates_used": int(sum(s == "valid" for s in trace.filter_run.accel_status)),
        "accelerometer_updates_rejected": int(sum(s in ("nonfinite", "near_zero")
                                                  for s in trace.filter_run.accel_status)),
        "numerical_covariance_corrections": int(trace.numerical_corrections[-1]),
        "max_covariance_correction_norm_rad2": float(trace.max_correction_norm_rad2[-1]),
        "heading_std_initial_deg": float(trace.heading_std_deg[0]),
        "heading_std_final_deg": float(trace.heading_std_deg[-1]),
        "final_roll_deg": float(trace.filter_run.euler_deg[-1, 0]),
        "final_pitch_deg": float(trace.filter_run.euler_deg[-1, 1]),
    })
    return metrics, series


def _write_csv(path: Path, columns: list[str], values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",", header=",".join(columns), comments="", fmt="%.12g")


def _save_estimate(directory: Path, prefix: str, run: FilterRun) -> None:
    _write_csv(directory / f"{prefix}estimates.csv",
               ["timestamp_s", *Q_FIELDS, "roll_deg", "pitch_deg", "yaw_deg"],
               np.column_stack((run.time_s, run.quaternion_bw, run.euler_deg)))


def _save_error(directory: Path, prefix: str, run: FilterRun, series: dict) -> None:
    _write_csv(directory / f"{prefix}errors.csv",
               ["timestamp_s", "true_roll_deg", "true_pitch_deg", "true_yaw_deg",
                "geodesic_error_deg", "relative_yaw_error_deg", "quaternion_norm_deviation"],
               np.column_stack((run.time_s, series["true_euler_deg"],
                                series["geodesic_error_deg"], series["relative_yaw_error_deg"],
                                series["norm_error"])))


def save_case(directory: Path, title: str, readings: list[ImuMeasurement], truth_q: np.ndarray,
              ukf_config: QuaternionUkfConfig, madgwick_config, source: dict) -> dict:
    """Run every method before truth scoring and export separate data boundaries."""
    directory.mkdir(parents=True, exist_ok=True)
    ukf = run_ukf(readings, ukf_config)
    madgwick = run_filter(readings, madgwick_config)
    gyro_only = run_filter(readings, replace(madgwick_config, beta_per_s=0.0))
    scored = {}
    series = {}
    for name, run in (("ukf", ukf.filter_run), ("madgwick", madgwick), ("gyro_only", gyro_only)):
        scored[name], series[name] = (score_ukf(ukf, truth_q) if name == "ukf"
                                       else evaluate_run(run, truth_q))
        prefix = "" if name == "ukf" else f"{name}_"
        _save_estimate(directory, prefix, run)
        _save_error(directory, prefix, run, series[name])
        (directory / f"{prefix}metrics.json").write_text(
            json.dumps(scored[name], indent=2) + "\n", encoding="utf-8")
    run = ukf.filter_run
    _write_csv(directory / "diagnostics.csv",
               ["timestamp_s", "covariance_xx_rad2", "covariance_yy_rad2", "covariance_zz_rad2",
                "heading_std_deg", "minimum_eigenvalue_rad2", "symmetry_residual",
                "covariance_condition", "innovation_norm", "innovation_condition",
                "mean_iterations", "sigma_max_norm_deviation", "numerical_corrections"],
               np.column_stack((run.time_s, np.diagonal(ukf.covariance_rad2, axis1=1, axis2=2),
                                ukf.heading_std_deg, ukf.minimum_eigenvalue_rad2,
                                ukf.covariance_symmetry_residual, ukf.covariance_condition,
                                ukf.innovation_norm, ukf.innovation_condition,
                                ukf.mean_iterations, ukf.sigma_max_norm_deviation,
                                ukf.numerical_corrections)))
    metadata = {
        "source": source,
        "estimator_visible": "estimates.csv, comparator estimates and diagnostics.csv use only IMU data",
        "evaluation_only": "errors/metrics and plots use separate truth; no estimator receives it",
        "frames": "ENU world; FLU body; scalar-first Hamilton body-to-world quaternion",
        "initial_quaternion_bw": ukf_config.initial_quaternion_bw.tolist(),
        "ukf_config": {
            "initial_attitude_std_rad": ukf_config.initial_attitude_std_rad.tolist(),
            "gyro_noise_std_rad_s_per_sample": ukf_config.gyro_noise_std_rad_s.tolist(),
            "accel_direction_std_unitless_per_sample": ukf_config.accel_direction_std.tolist(),
            "alpha": ukf_config.alpha, "beta": ukf_config.beta, "kappa": ukf_config.kappa,
            "covariance_floor_rad2": ukf_config.covariance_floor_rad2,
        },
        "madgwick_beta_per_s": madgwick_config.beta_per_s,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_attitude_plot(directory / "comparison.png", title, run.time_s,
                       series["ukf"]["true_euler_deg"], run.euler_deg,
                       series["ukf"]["geodesic_error_deg"], series["ukf"]["norm_error"],
                       run.accel_norm_m_s2, run.gyro_norm_rad_s,
                       {"Madgwick": (madgwick.euler_deg, series["madgwick"]["geodesic_error_deg"]),
                        "gyro_only": (gyro_only.euler_deg, series["gyro_only"]["geodesic_error_deg"])},
                       primary_label="UKF")
    save_covariance_plot(directory / "covariance.png", run.time_s, ukf.covariance_rad2,
                         ukf.heading_std_deg, ukf.minimum_eigenvalue_rad2,
                         ukf.innovation_norm, ukf.covariance_condition)
    return scored


def _regular_grid(duration_s: float, sample_hz: float) -> np.ndarray:
    if not np.isfinite(duration_s * sample_hz) or duration_s <= 0 or sample_hz <= 0:
        raise ValueError("duration and sample rate must be positive and finite")
    steps = duration_s * sample_hz
    if not np.isclose(steps, round(steps), atol=1e-10, rtol=0):
        raise ValueError("duration must contain a whole number of sample periods")
    return np.arange(round(steps) + 1) / sample_hz


def run_recorded(name: str, input_root: Path, output_root: Path,
                 ukf_config: QuaternionUkfConfig, madgwick_config) -> dict:
    source_dir = input_root / name
    readings = load_measurements(source_dir / "measurements.csv")
    truth_q = load_aligned_truth(source_dir / "truth.csv",
                                 np.array([item.timestamp_s for item in readings]))
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    source = {"kind": "M2 plant/controller flight", "scenario": name,
              "measurement_file": str(source_dir / "measurements.csv"),
              "truth_file_evaluation_only": str(source_dir / "truth.csv"),
              "seed": source_metadata["imu"]["seed"]}
    return save_case(output_root / name, f"Shared IMU comparison: {name}", readings,
                     truth_q, ukf_config, madgwick_config, source)


def run_static_tilt(output_root: Path, ukf_config: QuaternionUkfConfig,
                    madgwick_config, settings: dict) -> dict:
    times = _regular_grid(float(settings["duration_s"]), float(settings["sample_hz"]))
    truth = euler_to_quaternion(np.deg2rad(settings["roll_deg"]),
                                np.deg2rad(settings["pitch_deg"]), 0.0)
    truth_q = np.tile(truth, (len(times), 1))
    ideal = specific_force_body(truth, np.zeros(3), madgwick_config.gravity_m_s2)
    readings = [ImuMeasurement(float(t), ideal.copy(), np.zeros(3)) for t in times]
    source = {"kind": "analytical externally supported static tilt; not powered flight",
              "roll_deg": settings["roll_deg"], "pitch_deg": settings["pitch_deg"],
              "duration_s": settings["duration_s"], "sample_hz": settings["sample_hz"]}
    results = save_case(output_root / "static_tilt", "Externally supported static tilt",
                        readings, truth_q, ukf_config, madgwick_config, source)
    results["ukf"]["final_roll_error_deg"] = float(
        results["ukf"]["final_roll_deg"] - settings["roll_deg"])
    results["ukf"]["final_pitch_error_deg"] = float(
        results["ukf"]["final_pitch_deg"] - settings["pitch_deg"])
    (output_root / "static_tilt" / "metrics.json").write_text(
        json.dumps(results["ukf"], indent=2) + "\n", encoding="utf-8")
    return results


def run_rotation(name: str, output_root: Path, ukf_config: QuaternionUkfConfig,
                 madgwick_config, duration_s: float, sample_hz: float,
                 body_rate_rad_s: np.ndarray) -> dict:
    times = _regular_grid(duration_s, sample_hz)
    rate = np.asarray(body_rate_rad_s, dtype=float)
    if rate.shape != (3,) or not np.all(np.isfinite(rate)):
        raise ValueError("body rate must be a finite three-vector")
    truth_q = np.array([integrate_quaternion(ukf_config.initial_quaternion_bw, rate, t)
                        for t in times])
    readings = [ImuMeasurement(float(t), specific_force_body(
        q, np.zeros(3), madgwick_config.gravity_m_s2),
                               rate.copy()) for t, q in zip(times, truth_q)]
    source = {"kind": "analytical constant body-rate orientation; not plant flight",
              "initial_quaternion_bw": ukf_config.initial_quaternion_bw.tolist(),
              "body_rate_rad_s": rate.tolist(), "duration_s": duration_s,
              "sample_hz": sample_hz}
    return save_case(output_root / name, f"Analytical rotation: {name}", readings,
                     truth_q, ukf_config, madgwick_config, source)


def run_sensitivity(output_root: Path, ukf_config: QuaternionUkfConfig,
                    madgwick_config, imu_path: Path, cases: list[dict], seeds: list[int]) -> dict:
    """Thirty predetermined shared-input trials; no truth enters any filter."""
    base = load_imu_experiment(imu_path)
    rows = []
    for case in cases:
        for seed in seeds:
            imu = replace(base.imu, seed=seed,
                          accel_bias_m_s2=base.imu.accel_bias_m_s2 * case["bias_scale"],
                          gyro_bias_rad_s=base.imu.gyro_bias_rad_s * case["bias_scale"],
                          accel_noise_std_m_s2=base.imu.accel_noise_std_m_s2 * case["noise_scale"],
                          gyro_noise_std_rad_s=base.imu.gyro_noise_std_rad_s * case["noise_scale"])
            experiment = replace(base, imu=imu)
            for scenario in ("hover", "lateral_waypoint"):
                trace = simulate_imu_scenario(experiment, scenario)
                readings = [ImuMeasurement(float(t), a.copy(), g.copy())
                            for t, a, g in zip(trace.measurement_time_s,
                                               trace.measured_accel_m_s2,
                                               trace.measured_gyro_rad_s)]
                ukf = run_ukf(readings, ukf_config)
                madgwick = run_filter(readings, madgwick_config)
                gyro_only = run_filter(readings, replace(madgwick_config, beta_per_s=0.0))
                truth = trace.quaternion_bw[trace.sample_indices]
                row = {"case": case["name"], "seed": seed, "scenario": scenario,
                       "bias_scale": case["bias_scale"], "noise_scale": case["noise_scale"]}
                for name, run in (("ukf", ukf.filter_run), ("madgwick", madgwick),
                                  ("gyro_only", gyro_only)):
                    scored, _ = evaluate_run(run, truth)
                    for metric in ("roll_rmse_deg", "pitch_rmse_deg",
                                   "relative_yaw_final_error_deg", "geodesic_rmse_deg"):
                        row[f"{name}_{metric}"] = scored[metric]
                rows.append(row)
        LOGGER.info("sensitivity case %s complete", case["name"])
    aggregate = []
    for case in cases:
        for scenario in ("hover", "lateral_waypoint"):
            group = [row for row in rows if row["case"] == case["name"]
                     and row["scenario"] == scenario]
            summary = {"case": case["name"], "scenario": scenario, "trials": len(group)}
            for name in ("ukf", "madgwick", "gyro_only"):
                for metric in ("roll_rmse_deg", "pitch_rmse_deg",
                               "relative_yaw_final_error_deg", "geodesic_rmse_deg"):
                    values = np.array([row[f"{name}_{metric}"] for row in group])
                    summary[f"{name}_{metric}_mean"] = float(np.mean(values))
                    summary[f"{name}_{metric}_std"] = float(np.std(values))
            aggregate.append(summary)
    result = {"source_imu_config": str(imu_path), "seeds": seeds, "cases": cases,
              "ukf_filter": {"alpha": ukf_config.alpha, "beta": ukf_config.beta,
                             "kappa": ukf_config.kappa,
                             "initial_attitude_std_rad": ukf_config.initial_attitude_std_rad.tolist(),
                             "gyro_noise_std_rad_s_per_sample": ukf_config.gyro_noise_std_rad_s.tolist(),
                             "accel_direction_std": ukf_config.accel_direction_std.tolist()},
              "madgwick_beta_per_s": madgwick_config.beta_per_s,
              "individual_trials": rows, "aggregate": aggregate,
              "metric_window": "all samples; shared sensor sequence and declared initial orientation"}
    (output_root / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_sensitivity_plot(output_root / "sensitivity.png", aggregate)
    return result


def load_config(path: Path) -> tuple[QuaternionUkfConfig, dict]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    f = raw["filter"]
    config = QuaternionUkfConfig(
        initial_quaternion_bw=np.asarray(f["initial_quaternion_bw"], dtype=float),
        initial_attitude_std_rad=np.deg2rad(f["initial_attitude_std_deg"]),
        gyro_noise_std_rad_s=np.asarray(f["gyro_noise_std_rad_s"], dtype=float),
        accel_direction_std=np.asarray(f["accel_direction_std"], dtype=float),
        alpha=float(f["alpha"]), beta=float(f["beta"]), kappa=float(f["kappa"]),
        covariance_floor_rad2=float(f["covariance_floor_rad2"]),
        mean_tolerance_rad=float(f["mean_tolerance_rad"]),
        mean_max_iterations=int(f["mean_max_iterations"]),
    )
    return config, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ukf.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ukf"))
    args = parser.parse_args()
    configure_logging()
    ukf_config, raw = load_config(args.config)
    madgwick_config, m3_raw = load_experiment_config(Path(raw["evaluation"]["madgwick_config"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("hover", "lateral_waypoint"):
        scored = run_recorded(name, Path(raw["evaluation"]["input_root"]),
                              args.output_dir, ukf_config, madgwick_config)
        summary[name] = scored
        LOGGER.info("%s: UKF %.3f, Madgwick %.3f, gyro-only %.3f deg geodesic RMSE",
                    name, *(scored[method]["geodesic_rmse_deg"]
                            for method in ("ukf", "madgwick", "gyro_only")))
    summary["static_tilt"] = run_static_tilt(
        args.output_dir, ukf_config, madgwick_config, m3_raw["static_tilt"])
    known = m3_raw["known_rotation"]
    summary["known_rotation"] = run_rotation(
        "known_rotation", args.output_dir, ukf_config, madgwick_config,
        float(known["duration_s"]), float(known["sample_hz"]),
        np.asarray(known["body_rate_rad_s"], dtype=float))
    nonlevel = raw["nonlevel_rotation"]
    q0 = euler_to_quaternion(*np.deg2rad(nonlevel["initial_roll_pitch_yaw_deg"]))
    summary["nonlevel_rotation"] = run_rotation(
        "nonlevel_rotation", args.output_dir,
        replace(ukf_config, initial_quaternion_bw=q0),
        replace(madgwick_config, initial_quaternion_bw=q0),
        float(nonlevel["duration_s"]), float(nonlevel["sample_hz"]),
        np.asarray(nonlevel["body_rate_rad_s"], dtype=float))
    (args.output_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sweep = m3_raw["sensitivity"]
    sensitivity = run_sensitivity(args.output_dir, ukf_config, madgwick_config,
                                  Path(raw["evaluation"]["imu_config"]),
                                  sweep["cases"], sweep["seeds"])
    LOGGER.info("sensitivity: %d shared-input trials", len(sensitivity["individual_trials"]))
    LOGGER.info("Saved %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
