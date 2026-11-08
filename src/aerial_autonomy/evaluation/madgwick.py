"""Evaluate a six-axis Madgwick filter against separate A2 truth files."""

import argparse
import csv
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from time import perf_counter
import tomllib

import numpy as np

from aerial_autonomy.config import load_imu_experiment
from aerial_autonomy.estimation.madgwick import MadgwickConfig, MadgwickImu
from aerial_autonomy.evaluation.imu import simulate_imu_scenario
from aerial_autonomy.logging_setup import configure_logging
from aerial_autonomy.sensors.imu import ImuMeasurement, specific_force_body
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, integrate_quaternion, quaternion_to_euler, quaternion_to_matrix,
)
from aerial_autonomy.visualization.attitude_plot import save_attitude_plot, save_sensitivity_plot


LOGGER = logging.getLogger(__name__)
MEASUREMENT_FIELDS = ("timestamp_s", "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                      "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
QUATERNION_FIELDS = ("quaternion_bw_w", "quaternion_bw_x", "quaternion_bw_y", "quaternion_bw_z")


@dataclass(frozen=True)
class FilterRun:
    time_s: np.ndarray
    quaternion_bw: np.ndarray
    euler_deg: np.ndarray
    accel_norm_m_s2: np.ndarray
    gyro_norm_rad_s: np.ndarray
    correction_used: np.ndarray
    accel_status: tuple[str, ...]
    runtime_s: float


def load_measurements(path: Path) -> list[ImuMeasurement]:
    """Read estimator-visible columns only; no truth path or column is used."""
    readings: list[ImuMeasurement] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MEASUREMENT_FIELDS:
            raise ValueError(f"Unexpected IMU measurement columns in {path}")
        for row in reader:
            try:
                readings.append(ImuMeasurement(
                    timestamp_s=float(row["timestamp_s"]),
                    accel_m_s2=np.array([float(row[f"accel_{axis}_m_s2"]) for axis in "xyz"]),
                    gyro_rad_s=np.array([float(row[f"gyro_{axis}_rad_s"]) for axis in "xyz"]),
                ))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid IMU row {reader.line_num} in {path}") from error
    if not readings:
        raise ValueError(f"No IMU measurements in {path}")
    return readings


def run_filter(readings: list[ImuMeasurement], config: MadgwickConfig) -> FilterRun:
    """Filter sensor readings alone; this function has no truth argument."""
    if not readings:
        raise ValueError("At least one IMU reading is required")
    estimator = MadgwickImu(config)
    quaternions = []
    norms = []
    corrections = []
    statuses = []
    start = perf_counter()
    for reading in readings:
        update = estimator.update(reading)
        quaternions.append(update.quaternion_bw)
        norms.append(update.accel_norm_m_s2)
        corrections.append(update.correction_used)
        statuses.append(update.accel_status)
    runtime_s = perf_counter() - start
    q = np.asarray(quaternions)
    return FilterRun(
        time_s=np.array([item.timestamp_s for item in readings]),
        quaternion_bw=q,
        euler_deg=np.rad2deg(np.array([quaternion_to_euler(item) for item in q])),
        accel_norm_m_s2=np.asarray(norms),
        gyro_norm_rad_s=np.array([np.linalg.norm(item.gyro_rad_s) for item in readings]),
        correction_used=np.asarray(corrections, dtype=bool),
        accel_status=tuple(statuses),
        runtime_s=runtime_s,
    )


def load_aligned_truth(path: Path, sample_times_s: np.ndarray) -> np.ndarray:
    """Evaluation-only lookup of the body-to-world truth quaternion."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    table = np.atleast_1d(table)
    if not all(field in table.dtype.names for field in ("timestamp_s", *QUATERNION_FIELDS)):
        raise ValueError(f"Missing truth time or quaternion fields in {path}")
    truth_times = table["timestamp_s"]
    if np.any(np.diff(truth_times) <= 0):
        raise ValueError("Truth timestamps must increase strictly")
    indices = np.searchsorted(truth_times, sample_times_s)
    if np.any(indices >= len(truth_times)) or not np.allclose(
        truth_times[np.minimum(indices, len(truth_times) - 1)], sample_times_s,
        rtol=0, atol=1e-9,
    ):
        raise ValueError("IMU and truth timestamps do not align")
    return np.column_stack([table[field][indices] for field in QUATERNION_FIELDS])


def geodesic_error_deg(estimated_quaternion_bw: np.ndarray,
                       true_quaternion_bw: np.ndarray) -> np.ndarray:
    """Shortest q/-q invariant rotation angle for matching quaternion arrays."""
    estimate = np.asarray(estimated_quaternion_bw, dtype=float)
    truth = np.asarray(true_quaternion_bw, dtype=float)
    if estimate.shape != truth.shape or estimate.ndim != 2 or estimate.shape[1] != 4:
        raise ValueError("Quaternion arrays must have matching shape (samples, 4)")
    estimate_norm = np.linalg.norm(estimate, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    if (not np.all(np.isfinite(estimate_norm)) or not np.all(np.isfinite(truth_norm))
            or np.any(estimate_norm < 1e-12) or np.any(truth_norm < 1e-12)):
        raise ValueError("Quaternion arrays contain zero or nonfinite norms")
    dot = np.sum(estimate * truth, axis=1) / (estimate_norm * truth_norm)
    return np.rad2deg(2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def evaluate_run(run: FilterRun, true_quaternion_bw: np.ndarray) -> tuple[dict, dict]:
    """Compute all-sample errors, including the initial sample and transients."""
    if len(run.time_s) != len(true_quaternion_bw):
        raise ValueError("Truth and estimate lengths differ")
    true_euler = np.rad2deg(np.array([quaternion_to_euler(q) for q in true_quaternion_bw]))
    angle_error = geodesic_error_deg(run.quaternion_bw, true_quaternion_bw)
    roll_pitch_error = (run.euler_deg[:, :2] - true_euler[:, :2] + 180) % 360 - 180
    estimate_yaw = np.rad2deg(np.unwrap(np.deg2rad(run.euler_deg[:, 2])))
    truth_yaw = np.rad2deg(np.unwrap(np.deg2rad(true_euler[:, 2])))
    relative_yaw_error = (estimate_yaw - estimate_yaw[0]) - (truth_yaw - truth_yaw[0])
    norm_error = np.abs(np.linalg.norm(run.quaternion_bw, axis=1) - 1)
    metrics = {
        "samples": int(len(run.time_s)),
        "metric_window": "all samples, including initialization and any transients",
        "initial_geodesic_error_deg": float(angle_error[0]),
        "roll_rmse_deg": float(np.sqrt(np.mean(roll_pitch_error[:, 0] ** 2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(roll_pitch_error[:, 1] ** 2))),
        "relative_yaw_rmse_deg": float(np.sqrt(np.mean(relative_yaw_error ** 2))),
        "relative_yaw_final_error_deg": float(relative_yaw_error[-1]),
        "geodesic_rmse_deg": float(np.sqrt(np.mean(angle_error ** 2))),
        "geodesic_max_deg": float(np.max(angle_error)),
        "max_quaternion_norm_deviation": float(np.max(norm_error)),
        "correction_samples": int(np.count_nonzero(run.correction_used)),
        "accelerometer_fallback_samples": int(sum(status in ("nonfinite", "near_zero", "magnitude_gated")
                                                  for status in run.accel_status)),
        "runtime_s": run.runtime_s,
        "throughput_samples_per_s": float(len(run.time_s) / run.runtime_s),
    }
    series = {"true_euler_deg": true_euler, "geodesic_error_deg": angle_error,
              "relative_yaw_error_deg": relative_yaw_error, "norm_error": norm_error}
    return metrics, series


def _write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="", fmt="%.12g")


def _save_run(directory: Path, stem: str, run: FilterRun,
              truth_q: np.ndarray) -> tuple[dict, dict]:
    """Write a sensor-derived estimate and a separate truth-scored error trace."""
    metrics, series = evaluate_run(run, truth_q)
    _write_csv(directory / f"{stem}estimates.csv", ["timestamp_s", *QUATERNION_FIELDS,
                "roll_deg", "pitch_deg", "yaw_deg", "accel_norm_m_s2", "gyro_norm_rad_s",
                "correction_used"], np.column_stack((run.time_s, run.quaternion_bw,
                run.euler_deg, run.accel_norm_m_s2, run.gyro_norm_rad_s,
                run.correction_used.astype(int))))
    _write_csv(directory / f"{stem}errors.csv", ["timestamp_s", "true_roll_deg", "true_pitch_deg",
                "true_yaw_deg", "geodesic_error_deg", "relative_yaw_error_deg",
                "quaternion_norm_deviation"], np.column_stack((run.time_s,
                series["true_euler_deg"], series["geodesic_error_deg"],
                series["relative_yaw_error_deg"], series["norm_error"])))
    (directory / f"{stem}metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics, series


def save_case(directory: Path, title: str, run: FilterRun, truth_q: np.ndarray,
              config: MadgwickConfig, source: dict,
              comparators: dict[str, FilterRun] | None = None) -> dict:
    """Keep estimator-only trace separate from truth-based evaluation outputs."""
    directory.mkdir(parents=True, exist_ok=True)
    metrics, series = _save_run(directory, "", run, truth_q)
    comparator_plot = {}
    for label, other in (comparators or {}).items():
        if not np.array_equal(other.time_s, run.time_s):
            raise ValueError(f"Comparator {label} timestamps differ")
        _, other_series = _save_run(directory, f"{label}_", other, truth_q)
        comparator_plot[label] = (other.euler_deg, other_series["geodesic_error_deg"])
    metadata = {
        "source": source,
        "estimator_visible": "estimates.csv is derived only from IMU measurements and declared initial orientation",
        "evaluation_only": "errors/metrics files and comparison.png use separate truth; all estimates files are sensor-derived",
        "comparators": list((comparators or {}).keys()),
        "initial_quaternion_bw": config.initial_quaternion_bw.tolist(),
        "initial_yaw_convention": "zero yaw at first sample; relative yaw evaluated from each trace's first sample",
        "beta_per_s": config.beta_per_s,
        "accel_norm_gate_m_s2": config.accel_norm_gate_m_s2,
        "gravity_m_s2": config.gravity_m_s2,
        "frames": "ENU world, FLU body, scalar-first Hamilton body-to-world quaternion",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_attitude_plot(directory / "comparison.png", title, run.time_s,
                       series["true_euler_deg"], run.euler_deg,
                       series["geodesic_error_deg"], series["norm_error"],
                       run.accel_norm_m_s2, run.gyro_norm_rad_s, comparator_plot)
    return metrics


def run_recorded_case(name: str, input_root: Path, output_root: Path,
                      config: MadgwickConfig, diagnostic_gate_m_s2: float | None = None,
                      diagnostic_time_s: float = 2.88) -> dict:
    source_dir = input_root / name
    readings = load_measurements(source_dir / "measurements.csv")
    run = run_filter(readings, config)
    comparators = {"gyro_only": run_filter(readings, replace(config, beta_per_s=0.0))}
    if diagnostic_gate_m_s2 is not None:
        comparators["magnitude_gated"] = run_filter(
            readings, replace(config, accel_norm_gate_m_s2=diagnostic_gate_m_s2))
    truth_q = load_aligned_truth(source_dir / "truth.csv", run.time_s)
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    source = {"kind": "A2 plant/controller dataset", "scenario": name,
              "measurement_file": str(source_dir / "measurements.csv"),
              "truth_file_for_evaluation_only": str(source_dir / "truth.csv"),
              "imu_seed": source_metadata["imu"]["seed"]}
    directory = output_root / name
    metrics = save_case(directory, f"Madgwick vs plant truth: {name}",
                        run, truth_q, config, source, comparators)
    if name == "lateral_waypoint" and diagnostic_gate_m_s2 is not None:
        save_dynamic_acceleration_diagnostic(source_dir / "truth.csv", directory,
                                             readings, truth_q, comparators["magnitude_gated"],
                                             config.gravity_m_s2, diagnostic_gate_m_s2,
                                             diagnostic_time_s)
    return metrics


def save_dynamic_acceleration_diagnostic(truth_path: Path, directory: Path,
                                         readings: list[ImuMeasurement], truth_q: np.ndarray,
                                         gated: FilterRun, gravity_m_s2: float,
                                         threshold_m_s2: float, target_time_s: float) -> dict:
    """Offline truth diagnostic; the operational filters never receive these fields."""
    if not np.isfinite(target_time_s):
        raise ValueError("Diagnostic time must be finite")
    times = np.array([item.timestamp_s for item in readings])
    index = int(np.argmin(np.abs(times - target_time_s)))
    truth = np.atleast_1d(np.genfromtxt(truth_path, delimiter=",", names=True, dtype=float))
    if "timestamp_s" not in truth.dtype.names or not all(
        f"specific_force_body_{axis}_m_s2" in truth.dtype.names for axis in "xyz"
    ):
        raise ValueError("Diagnostic requires aligned A2 specific-force truth")
    truth_times = truth["timestamp_s"]
    aligned = np.searchsorted(truth_times, times[index])
    if aligned >= len(truth_times) or not np.isclose(truth_times[aligned], times[index], atol=1e-9):
        raise ValueError("Diagnostic truth timestamp does not align")
    ideal = np.array([truth[f"specific_force_body_{axis}_m_s2"][aligned] for axis in "xyz"])
    measured = readings[index].accel_m_s2
    gravity_reference = gravity_m_s2 * quaternion_to_matrix(truth_q[index]).T @ [0, 0, 1]
    cosine = np.dot(ideal, gravity_reference) / (np.linalg.norm(ideal) * gravity_m_s2)
    mismatch_deg = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    result = {
        "role": "offline truth diagnostic only; no truth or translational acceleration enters either estimator",
        "requested_time_s": target_time_s,
        "sample_time_s": float(times[index]),
        "true_pitch_deg": float(np.rad2deg(quaternion_to_euler(truth_q[index])[1])),
        "ideal_specific_force_body_m_s2": ideal.tolist(),
        "gravity_reference_body_m_s2": gravity_reference.tolist(),
        "measured_specific_force_body_m_s2": measured.tolist(),
        "ideal_direction_mismatch_deg": mismatch_deg,
        "ideal_norm_minus_g_m_s2": float(np.linalg.norm(ideal) - gravity_m_s2),
        "measured_norm_minus_g_m_s2": float(np.linalg.norm(measured) - gravity_m_s2),
        "magnitude_gate_threshold_m_s2": threshold_m_s2,
        "ideal_direction_mismatch_rejected_by_norm_gate": bool(
            abs(np.linalg.norm(ideal) - gravity_m_s2) > threshold_m_s2),
        "measured_sample_rejected_by_norm_gate": gated.accel_status[index] == "magnitude_gated",
        "measured_sample_gate_status": gated.accel_status[index],
    }
    (directory / "dynamic_acceleration_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_known_rotation(output_root: Path, config: MadgwickConfig,
                       duration_s: float, sample_hz: float,
                       body_rate_rad_s: np.ndarray) -> dict:
    """Analytical sensor trajectory, not a quadrotor plant simulation."""
    if duration_s <= 0 or sample_hz <= 0 or not np.isfinite(duration_s * sample_hz):
        raise ValueError("Known rotation duration and sample rate must be positive and finite")
    steps = duration_s * sample_hz
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-10):
        raise ValueError("Known rotation duration must contain a whole number of samples")
    rate = np.asarray(body_rate_rad_s, dtype=float)
    if rate.shape != (3,) or not np.all(np.isfinite(rate)):
        raise ValueError("Known body rate must be a finite three-vector")
    times = np.arange(round(steps) + 1) / sample_hz
    truth_q = np.array([integrate_quaternion(config.initial_quaternion_bw, rate, t) for t in times])
    readings = [ImuMeasurement(float(t),
                specific_force_body(q, np.zeros(3), config.gravity_m_s2), rate.copy())
                for t, q in zip(times, truth_q)]
    run = run_filter(readings, config)
    source = {"kind": "analytical rotation with ideal sensors; not generated by the quadrotor plant",
              "body_rate_rad_s": rate.tolist(), "duration_s": duration_s,
              "sample_hz": sample_hz, "seed": None}
    directory = output_root / "known_rotation"
    gyro_only = run_filter(readings, replace(config, beta_per_s=0.0))
    metrics = save_case(directory, "Analytical known rotation (ideal IMU)",
                        run, truth_q, config, source, {"gyro_only": gyro_only})
    return metrics


def run_static_tilt(output_root: Path, config: MadgwickConfig, duration_s: float,
                    sample_hz: float, roll_deg: float, pitch_deg: float) -> dict:
    """Evaluate an externally supported stationary IMU, not tilted rotor flight."""
    if duration_s <= 0 or sample_hz <= 0 or not np.isfinite(duration_s * sample_hz):
        raise ValueError("Static tilt duration and sample rate must be positive and finite")
    steps = duration_s * sample_hz
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-10):
        raise ValueError("Static tilt duration must contain a whole number of samples")
    if not np.all(np.isfinite([roll_deg, pitch_deg])):
        raise ValueError("Static tilt angles must be finite")
    times = np.arange(round(steps) + 1) / sample_hz
    tilted_q = euler_to_quaternion(np.deg2rad(roll_deg), np.deg2rad(pitch_deg), 0)
    truth_q = np.tile(tilted_q, (len(times), 1))
    ideal_accel = specific_force_body(tilted_q, np.zeros(3), config.gravity_m_s2)
    readings = [ImuMeasurement(float(t), ideal_accel.copy(), np.zeros(3)) for t in times]
    filtered = run_filter(readings, config)
    gyro_only = run_filter(readings, replace(config, beta_per_s=0.0))
    source = {"kind": "analytic externally supported stationary tilted IMU; not powered quadrotor flight",
              "roll_deg": roll_deg, "pitch_deg": pitch_deg, "duration_s": duration_s,
              "sample_hz": sample_hz, "seed": None}
    directory = output_root / "static_tilt"
    metrics = save_case(directory, "Externally supported static tilt (ideal IMU)",
                        filtered, truth_q, config, source, {"gyro_only": gyro_only})
    metrics["final_roll_error_deg"] = float(filtered.euler_deg[-1, 0] - roll_deg)
    metrics["final_pitch_error_deg"] = float(filtered.euler_deg[-1, 1] - pitch_deg)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_sensitivity(output_root: Path, config: MadgwickConfig, imu_config_path: Path,
                    cases: list[dict], seeds: list[int]) -> dict:
    """Small predetermined bias/noise sweep; truth is used only for scoring."""
    base = load_imu_experiment(imu_config_path)
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
                plant_trace = simulate_imu_scenario(experiment, scenario)
                readings = [ImuMeasurement(float(t), accel.copy(), gyro.copy()) for t, accel, gyro in zip(
                    plant_trace.measurement_time_s, plant_trace.measured_accel_m_s2,
                    plant_trace.measured_gyro_rad_s)]
                filtered = run_filter(readings, config)
                gyro_only = run_filter(readings, replace(config, beta_per_s=0.0))
                sampled_truth = plant_trace.quaternion_bw[plant_trace.sample_indices]
                scored, _ = evaluate_run(filtered, sampled_truth)
                gyro_scored, _ = evaluate_run(gyro_only, sampled_truth)
                rows.append({"case": case["name"], "scenario": scenario, "seed": seed,
                             "bias_scale": case["bias_scale"], "noise_scale": case["noise_scale"],
                             "roll_rmse_deg": scored["roll_rmse_deg"],
                             "pitch_rmse_deg": scored["pitch_rmse_deg"],
                             "relative_yaw_final_error_deg": scored["relative_yaw_final_error_deg"],
                             "geodesic_rmse_deg": scored["geodesic_rmse_deg"],
                             "gyro_only_geodesic_rmse_deg": gyro_scored["geodesic_rmse_deg"]})
    aggregate = []
    for case in cases:
        for scenario in ("hover", "lateral_waypoint"):
            group = [row for row in rows if row["case"] == case["name"] and row["scenario"] == scenario]
            summary = {"case": case["name"], "scenario": scenario, "trials": len(group)}
            for key in ("roll_rmse_deg", "pitch_rmse_deg", "relative_yaw_final_error_deg",
                        "geodesic_rmse_deg", "gyro_only_geodesic_rmse_deg"):
                values = np.array([row[key] for row in group])
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
            aggregate.append(summary)
    result = {"imu_config": str(imu_config_path),
              "base_imu": {
                  "sample_hz": base.imu.sample_hz,
                  "accel_bias_m_s2": base.imu.accel_bias_m_s2.tolist(),
                  "gyro_bias_rad_s": base.imu.gyro_bias_rad_s.tolist(),
                  "accel_noise_std_m_s2": base.imu.accel_noise_std_m_s2.tolist(),
                  "gyro_noise_std_rad_s": base.imu.gyro_noise_std_rad_s.tolist(),
                  "noise_standard_deviation_interpretation": "independent per sample",
              },
              "filter": {"beta_per_s": config.beta_per_s,
                         "initial_quaternion_bw": config.initial_quaternion_bw.tolist(),
                         "gravity_m_s2": config.gravity_m_s2,
                         "accel_norm_gate_m_s2": config.accel_norm_gate_m_s2},
              "seeds": seeds, "cases": cases,
              "individual_trials": rows, "aggregate": aggregate,
              "metric_window": "all samples; predetermined filter gain, no truth-based tuning"}
    (output_root / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_sensitivity_plot(output_root / "sensitivity.png", aggregate)
    return result


def load_experiment_config(path: Path) -> tuple[MadgwickConfig, dict]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    filter_fields = raw["filter"]
    setting = MadgwickConfig(
        beta_per_s=float(filter_fields["beta_per_s"]),
        initial_quaternion_bw=np.asarray(filter_fields["initial_quaternion_bw"], dtype=float),
        accel_norm_gate_m_s2=(float(filter_fields["accel_norm_gate_m_s2"])
                              if "accel_norm_gate_m_s2" in filter_fields else None),
        gravity_m_s2=float(filter_fields["gravity_m_s2"]),
    )
    return setting, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/madgwick.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/madgwick"))
    args = parser.parse_args()
    configure_logging()
    config, raw = load_experiment_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_root = Path(raw["evaluation"]["input_root"])
    gate = raw["evaluation"].get("diagnostic_gate_m_s2")
    diagnostic_gate = float(gate) if gate is not None else None
    diagnostic_time = float(raw["evaluation"].get("diagnostic_time_s", 2.88))
    for scenario in ("hover", "lateral_waypoint"):
        metrics = run_recorded_case(scenario, input_root, args.output_dir, config,
                                    diagnostic_gate, diagnostic_time)
        LOGGER.info("%s: roll RMSE %.3f deg, pitch RMSE %.3f deg, geodesic RMSE %.3f deg",
                    scenario, metrics["roll_rmse_deg"], metrics["pitch_rmse_deg"],
                    metrics["geodesic_rmse_deg"])
    tilt = raw["static_tilt"]
    static = run_static_tilt(args.output_dir, config, float(tilt["duration_s"]),
                             float(tilt["sample_hz"]), float(tilt["roll_deg"]),
                             float(tilt["pitch_deg"]))
    LOGGER.info("static_tilt: final roll error %.3f deg, pitch error %.3f deg",
                static["final_roll_error_deg"], static["final_pitch_error_deg"])
    rotation = raw["known_rotation"]
    known = run_known_rotation(args.output_dir, config, float(rotation["duration_s"]),
                               float(rotation["sample_hz"]),
                               np.asarray(rotation["body_rate_rad_s"], dtype=float))
    LOGGER.info("known_rotation: geodesic RMSE %.4f deg", known["geodesic_rmse_deg"])
    sweep = raw["sensitivity"]
    sensitivity = run_sensitivity(args.output_dir, config,
                                  Path(raw["evaluation"]["imu_config"]),
                                  sweep["cases"], sweep["seeds"])
    LOGGER.info("sensitivity: %d seeded trials", len(sensitivity["individual_trials"]))
    LOGGER.info("Saved %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
