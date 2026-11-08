"""Offline M4 yaw-prior, gravity-observation, and covariance diagnostics.

The operational QuaternionUkf and its original configuration are unchanged.
Only this evaluation module receives truth. Its gravity-reference oracle is
explicitly offline and is never an operational estimator input.
"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerial_autonomy.config import load_imu_experiment
from aerial_autonomy.estimation.quaternion_ukf import (
    QuaternionUkf, QuaternionUkfConfig, local_sigma_points, quaternion_sigma_mean,
    relative_rotation_vector, right_jacobian_so3, rotation_vector_to_quaternion,
    stabilize_covariance, unscented_weights,
)
from aerial_autonomy.evaluation.madgwick import (
    evaluate_run, geodesic_error_deg, load_aligned_truth, load_measurements,
)
from aerial_autonomy.evaluation.ukf import load_config, run_ukf
from aerial_autonomy.sensors.imu import ImuMeasurement
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, normalize_quaternion, quaternion_product, quaternion_to_euler,
    quaternion_to_matrix,
)


UP = np.array([0.0, 0.0, 1.0])


def gravity_direction(quaternion_bw: np.ndarray) -> np.ndarray:
    """Predicted body-frame up direction for a body-to-world quaternion."""
    return quaternion_to_matrix(quaternion_bw).T @ UP


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def gravity_right_jacobian(quaternion_bw: np.ndarray) -> np.ndarray:
    """dh(q Exp(delta))/d delta at zero: [h(q)]x, rank two."""
    return skew(gravity_direction(quaternion_bw))


def direction_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def wrapped_difference_deg(first_deg: float, second_deg: float) -> float:
    return float((first_deg - second_deg + 180.0) % 360.0 - 180.0)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot export an empty diagnostic trace")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurement_variants(readings: list[ImuMeasurement], truth_q: np.ndarray,
                         gravity_m_s2: float) -> dict[str, list[ImuMeasurement]]:
    """Two offline substitutions plus the untouched operational sequence."""
    if len(readings) != len(truth_q):
        raise ValueError("truth must align with IMU readings for offline diagnostics")
    return {
        "operational": readings,
        "gyro_prediction_only": [ImuMeasurement(
            sample.timestamp_s, np.zeros(3), sample.gyro_rad_s.copy())
            for sample in readings],
        "offline_truth_gravity_oracle": [ImuMeasurement(
            sample.timestamp_s, gravity_m_s2 * gravity_direction(q),
            sample.gyro_rad_s.copy()) for sample, q in zip(readings, truth_q)],
    }


def run_variants(readings: list[ImuMeasurement], truth_q: np.ndarray,
                 config: QuaternionUkfConfig, gravity_m_s2: float) -> tuple[dict, dict]:
    metrics, traces = {}, {}
    for name, sequence in measurement_variants(readings, truth_q, gravity_m_s2).items():
        trace = run_ukf(sequence, config)
        scored, series = evaluate_run(trace.filter_run, truth_q)
        metrics[name] = {
            "geodesic_rmse_deg": scored["geodesic_rmse_deg"],
            "relative_yaw_final_error_deg": scored["relative_yaw_final_error_deg"],
            "roll_rmse_deg": scored["roll_rmse_deg"],
            "pitch_rmse_deg": scored["pitch_rmse_deg"],
            "accelerometer_updates_used": int(np.count_nonzero(trace.filter_run.correction_used)),
        }
        traces[name] = (trace, series)
    return metrics, traces


def run_yaw_prior_sweep(readings: list[ImuMeasurement], truth_q: np.ndarray,
                        config: QuaternionUkfConfig, priors_deg: list[float]) -> list[dict]:
    rows = []
    for prior in priors_deg:
        if not np.isfinite(prior) or prior <= 0:
            raise ValueError("yaw prior standard deviations must be finite and positive")
        std = config.initial_attitude_std_rad.copy()
        std[2] = np.deg2rad(prior)
        trace = run_ukf(readings, replace(config, initial_attitude_std_rad=std))
        metrics, _ = evaluate_run(trace.filter_run, truth_q)
        rows.append({
            "initial_yaw_std_deg": float(prior),
            "geodesic_rmse_deg": metrics["geodesic_rmse_deg"],
            "final_relative_yaw_error_deg": metrics["relative_yaw_final_error_deg"],
            "final_heading_projection_std_deg": float(trace.heading_std_deg[-1]),
            "max_covariance_condition": float(np.max(trace.covariance_condition)),
            "covariance_corrections": int(trace.numerical_corrections[-1]),
        })
    return rows


def audit_update(prior_q: np.ndarray, prior_P: np.ndarray, reading: ImuMeasurement,
                 dt_s: float, config: QuaternionUkfConfig) -> dict:
    """Reconstruct one UKF step independently of the filter's update method.

    Shared SO(3) primitives are reused; the sigma propagation, cross covariance,
    innovation, gain, and covariance reset are assembled again here and checked
    against the real filter output in ``lateral_trace``.
    """
    scale, wm, wc = unscented_weights(config.alpha, config.beta, config.kappa)
    offsets = local_sigma_points(prior_P, scale)
    increment = rotation_vector_to_quaternion(reading.gyro_rad_s * dt_s)
    sigma = np.array([normalize_quaternion(quaternion_product(
        quaternion_product(prior_q, rotation_vector_to_quaternion(delta)), increment))
        for delta in offsets])
    prediction, errors, _ = quaternion_sigma_mean(
        sigma, wm, config.mean_tolerance_rad, config.mean_max_iterations)
    predicted_P = sum(weight * np.outer(error, error)
                      for weight, error in zip(wc, errors))
    predicted_P += np.diag((config.gyro_noise_std_rad_s * dt_s) ** 2)
    predicted_P, _ = stabilize_covariance(predicted_P, config.covariance_floor_rad2)
    measurement_offsets = local_sigma_points(predicted_P, scale)
    measurement_sigma = np.array([normalize_quaternion(quaternion_product(
        prediction, rotation_vector_to_quaternion(delta)))
        for delta in measurement_offsets])
    directions = np.array([gravity_direction(q) for q in measurement_sigma])
    predicted_direction = wm @ directions
    direction_errors = directions - predicted_direction
    S = np.diag(config.accel_direction_std ** 2)
    S += sum(weight * np.outer(delta, delta)
             for weight, delta in zip(wc, direction_errors))
    cross = sum(weight * np.outer(error, delta)
                for weight, error, delta in zip(wc, measurement_offsets, direction_errors))
    gain = np.linalg.solve(S, cross.T).T
    residual = reading.accel_m_s2 / np.linalg.norm(reading.accel_m_s2) - predicted_direction
    correction = gain @ residual
    posterior = normalize_quaternion(quaternion_product(
        prediction, rotation_vector_to_quaternion(correction)))
    local_P = predicted_P - gain @ S @ gain.T
    J = right_jacobian_so3(correction)
    posterior_P, _ = stabilize_covariance(J @ local_P @ J.T, config.covariance_floor_rad2)
    return {"prediction": prediction, "posterior": posterior,
            "predicted_P": predicted_P, "posterior_P": posterior_P,
            "predicted_direction": predicted_direction, "innovation": residual,
            "S": S, "gain": gain, "correction": correction}


def lateral_trace(readings: list[ImuMeasurement], truth_q: np.ndarray,
                  truth_specific_force: np.ndarray, config: QuaternionUkfConfig,
                  gravity_m_s2: float) -> tuple[list[dict], dict]:
    """Quantify prediction versus accelerometer yaw changes, with truth only for scoring."""
    if not (len(readings) == len(truth_q) == len(truth_specific_force)):
        raise ValueError("diagnostic truth and measurements must align")
    estimator = QuaternionUkf(config)
    rows = []
    posterior_quaternions = []
    maximum_quaternion_disagreement = 0.0
    maximum_covariance_disagreement = 0.0
    for index, (reading, q_true, ideal_force) in enumerate(
            zip(readings, truth_q, truth_specific_force)):
        prior_q, prior_P = estimator.quaternion_bw, estimator.covariance_rad2
        diagnostic = None if index == 0 else audit_update(
            prior_q, prior_P, reading,
            reading.timestamp_s - readings[index - 1].timestamp_s, config)
        update = estimator.update(reading)
        posterior_quaternions.append(update.quaternion_bw)
        if diagnostic is not None:
            quaternion_disagreement = np.linalg.norm(relative_rotation_vector(
                diagnostic["posterior"], update.quaternion_bw))
            covariance_disagreement = np.max(abs(diagnostic["posterior_P"] - update.covariance_rad2))
            maximum_quaternion_disagreement = max(maximum_quaternion_disagreement,
                                                  quaternion_disagreement)
            maximum_covariance_disagreement = max(maximum_covariance_disagreement,
                                                  covariance_disagreement)
            if quaternion_disagreement > 1e-9 or covariance_disagreement > 1e-9:
                raise AssertionError("independent UKF step audit disagrees with estimator")
        predicted_q = prior_q if diagnostic is None else diagnostic["prediction"]
        correction = np.zeros(3) if diagnostic is None else diagnostic["correction"]
        innovation = np.zeros(3) if diagnostic is None else diagnostic["innovation"]
        S = np.full((3, 3), np.nan) if diagnostic is None else diagnostic["S"]
        gain = np.zeros((3, 3)) if diagnostic is None else diagnostic["gain"]
        heading = gravity_direction(predicted_q)
        estimate_yaw = float(np.rad2deg(quaternion_to_euler(update.quaternion_bw)[2]))
        predicted_yaw = float(np.rad2deg(quaternion_to_euler(predicted_q)[2]))
        prior_yaw = float(np.rad2deg(quaternion_to_euler(prior_q)[2]))
        P = update.covariance_rad2
        correlation = P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
        real_force = reading.accel_m_s2
        true_gravity = gravity_direction(q_true)
        rows.append({
            "timestamp_s": float(reading.timestamp_s),
            "true_yaw_deg": float(np.rad2deg(quaternion_to_euler(q_true)[2])),
            "estimated_yaw_deg": estimate_yaw,
            "predicted_yaw_deg": predicted_yaw,
            "gyro_prediction_yaw_change_deg": 0.0 if index == 0 else wrapped_difference_deg(predicted_yaw, prior_yaw),
            "accelerometer_update_yaw_change_deg": 0.0 if index == 0 else wrapped_difference_deg(estimate_yaw, predicted_yaw),
            "correction_x_rad": float(correction[0]),
            "correction_y_rad": float(correction[1]),
            "correction_z_rad": float(correction[2]),
            "correction_global_heading_component_rad": float(heading @ correction),
            "innovation_x": float(innovation[0]),
            "innovation_y": float(innovation[1]),
            "innovation_z": float(innovation[2]),
            "innovation_norm": float(np.linalg.norm(innovation)),
            "innovation_covariance_xx": float(S[0, 0]),
            "innovation_covariance_yy": float(S[1, 1]),
            "innovation_covariance_zz": float(S[2, 2]),
            "innovation_covariance_xy": float(S[0, 1]),
            "innovation_covariance_xz": float(S[0, 2]),
            "innovation_covariance_yz": float(S[1, 2]),
            "innovation_covariance_condition": float("nan") if index == 0 else float(np.linalg.cond(S)),
            "gain_frobenius_norm": float(np.linalg.norm(gain)),
            "gain_heading_norm": float(np.linalg.norm(heading @ gain)),
            "gain_heading_x": float((heading @ gain)[0]),
            "gain_heading_y": float((heading @ gain)[1]),
            "gain_heading_z": float((heading @ gain)[2]),
            "covariance_xx_rad2": float(P[0, 0]),
            "covariance_yy_rad2": float(P[1, 1]),
            "covariance_zz_rad2": float(P[2, 2]),
            "covariance_xy_rad2": float(P[0, 1]),
            "covariance_xz_rad2": float(P[0, 2]),
            "covariance_yz_rad2": float(P[1, 2]),
            "correlation_xy": float(correlation[0, 1]),
            "correlation_xz": float(correlation[0, 2]),
            "correlation_yz": float(correlation[1, 2]),
            "heading_projection_std_deg": float(np.rad2deg(np.sqrt(heading @ P @ heading))),
            "measured_specific_force_norm_minus_g_m_s2": float(np.linalg.norm(real_force) - gravity_m_s2),
            "measured_gravity_direction_mismatch_deg": direction_angle_deg(real_force, true_gravity),
            "ideal_gravity_direction_mismatch_deg": direction_angle_deg(ideal_force, true_gravity),
        })
    trace_q = np.array([row["estimated_yaw_deg"] for row in rows])
    truth_yaw = np.array([row["true_yaw_deg"] for row in rows])
    relative_error = np.rad2deg(np.unwrap(np.deg2rad(trace_q))) - np.rad2deg(np.unwrap(np.deg2rad(truth_yaw)))
    relative_error -= relative_error[0]
    orientation_error = geodesic_error_deg(np.array(posterior_quaternions), truth_q)
    for row, yaw_error, full_error in zip(rows, relative_error, orientation_error):
        row["relative_yaw_error_deg"] = float(yaw_error)
        row["geodesic_error_deg"] = float(full_error)
    return rows, {"max_step_audit_quaternion_error_rad": maximum_quaternion_disagreement,
                  "max_step_audit_covariance_error_rad2": maximum_covariance_disagreement}


def summarize_lateral(rows: list[dict], switch_time_s: float,
                      gate_threshold_m_s2: float = 0.5) -> dict:
    after = [row for row in rows if row["timestamp_s"] >= switch_time_s]
    onset = {}
    for threshold in (2.0, 5.0, 10.0):
        crossing = next((row["timestamp_s"] for row in after
                         if abs(row["relative_yaw_error_deg"]) >= threshold), None)
        onset[f"first_{int(threshold)}deg_yaw_error_s"] = crossing
    peak = max(rows, key=lambda row: abs(row["accelerometer_update_yaw_change_deg"]))
    peak_mismatch = max(rows, key=lambda row: row["ideal_gravity_direction_mismatch_deg"])
    near = min(rows, key=lambda row: abs(row["timestamp_s"] - 2.88))
    return {
        "maneuver_switch_time_s": switch_time_s,
        **onset,
        "sum_gyro_prediction_yaw_change_after_switch_deg": float(sum(
            row["gyro_prediction_yaw_change_deg"] for row in after)),
        "sum_accelerometer_update_yaw_change_after_switch_deg": float(sum(
            row["accelerometer_update_yaw_change_deg"] for row in after)),
        "peak_absolute_accelerometer_yaw_change_per_step_deg": abs(peak["accelerometer_update_yaw_change_deg"]),
        "peak_accelerometer_yaw_change_time_s": peak["timestamp_s"],
        "peak_ideal_direction_mismatch_deg": peak_mismatch["ideal_gravity_direction_mismatch_deg"],
        "peak_ideal_direction_mismatch_time_s": peak_mismatch["timestamp_s"],
        "at_2_88_s": {
            "ideal_direction_mismatch_deg": near["ideal_gravity_direction_mismatch_deg"],
            "measured_direction_mismatch_deg": near["measured_gravity_direction_mismatch_deg"],
            "measured_norm_minus_g_m_s2": near["measured_specific_force_norm_minus_g_m_s2"],
            "norm_gate_threshold_m_s2": gate_threshold_m_s2,
            "norm_gate_rejects_measured": abs(near["measured_specific_force_norm_minus_g_m_s2"]) > gate_threshold_m_s2,
        },
        "max_gain_heading_norm": max(row["gain_heading_norm"] for row in rows),
        "max_abs_covariance_correlation": max(abs(row[key]) for row in rows
            for key in ("correlation_xy", "correlation_xz", "correlation_yz")),
        "heading_std_initial_final_deg": [rows[0]["heading_projection_std_deg"],
                                          rows[-1]["heading_projection_std_deg"]],
    }


def yaw_prior_coupling(readings: list[ImuMeasurement], truth_q: np.ndarray,
                       ideal_force: np.ndarray, config: QuaternionUkfConfig,
                       gravity_m_s2: float, switch_time_s: float,
                       priors_deg: list[float]) -> list[dict]:
    """Replay each declared prior to expose nonlinear heading/measurement coupling."""
    results = []
    for prior in priors_deg:
        std = config.initial_attitude_std_rad.copy()
        std[2] = np.deg2rad(prior)
        rows, _ = lateral_trace(readings, truth_q, ideal_force,
                                replace(config, initial_attitude_std_rad=std), gravity_m_s2)
        mechanism = summarize_lateral(rows, switch_time_s)
        near = min(rows, key=lambda row: abs(row["timestamp_s"] - 2.88))
        results.append({
            "initial_yaw_std_deg": float(prior),
            "gain_heading_norm_at_2_88_s": near["gain_heading_norm"],
            "summed_accelerometer_yaw_change_after_switch_deg":
                mechanism["sum_accelerometer_update_yaw_change_after_switch_deg"],
            "max_abs_covariance_correlation": mechanism["max_abs_covariance_correlation"],
        })
    return results


def _tangent_basis(direction: np.ndarray) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(direction, first)))


def observable_direction_nees(estimate_q: np.ndarray, truth_q: np.ndarray,
                              covariance_rad2: np.ndarray) -> float:
    """Small-angle two-dimensional NEES of gravity direction; excludes heading."""
    estimated = gravity_direction(estimate_q)
    true = gravity_direction(truth_q)
    basis = _tangent_basis(estimated)
    H = gravity_right_jacobian(estimate_q)
    projected = basis.T @ H @ covariance_rad2 @ H.T @ basis
    residual = basis.T @ (true - estimated)
    return float(residual @ np.linalg.solve(projected, residual))


def geometry_audit(config: QuaternionUkfConfig) -> dict:
    """Finite-difference checks separate from the recorded flight experiment."""
    q = euler_to_quaternion(.3, -.2, .4)
    epsilon = 1e-6
    numeric_H = np.column_stack([
        (gravity_direction(quaternion_product(q, rotation_vector_to_quaternion(
            epsilon * np.eye(3)[axis])))
         - gravity_direction(quaternion_product(q, rotation_vector_to_quaternion(
             -epsilon * np.eye(3)[axis])))) / (2 * epsilon)
        for axis in range(3)])
    analytic_H = gravity_right_jacobian(q)
    singular_values = np.linalg.svd(analytic_H, compute_uv=False)
    direction = gravity_direction(q)
    yaw_rotation = euler_to_quaternion(0, 0, .7)
    global_yaw_difference = np.linalg.norm(gravity_direction(q) - gravity_direction(
        quaternion_product(yaw_rotation, q)))
    correction = np.array([.3, -.2, .4])
    perturbation = np.array([1e-6, 2e-6, -3e-6])
    actual_reset = relative_rotation_vector(
        rotation_vector_to_quaternion(correction),
        rotation_vector_to_quaternion(correction + perturbation))
    expected_reset = right_jacobian_so3(correction) @ perturbation
    covariance = np.diag([.01, .02, .03])
    scale, wm, wc = unscented_weights(config.alpha, config.beta, config.kappa)
    sigma = local_sigma_points(covariance, scale)
    quaternions = np.array([quaternion_product(q, rotation_vector_to_quaternion(delta))
                            for delta in sigma])
    recovered_mean, local_errors, iterations = quaternion_sigma_mean(
        quaternions, wm, config.mean_tolerance_rad, config.mean_max_iterations)
    recovered_P = sum(weight * np.outer(error, error)
                      for weight, error in zip(wc, local_errors))
    untouched, ordinary_correction = stabilize_covariance(covariance, config.covariance_floor_rad2)
    regularized, singular_correction = stabilize_covariance(
        np.diag([0.0, .01, .02]), config.covariance_floor_rad2)
    return {
        "gravity_jacobian_max_finite_difference_error": float(np.max(abs(numeric_H - analytic_H))),
        "gravity_jacobian_singular_values": singular_values.tolist(),
        "gravity_jacobian_times_heading_norm": float(np.linalg.norm(analytic_H @ direction)),
        "global_yaw_gravity_prediction_difference": float(global_yaw_difference),
        "right_jacobian_reset_finite_difference_error_rad": float(np.linalg.norm(actual_reset - expected_reset)),
        "sigma_mean_reconstruction_error_rad": float(np.linalg.norm(relative_rotation_vector(q, recovered_mean))),
        "sigma_mean_iterations": iterations,
        "sigma_covariance_reconstruction_error_rad2": float(np.linalg.norm(recovered_P - covariance)),
        "ordinary_covariance_stabilization_norm_rad2": ordinary_correction,
        "ordinary_covariance_unchanged": bool(np.array_equal(untouched, covariance)),
        "singular_covariance_stabilization_norm_rad2": singular_correction,
        "regularized_min_eigenvalue_rad2": float(np.linalg.eigvalsh(regularized)[0]),
        "noise_interpretation": "gyro rad/s per sample -> local Q=diag((sigma*dt)^2); normalized acceleration uses approximate full-rank Cartesian covariance",
    }


def level_heading_check(config: QuaternionUkfConfig, priors_deg: list[float],
                        gravity_m_s2: float) -> list[dict]:
    """Exact level gravity cannot contract the yaw prior with zero process noise."""
    readings = [ImuMeasurement(k * .02, gravity_m_s2 * UP, np.zeros(3))
                for k in range(151)]
    rows = []
    for prior in priors_deg:
        std = config.initial_attitude_std_rad.copy()
        std[2] = np.deg2rad(prior)
        arm = replace(config, initial_attitude_std_rad=std,
                      gyro_noise_std_rad_s=np.zeros(3))
        trace = run_ukf(readings, arm)
        rows.append({
            "initial_yaw_std_deg": float(prior),
            "final_heading_projection_std_deg": float(trace.heading_std_deg[-1]),
            "initial_local_heading_variance_rad2": float(trace.covariance_rad2[0, 2, 2]),
            "final_local_heading_variance_rad2": float(trace.covariance_rad2[-1, 2, 2]),
            "final_tilt_variance_x_rad2": float(trace.covariance_rad2[-1, 0, 0]),
            "final_tilt_variance_y_rad2": float(trace.covariance_rad2[-1, 1, 1]),
        })
    return rows


def static_consistency(config: QuaternionUkfConfig, seeds: list[int],
                       duration_s: float, sample_hz: float, gravity_m_s2: float,
                       priors_deg: list[float]) -> dict:
    """Controlled static, bias-free Monte Carlo; not a powered-flight model."""
    steps = round(duration_s * sample_hz)
    if steps <= 0 or not np.isclose(steps, duration_s * sample_hz, atol=1e-9, rtol=0):
        raise ValueError("static duration must contain whole sample periods")
    rows = []
    for yaw_std_deg in priors_deg:
        std = config.initial_attitude_std_rad.copy()
        std[2] = np.deg2rad(yaw_std_deg)
        arm = replace(config, initial_attitude_std_rad=std)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            truth = rotation_vector_to_quaternion(rng.normal(0, std))
            readings = [ImuMeasurement(
                k / sample_hz,
                gravity_m_s2 * (gravity_direction(truth)
                                 + rng.normal(0, config.accel_direction_std)),
                rng.normal(0, config.gyro_noise_std_rad_s)) for k in range(steps + 1)]
            trace = run_ukf(readings, arm)
            q = trace.filter_run.quaternion_bw[-1]
            P = trace.covariance_rad2[-1]
            rows.append({
                "initial_yaw_std_deg": float(yaw_std_deg), "seed": int(seed),
                "final_tilt_direction_error_deg": direction_angle_deg(
                    gravity_direction(q), gravity_direction(truth)),
                "final_geodesic_error_deg": float(geodesic_error_deg(q[None, :], truth[None, :])[0]),
                "final_observable_direction_nees_2d": observable_direction_nees(q, truth, P),
                "initial_heading_projection_std_deg": float(trace.heading_std_deg[0]),
                "final_heading_projection_std_deg": float(trace.heading_std_deg[-1]),
                "final_covariance_min_eigenvalue_rad2": float(trace.minimum_eigenvalue_rad2[-1]),
            })
    aggregate = []
    for prior in priors_deg:
        group = [row for row in rows if row["initial_yaw_std_deg"] == prior]
        aggregate.append({
            "initial_yaw_std_deg": float(prior), "trials": len(group),
            **{metric + suffix: float(function([row[metric] for row in group]))
               for metric in ("final_tilt_direction_error_deg", "final_geodesic_error_deg",
                              "final_observable_direction_nees_2d",
                              "final_heading_projection_std_deg")
               for suffix, function in (("_mean", np.mean), ("_std", np.std))},
        })
    return {"description": "Externally supported stationary true attitude drawn from declared initial prior; bias-free sampled gyro and gravity-direction noise. Not powered flight.",
            "observable_direction_nees_definition": "e=B^T(h_true-h_est), C=B^T H P H^T B, NEES=e^T C^-1 e; B spans the plane perpendicular to h_est, H=[h_est]x. Two local observable directions; small-angle approximation; excludes global yaw.",
            "seeds": seeds, "duration_s": duration_s, "sample_hz": sample_hz,
            "individual_trials": rows, "aggregate": aggregate}


def save_plots(output: Path, prior_rows: list[dict], coupling_rows: list[dict],
               variant_traces: dict, rows: list[dict], static: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [row["initial_yaw_std_deg"] for row in prior_rows]
    axes[0].plot(x, [row["geodesic_rmse_deg"] for row in prior_rows], "o-")
    axes[0].set_ylabel("geodesic RMSE (deg)")
    axes[1].plot(x, [row["final_relative_yaw_error_deg"] for row in prior_rows], "o-", label="yaw error")
    axes[1].plot(x, [row["final_heading_projection_std_deg"] for row in prior_rows], "s--", label="projected std")
    axes[1].legend()
    axes[1].set_ylabel("final angle (deg)")
    for axis in axes:
        axis.set_xlabel("initial yaw std (deg)"); axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output / "yaw_prior_sweep.png", dpi=150); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    prior = [row["initial_yaw_std_deg"] for row in coupling_rows]
    axes[0].plot(prior, [row["gain_heading_norm_at_2_88_s"] for row in coupling_rows], "o-")
    axes[0].set_ylabel("||heading projection of K|| at 2.88 s")
    axes[1].plot(prior, [row["summed_accelerometer_yaw_change_after_switch_deg"]
                          for row in coupling_rows], "o-")
    axes[1].set_ylabel("summed accelerometer yaw change (deg)")
    for axis in axes: axis.set_xlabel("initial yaw std (deg)"); axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output / "yaw_prior_coupling.png", dpi=150); plt.close(figure)

    time_s = np.array([row["timestamp_s"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for name, (trace, series) in variant_traces.items():
        axes[0, 0].plot(time_s, series["relative_yaw_error_deg"], label=name.replace("_", " "))
        axes[0, 1].plot(time_s, series["geodesic_error_deg"], label=name.replace("_", " "))
    axes[0, 0].set_ylabel("relative yaw error (deg)")
    axes[0, 1].set_ylabel("SO(3) geodesic error (deg)")
    axes[1, 0].plot(time_s, [row["gyro_prediction_yaw_change_deg"] for row in rows], label="gyro prediction")
    axes[1, 0].plot(time_s, [row["accelerometer_update_yaw_change_deg"] for row in rows], label="accelerometer update")
    axes[1, 0].set_ylabel("yaw change per sample (deg)")
    axes[1, 1].plot(time_s, [row["ideal_gravity_direction_mismatch_deg"] for row in rows], label="ideal force vs gravity")
    axes[1, 1].plot(time_s, [row["measured_gravity_direction_mismatch_deg"] for row in rows], label="measured vs gravity", alpha=.7)
    axes[1, 1].set_ylabel("direction mismatch (deg)")
    for axis in axes.flat:
        axis.grid(alpha=.3); axis.legend(fontsize=8)
    for axis in axes[1]: axis.set_xlabel("time (s)")
    figure.tight_layout(); figure.savefig(output / "lateral_mechanism.png", dpi=150); plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(time_s, [row["gain_heading_norm"] for row in rows])
    axes[0, 0].set_ylabel("||heading projection of K||")
    axes[0, 1].plot(time_s, [row["correction_global_heading_component_rad"] for row in rows])
    axes[0, 1].set_ylabel("correction along heading (rad)")
    for key in ("correlation_xy", "correlation_xz", "correlation_yz"):
        axes[1, 0].plot(time_s, [row[key] for row in rows], label=key)
    axes[1, 0].set_ylabel("covariance correlation"); axes[1, 0].legend()
    axes[1, 1].plot(time_s, [row["heading_projection_std_deg"] for row in rows])
    axes[1, 1].set_ylabel("projected heading std (deg)")
    for axis in axes.flat: axis.grid(alpha=.3)
    for axis in axes[1]: axis.set_xlabel("time (s)")
    figure.tight_layout(); figure.savefig(output / "lateral_covariance_coupling.png", dpi=150); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    aggregate = static["aggregate"]
    axes[0].errorbar([row["initial_yaw_std_deg"] for row in aggregate],
                     [row["final_tilt_direction_error_deg_mean"] for row in aggregate],
                     yerr=[row["final_tilt_direction_error_deg_std"] for row in aggregate], fmt="o-")
    axes[0].set_ylabel("static tilt direction error (deg)")
    axes[1].errorbar([row["initial_yaw_std_deg"] for row in aggregate],
                     [row["final_observable_direction_nees_2d_mean"] for row in aggregate],
                     yerr=[row["final_observable_direction_nees_2d_std"] for row in aggregate], fmt="o-")
    axes[1].axhline(2.0, color="black", linestyle="--", label="2D Gaussian reference mean")
    axes[1].set_ylabel("observable 2D directional NEES"); axes[1].legend(fontsize=8)
    for axis in axes: axis.set_xlabel("initial yaw std (deg)"); axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output / "static_consistency.png", dpi=150); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ukf_validation.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ukf_validation"))
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        settings = tomllib.load(stream)
    paths = {key: Path(value) for key, value in settings["inputs"].items()}
    missing = [paths[key] for key in ("measurements", "truth", "baseline_metrics")
               if not paths[key].is_file()]
    if missing:
        parser.error("Missing prerequisite output(s): " + ", ".join(map(str, missing))
                     + ". Run the IMU evaluation, then the UKF evaluation, before UKF validation.")
    config, _ = load_config(paths["ukf_config"])
    experiment = load_imu_experiment(paths["imu_config"])
    readings = load_measurements(paths["measurements"])
    times = np.array([reading.timestamp_s for reading in readings])
    truth = load_aligned_truth(paths["truth"], times)
    truth_table = np.atleast_1d(np.genfromtxt(paths["truth"], delimiter=",", names=True))
    indices = np.searchsorted(truth_table["timestamp_s"], times)
    ideal_force = np.column_stack([truth_table[f"specific_force_body_{axis}_m_s2"][indices]
                                   for axis in "xyz"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_rows = run_yaw_prior_sweep(readings, truth, config,
                                     settings["yaw_prior_sweep"]["initial_yaw_std_deg"])
    _csv(args.output_dir / "yaw_prior_sweep.csv", prior_rows)
    _json(args.output_dir / "yaw_prior_sweep.json", {"rows": prior_rows,
          "sensor_inputs_unchanged": True, "only_initial_yaw_covariance_varies": True})
    variants, variant_traces = run_variants(readings, truth, config,
                                            experiment.base.vehicle.gravity_m_s2)
    _json(args.output_dir / "measurement_variants.json", {
        "variants": variants,
        "operational_input": "unchanged recorded accelerometer and gyro",
        "gyro_prediction_only_input": "recorded gyro; zero accelerometer skips correction",
        "offline_truth_gravity_oracle_input": "recorded gyro; true orientation constructs gravity direction; evaluation-only, not deployable"})
    variant_rows = []
    for index, time_s in enumerate(times):
        row = {"timestamp_s": float(time_s)}
        for name, (trace, series) in variant_traces.items():
            row[f"{name}_relative_yaw_error_deg"] = float(series["relative_yaw_error_deg"][index])
            row[f"{name}_geodesic_error_deg"] = float(series["geodesic_error_deg"][index])
        variant_rows.append(row)
    _csv(args.output_dir / "measurement_variants_errors.csv", variant_rows)
    rows, audit = lateral_trace(readings, truth, ideal_force, config,
                                experiment.base.vehicle.gravity_m_s2)
    _csv(args.output_dir / "lateral_diagnostics.csv", rows)
    mechanism = summarize_lateral(rows, experiment.maneuver.switch_time_s)
    _json(args.output_dir / "lateral_mechanism.json", {"summary": mechanism,
          "audit": audit, "coordinates": "right-local correction rad; heading component projects on predicted body-frame gravity axis",
          "truth_boundary": "truth is used only for error and mismatch scoring"})
    coupling = yaw_prior_coupling(
        readings, truth, ideal_force, config, experiment.base.vehicle.gravity_m_s2,
        experiment.maneuver.switch_time_s,
        settings["yaw_prior_sweep"]["initial_yaw_std_deg"])
    _csv(args.output_dir / "yaw_prior_coupling.csv", coupling)
    _json(args.output_dir / "yaw_prior_coupling.json", {"rows": coupling,
          "same_sensor_sequence": True,
          "heading_gain_projection": "predicted body gravity axis transposed times the 3x3 Kalman gain"})
    consistency = static_consistency(config, settings["static_consistency"]["seeds"],
                                     settings["static_consistency"]["duration_s"],
                                     settings["static_consistency"]["sample_hz"],
                                     experiment.base.vehicle.gravity_m_s2,
                                     settings["yaw_prior_sweep"]["initial_yaw_std_deg"])
    _json(args.output_dir / "static_consistency.json", consistency)
    _csv(args.output_dir / "static_consistency_trials.csv", consistency["individual_trials"])
    _json(args.output_dir / "geometry_audit.json", geometry_audit(config))
    level_rows = level_heading_check(config, [5.0, 30.0, 60.0],
                                     experiment.base.vehicle.gravity_m_s2)
    _json(args.output_dir / "level_heading_observability.json", {"rows": level_rows,
          "description": "Exact level gravity, zero gyro and process noise; only tilt variance should contract."})
    save_plots(args.output_dir, prior_rows, coupling, variant_traces, rows, consistency)
    baseline = json.loads(paths["baseline_metrics"].read_text(encoding="utf-8"))
    _json(args.output_dir / "reproduction.json", {
        "original_baseline_output_preserved": str(paths["baseline_metrics"]),
        "baseline_recorded_geodesic_rmse_deg": baseline["geodesic_rmse_deg"],
        "baseline_replayed_geodesic_rmse_deg": variants["operational"]["geodesic_rmse_deg"],
        "baseline_recorded_final_relative_yaw_error_deg": baseline["relative_yaw_final_error_deg"],
        "baseline_replayed_final_relative_yaw_error_deg": variants["operational"]["relative_yaw_final_error_deg"],
        "ukf_config": str(paths["ukf_config"]), "imu_config": str(paths["imu_config"]),
        "measurement_file": str(paths["measurements"]),
        "truth_file_evaluation_only": str(paths["truth"]),
        "validation_config": str(args.config)})
    print("M4 operational replay: geodesic %.6f deg, final relative yaw %.6f deg" % (
        variants["operational"]["geodesic_rmse_deg"],
        variants["operational"]["relative_yaw_final_error_deg"]))
    print("Prediction only: %.6f deg; offline gravity oracle: %.6f deg" % (
        variants["gyro_prediction_only"]["geodesic_rmse_deg"],
        variants["offline_truth_gravity_oracle"]["geodesic_rmse_deg"]))
    print("Saved", args.output_dir.resolve())


if __name__ == "__main__":
    main()
