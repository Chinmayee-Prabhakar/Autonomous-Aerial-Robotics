"""M4 SO(3) geometry, unscented update, observability, and shared-input checks."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aerial_autonomy.config import load_imu_experiment
from aerial_autonomy.estimation.quaternion_ukf import (
    QuaternionUkf, QuaternionUkfConfig, local_sigma_points,
    quaternion_sigma_mean, quaternion_to_rotation_vector, relative_rotation_vector,
    right_jacobian_so3, rotation_vector_to_quaternion, stabilize_covariance,
    unscented_weights,
)
from aerial_autonomy.evaluation.imu import simulate_imu_scenario
from aerial_autonomy.evaluation.madgwick import (
    evaluate_run, load_experiment_config, load_measurements, run_filter,
)
from aerial_autonomy.evaluation.ukf import run_ukf
from aerial_autonomy.sensors.imu import ImuMeasurement
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, integrate_quaternion, quaternion_product,
    quaternion_to_euler, quaternion_to_matrix,
)


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
G = 9.81


def setting(**overrides) -> QuaternionUkfConfig:
    options = dict(initial_quaternion_bw=IDENTITY,
                   initial_attitude_std_rad=np.deg2rad([10, 10, 30]),
                   gyro_noise_std_rad_s=np.array([.003, .003, .004]),
                   accel_direction_std=np.array([.05, .05, .05]))
    options.update(overrides)
    return QuaternionUkfConfig(**options)


def sample(time_s: float, acceleration=(0, 0, G), gyro=(0, 0, 0)) -> ImuMeasurement:
    return ImuMeasurement(time_s, np.asarray(acceleration, dtype=float),
                          np.asarray(gyro, dtype=float))


class RotationGeometryTests(unittest.TestCase):
    def test_exp_log_round_trip_for_coupled_rotations(self) -> None:
        for vector in (np.array([.2, -.4, .3]), np.array([1.1, .4, -1.2])):
            np.testing.assert_allclose(quaternion_to_rotation_vector(
                rotation_vector_to_quaternion(vector)), vector, atol=1e-14)

    def test_small_angle_limit(self) -> None:
        tiny = np.array([1e-12, -2e-12, 3e-12])
        q = rotation_vector_to_quaternion(tiny)
        np.testing.assert_allclose(q[1:], tiny / 2, atol=1e-23)
        np.testing.assert_allclose(quaternion_to_rotation_vector(q), tiny, atol=1e-23)

    def test_antipodal_sign_has_same_log_and_relative_error(self) -> None:
        q = rotation_vector_to_quaternion([.3, .2, -.1])
        np.testing.assert_allclose(quaternion_to_rotation_vector(q),
                                   quaternion_to_rotation_vector(-q), atol=1e-15)
        np.testing.assert_allclose(relative_rotation_vector(IDENTITY, q),
                                   relative_rotation_vector(IDENTITY, -q), atol=1e-15)

    def test_right_multiplicative_error(self) -> None:
        reference = euler_to_quaternion(.2, -.3, .4)
        error = np.array([.05, -.07, .09])
        perturbed = quaternion_product(reference, rotation_vector_to_quaternion(error))
        np.testing.assert_allclose(relative_rotation_vector(reference, perturbed), error,
                                   atol=1e-14)

    def test_right_jacobian_matches_finite_difference(self) -> None:
        correction = np.array([.3, -.2, .4])
        epsilon = np.array([1e-6, 2e-6, -3e-6])
        actual = relative_rotation_vector(rotation_vector_to_quaternion(correction),
                                          rotation_vector_to_quaternion(correction + epsilon))
        np.testing.assert_allclose(actual, right_jacobian_so3(correction) @ epsilon,
                                   atol=1e-12)

    def test_global_yaw_rotation_does_not_change_gravity_observation(self) -> None:
        q = euler_to_quaternion(.3, -.2, .4)
        global_yaw = euler_to_quaternion(0, 0, 1.0)
        rotated = quaternion_product(global_yaw, q)
        np.testing.assert_allclose(quaternion_to_matrix(q).T @ [0, 0, 1],
                                   quaternion_to_matrix(rotated).T @ [0, 0, 1], atol=1e-14)


class SigmaPointTests(unittest.TestCase):
    def test_scaled_weights_and_covariance_reconstruction(self) -> None:
        scale, wm, wc = unscented_weights(1.0, 2.0, 0.0)
        covariance = np.diag([.01, .02, .03])
        errors = local_sigma_points(covariance, scale)
        quaternions = np.array([rotation_vector_to_quaternion(error) for error in errors])
        mean, local, _ = quaternion_sigma_mean(quaternions, wm, 1e-12, 20)
        np.testing.assert_allclose(mean, IDENTITY, atol=1e-14)
        reconstructed = sum(weight * np.outer(error, error)
                            for weight, error in zip(wc, local))
        np.testing.assert_allclose(reconstructed, covariance, atol=1e-14)

    def test_mean_of_nontrivial_attitudes(self) -> None:
        center = euler_to_quaternion(.2, -.3, .4)
        delta = np.array([.1, -.04, .05])
        pair = np.array([quaternion_product(center, rotation_vector_to_quaternion(delta)),
                         -quaternion_product(center, rotation_vector_to_quaternion(-delta))])
        mean, _, iterations = quaternion_sigma_mean(pair, np.array([.5, .5]), 1e-12, 20)
        self.assertLess(np.linalg.norm(relative_rotation_vector(center, mean)), 1e-12)
        self.assertLessEqual(iterations, 20)

    def test_invalid_ut_parameters_and_noise_are_rejected(self) -> None:
        for kwargs in ({"alpha": 0}, {"beta": -1}, {"kappa": -3},
                       {"accel_direction_std": np.zeros(3)},
                       {"gyro_noise_std_rad_s": np.array([-1, 0, 0])},
                       {"initial_quaternion_bw": np.ones(3)},
                       {"initial_quaternion_bw": np.full(4, np.nan)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                setting(**kwargs)

    def test_covariance_floor_only_changes_near_singular_matrix(self) -> None:
        ordinary = np.diag([.01, .02, .03])
        stable, correction = stabilize_covariance(ordinary, 1e-12)
        np.testing.assert_array_equal(stable, ordinary)
        self.assertEqual(correction, 0)
        singular, correction = stabilize_covariance(np.diag([0, .01, .02]), 1e-12)
        self.assertGreater(correction, 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(singular)[0], 1e-12)


class UkfUpdateTests(unittest.TestCase):
    def test_constant_body_rate_prediction_without_accel(self) -> None:
        rate = np.array([.18, -.12, .23])
        config = setting(gyro_noise_std_rad_s=np.zeros(3))
        readings = [sample(i * .02, acceleration=(0, 0, 0), gyro=rate)
                    for i in range(301)]
        result = run_ukf(readings, config)
        expected = integrate_quaternion(IDENTITY, rate, 6.0)
        self.assertLess(np.linalg.norm(relative_rotation_vector(
            expected, result.filter_run.quaternion_bw[-1])), 1e-9)

    def test_nonlevel_coupled_rotation_prediction(self) -> None:
        initial = euler_to_quaternion(.3, -.2, .4)
        rate = np.array([.17, .11, -.19])
        readings = [sample(i * .02, acceleration=(0, 0, 0), gyro=rate)
                    for i in range(201)]
        result = run_ukf(readings, setting(initial_quaternion_bw=initial))
        expected = integrate_quaternion(initial, rate, 4.0)
        self.assertLess(np.linalg.norm(relative_rotation_vector(
            expected, result.filter_run.quaternion_bw[-1])), 1e-9)

    def test_externally_supported_static_tilt_converges(self) -> None:
        truth = euler_to_quaternion(.3, -.2, 0)
        acceleration = G * quaternion_to_matrix(truth).T @ [0, 0, 1]
        result = run_ukf([sample(i * .02, acceleration) for i in range(501)], setting())
        error = np.rad2deg(quaternion_to_euler(result.filter_run.quaternion_bw[-1])[:2]
                           - np.array([.3, -.2]))
        self.assertLess(np.max(abs(error)), .1)

    def test_yaw_drifts_with_unmodeled_gyro_bias(self) -> None:
        bias = .01
        result = run_ukf([sample(i * .02, gyro=(0, 0, bias)) for i in range(251)], setting())
        yaw = quaternion_to_euler(result.filter_run.quaternion_bw[-1])[2]
        self.assertAlmostEqual(yaw, bias * 5.0, delta=.01)

    def test_global_yaw_rotation_equivariance_of_entire_filter(self) -> None:
        yaw_rotation = euler_to_quaternion(0, 0, .7)
        readings = [sample(i * .02, acceleration=(.2, -.1, G),
                           gyro=(.03, -.02, .04)) for i in range(101)]
        level = run_ukf(readings, setting())
        yawed = run_ukf(readings, setting(initial_quaternion_bw=yaw_rotation))
        for base_q, shifted_q in zip(level.filter_run.quaternion_bw,
                                      yawed.filter_run.quaternion_bw):
            expected = quaternion_product(yaw_rotation, base_q)
            self.assertLess(np.linalg.norm(relative_rotation_vector(expected, shifted_q)),
                            1e-10)
        np.testing.assert_allclose(level.covariance_rad2, yawed.covariance_rad2,
                                   atol=1e-12)

    def test_process_uncertainty_enters_accelerometer_gain(self) -> None:
        tilt = euler_to_quaternion(0, .2, 0)
        acceleration = G * quaternion_to_matrix(tilt).T @ [0, 0, 1]
        readings = [sample(0), sample(.1, acceleration)]
        narrow = setting(initial_attitude_std_rad=np.full(3, .01),
                         gyro_noise_std_rad_s=np.zeros(3))
        broad = setting(initial_attitude_std_rad=np.full(3, .01),
                        gyro_noise_std_rad_s=np.full(3, 1.0))
        narrow_result = run_ukf(readings, narrow)
        broad_result = run_ukf(readings, broad)
        narrow_pitch = quaternion_to_euler(narrow_result.filter_run.quaternion_bw[-1])[1]
        broad_pitch = quaternion_to_euler(broad_result.filter_run.quaternion_bw[-1])[1]
        self.assertGreater(broad_pitch, narrow_pitch + .05)

    def test_covariance_stays_symmetric_positive_and_finite(self) -> None:
        generator = np.random.default_rng(4)
        readings = [sample(i * .02, np.array([0, 0, G]) + generator.normal(0, .2, 3),
                           generator.normal(0, .02, 3)) for i in range(500)]
        trace = run_ukf(readings, setting())
        self.assertTrue(np.all(np.isfinite(trace.filter_run.quaternion_bw)))
        self.assertTrue(np.all(np.isfinite(trace.covariance_rad2)))
        self.assertGreater(np.min(trace.minimum_eigenvalue_rad2), 0)
        self.assertLess(np.max(trace.covariance_symmetry_residual), 1e-14)
        np.testing.assert_allclose(np.linalg.norm(trace.filter_run.quaternion_bw, axis=1),
                                   1.0, atol=1e-12)

    def test_near_singular_initial_covariance_is_reported(self) -> None:
        config = setting(initial_attitude_std_rad=np.zeros(3))
        estimator = QuaternionUkf(config)
        report = estimator.update(sample(0))
        self.assertGreaterEqual(report.numerical_corrections, 1)
        self.assertGreater(report.max_correction_norm_rad2, 0)
        self.assertGreaterEqual(report.covariance_min_eigenvalue_rad2, 1e-12)
        normal = QuaternionUkf(setting()).update(sample(0))
        self.assertEqual(normal.numerical_corrections, 0)

    def test_invalid_accel_falls_back_and_gyro_continues(self) -> None:
        estimator = QuaternionUkf(setting())
        estimator.update(sample(0))
        first = estimator.update(sample(.02, acceleration=(np.nan, 0, 0), gyro=(0, .2, 0)))
        second = estimator.update(sample(.04, acceleration=(0, 0, 0), gyro=(0, .2, 0)))
        self.assertEqual(first.accel_status, "nonfinite")
        self.assertEqual(second.accel_status, "near_zero")
        expected = integrate_quaternion(IDENTITY, np.array([0, .2, 0]), .04)
        self.assertLess(np.linalg.norm(relative_rotation_vector(expected,
                           estimator.quaternion_bw)), 1e-9)

    def test_invalid_gyro_and_timestamps_leave_state_unchanged(self) -> None:
        estimator = QuaternionUkf(setting())
        estimator.update(sample(0))
        before_q, before_P = estimator.quaternion_bw, estimator.covariance_rad2
        for reading in (sample(0), sample(np.nan), sample(.02, gyro=(np.nan, 0, 0))):
            with self.assertRaises(ValueError):
                estimator.update(reading)
        np.testing.assert_array_equal(estimator.quaternion_bw, before_q)
        np.testing.assert_array_equal(estimator.covariance_rad2, before_P)

    def test_identical_readings_replay_identically(self) -> None:
        generator = np.random.default_rng(9)
        readings = [sample(i * .02, np.array([0, 0, G]) + generator.normal(0, .1, 3),
                           generator.normal(0, .01, 3)) for i in range(201)]
        first, second = run_ukf(readings, setting()), run_ukf(readings, setting())
        np.testing.assert_array_equal(first.filter_run.quaternion_bw,
                                      second.filter_run.quaternion_bw)
        np.testing.assert_array_equal(first.covariance_rad2, second.covariance_rad2)

    def test_sensor_only_loader_needs_no_truth_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "measurements.csv"
            path.write_text("timestamp_s,accel_x_m_s2,accel_y_m_s2,accel_z_m_s2,"
                            "gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s\n"
                            "0,0,0,9.81,0,0,0\n0.02,0,0,9.81,0,0,0\n", encoding="utf-8")
            run = run_ukf(load_measurements(path), setting())
            self.assertEqual(len(run.filter_run.time_s), 2)


class SharedBaselineTests(unittest.TestCase):
    def test_same_sensor_records_feed_all_three_estimators(self) -> None:
        readings = [sample(i * .02, gyro=(0, .1, 0)) for i in range(51)]
        ukf = run_ukf(readings, setting())
        madgwick_config, _ = load_experiment_config(Path("configs/madgwick.toml"))
        madgwick = run_filter(readings, madgwick_config)
        gyro_only = run_filter(readings, replace(madgwick_config, beta_per_s=0))
        np.testing.assert_array_equal(ukf.filter_run.time_s, madgwick.time_s)
        np.testing.assert_array_equal(madgwick.time_s, gyro_only.time_s)
        np.testing.assert_array_equal(ukf.filter_run.quaternion_bw[0],
                                      madgwick.quaternion_bw[0])

    def test_recorded_m3_baselines_are_preserved_by_shared_pipeline(self) -> None:
        base = load_imu_experiment(Path("configs/imu.toml"))
        madgwick_config, _ = load_experiment_config(Path("configs/madgwick.toml"))
        for scenario, expected in (("hover", .640526), ("lateral_waypoint", 1.43756)):
            with self.subTest(scenario=scenario):
                plant = simulate_imu_scenario(base, scenario)
                readings = [ImuMeasurement(float(t), a.copy(), g.copy())
                            for t, a, g in zip(plant.measurement_time_s,
                                               plant.measured_accel_m_s2,
                                               plant.measured_gyro_rad_s)]
                scored, _ = evaluate_run(run_filter(readings, madgwick_config),
                                         plant.quaternion_bw[plant.sample_indices])
                self.assertAlmostEqual(scored["geodesic_rmse_deg"], expected, delta=.002)


if __name__ == "__main__":
    unittest.main()
