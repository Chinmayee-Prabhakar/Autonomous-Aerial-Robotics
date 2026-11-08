"""A3 quaternion mathematics, sensor-only API, and numerical safeguards."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aerial_autonomy.estimation.madgwick import (
    MadgwickConfig, MadgwickImu, gravity_direction_and_jacobian,
)
from aerial_autonomy.evaluation.madgwick import geodesic_error_deg, load_measurements, run_filter
from aerial_autonomy.sensors.imu import ImuMeasurement
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion, integrate_quaternion, quaternion_to_euler, quaternion_to_matrix,
)


G = 9.81
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def config(beta: float = 0.08, initial: np.ndarray = IDENTITY) -> MadgwickConfig:
    return MadgwickConfig(beta, initial)


def sample(t: float, accel: np.ndarray, gyro: np.ndarray | None = None) -> ImuMeasurement:
    return ImuMeasurement(t, np.asarray(accel, dtype=float),
                          np.zeros(3) if gyro is None else np.asarray(gyro, dtype=float))


class GravityObjectiveTests(unittest.TestCase):
    def test_body_to_world_gravity_direction(self) -> None:
        q = euler_to_quaternion(0.3, -0.2, 0.4)
        predicted, _ = gravity_direction_and_jacobian(q)
        np.testing.assert_allclose(predicted, quaternion_to_matrix(q).T @ [0, 0, 1], atol=1e-14)

    def test_jacobian_and_descent_gradient_match_finite_difference(self) -> None:
        q = euler_to_quaternion(0.3, -0.2, 0.4)
        measured = np.array([0.1, -0.2, 0.97])
        measured /= np.linalg.norm(measured)
        predicted, jacobian = gravity_direction_and_jacobian(q)
        epsilon = 1e-7
        for component in range(4):
            displacement = np.eye(4)[component] * epsilon
            plus = gravity_direction_and_jacobian(q + displacement)[0]
            minus = gravity_direction_and_jacobian(q - displacement)[0]
            np.testing.assert_allclose((plus - minus) / (2 * epsilon),
                                       jacobian[:, component], atol=1e-9)
            cost_plus = 0.5 * np.linalg.norm(plus - measured) ** 2
            cost_minus = 0.5 * np.linalg.norm(minus - measured) ** 2
            self.assertAlmostEqual((cost_plus - cost_minus) / (2 * epsilon),
                                   float((jacobian.T @ (predicted - measured))[component]), places=8)


class MadgwickUpdateTests(unittest.TestCase):
    def test_static_level_ideal_imu_stays_level(self) -> None:
        readings = [sample(i * 0.01, [0, 0, G]) for i in range(201)]
        result = run_filter(readings, config())
        np.testing.assert_allclose(result.quaternion_bw, np.tile(IDENTITY, (201, 1)), atol=1e-12)
        self.assertEqual(np.count_nonzero(result.correction_used), 0)

    def test_static_tilt_converges_with_gravity_observation(self) -> None:
        # An externally supported, stationary tilted sensor, not powered tilted hover.
        truth = euler_to_quaternion(0.3, -0.2, 0.0)
        accel = G * quaternion_to_matrix(truth).T @ [0, 0, 1]
        result = run_filter([sample(i * 0.01, accel) for i in range(1001)], config())
        # Gravity determines tilt; it cannot determine the global yaw gauge.
        tilt_error_deg = np.rad2deg(quaternion_to_euler(result.quaternion_bw[-1])[:2] - [0.3, -0.2])
        self.assertLess(np.max(np.abs(tilt_error_deg)), 0.3)
        self.assertGreater(np.count_nonzero(result.correction_used), 0)

    def test_constant_body_rate_matches_analytic_quaternion(self) -> None:
        rate = np.array([0.18, -0.12, 0.23])
        readings = [sample(i * 0.01, [0, 0, G], rate) for i in range(501)]
        result = run_filter(readings, config(beta=0))
        expected = integrate_quaternion(IDENTITY, rate, 5.0)
        self.assertLess(geodesic_error_deg(result.quaternion_bw[-1:], expected[None, :])[0], 0.01)

    def test_body_rate_right_multiplies_nonlevel_start(self) -> None:
        initial = euler_to_quaternion(0.2, -0.3, 0.4)
        rate = np.array([0.0, 0.0, 1.0])
        estimator = MadgwickImu(config(beta=0, initial=initial))
        estimator.update(sample(0.0, [0, 0, G], rate))
        estimated = estimator.update(sample(0.02, [0, 0, G], rate)).quaternion_bw
        expected = integrate_quaternion(initial, rate, 0.02)
        self.assertLess(geodesic_error_deg(estimated[None, :], expected[None, :])[0], 0.001)

    def test_long_noisy_trace_stays_normalized_and_repeatable(self) -> None:
        generator = np.random.default_rng(42)
        readings = [sample(i * 0.01, [0, 0, G] + generator.normal(0, 3, 3),
                           generator.normal(0, 0.05, 3)) for i in range(1500)]
        first = run_filter(readings, config())
        second = run_filter(readings, config())
        np.testing.assert_array_equal(first.quaternion_bw, second.quaternion_bw)
        np.testing.assert_allclose(np.linalg.norm(first.quaternion_bw, axis=1), 1, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(first.quaternion_bw)))

    def test_invalid_accelerometer_falls_back_to_gyro(self) -> None:
        rate = np.array([0.0, 0.3, 0.0])
        estimator = MadgwickImu(config())
        estimator.update(sample(0, [0, 0, G]))
        first = estimator.update(sample(0.02, [np.nan, 0, G], rate))
        second = estimator.update(sample(0.04, [0, 0, 0], rate))
        self.assertEqual(first.accel_status, "nonfinite")
        self.assertEqual(second.accel_status, "near_zero")
        self.assertFalse(first.correction_used or second.correction_used)
        expected = integrate_quaternion(IDENTITY, rate, 0.04)
        self.assertLess(geodesic_error_deg(second.quaternion_bw[None, :], expected[None, :])[0], 0.001)

    def test_optional_magnitude_gate_is_not_standard_baseline(self) -> None:
        estimator = MadgwickImu(replace(config(), accel_norm_gate_m_s2=0.1))
        estimator.update(sample(0, [0, 0, G]))
        result = estimator.update(sample(0.02, [2, 0, 20]))
        self.assertEqual(result.accel_status, "magnitude_gated")
        self.assertFalse(result.correction_used)

    def test_norm_gate_misses_thrust_driven_direction_mismatch(self) -> None:
        # Representative A2 lateral-flight condition near t=2.88 s.
        truth = euler_to_quaternion(0, np.deg2rad(8.326), 0)
        ideal_force = np.array([-0.029, 0, 9.716])
        gravity_reference = G * quaternion_to_matrix(truth).T @ [0, 0, 1]
        mismatch_deg = np.rad2deg(np.arccos(np.clip(
            np.dot(ideal_force, gravity_reference) /
            (np.linalg.norm(ideal_force) * G), -1, 1)))
        self.assertGreater(mismatch_deg, 8.0)
        self.assertLess(abs(np.linalg.norm(ideal_force) - G), 0.5)
        estimator = MadgwickImu(replace(config(), accel_norm_gate_m_s2=0.5))
        estimator.update(sample(0, ideal_force))
        result = estimator.update(sample(0.02, ideal_force))
        self.assertNotEqual(result.accel_status, "magnitude_gated")
        self.assertTrue(result.correction_used)

    def test_variable_positive_intervals_and_rejected_repeated_time(self) -> None:
        rate = np.array([0, 0, 0.5])
        estimator = MadgwickImu(config(beta=0))
        estimator.update(sample(0.0, [0, 0, G]))
        estimator.update(sample(0.01, [0, 0, G], rate))
        before = estimator.quaternion_bw
        with self.assertRaisesRegex(ValueError, "strictly"):
            estimator.update(sample(0.01, [0, 0, G], rate))
        np.testing.assert_array_equal(estimator.quaternion_bw, before)
        last = estimator.update(sample(0.035, [0, 0, G], rate))
        self.assertAlmostEqual(last.delta_time_s, 0.025)
        expected = integrate_quaternion(IDENTITY, rate, 0.035)
        self.assertLess(geodesic_error_deg(last.quaternion_bw[None, :], expected[None, :])[0], 0.001)

    def test_invalid_configuration_measurements_and_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta"):
            config(beta=-0.1)
        with self.assertRaisesRegex(ValueError, "quaternion norm"):
            config(initial=np.zeros(4))
        estimator = MadgwickImu(config())
        with self.assertRaisesRegex(ValueError, "timestamp"):
            estimator.update(sample(np.nan, [0, 0, G]))
        with self.assertRaisesRegex(ValueError, "gyro"):
            estimator.update(sample(0, [0, 0, G], [np.nan, 0, 0]))
        with self.assertRaisesRegex(ValueError, "accel"):
            estimator.update(sample(0, [0, 0]))
        np.testing.assert_array_equal(estimator.quaternion_bw, IDENTITY)

    def test_unobservable_yaw_drifts_with_gyro_bias(self) -> None:
        bias = 0.03
        readings = [sample(i * 0.02, [0, 0, G], [0, 0, bias]) for i in range(501)]
        result = run_filter(readings, config())
        yaw = quaternion_to_euler(result.quaternion_bw[-1])[2]
        self.assertAlmostEqual(yaw, bias * 10.0, delta=0.002)


class DataBoundaryTests(unittest.TestCase):
    def test_q_and_negative_q_have_zero_geodesic_error(self) -> None:
        rotated = euler_to_quaternion(0, 0, np.pi / 2)
        self.assertAlmostEqual(geodesic_error_deg(rotated[None, :], -rotated[None, :])[0], 0)
        self.assertAlmostEqual(geodesic_error_deg(rotated[None, :], IDENTITY[None, :])[0], 90)

    def test_filter_loads_sensor_csv_without_truth_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "measurements.csv"
            path.write_text(",".join(("timestamp_s", "accel_x_m_s2", "accel_y_m_s2",
                                      "accel_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s",
                                      "gyro_z_rad_s")) + "\n0,0,0,9.81,0,0,0\n0.02,0,0,9.81,0,0,0\n",
                            encoding="utf-8")
            readings = load_measurements(path)
            result = run_filter(readings, config())
            self.assertEqual(len(result.time_s), 2)
            np.testing.assert_allclose(result.quaternion_bw[-1], IDENTITY)


if __name__ == "__main__":
    unittest.main()
