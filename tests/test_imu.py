"""Deterministic physics, sampling, and export checks for milestone A2."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aerial_autonomy.config import ImuConfig, load_imu_experiment
from aerial_autonomy.evaluation.imu import export_scenario, simulate_imu_scenario
from aerial_autonomy.sensors.imu import SimulatedImu, ideal_imu, specific_force_body
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state, world_acceleration
from aerial_autonomy.simulation.rotation import euler_to_quaternion


CONFIG = load_imu_experiment(Path(__file__).resolve().parents[1] / "configs" / "imu.toml")
VEHICLE = CONFIG.base.vehicle
DT = CONFIG.base.demo.step_s


def zero_noise_config(sample_hz: float = 100.0) -> ImuConfig:
    return ImuConfig(sample_hz, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 5)


class PlantValidationTests(unittest.TestCase):
    def test_zero_thrust_free_fall(self) -> None:
        state = VehicleState.at_rest()
        wrench = Wrench(0.0, np.zeros(3))
        np.testing.assert_allclose(world_acceleration(state, wrench, VEHICLE), [0, 0, -VEHICLE.gravity_m_s2])
        next_state = advance_state(state, wrench, VEHICLE, DT)
        self.assertAlmostEqual(next_state.velocity_m_s[2], -VEHICLE.gravity_m_s2 * DT)
        self.assertAlmostEqual(next_state.position_m[2], -VEHICLE.gravity_m_s2 * DT * DT)

    def test_level_hover_thrust_balances_gravity(self) -> None:
        wrench = Wrench(VEHICLE.mass_kg * VEHICLE.gravity_m_s2, np.zeros(3))
        np.testing.assert_allclose(world_acceleration(VehicleState.at_rest(), wrench, VEHICLE), 0, atol=1e-12)

    def test_pitch_rotates_thrust_into_world_x(self) -> None:
        pitch = np.deg2rad(30.0)
        state = VehicleState.at_rest()
        state.quaternion_bw = euler_to_quaternion(0, pitch, 0)
        thrust = VEHICLE.mass_kg * VEHICLE.gravity_m_s2
        acceleration = world_acceleration(state, Wrench(thrust, np.zeros(3)), VEHICLE)
        np.testing.assert_allclose(acceleration, [VEHICLE.gravity_m_s2 * np.sin(pitch), 0,
                                                   VEHICLE.gravity_m_s2 * (np.cos(pitch) - 1)], atol=1e-12)

    def test_torque_causes_correct_angular_acceleration_sign(self) -> None:
        next_state = advance_state(VehicleState.at_rest(), Wrench(0, np.array([0.1, 0, 0])), VEHICLE, DT)
        self.assertAlmostEqual(next_state.body_rate_rad_s[0], DT * 0.1 / VEHICLE.inertia_kg_m2[0])
        np.testing.assert_allclose(next_state.body_rate_rad_s[1:], 0)

    def test_constant_body_rate_matches_analytical_quaternion(self) -> None:
        state = VehicleState.at_rest()
        state.body_rate_rad_s = np.array([0.0, 0.0, 1.0])
        wrench = Wrench(0.0, np.zeros(3))
        for _ in range(100):
            state = advance_state(state, wrench, VEHICLE, DT)
        np.testing.assert_allclose(state.quaternion_bw, euler_to_quaternion(0, 0, 1.0), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(state.quaternion_bw), 1.0, places=12)

    def test_maneuver_is_finite_and_moves_in_positive_x(self) -> None:
        trace = simulate_imu_scenario(CONFIG, "lateral_waypoint")
        for values in (trace.position_m, trace.velocity_m_s, trace.quaternion_bw,
                       trace.body_rate_rad_s, trace.acceleration_world_m_s2,
                       trace.specific_force_body_m_s2, trace.measured_accel_m_s2, trace.measured_gyro_rad_s):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(trace.position_m[-1, 0], 0.8)
        self.assertGreater(np.max(np.abs(trace.body_rate_rad_s)), 0.05)
        self.assertGreater(np.max(np.abs(trace.acceleration_world_m_s2[:, 0])), 0.5)
        np.testing.assert_allclose(np.linalg.norm(trace.quaternion_bw, axis=1), 1.0, atol=1e-12)


class ImuPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = VehicleState.at_rest()
        self.hover_wrench = Wrench(VEHICLE.mass_kg * VEHICLE.gravity_m_s2, np.zeros(3))

    def test_stationary_level_hover_reports_positive_g(self) -> None:
        reading = SimulatedImu(zero_noise_config(), DT).sample(0, self.state, self.hover_wrench, VEHICLE)
        np.testing.assert_allclose(reading.accel_m_s2, [0, 0, VEHICLE.gravity_m_s2], atol=1e-12)
        np.testing.assert_allclose(reading.gyro_rad_s, 0, atol=1e-12)

    def test_free_fall_reports_zero_specific_force(self) -> None:
        reading = SimulatedImu(zero_noise_config(), DT).sample(0, self.state, Wrench(0, np.zeros(3)), VEHICLE)
        np.testing.assert_allclose(reading.accel_m_s2, 0, atol=1e-12)

    def test_known_rotation_transforms_specific_force(self) -> None:
        q = euler_to_quaternion(np.pi / 2, 0, 0)
        np.testing.assert_allclose(specific_force_body(q, np.zeros(3), VEHICLE.gravity_m_s2),
                                   [0, VEHICLE.gravity_m_s2, 0], atol=1e-12)

    def test_gyro_reports_body_rate_not_euler_derivative(self) -> None:
        self.state.quaternion_bw = euler_to_quaternion(0.2, 0.3, 0.4)
        self.state.body_rate_rad_s = np.array([0.1, -0.2, 0.3])
        reading = SimulatedImu(zero_noise_config(), DT).sample(0, self.state, self.hover_wrench, VEHICLE)
        np.testing.assert_allclose(reading.gyro_rad_s, self.state.body_rate_rad_s)

    def test_constant_biases_are_exact_without_noise(self) -> None:
        config = replace(zero_noise_config(), accel_bias_m_s2=np.array([0.1, -0.2, 0.3]),
                         gyro_bias_rad_s=np.array([0.01, -0.02, 0.03]))
        reading = SimulatedImu(config, DT).sample(0, self.state, self.hover_wrench, VEHICLE)
        true_accel, true_rate = ideal_imu(self.state, self.hover_wrench, VEHICLE)
        np.testing.assert_allclose(reading.accel_m_s2 - true_accel, config.accel_bias_m_s2)
        np.testing.assert_allclose(reading.gyro_rad_s - true_rate, config.gyro_bias_rad_s)

    def test_drag_is_in_specific_force(self) -> None:
        self.state.velocity_m_s = np.array([1.0, 0.0, 0.0])
        true_accel, _ = ideal_imu(self.state, self.hover_wrench, VEHICLE)
        np.testing.assert_allclose(true_accel,
                                   [-VEHICLE.linear_drag_n_s_m / VEHICLE.mass_kg, 0, VEHICLE.gravity_m_s2])


class ImuSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = VehicleState.at_rest()
        self.wrench = Wrench(VEHICLE.mass_kg * VEHICLE.gravity_m_s2, np.zeros(3))

    def test_same_seed_matches_and_different_seed_changes_noise(self) -> None:
        def readings(seed: int) -> np.ndarray:
            sensor = SimulatedImu(replace(CONFIG.imu, seed=seed), DT)
            return np.array([sensor.sample(2 * i, self.state, self.wrench, VEHICLE).accel_m_s2
                             for i in range(20)])
        np.testing.assert_array_equal(readings(19), readings(19))
        self.assertFalse(np.array_equal(readings(19), readings(20)))

    def test_full_scenario_replays_identically(self) -> None:
        first = simulate_imu_scenario(CONFIG, "lateral_waypoint")
        second = simulate_imu_scenario(CONFIG, "lateral_waypoint")
        np.testing.assert_array_equal(first.measured_accel_m_s2, second.measured_accel_m_s2)
        np.testing.assert_array_equal(first.measured_gyro_rad_s, second.measured_gyro_rad_s)
        np.testing.assert_array_equal(first.position_m, second.position_m)

    def test_noise_standard_deviation_matches_config_over_seeded_samples(self) -> None:
        config = ImuConfig(100.0, np.zeros(3), np.zeros(3),
                           np.array([0.2, 0.3, 0.4]), np.array([0.02, 0.03, 0.04]), 124)
        sensor = SimulatedImu(config, DT)
        measurements = [sensor.sample(i, self.state, self.wrench, VEHICLE) for i in range(5000)]
        accel = np.array([item.accel_m_s2 for item in measurements]) - np.array([0, 0, VEHICLE.gravity_m_s2])
        gyro = np.array([item.gyro_rad_s for item in measurements])
        np.testing.assert_allclose(accel.std(axis=0), config.accel_noise_std_m_s2, rtol=0.08)
        np.testing.assert_allclose(gyro.std(axis=0), config.gyro_noise_std_rad_s, rtol=0.08)

    def test_sampling_times_are_monotonic_and_exact_grid(self) -> None:
        sensor = SimulatedImu(CONFIG.imu, DT)
        self.assertEqual(sensor.sample_stride, 2)
        readings = [sensor.sample(i, self.state, self.wrench, VEHICLE) for i in range(11)
                    if sensor.should_sample(i)]
        times = np.array([reading.timestamp_s for reading in readings])
        np.testing.assert_allclose(times, np.arange(6) / CONFIG.imu.sample_hz, atol=1e-12)
        self.assertTrue(np.all(np.diff(times) > 0))
        with self.assertRaisesRegex(ValueError, "sampling grid"):
            sensor.sample(1, self.state, self.wrench, VEHICLE)

    def test_invalid_parameters_and_unsupported_rate_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_hz"):
            replace(CONFIG.imu, sample_hz=0)
        with self.assertRaisesRegex(ValueError, "negative"):
            replace(CONFIG.imu, accel_noise_std_m_s2=np.array([-0.1, 0, 0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(CONFIG.imu, gyro_bias_rad_s=np.array([np.nan, 0, 0]))
        with self.assertRaisesRegex(ValueError, "seed"):
            replace(CONFIG.imu, seed=-1)
        with self.assertRaisesRegex(ValueError, "seed"):
            replace(CONFIG.imu, seed=1.5)
        with self.assertRaisesRegex(ValueError, "three finite"):
            replace(CONFIG.imu, accel_bias_m_s2=[0, 0])
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            SimulatedImu(replace(CONFIG.imu, sample_hz=60.0), DT)

    def test_hover_steady_truth_and_separate_export_schema(self) -> None:
        trace = simulate_imu_scenario(CONFIG, "hover")
        steady = trace.time_s >= CONFIG.base.demo.duration_s - 2.0
        np.testing.assert_allclose(trace.specific_force_body_m_s2[steady].mean(axis=0),
                                   [0, 0, VEHICLE.gravity_m_s2], atol=0.02)
        np.testing.assert_allclose(trace.body_rate_rad_s[steady], 0, atol=1e-12)
        with tempfile.TemporaryDirectory() as temp:
            export_scenario(trace, CONFIG, Path(temp))
            measurement_header = (Path(temp) / "measurements.csv").read_text(encoding="utf-8").splitlines()[0]
            truth_header = (Path(temp) / "truth.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("accel_x_m_s2", measurement_header)
            self.assertNotIn("quaternion", measurement_header)
            self.assertIn("quaternion_bw_w", truth_header)
            self.assertTrue((Path(temp) / "comparison.png").exists())
            self.assertTrue((Path(temp) / "metadata.json").exists())
            self.assertTrue((Path(temp) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
