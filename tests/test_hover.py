"""Small standard-library test suite for the first milestone."""

from pathlib import Path
import unittest

import numpy as np

from aerial_autonomy.config import load_config
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.hover import run_hover
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state
from aerial_autonomy.simulation.rotation import euler_to_quaternion, quaternion_to_euler


CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "hover.toml")


class DynamicsTests(unittest.TestCase):
    def test_hover_wrench_preserves_rest_state(self) -> None:
        state = VehicleState.at_rest()
        wrench = Wrench(CONFIG.vehicle.mass_kg * CONFIG.vehicle.gravity_m_s2, np.zeros(3))
        next_state = advance_state(state, wrench, CONFIG.vehicle, 0.01)
        np.testing.assert_allclose(next_state.position_m, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(next_state.velocity_m_s, np.zeros(3), atol=1e-12)

    def test_upward_force_changes_velocity_and_position(self) -> None:
        state = VehicleState.at_rest()
        wrench = Wrench(1.2 * CONFIG.vehicle.mass_kg * CONFIG.vehicle.gravity_m_s2, np.zeros(3))
        next_state = advance_state(state, wrench, CONFIG.vehicle, 0.01)
        self.assertGreater(next_state.velocity_m_s[2], 0)
        self.assertGreater(next_state.position_m[2], 0)

    def test_quaternion_remains_normalized(self) -> None:
        state = VehicleState.at_rest()
        state.body_rate_rad_s = np.array([0.8, -0.4, 0.6])
        wrench = Wrench(CONFIG.vehicle.mass_kg * CONFIG.vehicle.gravity_m_s2, np.zeros(3))
        for _ in range(1000):
            state = advance_state(state, wrench, CONFIG.vehicle, 0.01)
        self.assertAlmostEqual(float(np.linalg.norm(state.quaternion_bw)), 1.0, places=12)

    def test_euler_quaternion_round_trip(self) -> None:
        angles = np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(quaternion_to_euler(euler_to_quaternion(*angles)), angles, atol=1e-12)


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = CascadedHoverController(CONFIG.vehicle, CONFIG.controller)

    def test_hover_command_balances_weight(self) -> None:
        state = VehicleState.at_rest()
        command = self.controller.command(state, np.zeros(3), 0.0)
        self.assertAlmostEqual(command.wrench.thrust_n, CONFIG.vehicle.mass_kg * CONFIG.vehicle.gravity_m_s2)
        np.testing.assert_allclose(command.wrench.torque_nm, np.zeros(3), atol=1e-12)

    def test_takeoff_requests_more_than_weight(self) -> None:
        command = self.controller.command(VehicleState.at_rest(), np.array([0.0, 0.0, 1.0]), 0.0)
        self.assertGreater(command.wrench.thrust_n, CONFIG.vehicle.mass_kg * CONFIG.vehicle.gravity_m_s2)

    def test_lateral_target_tilts_within_limit(self) -> None:
        command = self.controller.command(VehicleState.at_rest(), np.array([10.0, 0.0, 0.0]), 0.0)
        self.assertGreater(command.desired_euler_rad[1], 0)
        self.assertLessEqual(command.desired_euler_rad[1], np.deg2rad(CONFIG.controller.max_tilt_deg) + 1e-12)
        self.assertTrue(np.all(np.abs(command.wrench.torque_nm) <= CONFIG.controller.max_torque_nm))


class IntegrationTests(unittest.TestCase):
    def test_reproducible_hover_converges(self) -> None:
        first = run_hover(CONFIG)
        second = run_hover(CONFIG)
        np.testing.assert_array_equal(first.position_m, second.position_m)
        self.assertLess(np.linalg.norm(first.position_m[-1] - CONFIG.demo.target_position_m), 0.05)
        self.assertLess(np.max(np.abs(first.euler_rad)), np.deg2rad(1.0))


if __name__ == "__main__":
    unittest.main()

