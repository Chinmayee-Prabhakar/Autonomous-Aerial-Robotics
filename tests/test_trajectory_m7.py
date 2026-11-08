"""M7 reference, collision, retiming, plant integration, and replay checks."""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from aerial_autonomy.config import load_config
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.rrt_star import load_benchmark
from aerial_autonomy.evaluation.trajectory_m7 import fly, load_m7_config
from aerial_autonomy.planning.geometry import Sphere, Workspace, World
from aerial_autonomy.planning.rrt_star import plan_rrt_star
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.trajectory.quintic import TrajectoryLimits, generate_trajectory, reference_clearance


ROOT = Path(__file__).resolve().parents[1]


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vehicle = load_config(ROOT/"configs"/"hover.toml")
        cls.world = World(Workspace((0,0,0),(10,6,5)), (), .18, .07)
        cls.limits = TrajectoryLimits(1.25, 1.8, 20, 5, 16, 1, 20)

    def test_straight_start_end_and_continuous_evaluation(self):
        trajectory, report = generate_trajectory(((1,3,1),(9,3,1)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        np.testing.assert_allclose(trajectory.evaluate(0).position_m, (1,3,1))
        np.testing.assert_allclose(trajectory.evaluate(trajectory.duration_s).position_m, (9,3,1))
        for t in (0, trajectory.duration_s):
            np.testing.assert_allclose(trajectory.evaluate(t).velocity_m_s, 0, atol=1e-12)
            np.testing.assert_allclose(trajectory.evaluate(t).acceleration_m_s2, 0, atol=1e-12)
        middle = trajectory.evaluate(trajectory.duration_s/2)
        self.assertAlmostEqual(middle.position_m[0], 5)
        self.assertGreater(middle.velocity_m_s[0], 0)
        with self.assertRaises(ValueError):
            trajectory.evaluate(-.01)

    def test_multi_waypoint_position_velocity_acceleration_continuity(self):
        trajectory, _ = generate_trajectory(((1,1,1),(3,1,1),(3,3,2)), self.world,
                                            self.vehicle.vehicle, self.limits)
        t = trajectory.durations_s[0]
        for offset in (-1e-6, 0, 1e-6):
            sample = trajectory.evaluate(t+offset)
            np.testing.assert_allclose(sample.position_m, (3,1,1), atol=1e-8)
            np.testing.assert_allclose(sample.velocity_m_s, 0, atol=1e-8)
            np.testing.assert_allclose(sample.acceleration_m_s2, 0, atol=1e-5)

    def test_analytic_speed_acceleration_thrust_tilt_and_retiming(self):
        trajectory, report = generate_trajectory(((1,3,1),(9,3,1)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertGreater(trajectory.duration_s, self.limits.initial_segment_s)
        self.assertLessEqual(report["max_speed_m_s"], self.limits.max_speed_m_s)
        self.assertLessEqual(report["max_acceleration_m_s2"], self.limits.max_acceleration_m_s2)
        self.assertLessEqual(report["max_tilt_deg"], self.limits.max_tilt_deg)
        self.assertGreaterEqual(report["min_thrust_n"], self.limits.min_thrust_n)
        self.assertLessEqual(report["max_thrust_n"], self.limits.max_thrust_n)
        self.assertAlmostEqual(report["max_speed_m_s"], 8*15/8/trajectory.duration_s)
        self.assertGreater(report["min_thrust_n"], 0)
        samples = [trajectory.evaluate(t) for t in np.linspace(0, trajectory.duration_s, 1001)]
        self.assertLessEqual(max(np.linalg.norm(s.velocity_m_s) for s in samples), report["max_speed_m_s"]+1e-10)
        self.assertLessEqual(max(np.linalg.norm(s.acceleration_m_s2) for s in samples), report["max_acceleration_m_s2"]+1e-5)

    def test_required_force_extrema_include_drag_and_gravity(self):
        trajectory, report = generate_trajectory(((1,1,1),(3,2,2)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        thrusts, tilts = [], []
        for time in np.linspace(0, trajectory.duration_s, 2001):
            reference = trajectory.evaluate(float(time))
            force = (self.vehicle.vehicle.mass_kg *
                     (reference.acceleration_m_s2 + np.array((0,0,self.vehicle.vehicle.gravity_m_s2)))
                     + self.vehicle.vehicle.linear_drag_n_s_m*reference.velocity_m_s)
            thrusts.append(np.linalg.norm(force))
            tilts.append(np.degrees(np.arctan2(np.linalg.norm(force[:2]),force[2])))
        self.assertGreater(report["max_tilt_deg"], 0)
        self.assertAlmostEqual(report["min_thrust_n"], min(thrusts), places=4)
        self.assertAlmostEqual(report["max_thrust_n"], max(thrusts), places=4)
        self.assertAlmostEqual(report["max_tilt_deg"], max(tilts), places=4)

    def test_collision_rejected_and_original_polyline_fallback(self):
        obstacle_world = World(self.world.workspace, (Sphere((5,3,1),1.1),), .18, .07)
        crossing = ((1,3,1),(9,3,1))
        trajectory, report = generate_trajectory(crossing, obstacle_world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertIsNone(trajectory)
        self.assertEqual(report["reason"], "reference_collision")
        # A corner cut would cross the sphere; the safe certified polyline survives.
        safe = ((1,3,1),(3,5,1),(7,5,1),(9,3,1))
        self.assertGreater(reference_clearance(obstacle_world, safe), 0)
        trajectory, report = generate_trajectory(safe, obstacle_world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        self.assertEqual(trajectory.waypoints_m, safe)

    def test_explicit_dynamic_and_path_failures(self):
        impossible = replace(self.limits, max_thrust_n=8)
        trajectory, report = generate_trajectory(((1,3,1),(2,3,1)), self.world,
                                                  self.vehicle.vehicle, impossible)
        self.assertIsNone(trajectory)
        self.assertEqual(report["reason"], "dynamic_limits_unmet")
        trajectory, report = generate_trajectory(((1,3,1),(1,3,1)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertIsNone(trajectory)
        self.assertEqual(report["reason"], "invalid_path")

    def test_controller_legacy_mode_and_feedforward(self):
        controller = CascadedHoverController(self.vehicle.vehicle, self.vehicle.controller)
        state = VehicleState.at_rest()
        target = np.array((1.,0.,1.))
        old = controller.command(state, target, 0)
        legacy = controller.command(state, target, 0, np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(old.wrench.thrust_n, legacy.wrench.thrust_n)
        np.testing.assert_allclose(old.wrench.torque_nm, legacy.wrench.torque_nm)
        guided = controller.command(state, target, 0, np.array((.5,0,0)), np.zeros(3))
        self.assertGreater(guided.desired_euler_rad[1], old.desired_euler_rad[1])

    def test_seeded_planner_and_flight_replay(self):
        raw, planner, scenarios, vehicle, limits, _, _ = load_m7_config(ROOT/"configs"/"trajectory_m7.toml")
        scenario = next(x for x in scenarios if x.name == "detour")
        world = replace(scenario.world, clearance_m=raw["planning"]["clearance_m"])
        first = plan_rrt_star(world, scenario.start, scenario.goal, planner, 11)
        second = plan_rrt_star(world, scenario.start, scenario.goal, planner, 11)
        self.assertEqual(first.path_m, second.path_m)
        trajectory, report = generate_trajectory(first.path_m, scenario.world, vehicle.vehicle, limits)
        self.assertTrue(report["success"])
        self.assertLessEqual(report["max_speed_m_s"], limits.retiming_target_speed_m_s)
        self.assertLessEqual(report["max_acceleration_m_s2"], limits.retiming_target_acceleration_m_s2)
        trace_a, metrics_a = fly(trajectory, scenario.world, vehicle, raw["simulation"])
        trace_b, metrics_b = fly(trajectory, scenario.world, vehicle, raw["simulation"])
        np.testing.assert_array_equal(trace_a["actual_position_m"], trace_b["actual_position_m"])
        self.assertEqual(metrics_a, metrics_b)
        self.assertTrue(metrics_a["flight_success"])
        self.assertTrue(metrics_a["actual_dynamic_limits_satisfied"])
        self.assertGreater(metrics_a["min_actual_clearance_m"], 0)

    def test_actual_dynamic_limit_failure_is_distinct_from_reference(self):
        trajectory, report = generate_trajectory(((1,3,1),(3,3,1)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        _, metrics = fly(trajectory, self.world, self.vehicle,
                         {"step_s": .01, "settle_s": 1, "yaw_deg": 0,
                          "max_position_error_m": 2, "final_error_m": .15,
                          "settle_mean_error_m": .15},
                         replace(self.limits, max_speed_m_s=.05))
        self.assertFalse(metrics["actual_dynamic_limits_satisfied"])
        self.assertFalse(metrics["flight_success"])
        self.assertEqual(metrics["failure_reason"], "actual_dynamic_limit")

    def test_swept_collision_reports_failure(self):
        trajectory, report = generate_trajectory(((1,3,1),(3,3,1)), self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        # Deliberately substitute a blocked flight world to exercise actual collision reporting.
        blocked = World(self.world.workspace, (Sphere((2,3,1),.5),), .18, .07)
        trace, metrics = fly(trajectory, blocked, self.vehicle,
                             {"step_s": .01, "settle_s": 1, "yaw_deg": 0,
                              "max_position_error_m": 2, "final_error_m": .15,
                              "settle_mean_error_m": .15})
        self.assertFalse(metrics["flight_success"])
        self.assertTrue(metrics["actual_collision"])
        self.assertIn(metrics["failure_reason"], ("actual_collision", "actual_swept_collision"))


if __name__ == "__main__":
    unittest.main()
