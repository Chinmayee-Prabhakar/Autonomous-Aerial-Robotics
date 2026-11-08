"""Source-only M11 geometry, perception boundary and integrated flight checks."""

import unittest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import (CompleteMission, EXPECTED_CROSSINGS,
    MissionCondition, geometry_study, load_m11_config, physical_crossings, save_trial)
from aerial_autonomy.perception.mission_m11_detect import TargetColorDetector
from aerial_autonomy.perception.window_m9_render import RenderSettings
from aerial_autonomy.perception.window_m10_arm import (ArmShape, RotatingArm,
    arm_clearance_m, swept_arm_clearance_m)
from aerial_autonomy.perception.window_m10_timing import MotionEstimate, choose_crossing
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state
from aerial_autonomy.simulation.rotation import euler_to_quaternion
from aerial_autonomy.trajectory.quintic import generate_trajectory


class MissionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.raw, cls.vehicle, cls.limits, cls.camera, cls.world, cls.arm,
         cls.colors, cls.study) = load_m11_config("configs/mission_m11.toml")

    def test_m10_checkpoint_envelope(self):
        self.assertAlmostEqual(self.world.envelope_m, .25)

    def test_larger_arm_stays_inside_frame(self):
        self.assertGreaterEqual(self.study["arm_tip_to_frame_m"], .04)
        self.assertGreaterEqual(self.arm.shape.length_m, .85)
        self.assertGreater(self.arm.shape.arm_radius_m, .075)

    def test_physical_pivot_is_opening_center(self):
        self.assertEqual((self.arm.shape.pivot_right_m,
                          self.arm.shape.pivot_up_m), (0., 0.))
        pivot, _ = self.arm.endpoints_world(1.3)
        np.testing.assert_allclose(pivot, self.arm.window.center_world_m, atol=1e-12)

    def test_rendered_hub_matches_physical_pivot_and_pixel_detector(self):
        mission = CompleteMission(self.raw, self.vehicle, self.limits, self.camera,
            self.world, self.arm, self.colors, MissionCondition(11, .9))
        window = self.arm.window
        state = VehicleState(window.center_world_m+window.normal_world*2.25,
            np.zeros(3), euler_to_quaternion(0., 0., 0.), np.zeros(3))
        image = mission.renderer.render(state, 0., RenderSettings(0.),
                                        np.random.default_rng(1))
        pivot_world, _ = self.arm.endpoints_world(0.)
        pixel, _, visible = self.camera.project(self.camera.world_to_camera(
            state, pivot_world[None, :]))
        self.assertTrue(visible[0])
        magenta = ((image[:, :, 0] > 155) & (image[:, :, 2] > 140) &
                   (image[:, :, 1] < 115))
        y, x = np.nonzero(magenta)
        self.assertGreater(len(x), 3)
        self.assertLess(np.linalg.norm(np.median(np.column_stack((x, y)), axis=0)-
                                       pixel[0]), 3.)
        fix = mission._guidance("delta", delta=True).perceive(image, state, "delta", 0.)
        detection = mission.arm_detector.detect(image, fix, 0.)
        self.assertTrue(fix.valid)
        self.assertTrue(detection.valid, detection.reason)
        self.assertLess(np.linalg.norm(detection.pivot_local_m), .06)
        self.assertLess(abs((detection.angle_rad-self.arm.angle_at(0.)+math.pi) %
                            (2*math.pi)-math.pi), .1)

    def test_centered_arm_rejects_old_corridor(self):
        old_corridor = dict(self.raw, guidance=dict(self.raw["guidance"],
            delta_corridor_right_m=-.35))
        study = geometry_study(old_corridor, self.world, self.arm,
                               self.vehicle, self.limits)
        self.assertFalse(study["feasible"])
        self.assertEqual(study["corridor_safe_phase_fraction_at_plane"], 0.)
        self.assertEqual(study["full_crossing_safe_phase_fraction"], 0.)

    def test_relocated_corridor_preserves_timed_full_crossing(self):
        self.assertLess(self.study["corridor_min_arm_clearance_at_plane_m"], 0.)
        self.assertGreater(self.study["corridor_max_arm_clearance_at_plane_m"], .25)
        self.assertGreater(self.study["frame_clearance_after_tracking_reserve_m"], .3)
        self.assertTrue(self.study["full_crossing_reference_feasible"])
        self.assertGreater(self.study["full_crossing_safe_phase_fraction"], .1)
        self.assertLess(self.study["full_crossing_safe_phase_fraction"], .9)
        self.assertGreater(self.study["full_crossing_max_predicted_arm_clearance_m"], .035)

    def test_centered_arm_swept_check_catches_between_sample_contact(self):
        point = (self.arm.window.center_world_m+self.arm.window.right_world*
                 self.raw["guidance"]["delta_corridor_right_m"])
        moving = RotatingArm(self.arm.window, self.arm.shape, math.pi/2, math.pi)
        for t in (0., 1.):
            self.assertGreater(arm_clearance_m(point, moving.window, moving.shape,
                               moving.angle_at(t), self.world.envelope_m), 0.)
        self.assertLess(swept_arm_clearance_m(point, point, 0., 1., moving,
                                              self.world.envelope_m), 0.)

    def test_arm_creates_both_safe_and_unsafe_phases(self):
        self.assertLess(self.study["corridor_min_arm_clearance_at_plane_m"], 0)
        self.assertGreater(self.study["corridor_max_arm_clearance_at_plane_m"], .15)

    def test_bypass_is_outside_finite_wall(self):
        self.assertGreater(self.study["minimum_bypass_clearance_m"], .5)
        self.assertGreater(self.study["vehicle_corridor_to_frame_m"], 0)

    def test_overlarge_arm_rejected(self):
        shape = ArmShape(.5, 0., 1.3, .1)
        study = geometry_study(self.raw, self.world, RotatingArm(
            self.arm.window, shape, .9, .35))
        self.assertFalse(study["feasible"])

    def test_fourth_color_is_isolated_from_other_frames(self):
        detector = TargetColorDetector({"delta": self.colors["delta"]})
        self.assertEqual(set(detector.colors_by_id), {"delta"})

    def test_stationary_blocker_has_no_safe_crossing_choice(self):
        window = self.arm.window
        corridor = self.raw["guidance"]["delta_corridor_right_m"]
        start = (window.center_world_m+window.normal_world*2.25-
                 window.right_world*abs(corridor))
        end = (window.center_world_m-window.normal_world*.72-
               window.right_world*abs(corridor))
        trajectory, report = generate_trajectory([start, end], self.world,
                                                  self.vehicle.vehicle, self.limits)
        self.assertTrue(report["success"])
        estimate = MotionEstimate(True, 0., np.pi, 0., .045, .025, 0., None, 7)
        choice = choose_crossing(trajectory, 0., estimate, window,
            self.arm.shape, self.world.envelope_m, 8.)
        self.assertFalse(choice.valid)

    def test_invalid_image_cannot_create_pose(self):
        mission = CompleteMission(self.raw, self.vehicle, self.limits, self.camera,
            self.world, self.arm, self.colors, MissionCondition(11, .9,
            forced_visual_loss=True))
        guidance = mission._guidance("alpha")
        blank = np.zeros((self.camera.height_px, self.camera.width_px, 3), dtype=np.uint8)
        fix = guidance.perceive(blank, mission.state, "alpha", 0.)
        self.assertFalse(fix.valid)
        self.assertIsNone(fix.estimated_window)


class CompleteMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.raw, cls.vehicle, cls.limits, cls.camera, cls.world, cls.arm,
         cls.colors, cls.study) = load_m11_config("configs/mission_m11.toml")
        cls.mission = CompleteMission(cls.raw, cls.vehicle, cls.limits, cls.camera,
            cls.world, cls.arm, cls.colors, MissionCondition(101, .9))
        cls.result = cls.mission.run()

    def test_complete_mission(self):
        self.assertTrue(self.result["success"], self.result["failure_reason"])

    def test_state_progression(self):
        states = [x["state"] for x in self.mission.transitions]
        for required in ("TAKEOFF", "APPROACH_STATIC", "TRAVERSE_STATIC",
                         "WAIT_FOR_SAFE_PHASE", "TRAVERSE_DELTA", "BYPASS_DELTA",
                         "RETURN_HOME", "LAND", "COMPLETE"):
            self.assertIn(required, states)

    def test_crossing_order_and_single_delta(self):
        self.assertEqual(tuple(map(tuple, self.result["crossing_order"])), EXPECTED_CROSSINGS)
        self.assertEqual(self.result["delta_opening_count"], 1)

    def test_external_bypass_classified_separately(self):
        self.assertEqual(self.result["delta_external_bypass_count"], 1)
        events = physical_crossings(self.mission.records, self.world)
        self.assertEqual(sum(e["classification"] == "external_plane_bypass"
                             for e in events), 1)

    def test_every_crossing_has_envelope_entry_exit(self):
        events = [e for e in self.result["crossing_events"]
                  if e["classification"] == "opening"]
        self.assertEqual(len(events), 7)
        self.assertTrue(all(e["entry_time_s"] < e["plane_crossing_time_s"] <
                            e["exit_time_s"] for e in events))

    def test_reverse_poses_are_new_camera_observations(self):
        return_observations = [x for x in self.mission.observations
                               if x["phase"] in ("ACQUIRE_WINDOW", "APPROACH_STATIC") and
                               x["time_s"] > 114 and x["window_valid"]]
        self.assertGreaterEqual(len({x["target_id"] for x in return_observations}), 3)

    def test_reverse_heading(self):
        self.assertTrue(any(abs(x["desired_quaternion_bw"][0]) < .2 and
                            x["phase"] == "TRAVERSE_STATIC" and x["time_s"] > 120
                            for x in self.mission.records))

    def test_perception_required_for_traversal(self):
        for target in ("alpha", "beta", "gamma", "delta"):
            first = next(x for x in self.mission.transitions if x["state"].startswith("TRAVERSE")
                         and x["target_id"] == target)
            self.assertTrue(any(x["target_id"] == target and x["window_valid"] and
                0 <= first["time_s"]-x["time_s"] < 10 for x in self.mission.observations))

    def test_arm_tracker_and_timed_choice(self):
        self.assertGreater(self.result["valid_arm_observation_count"], 5)
        self.assertTrue(any(x["valid"] for x in self.mission.choices))

    def test_monotonic_single_clock(self):
        times = np.asarray([x["time_s"] for x in self.mission.records])
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertAlmostEqual(times[-1], self.result["duration_s"])

    def test_actual_controls_replay_plant(self):
        rows = self.mission.records
        for i in range(1, len(rows), 113):
            a, b = rows[i-1], rows[i]
            prior = VehicleState(a["position_m"], a["velocity_m_s"],
                                 a["quaternion_bw"], a["body_rate_rad_s"])
            applied = Wrench(b["thrust_n"], b["torque_nm"])
            replay = advance_state(prior, applied, self.vehicle.vehicle,
                                   b["time_s"]-a["time_s"])
            np.testing.assert_allclose(replay.position_m, b["position_m"], atol=1e-9)
            np.testing.assert_allclose(replay.quaternion_bw, b["quaternion_bw"], atol=1e-9)

    def test_collision_and_dynamic_gates(self):
        self.assertGreater(self.result["minimum_static_clearance_m"], 0)
        self.assertGreater(self.result["minimum_dynamic_clearance_m"], 0)
        self.assertLess(self.result["peak_speed_m_s"], 1.25)
        self.assertLess(self.result["peak_acceleration_m_s2"], 1.8)
        self.assertLess(self.result["peak_thrust_n"], 16.)
        self.assertLess(self.result["peak_tilt_deg"], 20.)

    def test_touchdown_tolerances(self):
        self.assertTrue(self.result["landing_success"])
        self.assertLess(self.result["landing_horizontal_error_m"], .2)
        self.assertLess(abs(self.result["landing_velocity_world_m_s"][2]), .18)
        self.assertGreaterEqual(self.result["landing_position_world_m"][2], 0.)

    def test_trace_visualization_uses_actual_poses(self):
        with TemporaryDirectory() as temp:
            save_trial(self.mission, Path(temp))
            with np.load(Path(temp) / "physical_trace.npz") as trace:
                self.assertEqual(len(trace["time_s"]), len(self.mission.records))
                np.testing.assert_array_equal(trace["position_m"][-1],
                                              self.mission.state.position_m)
                np.testing.assert_array_equal(trace["quaternion_bw"][-1],
                                              self.mission.state.quaternion_bw)

    def test_complete_seed_replay_is_deterministic(self):
        repeated = CompleteMission(self.raw, self.vehicle, self.limits, self.camera,
            self.world, self.arm, self.colors, MissionCondition(101, .9))
        self.assertEqual(repeated.run(), self.result)


class FailureMissionTests(unittest.TestCase):
    def test_total_visual_loss_aborts_before_first_crossing(self):
        raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
            "configs/mission_m11.toml")
        mission = CompleteMission(raw, vehicle, limits, camera, world, arm, colors,
            MissionCondition(201, .9, forced_visual_loss=True))
        result = mission.run()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "persistent_visual_loss")
        self.assertEqual(result["crossing_order"], [])
        self.assertGreater(result["minimum_static_clearance_m"], 0)
        self.assertGreater(result["minimum_dynamic_clearance_m"], 0)


if __name__ == "__main__":
    unittest.main()
