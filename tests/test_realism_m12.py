"""Source-only M12 coordinate, scene-plan, and pixel-mask checks."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import CompleteMission, load_m11_config
from aerial_autonomy.perception.mission_m11_detect import TargetColorDetector
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.realism.bridge import (
    OPTICAL_RDF_FROM_BLENDER_CAMERA, camera_blender_pose, decor_spec,
    nearest_index, project_blender_camera, project_path, window_face_corners,
)
from aerial_autonomy.realism.detector import AdaptiveColorWindowDetector
from aerial_autonomy.realism.evaluate import (_native_truth_geometry,
    _window_detectors)
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import euler_to_quaternion


class RealismBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, _, cls.camera, cls.world, cls.arm, cls.colors, _ = load_m11_config(
            'configs/mission_m11.toml')

    def test_optical_conversion_is_proper_rotation(self):
        matrix = OPTICAL_RDF_FROM_BLENDER_CAMERA
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(matrix), 1.)
        np.testing.assert_allclose(matrix @ [0., 0., -1.], [0., 0., 1.])

    def test_front_and_reverse_projection_matches_authoritative_camera(self):
        points = np.array([[5., 0., 1.5], [5., -.5, 1.], [5., .5, 2.],
                           [10., .7, 2.], [10., -.8, 1.2]])
        for xyz, yaw in [([1., 0., 1.5], 0.), ([7., .1, 1.5], math.pi),
                         ([2., -.3, 1.3], .16), ([8., .2, 1.7], 3.0)]:
            state = VehicleState(np.array(xyz), np.zeros(3),
                euler_to_quaternion(.03, -.05, yaw), np.zeros(3))
            center, rotation = camera_blender_pose(state, self.camera)
            uv, depth = project_blender_camera(points, center, rotation,
                                               self.camera)
            optical = self.camera.world_to_camera(state, points)
            reference_uv, ahead, _ = self.camera.project(optical)
            np.testing.assert_allclose(depth, optical[:, 2], atol=1e-10)
            np.testing.assert_allclose(uv[ahead], reference_uv[ahead], atol=1e-9)

    def test_window_face_dimensions_and_reverse_physical_identity(self):
        window = {'center_world_m': [5., 0., 1.6],
                  'right_world': [0., 1., 0.], 'up_world': [0., 0., 1.],
                  'opening_width_m': 1.4, 'opening_height_m': 1.1,
                  'frame_thickness_m': .16}
        positive, side = window_face_corners(window, [6., 0., 1.6])
        self.assertEqual(side, 'positive_normal')
        negative, side = window_face_corners(window, [4., 0., 1.6])
        self.assertEqual(side, 'negative_normal')
        self.assertAlmostEqual(np.linalg.norm(positive[1]-positive[0]), 1.4)
        self.assertAlmostEqual(np.linalg.norm(positive[2]-positive[1]), 1.1)
        self.assertAlmostEqual(positive[0, 0]-negative[0, 0], .16)
        np.testing.assert_allclose(positive[0, 1:], negative[0, 1:])

    def test_centered_arm_and_camera_intrinsics_match_m11(self):
        self.assertEqual((self.arm.shape.pivot_right_m,
                          self.arm.shape.pivot_up_m), (0., 0.))
        self.assertAlmostEqual(self.arm.shape.length_m, .88)
        self.assertAlmostEqual(self.arm.shape.arm_radius_m, .10)
        self.assertEqual((self.camera.width_px, self.camera.height_px), (320, 240))
        self.assertEqual((self.camera.fx_px, self.camera.fy_px), (220, 220))
        self.assertAlmostEqual(self.world.envelope_m, .25)

    def test_saved_time_index_selection_and_decor_clearance(self):
        self.assertEqual(nearest_index(np.array([0., .01, .02]), .014), 1)
        trace = {'position_m': np.tile(np.array([[0., 0., 0.]]), (20, 1))}
        decor = decor_spec(trace)
        self.assertEqual(len(decor), 10)
        self.assertTrue(all(x['minimum_extra_clearance_m'] > .1 for x in decor))

    def test_serialized_paths_are_portable(self):
        plan = 'outputs/realism_m12/scene_plan.json'
        expected = project_path(plan, 'configs/mission_m11.toml')
        self.assertEqual(expected,
                         project_path(plan, r'configs\mission_m11.toml'))
        with self.assertRaises(ValueError):
            project_path(plan, '../outside.toml')


class AdaptivePixelMaskTests(unittest.TestCase):
    def test_luminance_changes_preserve_target_hue(self):
        detector = AdaptiveColorWindowDetector({'alpha': (255, 0, 0),
            'delta': (0, 230, 230)})
        image = np.array([[[210, 48, 42], [120, 18, 15], [42, 160, 167],
                           [125, 125, 125]]], dtype=np.uint8)
        np.testing.assert_array_equal(detector.color_mask(image, 'alpha')[0],
                                      [True, True, False, False])
        np.testing.assert_array_equal(detector.color_mask(image, 'delta')[0],
                                      [False, False, True, False])


class IndependentEvaluationTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
            'configs/mission_m11.toml')
        cls.raw, cls.vehicle, cls.limits, cls.camera = raw, vehicle, limits, camera
        cls.world, cls.colors = world, colors

    def _native_fixture(self):
        wall = self.world.wall_for_id('alpha')
        physical = wall.window
        target = {'center_world_m': physical.center_world_m.tolist(),
            'right_world': physical.right_world.tolist(),
            'up_world': physical.up_world.tolist(),
            'opening_width_m': physical.model.width_m,
            'opening_height_m': physical.model.height_m,
            'frame_thickness_m': physical.frame_thickness_m}
        trace = {'time_s': np.array([1.25]),
            'position_m': np.array([[2., 0., 1.6]]),
            'velocity_m_s': np.zeros((1, 3)),
            'quaternion_bw': np.array([[1., 0., 0., 0.]]),
            'body_rate_rad_s': np.zeros((1, 3))}
        frames = {'time_s': np.array([1.25]),
                  'target_id': np.array(['alpha'])}
        job = {'job_id': 'native_truth_test', 'source_camera_frame': 0,
            'time_s': 1.25, 'trace_index': 0, 'target_id': 'alpha',
            'expected_side': 'positive_normal'}
        return target, trace, frames, job

    def test_exact_m11_detector_matches_operational_guidance(self):
        holder = SimpleNamespace(raw=self.raw, vehicle=self.vehicle,
            limits=self.limits, camera=self.camera, world=self.world,
            colors=self.colors)
        adapted = AdaptiveColorWindowDetector(self.colors)
        for target in self.colors:
            mission = CompleteMission._guidance(holder, target, delta=target == 'delta')
            compared = _window_detectors(target, self.colors, adapted)
            self.assertIsInstance(mission.detector, TargetColorDetector)
            self.assertIsInstance(compared['m11_operational'],
                                  type(mission.detector))
            self.assertEqual(compared['m11_operational'].colors_by_id,
                             mission.detector.colors_by_id)
            self.assertIsInstance(compared['m9_comparator'], ColorWindowDetector)

    def test_saved_detector_corners_cannot_change_native_truth(self):
        # No accepted detector output or generated file enters this test.
        target, trace, essential, job = self._native_fixture()
        reference, corners_world = _native_truth_geometry(
            job, target, self.camera, trace, essential)
        self.assertTrue(reference[2])
        corrupted = dict(essential, corners_uv=np.full((1, 4, 2), 1e9))
        repeated, repeated_world = _native_truth_geometry(
            job, target, self.camera, trace, corrupted)
        self.assertEqual(reference[2], repeated[2])
        np.testing.assert_array_equal(reference[1], repeated[1])
        np.testing.assert_array_equal(corners_world, repeated_world)

    def test_native_timestamp_mismatch_rejected(self):
        target, trace, frames, job = self._native_fixture()
        job['time_s'] += .01
        with self.assertRaisesRegex(AssertionError, 'time/target mismatch'):
            _native_truth_geometry(job, target, self.camera, trace, frames)


if __name__ == '__main__':
    unittest.main()
