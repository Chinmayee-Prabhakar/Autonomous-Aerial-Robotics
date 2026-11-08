"""M8 camera geometry, image-only pose, failures and data-boundary regression."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aerial_autonomy.evaluation.window_m8 import load_experiment, state_at, truth_relative_pose, evaluate_one
from aerial_autonomy.perception.camera import CameraCalibration, FORWARD_OPTICAL_TO_BODY
from aerial_autonomy.perception.window_geometry import WindowModel, StaticWindow
from aerial_autonomy.perception.window_observation import ObservationConfig, WindowObservation, observe_window
from aerial_autonomy.perception.window_pose import estimate_window_pose, reproject_estimate
from aerial_autonomy.simulation.rotation import euler_to_quaternion


class WindowM8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.camera,cls.windows = load_experiment("configs/window_m8.toml")
        cls.window = cls.windows[0]
        cls.frontal = state_at([0.,0.,1.5])

    def observe(self,state=None,window=None,config=None,seed=11):
        return observe_window(window or self.window,self.camera,state or self.frontal,
                              config or ObservationConfig(),np.random.default_rng(seed))

    def test_optical_axes_in_body_flu(self):
        np.testing.assert_array_equal(FORWARD_OPTICAL_TO_BODY[:,0],[0,-1,0])
        np.testing.assert_array_equal(FORWARD_OPTICAL_TO_BODY[:,1],[0,0,-1])
        np.testing.assert_array_equal(FORWARD_OPTICAL_TO_BODY[:,2],[1,0,0])
        self.assertAlmostEqual(np.linalg.det(FORWARD_OPTICAL_TO_BODY),1)

    def test_pinhole_and_behind_camera(self):
        pixels,ahead,inside = self.camera.project(np.array([[0.,0.,2.],[.2,-.4,2.],
                                                            [0.,0.,-1.]]))
        np.testing.assert_allclose(pixels[:2],[[320,240],[370,140]])
        np.testing.assert_array_equal(ahead,[True,True,False])
        np.testing.assert_array_equal(inside,[True,True,False])
        self.assertTrue(np.isnan(pixels[2]).all())

    def test_extrinsic_translation_and_inverse(self):
        camera_points = np.array([[.2,-.3,3.],[0.,0.,1.]])
        state = state_at([2.,-1.,1.],.3,.1)
        world = self.camera.camera_to_world_points(state,camera_points)
        np.testing.assert_allclose(self.camera.world_to_camera(state,world),camera_points,atol=1e-12)
        rotation,center = self.camera.camera_to_world(state)
        np.testing.assert_allclose(center,state.position_m+
                                   rotation@self.camera.rotation_camera_to_body.T@
                                   self.camera.translation_camera_in_body_m)

    def test_quaternion_yaw_rotates_forward_axis(self):
        state = state_at([0.,0.,0.],np.pi/2)
        rotation,_ = self.camera.camera_to_world(state)
        np.testing.assert_allclose(rotation[:,2],[0.,1.,0.],atol=1e-12)
        np.testing.assert_allclose(state.quaternion_bw,euler_to_quaternion(0,0,np.pi/2))

    def test_window_corner_order_plane_and_clearance(self):
        points = self.window.world_corners_m()
        np.testing.assert_allclose(points[0],[4.,.7,.95])
        np.testing.assert_allclose(points[2],[4.,-.7,2.05])
        np.testing.assert_allclose(self.window.normal_world,[-1,0,0])
        for point in points:
            self.assertAlmostEqual(np.dot(point-self.window.center_world_m,
                                           self.window.normal_world),0)
        self.assertTrue(self.window.clears_opening([4.,0.,1.5],.2))
        self.assertFalse(self.window.clears_opening([4.,.65,1.5],.2))

    def test_frontal_analytical_projection_and_pose(self):
        observation = self.observe()
        self.assertEqual(observation.visible_count,4)
        expected_width = 500*1.4/3.92
        self.assertAlmostEqual(observation.pixels_uv[1,0]-observation.pixels_uv[0,0],
                               expected_width,places=10)
        estimate = estimate_window_pose(observation,self.window.model,self.camera)
        self.assertTrue(estimate.valid,estimate.reason)
        np.testing.assert_allclose(estimate.translation_window_in_camera_m,
                                   [0,.02,3.92],atol=1e-9)
        self.assertLess(estimate.reprojection_rmse_px,1e-8)
        self.assertEqual(estimate.facing_side,"positive_normal")

    def test_oblique_pose_against_independent_transform(self):
        state = state_at([0.,-.8,1.5],.3,.08)
        observation = self.observe(state=state)
        estimate = estimate_window_pose(observation,self.window.model,self.camera)
        truth_rotation,truth_translation = truth_relative_pose(self.camera,state,self.window)
        self.assertTrue(estimate.valid,estimate.reason)
        np.testing.assert_allclose(estimate.translation_window_in_camera_m,truth_translation,atol=1e-8)
        np.testing.assert_allclose(estimate.rotation_window_to_camera,truth_rotation,atol=1e-8)
        np.testing.assert_allclose(reproject_estimate(estimate,self.window.model,self.camera),
                                   observation.pixels_uv,atol=1e-8)

    def test_controlled_noise_bounded_and_replay(self):
        config = ObservationConfig(pixel_noise_std=1.5)
        state = state_at([0.,-.8,1.5],.3,.08)
        first = self.observe(state,config=config,seed=19)
        second = self.observe(state,config=config,seed=19)
        np.testing.assert_array_equal(first.pixels_uv,second.pixels_uv)
        estimate = estimate_window_pose(first,self.window.model,self.camera)
        self.assertTrue(estimate.valid,estimate.reason)
        _,t = truth_relative_pose(self.camera,state,self.window)
        self.assertLess(np.linalg.norm(estimate.translation_window_in_camera_m-t),.35)
        self.assertLess(estimate.reprojection_rmse_px,3)

    def test_reverse_side_order_and_normal(self):
        state = state_at([8.,0.,1.5],np.pi)
        observation = self.observe(state)
        self.assertEqual(observation.visible_count,4)
        self.assertGreater(observation.pixels_uv[0,0],observation.pixels_uv[1,0])
        estimate = estimate_window_pose(observation,self.window.model,self.camera)
        self.assertTrue(estimate.valid,estimate.reason)
        self.assertEqual(estimate.facing_side,"negative_normal")
        true_rotation,_ = truth_relative_pose(self.camera,state,self.window)
        np.testing.assert_allclose(estimate.rotation_window_to_camera,true_rotation,atol=1e-8)

    def test_multiple_window_id_isolation(self):
        secondary = self.windows[1]
        observation = self.observe(window=secondary)
        self.assertTrue(estimate_window_pose(observation,secondary.model,self.camera).valid)
        self.assertEqual(estimate_window_pose(observation,self.window.model,self.camera).reason,
                         "window_id_mismatch")

    def test_visibility_and_dropout_fail_closed(self):
        states = [(state_at([0,0,1.5],np.pi),"behind_camera"),
                  (state_at([0,8,1.5]),"outside_image"),
                  (state_at([0,-2.5,1.5]),"partial_visibility")]
        for state,reason in states:
            with self.subTest(reason=reason):
                observation = self.observe(state)
                self.assertEqual(observation.reason,reason)
                estimate = estimate_window_pose(observation,self.window.model,self.camera)
                self.assertFalse(estimate.valid)
                self.assertIsNone(estimate.translation_window_in_camera_m)
        for config,reason in ((ObservationConfig(frame_dropout_probability=1),"frame_dropout"),
                              (ObservationConfig(corner_dropout_probability=1),"no_visible_corners")):
            observation = self.observe(config=config)
            self.assertEqual(observation.reason,reason)
            self.assertEqual(estimate_window_pose(observation,self.window.model,self.camera).reason,reason)

    def test_degenerate_collinear_and_planar_grazing_rejected(self):
        pixels = np.array([[100.,100.],[200.,100.],[300.,100.],[400.,100.]])
        observation = WindowObservation("alpha",0.,pixels,np.ones(4,dtype=bool),None)
        self.assertEqual(estimate_window_pose(observation,self.window.model,self.camera).reason,
                         "degenerate_geometry")
        # A grazing projected quadrilateral has very little normal information.
        pixels = np.array([[200.,200.],[400.,200.],[402.,203.],[198.,203.]])
        observation = WindowObservation("alpha",0.,pixels,np.ones(4,dtype=bool),None)
        self.assertEqual(estimate_window_pose(observation,self.window.model,self.camera).reason,
                         "degenerate_geometry")

    def test_competing_planar_branches_fail_closed(self):
        # A distant small near-frontal window admits two similarly plausible
        # tilt branches under 1.5 px image noise; no arbitrary normal is returned.
        window = StaticWindow(WindowModel("small",.8,.6),np.array([8.,0.,1.5]),
                              np.array([0.,-1.,0.]),np.array([0.,0.,1.]))
        observation = self.observe(window=window,config=ObservationConfig(1.5),seed=1)
        estimate = estimate_window_pose(observation,window.model,self.camera)
        self.assertEqual(estimate.reason,"ambiguous_planar_pose")
        self.assertIsNone(estimate.rotation_window_to_camera)

    def test_observation_boundary_separates_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            row = evaluate_one("test","isolation",11,self.window,self.camera,self.frontal,
                               ObservationConfig(),Path(directory))
            self.assertTrue(row["success"])
            observation = json.loads(next((Path(directory)/"observations").glob("*.json")).read_text())
            estimate = json.loads(next((Path(directory)/"estimates").glob("*.json")).read_text())
            truth = json.loads(next((Path(directory)/"evaluation_truth").glob("*.json")).read_text())
            self.assertNotIn("center_world_m",str(observation))
            self.assertNotIn("center_world_m",str(estimate))
            self.assertIn("window_center_world_m",truth)

    def test_invalid_calibration_and_geometry(self):
        with self.assertRaises(ValueError):
            CameraCalibration(-1,500,320,240,640,480,FORWARD_OPTICAL_TO_BODY,np.zeros(3))
        with self.assertRaises(ValueError):
            WindowModel("",1,1)
        with self.assertRaises(ValueError):
            StaticWindow(self.window.model,np.zeros(3),np.array([1.,0.,0.]),
                         np.array([1.,0.,0.]))


if __name__ == "__main__":
    unittest.main()
