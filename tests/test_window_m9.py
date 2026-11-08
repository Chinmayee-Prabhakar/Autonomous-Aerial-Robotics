"""M9 pixel-to-pose-to-flight integration and simulator-truth boundary tests."""

import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aerial_autonomy.evaluation.window_m8 import state_at
from aerial_autonomy.evaluation.window_m9 import FlightCondition, WindowMission, load_m9_config
from aerial_autonomy.evaluation.window_m9_study import run_detector_study
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m9_guidance import ImagePoseFix, ImageWindowGuidance, TargetModel
from aerial_autonomy.perception.window_m9_render import RenderSettings, WindowImageRenderer
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld, WindowWall


class M9VisionAndGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,cls.vehicle,cls.limits,cls.camera,cls.world,cls.targets,cls.colors=load_m9_config(
            "configs/window_m9.toml")

    def make_guidance(self,targets=None):
        return ImageWindowGuidance(self.camera,ColorWindowDetector(self.colors),
            targets or self.targets,self.world.workspace,self.vehicle.vehicle,self.limits)

    def render(self,state=None,seed=11,settings=None,world=None):
        return WindowImageRenderer(self.camera,world or self.world,self.colors).render(
            state or state_at([0,0,1.6]),settings or RenderSettings(),np.random.default_rng(seed))

    def test_renderer_produces_frame_pixels_and_open_hole(self):
        image=self.render()
        self.assertEqual(image.shape,(240,320,3))
        self.assertEqual(image.dtype,np.uint8)
        self.assertGreater(int(image[120,115,0]),150)  # red left frame
        self.assertLess(int(image[120,160,0]),100)  # open central passage
        self.assertGreater(int(image[120,20,0]),80)  # solid wall

    def test_renderer_seed_noise_lighting_and_blur(self):
        state=state_at([0,0,1.6])
        settings=RenderSettings(8,.8,True)
        np.testing.assert_array_equal(self.render(state,11,settings),self.render(state,11,settings))
        self.assertFalse(np.array_equal(self.render(state,11,settings),self.render(state,12,settings)))

    def test_study_keeps_oracle_diagnostics_separate_from_failed_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows,_=run_detector_study(self.raw,self.vehicle,self.limits,self.camera,
                self.world,self.targets,self.colors,Path(temporary))
        self.assertEqual(len(rows),12)
        self.assertEqual(sum(r["detected"] for r in rows),8)
        self.assertEqual(sum(r["false_positive"] for r in rows),0)
        solid=next(r for r in rows if r["case"]=="solid_color_false_positive")
        self.assertFalse(solid["detected"])
        self.assertTrue(solid["oracle_corner_pose_valid"])
        self.assertIsNone(solid["translation_error_m"])
        self.assertIsNotNone(solid["oracle_translation_error_m"])

    def test_behind_camera_rejected_from_pixels(self):
        image=self.render(state_at([0,0,1.6],np.pi))
        detection=ColorWindowDetector(self.colors).detect(image,self.camera,"alpha")
        self.assertFalse(detection.valid)

    def test_detector_orders_front_and_reverse_physical_corners(self):
        detector=ColorWindowDetector(self.colors)
        front=detector.detect(self.render(state_at([0,0,1.6])),self.camera,"alpha")
        reverse=detector.detect(self.render(state_at([8,0,1.6],np.pi)),self.camera,
                                "alpha","negative_normal")
        self.assertTrue(front.valid,front.reason)
        self.assertTrue(reverse.valid,reverse.reason)
        self.assertLess(front.corners_uv[0,0],front.corners_uv[1,0])
        self.assertGreater(reverse.corners_uv[0,0],reverse.corners_uv[1,0])

    def test_detector_rejects_blank_and_solid_color(self):
        image=np.full((240,320,3),40,dtype=np.uint8)
        detector=ColorWindowDetector(self.colors)
        self.assertFalse(detector.detect(image,self.camera,"alpha").valid)
        image[70:170,100:220]=self.colors["alpha"]
        self.assertFalse(detector.detect(image,self.camera,"alpha").valid)

    def test_partial_occlusion_and_out_of_frame_fail(self):
        detector=ColorWindowDetector(self.colors)
        image=self.render();image[:,0:150]=40
        self.assertFalse(detector.detect(image,self.camera,"alpha").valid)
        image=self.render(state_at([0,0,1.6],.9))
        self.assertFalse(detector.detect(image,self.camera,"alpha").valid)

    def test_multiple_color_id_isolation(self):
        image=self.render(state_at([6.5,0,1.7],.1))
        detector=ColorWindowDetector(self.colors)
        beta=detector.detect(image,self.camera,"beta")
        self.assertTrue(beta.valid,beta.reason)
        self.assertFalse(detector.detect(image,self.camera,"alpha").valid)

    def test_image_to_accepted_m8_pose_and_world_transform(self):
        state=state_at([0,0,1.6])
        fix=self.make_guidance().perceive(self.render(state),state,"alpha",0.)
        self.assertTrue(fix.valid,fix.reason)
        self.assertEqual(fix.estimate.window_id,"alpha")
        np.testing.assert_allclose(fix.estimated_window.center_world_m,
            self.world.wall_for_id("alpha").window.center_world_m,atol=.08)

    def test_blank_image_cannot_fall_back_to_true_window(self):
        state=state_at([0,0,1.6])
        guidance=self.make_guidance()
        valid=guidance.perceive(self.render(state),state,"alpha",0.)
        self.assertTrue(valid.valid)
        blank=np.zeros((240,320,3),dtype=np.uint8)
        failed=guidance.perceive(blank,state,"alpha",.35)
        self.assertFalse(failed.valid)
        self.assertIsNone(failed.estimated_window)
        trajectory,report=guidance.plan_segment(state,failed,np.array([3.,0,1.6]))
        self.assertIsNone(trajectory)
        self.assertEqual(report["reason"],"invalid_image_pose")

    def test_guidance_target_changes_when_only_rendered_window_moves(self):
        state=state_at([0,0,1.6])
        guidance=self.make_guidance()
        original=guidance.perceive(self.render(state),state,"alpha",0.)
        wall=self.world.wall_for_id("alpha")
        shifted=StaticWindow(wall.window.model,wall.window.center_world_m+np.array([0,.2,0]),
            wall.window.right_world,wall.window.up_world,wall.window.frame_thickness_m)
        shifted_world=WindowCourseWorld(self.world.workspace,
            (WindowWall(shifted,wall.outer_width_m,wall.outer_height_m),*self.world.walls[1:]),
            self.world.vehicle_radius_m,self.world.clearance_m)
        moved=guidance.perceive(self.render(state,world=shifted_world),state,"alpha",0.)
        self.assertTrue(moved.valid,moved.reason)
        self.assertGreater(np.linalg.norm(moved.estimated_window.center_world_m-
                                          original.estimated_window.center_world_m),.12)

    def test_temporal_jump_rejected_before_navigation(self):
        sim=WindowMission(self.raw,self.vehicle,self.limits,self.camera,self.world,
            self.targets,self.colors,FlightCondition("jump",11,0,1,False,0),["alpha"])
        sim.state=state_at([0,0,1.6])
        first=sim._observe("alpha",force=True)
        self.assertTrue(first.valid,first.reason)
        wall=self.world.wall_for_id("alpha")
        moved=StaticWindow(wall.window.model,
            wall.window.center_world_m+np.array([0,.6,0]),
            wall.window.right_world,wall.window.up_world,wall.window.frame_thickness_m)
        rendered_world=WindowCourseWorld(self.world.workspace,
            (WindowWall(moved,wall.outer_width_m,wall.outer_height_m),*self.world.walls[1:]),
            self.world.vehicle_radius_m,self.world.clearance_m)
        sim.renderer=WindowImageRenderer(self.camera,rendered_world,self.colors)
        second=sim._observe("alpha",force=True)
        self.assertFalse(second.valid)
        self.assertEqual(second.reason,"temporal_pose_jump")
        self.assertIsNone(second.estimated_window)

    def test_wall_and_envelope_collision_whole_segment(self):
        wall=self.world.wall_for_id("alpha")
        self.assertTrue(self.world.segment_free([4.,0,1.6],[6.,0,1.6]))
        self.assertFalse(self.world.segment_free([4.,.8,1.6],[6.,.8,1.6]))
        self.assertFalse(self.world.point_free([5.,2.,1.6]))
        self.assertGreater(self.world.segment_clearance_m([4.,0,1.6],[6.,0,1.6]),0)
        self.assertLess(self.world.segment_clearance_m([4.,.8,1.6],[6.,.8,1.6]),0)

    def test_image_derived_approach_and_m7_feasibility(self):
        state=state_at([0,0,1.6])
        guidance=self.make_guidance()
        fix=guidance.perceive(self.render(state),state,"alpha",0.)
        entry,exit_point=guidance.entry_exit(fix)
        self.assertLess(entry[0],exit_point[0])
        trajectory,report=guidance.plan_segment(state,fix,entry)
        self.assertTrue(report["success"],report["reason"])
        self.assertTrue(report["continuous_collision_checked"])
        self.assertLessEqual(report["max_speed_m_s"],self.limits.max_speed_m_s)
        self.assertGreater(trajectory.duration_s,0)

    def test_narrow_opening_is_not_silently_enlarged(self):
        state=state_at([3.,0,1.6])
        guidance=self.make_guidance()
        valid=guidance.perceive(self.render(state_at([0,0,1.6])),state_at([0,0,1.6]),"alpha",0.)
        narrow=StaticWindow(WindowModel("alpha",.4,.4),valid.estimated_window.center_world_m,
            valid.estimated_window.right_world,valid.estimated_window.up_world,.16)
        fix=ImagePoseFix("alpha",True,None,valid.detection,valid.estimate,narrow)
        trajectory,report=guidance.plan_segment(state,fix,np.array([6.,0,1.6]))
        self.assertIsNone(trajectory)
        self.assertEqual(report["reason"],"reference_collision")


class M9FlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,cls.vehicle,cls.limits,cls.camera,cls.world,cls.targets,cls.colors=load_m9_config(
            "configs/window_m9.toml")

    def flight(self,condition,mission_ids=None,reverse=False):
        sim=WindowMission(self.raw,self.vehicle,self.limits,self.camera,self.world,
            self.targets,self.colors,condition,mission_ids,reverse)
        result=sim.run()
        return sim,result

    def test_single_window_state_sequence_and_crossing(self):
        sim,result=self.flight(FlightCondition("single",11,3,1,False,0),["alpha"])
        self.assertTrue(result["completed"],result["failure_reason"])
        self.assertEqual(result["crossing_order"],["alpha"])
        self.assertEqual(result["collision_count"],0)
        self.assertGreater(result["min_actual_clearance_m"],.01)
        states=[row["state"] for row in sim.transitions]
        self.assertIn("ACQUIRE",states)
        self.assertIn("APPROACH",states)
        self.assertIn("TRAVERSE",states)
        self.assertIn("CONFIRM_EXIT",states)

    def test_three_windows_one_continuous_flight(self):
        sim,result=self.flight(FlightCondition("nominal",11,3,1,False,0))
        self.assertTrue(result["completed"],result["failure_reason"])
        self.assertEqual(result["crossing_order"],["alpha","beta","gamma"])
        self.assertEqual(len(set(x["window_id"] for x in sim.crossings)),3)
        self.assertEqual(result["collision_count"],0)
        self.assertLess(result["max_actual_speed_m_s"],self.limits.max_speed_m_s)
        self.assertLess(result["max_actual_acceleration_m_s2"],self.limits.max_acceleration_m_s2)

    def test_reverse_image_guided_traversal(self):
        _,result=self.flight(FlightCondition("reverse",11,3,1,False,0),["alpha"],True)
        self.assertTrue(result["completed"],result["failure_reason"])
        self.assertEqual(result["crossing_order"],["alpha"])

    def test_total_visual_loss_aborts_without_crossing(self):
        _,result=self.flight(FlightCondition("loss",101,3,1,False,1),["alpha"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["crossing_count"],0)
        self.assertEqual(result["failure_reason"],"persistent_visual_loss")
        self.assertEqual(result["collision_count"],0)

    def test_fixed_seed_replays_flight_and_perception(self):
        condition=FlightCondition("replay",19,9,.85,False,.05,.08)
        first,first_result=self.flight(condition,["alpha"])
        second,second_result=self.flight(condition,["alpha"])
        self.assertEqual(first_result,second_result)
        self.assertEqual(first.perception,second.perception)
        np.testing.assert_array_equal(
            np.asarray([r["position_m"] for r in first.actual]),
            np.asarray([r["position_m"] for r in second.actual]))


if __name__=="__main__":
    unittest.main()
