"""Source-only M10 perception, motion, timed safety and plant integration."""

import math
import unittest
import numpy as np

from aerial_autonomy.evaluation.window_m10 import load_m10_config,RotatingWindowMission
from aerial_autonomy.perception.window_m9_render import RenderSettings
from aerial_autonomy.perception.window_m10_arm import (RotatingArm,arm_clearance_m,
    swept_arm_clearance_m)
from aerial_autonomy.perception.window_m10_timing import (AngularTracker,
    MotionEstimate,choose_crossing,wrap_angle)
from aerial_autonomy.perception.window_m10_vision import ArmDetection
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import euler_to_quaternion
from aerial_autonomy.trajectory.quintic import Trajectory
from aerial_autonomy.visualization.window_m10_plot import quadrotor_geometry


class M10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,cls.vehicle,cls.limits,cls.camera,cls.world,cls.arm=load_m10_config(
            "configs/window_m10.toml")

    def mission(self,**kwargs):
        return RotatingWindowMission(self.raw,self.vehicle,self.limits,self.camera,
                                     self.world,self.arm,**kwargs)

    def test_stationary_frame_and_normal_axis(self):
        self.assertTrue(np.allclose(self.arm.window.center_world_m,[21,0,1.6]))
        self.assertTrue(np.allclose(self.arm.window.normal_world,[-1,0,0]))
        self.assertTrue(np.allclose(self.arm.endpoints_world(0)[0],
                                    self.arm.endpoints_world(5)[0]))

    def test_angle_convention_and_time_update(self):
        p,q=self.arm.shape.endpoints_local(math.pi/2)
        self.assertAlmostEqual(q[1]-p[1],self.arm.shape.length_m)
        self.assertAlmostEqual(self.arm.angle_at(2)-self.arm.angle_at(0),.7)

    def test_arm_blocks_center_in_one_phase_and_clears_in_another(self):
        center=self.arm.window.center_world_m
        blocked=arm_clearance_m(center,self.arm.window,self.arm.shape,math.pi,.25)
        clear=arm_clearance_m(center,self.arm.window,self.arm.shape,math.pi/2,.25)
        self.assertLess(blocked,0)
        self.assertGreater(clear,.1)

    def test_swept_collision_finds_between_endpoint_contact(self):
        moving=RotatingArm(self.arm.window,self.arm.shape,math.pi/2,math.pi)
        point=self.arm.window.center_world_m
        self.assertGreater(arm_clearance_m(point,moving.window,moving.shape,moving.angle_at(0),.25),0)
        self.assertGreater(arm_clearance_m(point,moving.window,moving.shape,moving.angle_at(1),.25),0)
        self.assertLess(swept_arm_clearance_m(point,point,0,1,moving,.25),0)

    def test_reverse_side_point_has_same_capsule_clearance(self):
        center=self.arm.window.center_world_m
        normal=self.arm.window.normal_world
        a=arm_clearance_m(center+.4*normal,self.arm.window,self.arm.shape,math.pi,.25)
        b=arm_clearance_m(center-.4*normal,self.arm.window,self.arm.shape,math.pi,.25)
        self.assertAlmostEqual(a,b)

    def test_window_and_arm_detect_from_pixels(self):
        mission=self.mission()
        image=mission.renderer.render(mission.state,0,RenderSettings(3.),np.random.default_rng(1))
        fix=mission.guidance.perceive(image,mission.state,mission.target_id,0.)
        det=mission.detector.detect(image,fix,0.)
        self.assertTrue(fix.valid)
        self.assertTrue(det.valid)
        self.assertLess(abs(wrap_angle(det.angle_rad-self.arm.angle_at(0))),.1)
        self.assertLess(np.linalg.norm(np.array(det.pivot_local_m)-[.55,0]),.06)

    def test_blank_image_fails_without_truth_fallback(self):
        mission=self.mission()
        image=np.zeros((self.camera.height_px,self.camera.width_px,3),dtype=np.uint8)
        fix=mission.guidance.perceive(image,mission.state,mission.target_id,0.)
        det=mission.detector.detect(image,fix,0.)
        self.assertFalse(fix.valid)
        self.assertFalse(det.valid)

    def test_pixel_measurement_tracks_changed_rendered_phase(self):
        mission=self.mission()
        angles=[]
        for t in (0.,2.):
            image=mission.renderer.render(mission.state,t,RenderSettings(2.),
                                          np.random.default_rng(5))
            fix=mission.guidance.perceive(image,mission.state,mission.target_id,t)
            angles.append(mission.detector.detect(image,fix,t).angle_rad)
        self.assertAlmostEqual(wrap_angle(angles[1]-angles[0]),.7,delta=.1)

    def test_tracker_wrap_and_signed_rate(self):
        tracker=AngularTracker()
        for t,a in ((0,3.0),(.2,3.08),(.4,-3.12318530718),(.6,-3.04318530718)):
            tracker.update(ArmDetection(True,t,a,(0,0),(1,0),1.,None,30))
        est=tracker.estimate(.6)
        self.assertTrue(est.valid)
        self.assertAlmostEqual(est.angular_velocity_rad_s,.4,delta=.03)
        self.assertGreater(est.angle_rad,math.pi)

    def test_tracker_dropouts_and_stale_age(self):
        tracker=AngularTracker(max_age_s=.5)
        for t in (0,.2,.4):
            tracker.update(ArmDetection(True,t,t,(0,0),(1,0),1.,None,30))
        tracker.update(ArmDetection(False,.6,None,None,None,0.,"missing",0))
        self.assertTrue(tracker.estimate(.6).valid)
        self.assertFalse(tracker.estimate(1.).valid)
        self.assertEqual(tracker.misses,1)

    def test_tracker_rejects_nonincreasing_timestamp(self):
        tracker=AngularTracker()
        tracker.update(ArmDetection(True,1.,0.,(0,0),(1,0),1.,None,30))
        with self.assertRaises(ValueError):
            tracker.update(ArmDetection(True,1.,.1,(0,0),(1,0),1.,None,30))

    def test_prediction_error_grows_with_age_bound(self):
        estimate=MotionEstimate(True,0.,.4,.3,.05,.03,0.,None,4)
        near,bound_near=estimate.predict(.5)
        far,bound_far=estimate.predict(3.)
        self.assertAlmostEqual(near,.55)
        self.assertAlmostEqual(far,1.3)
        self.assertGreater(bound_far,bound_near)

    def test_safe_and_unsafe_phase_selection(self):
        trajectory=Trajectory(((18.75,0,1.6),(21.72,0,1.6)),(6.25,))
        safe=MotionEstimate(True,0.,math.pi/2,0.,.045,.025,0.,None,5)
        blocked=MotionEstimate(True,0.,math.pi,0.,.045,.025,0.,None,5)
        self.assertTrue(choose_crossing(trajectory,0.,safe,self.arm.window,
                                        self.arm.shape,.25,2.).valid)
        self.assertFalse(choose_crossing(trajectory,0.,blocked,self.arm.window,
                                         self.arm.shape,.25,2.).valid)

    def test_uncertain_motion_rejects_crossing(self):
        trajectory=Trajectory(((18.75,0,1.6),(21.72,0,1.6)),(6.25,))
        uncertain=MotionEstimate(True,0.,math.pi/2,0.,.9,.2,0.,None,5)
        self.assertFalse(choose_crossing(trajectory,0.,uncertain,self.arm.window,
                                         self.arm.shape,.25,2.).valid)

    def test_far_side_connection_avoids_frame(self):
        route=self.raw["far_side_connection"]["waypoints_world_m"]
        self.assertGreater(min(self.world.segment_clearance_m(a,b)
                               for a,b in zip(route[:-1],route[1:])),.1)
        self.assertTrue(any(a[0]>21 and b[0]<21 for a,b in zip(route[:-1],route[1:])))

    def test_full_dropout_aborts_without_crossing(self):
        result=self.mission(seed=5,dropout_probability=1.).run()
        self.assertEqual(result["failure_reason"],"window_not_detected")
        self.assertEqual(result["crossing_count"],0)

    def test_rendered_quadrotor_uses_saved_attitude(self):
        q=euler_to_quaternion(0,0,math.pi/2)
        mesh=quadrotor_geometry(np.array([1,2,3]),q)
        self.assertTrue(np.allclose(mesh["front"],[1,2.12,3.025],atol=1e-10))
        self.assertEqual(len(mesh["rings"]),4)
        self.assertEqual(len(mesh["arms"]),4)

    def test_hold_then_single_integrated_crossing(self):
        mission=self.mission(seed=101)
        result=mission.run()
        self.assertTrue(result["success"],result["failure_reason"])
        self.assertEqual(result["crossing_count"],1)
        self.assertEqual(result["collision_count"],0)
        self.assertGreater(result["waiting_s"],.2)
        self.assertGreater(result["minimum_actual_arm_clearance_m"],0)
        self.assertGreater(result["minimum_actual_frame_clearance_m"],0)
        self.assertGreater(np.ptp([x["position_m"][0] for x in mission.states]),2.)
        self.assertTrue(any(x["phase"]=="COMMIT_TO_TRAVERSE" for x in mission.states))

    def test_unsafe_stationary_arm_aborts_without_crossing(self):
        mission=self.mission(seed=211,initial_angle_rad=math.pi,
                             angular_velocity_rad_s=0.,max_wait_s=2.)
        result=mission.run()
        self.assertFalse(result["success"])
        self.assertEqual(result["crossing_count"],0)
        self.assertEqual(result["collision_count"],0)
        self.assertEqual(result["failure_reason"],"no_safe_interval_within_wait_limit")

    def test_seeded_replay(self):
        a=self.mission(seed=101).run();b=self.mission(seed=101).run()
        self.assertEqual(a,b)


if __name__=="__main__":unittest.main()
