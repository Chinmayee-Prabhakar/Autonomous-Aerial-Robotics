"""Image-derived target guidance; no true wall or window pose is accepted."""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.config import VehicleConfig
from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector, WindowDetection
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld, WindowWall
from aerial_autonomy.perception.window_observation import WindowObservation
from aerial_autonomy.perception.window_pose import WindowPoseEstimate, estimate_window_pose
from aerial_autonomy.planning.geometry import Workspace
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.trajectory.quintic import TrajectoryLimits, generate_trajectory


@dataclass(frozen=True)
class TargetModel:
    model: WindowModel
    frame_thickness_m: float
    outer_width_m: float
    outer_height_m: float
    expected_side: str

    def __post_init__(self):
        if (self.expected_side not in ("positive_normal","negative_normal")
                or self.frame_thickness_m <= 0
                or self.outer_width_m <= self.model.width_m
                or self.outer_height_m <= self.model.height_m):
            raise ValueError("invalid known target dimensions or side")


@dataclass(frozen=True)
class ImagePoseFix:
    target_id: str
    valid: bool
    reason: str | None
    detection: WindowDetection
    estimate: WindowPoseEstimate | None
    estimated_window: StaticWindow | None


class ImageWindowGuidance:
    """Consumes RGB, onboard state feedback, calibration, and metric target models.

    Constructor deliberately has no truth world, true window pose, or renderer.
    A physical wall is instantiated only around an estimated image-derived pose.
    """

    def __init__(self,camera: CameraCalibration,detector: ColorWindowDetector,
                 targets: dict[str,TargetModel],workspace: Workspace,
                 vehicle: VehicleConfig,limits: TrajectoryLimits,
                 vehicle_radius_m: float=.18,planning_clearance_m: float=.14,
                 approach_distance_m: float=2.25,exit_distance_m: float=1.3):
        self.camera=camera
        self.detector=detector
        self.targets=dict(targets)
        self.workspace=workspace
        self.vehicle=vehicle
        self.limits=limits
        self.vehicle_radius_m=vehicle_radius_m
        self.planning_clearance_m=planning_clearance_m
        self.approach_distance_m=approach_distance_m
        self.exit_distance_m=exit_distance_m
        if (set(targets)!=set(detector.colors_by_id) or
                min(vehicle_radius_m,approach_distance_m,exit_distance_m)<=0
                or planning_clearance_m<0):
            raise ValueError("invalid guidance catalog or safety settings")

    def perceive(self,image_rgb: np.ndarray,state: VehicleState,target_id: str,
                 timestamp_s: float) -> ImagePoseFix:
        if target_id not in self.targets:
            raise ValueError("target unknown to declared mission catalog")
        target=self.targets[target_id]
        detection=self.detector.detect(image_rgb,self.camera,target_id,target.expected_side)
        def fail(reason,estimate=None):
            return ImagePoseFix(target_id,False,reason,detection,estimate,None)
        if not detection.valid:
            return fail("detector_"+str(detection.reason))
        observation=WindowObservation(target_id,timestamp_s,detection.corners_uv,
                                      np.ones(4,dtype=bool),None)
        estimate=estimate_window_pose(observation,target.model,self.camera)
        if not estimate.valid:
            return fail("pose_"+str(estimate.reason),estimate)
        if estimate.facing_side!=target.expected_side:
            return fail("pose_wrong_side",estimate)
        if estimate.reprojection_rmse_px>4 or detection.confidence<.12:
            return fail("low_image_quality",estimate)
        camera_rotation,camera_center=self.camera.camera_to_world(state)
        rotation=camera_rotation@estimate.rotation_window_to_camera
        normal=rotation[:,2]
        visible_face_center=camera_center+camera_rotation@estimate.translation_window_in_camera_m
        sign=1 if target.expected_side=="positive_normal" else -1
        center=visible_face_center-sign*normal*target.frame_thickness_m/2
        window=StaticWindow(target.model,center,rotation[:,0],rotation[:,1],
                            target.frame_thickness_m)
        return ImagePoseFix(target_id,True,None,detection,estimate,window)

    def entry_exit(self,fix: ImagePoseFix):
        if not fix.valid or fix.estimated_window is None:
            raise ValueError("cannot guide from invalid image pose")
        side=self.targets[fix.target_id].expected_side
        sign=1 if side=="positive_normal" else -1
        normal=fix.estimated_window.normal_world
        center=fix.estimated_window.center_world_m
        return (center+sign*normal*self.approach_distance_m,
                center-sign*normal*self.exit_distance_m)

    def plan_segment(self,state: VehicleState,fix: ImagePoseFix,goal_world_m: np.ndarray):
        if not fix.valid or fix.estimated_window is None:
            return None,{"success":False,"reason":"invalid_image_pose"}
        target=self.targets[fix.target_id]
        estimated_world=WindowCourseWorld(self.workspace,
            (WindowWall(fix.estimated_window,target.outer_width_m,target.outer_height_m),),
            self.vehicle_radius_m,self.planning_clearance_m)
        start=np.asarray(state.position_m,dtype=float)
        goal=np.asarray(goal_world_m,dtype=float)
        if not np.all(np.isfinite(goal)) or np.linalg.norm(goal-start)<.05:
            return None,{"success":False,"reason":"degenerate_guidance_segment"}
        return generate_trajectory([start,goal],estimated_world,self.vehicle,self.limits)


class NavigationState:
    TAKEOFF="TAKEOFF"
    ACQUIRE="ACQUIRE"
    APPROACH="APPROACH"
    ALIGN="ALIGN"
    TRAVERSE="TRAVERSE"
    CONFIRM_EXIT="CONFIRM_EXIT"
    HOLD="HOLD_REACQUIRE"
    ABORT="ABORT"
    COMPLETE="COMPLETE"


@dataclass(frozen=True)
class NavigationPolicy:
    image_period_s: float = .35
    max_loss_s: float = 1.4
    max_tracking_error_m: float = .55
    final_segment_error_m: float = .16
    max_pose_jump_m: float = .35

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in vars(self).values()):
            raise ValueError("invalid navigation policy")
