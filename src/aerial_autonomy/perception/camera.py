"""Calibrated pinhole camera with optical RDF mounted on body FLU."""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import quaternion_to_matrix


# Columns map optical right, down, forward into body FLU at zero gimbal angle.
FORWARD_OPTICAL_TO_BODY = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])


@dataclass(frozen=True)
class CameraCalibration:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    rotation_camera_to_body: np.ndarray
    translation_camera_in_body_m: np.ndarray

    def __post_init__(self) -> None:
        intrinsic = (self.fx_px,self.fy_px,self.cx_px,self.cy_px)
        rotation = np.asarray(self.rotation_camera_to_body,dtype=float)
        translation = np.asarray(self.translation_camera_in_body_m,dtype=float)
        if (any(not math.isfinite(value) for value in intrinsic)
                or self.fx_px <= 0 or self.fy_px <= 0
                or type(self.width_px) is not int or self.width_px <= 0
                or type(self.height_px) is not int or self.height_px <= 0
                or rotation.shape != (3,3) or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10)
                or not np.isclose(np.linalg.det(rotation),1,atol=1e-10)
                or translation.shape != (3,) or not np.all(np.isfinite(translation))):
            raise ValueError("invalid camera calibration or extrinsics")
        object.__setattr__(self,"rotation_camera_to_body",rotation)
        object.__setattr__(self,"translation_camera_in_body_m",translation)

    def camera_to_world(self, state: VehicleState) -> tuple[np.ndarray,np.ndarray]:
        rotation_body_to_world = quaternion_to_matrix(state.quaternion_bw)
        rotation_camera_to_world = rotation_body_to_world @ self.rotation_camera_to_body
        camera_center_world = (np.asarray(state.position_m,dtype=float)
                               +rotation_body_to_world@self.translation_camera_in_body_m)
        return rotation_camera_to_world,camera_center_world

    def world_to_camera(self, state: VehicleState, world_points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(world_points_m,dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("world points must be finite Nx3")
        rotation,center = self.camera_to_world(state)
        return (points-center)@rotation

    def camera_to_world_points(self, state: VehicleState, camera_points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(camera_points_m,dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("camera points must be finite Nx3")
        rotation,center = self.camera_to_world(state)
        return points@rotation.T+center

    def project(self, camera_points_m: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        points = np.asarray(camera_points_m,dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("camera points must be finite Nx3")
        ahead = points[:,2] > 1e-9
        pixels = np.full((len(points),2),np.nan)
        pixels[ahead,0] = self.fx_px*points[ahead,0]/points[ahead,2]+self.cx_px
        pixels[ahead,1] = self.fy_px*points[ahead,1]/points[ahead,2]+self.cy_px
        in_image = (ahead & (pixels[:,0]>=0) & (pixels[:,0]<self.width_px)
                    & (pixels[:,1]>=0) & (pixels[:,1]<self.height_px))
        return pixels,ahead,in_image
