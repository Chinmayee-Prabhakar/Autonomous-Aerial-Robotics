"""Calibrated planar pose from four identified image corners, without world truth.

The window frame is (right, up, right-cross-up). R maps that frame into
camera optical RDF; t is the window center in camera metres. The DLT
homography provides an initial pose, refined by pixel reprojection.
"""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_geometry import WindowModel
from aerial_autonomy.perception.window_observation import WindowObservation


@dataclass(frozen=True)
class WindowPoseEstimate:
    window_id: str
    valid: bool
    reason: str | None
    rotation_window_to_camera: np.ndarray | None
    translation_window_in_camera_m: np.ndarray | None
    normal_camera: np.ndarray | None
    facing_side: str | None
    reprojection_rmse_px: float | None
    visible_count: int


def _failed(observation: WindowObservation, reason: str) -> WindowPoseEstimate:
    return WindowPoseEstimate(observation.window_id, False, reason, None, None,
                              None, None, None, observation.visible_count)


def _exp_so3(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    skew = np.array([[0.,-vector[2],vector[1]],
                     [vector[2],0.,-vector[0]],[-vector[1],vector[0],0.]])
    if angle < 1e-10:
        return np.eye(3)+skew
    return np.eye(3)+math.sin(angle)/angle*skew+(1-math.cos(angle))/angle**2*(skew@skew)


def _project(model: WindowModel, camera: CameraCalibration, rotation: np.ndarray,
             translation: np.ndarray) -> np.ndarray | None:
    points = model.local_corners_m()@rotation.T+translation
    if not np.all(np.isfinite(points)) or np.any(points[:,2] <= 1e-6):
        return None
    return np.column_stack((camera.fx_px*points[:,0]/points[:,2]+camera.cx_px,
                            camera.fy_px*points[:,1]/points[:,2]+camera.cy_px))


def reproject_estimate(estimate: WindowPoseEstimate, model: WindowModel,
                       camera: CameraCalibration) -> np.ndarray | None:
    if not estimate.valid or estimate.rotation_window_to_camera is None:
        return None
    return _project(model,camera,estimate.rotation_window_to_camera,
                    estimate.translation_window_in_camera_m)


def _refine(model: WindowModel, camera: CameraCalibration, pixels: np.ndarray,
            rotation: np.ndarray, translation: np.ndarray):
    if _project(model,camera,rotation,translation) is None:
        return None
    damping = 1e-3
    for _ in range(35):
        projected = _project(model,camera,rotation,translation)
        residual = (projected-pixels).ravel()
        jacobian = np.empty((8,6))
        for axis in range(6):
            perturb = np.zeros(6)
            perturb[axis] = 1e-5 if axis < 3 else 1e-5*max(1.,abs(translation[axis-3]))
            shifted_rotation = _exp_so3(perturb[:3])@rotation
            shifted_translation = translation+perturb[3:]
            shifted = _project(model,camera,shifted_rotation,shifted_translation)
            if shifted is None:
                return None
            jacobian[:,axis] = ((shifted-projected)/perturb[axis]).ravel()
        normal = jacobian.T@jacobian
        step = np.linalg.solve(normal+damping*np.diag(np.diag(normal)+1),
                               -jacobian.T@residual)
        candidate_rotation = _exp_so3(step[:3])@rotation
        candidate_translation = translation+step[3:]
        candidate = _project(model,camera,candidate_rotation,candidate_translation)
        if candidate is not None and np.sum((candidate-pixels)**2) < np.sum(residual**2):
            rotation,translation = candidate_rotation,candidate_translation
            damping = max(damping/3,1e-9)
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            damping = min(damping*10,1e9)
    projected = _project(model,camera,rotation,translation)
    return rotation,translation,float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))


def estimate_window_pose(observation: WindowObservation, model: WindowModel,
                         camera: CameraCalibration) -> WindowPoseEstimate:
    """Fit a cheirality-valid planar homography and refine six pose parameters.

    All inputs are image measurements, known dimensions, and camera calibration.
    Near-collapsed quadrilaterals and ill-conditioned planar solutions fail closed.
    """
    if observation.window_id != model.window_id:
        return _failed(observation,"window_id_mismatch")
    pixels = np.asarray(observation.pixels_uv,dtype=float)
    mask = np.asarray(observation.visible,dtype=bool)
    if pixels.shape != (4,2) or mask.shape != (4,):
        return _failed(observation,"malformed_observation")
    if not observation.usable or not np.all(np.isfinite(pixels)):
        return _failed(observation,observation.reason or "insufficient_visible_corners")
    edges = np.roll(pixels,-1,axis=0)-pixels
    area = abs(float(np.sum(pixels[:,0]*np.roll(pixels[:,1],-1)
                            -pixels[:,1]*np.roll(pixels[:,0],-1))))/2
    if area < 100 or np.min(np.linalg.norm(edges,axis=1)) < 5:
        return _failed(observation,"degenerate_geometry")
    local = model.local_corners_m()[:,:2]
    normalized = np.column_stack(((pixels[:,0]-camera.cx_px)/camera.fx_px,
                                   (pixels[:,1]-camera.cy_px)/camera.fy_px))
    rows = []
    for (x,y),(u,v) in zip(local,normalized):
        rows.extend(([x,y,1,0,0,0,-u*x,-u*y,-u],
                     [0,0,0,x,y,1,-v*x,-v*y,-v]))
    _,singular,vh = np.linalg.svd(np.asarray(rows))
    if singular[-2] < 1e-9 or singular[0]/singular[-2] > 1e8:
        return _failed(observation,"degenerate_geometry")
    homography = vh[-1].reshape((3,3))
    if homography[2,2] < 0:
        homography *= -1
    scale = (np.linalg.norm(homography[:,0])+np.linalg.norm(homography[:,1]))/2
    if scale < 1e-12:
        return _failed(observation,"degenerate_geometry")
    first = homography[:,0]/scale
    second = homography[:,1]/scale
    approx = np.column_stack((first,second,np.cross(first,second)))
    u,_,vh_r = np.linalg.svd(approx)
    rotation = u@np.diag([1,1,np.linalg.det(u@vh_r)])@vh_r
    translation = homography[:,2]/scale
    refined = _refine(model,camera,pixels,rotation,translation)
    if refined is None:
        return _failed(observation,"nonpositive_depth")
    rotation,translation,rmse = refined
    # Reflection of the plane tilt gives the standard competing planar branch.
    # Refine it independently and withhold a pose when both branches fit within
    # a subpixel margin but disagree materially in orientation.
    mirror = np.diag([-1.,-1.,1.])
    alternative = _refine(model,camera,pixels,mirror@rotation@mirror,translation.copy())
    if alternative is not None:
        alternative_rotation,_,alternative_rmse = alternative
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(alternative_rotation@rotation.T)-1)/2,-1,1))))
        if angle > 10 and alternative_rmse <= rmse+0.5:
            return _failed(observation,"ambiguous_planar_pose")
    # A tiny projected area relative to edge length has a poorly constrained
    # plane normal even if its reprojection error happens to be small.
    depth = (model.local_corners_m()@rotation.T+translation)[:,2]
    if np.max(depth)/np.min(depth) > 50 or abs(rotation[2,2]) < 0.025:
        return _failed(observation,"ambiguous_planar_pose")
    if rmse > 8:
        return _failed(observation,"high_reprojection_error")
    normal = rotation[:,2].copy()
    side = "positive_normal" if normal[2] < 0 else "negative_normal"
    return WindowPoseEstimate(observation.window_id,True,None,rotation,translation,
                              normal,side,rmse,4)
