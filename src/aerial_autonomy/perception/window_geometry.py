"""Stationary rectangular openings in ENU, with ordered local corners."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WindowModel:
    """Estimator-visible dimensions and declared corner correspondences only."""

    window_id: str
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        if (not self.window_id or not math.isfinite(self.width_m) or self.width_m <= 0
                or not math.isfinite(self.height_m) or self.height_m <= 0):
            raise ValueError("window needs an identifier and positive finite dimensions")

    def local_corners_m(self) -> np.ndarray:
        """Bottom-left, bottom-right, top-right, top-left from the +normal side."""
        half_w, half_h = self.width_m/2, self.height_m/2
        return np.array([[-half_w,-half_h,0], [half_w,-half_h,0],
                         [half_w,half_h,0], [-half_w,half_h,0]], dtype=float)


@dataclass(frozen=True)
class StaticWindow:
    model: WindowModel
    center_world_m: np.ndarray
    right_world: np.ndarray
    up_world: np.ndarray
    frame_thickness_m: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=float)
        right = np.asarray(self.right_world, dtype=float)
        up = np.asarray(self.up_world, dtype=float)
        if (any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in (center,right,up))
                or not np.isclose(np.linalg.norm(right),1,atol=1e-10)
                or not np.isclose(np.linalg.norm(up),1,atol=1e-10)
                or abs(float(right@up)) > 1e-10
                or not math.isfinite(self.frame_thickness_m) or self.frame_thickness_m < 0):
            raise ValueError("window frame requires finite orthonormal axes and thickness")
        object.__setattr__(self,"center_world_m",center)
        object.__setattr__(self,"right_world",right)
        object.__setattr__(self,"up_world",up)

    @property
    def normal_world(self) -> np.ndarray:
        return np.cross(self.right_world,self.up_world)

    @property
    def rotation_window_to_world(self) -> np.ndarray:
        return np.column_stack((self.right_world,self.up_world,self.normal_world))

    def world_corners_m(self) -> np.ndarray:
        return self.center_world_m + self.model.local_corners_m() @ self.rotation_window_to_world.T

    def opening_coordinates(self, point_world_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_world_m,dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("opening point must be a finite three-vector")
        return self.rotation_window_to_world.T @ (point-self.center_world_m)

    def clears_opening(self, point_world_m: np.ndarray, radius_m: float) -> bool:
        """Cross-section clearance for later M9; does not assert an actual crossing."""
        if not math.isfinite(radius_m) or radius_m < 0:
            raise ValueError("radius must be finite and nonnegative")
        local = self.opening_coordinates(point_world_m)
        return (abs(local[0])+radius_m < self.model.width_m/2
                and abs(local[1])+radius_m < self.model.height_m/2)
