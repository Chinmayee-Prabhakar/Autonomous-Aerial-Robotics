"""Seeded synthetic corner observations; world truth stays inside this sensor."""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_geometry import StaticWindow
from aerial_autonomy.simulation.dynamics import VehicleState


@dataclass(frozen=True)
class ObservationConfig:
    pixel_noise_std: float = 0.0
    corner_dropout_probability: float = 0.0
    frame_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if (not math.isfinite(self.pixel_noise_std) or self.pixel_noise_std < 0
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in
                       (self.corner_dropout_probability,self.frame_dropout_probability))):
            raise ValueError("invalid observation noise or dropout probability")


@dataclass(frozen=True)
class WindowObservation:
    window_id: str
    timestamp_s: float
    pixels_uv: np.ndarray
    visible: np.ndarray
    reason: str | None

    @property
    def visible_count(self) -> int:
        return int(np.count_nonzero(self.visible))

    @property
    def usable(self) -> bool:
        return self.visible_count == 4 and self.reason is None


def observe_window(window: StaticWindow, camera: CameraCalibration, state: VehicleState,
                   config: ObservationConfig, rng: np.random.Generator,
                   timestamp_s: float = 0.0) -> WindowObservation:
    if not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    camera_points = camera.world_to_camera(state,window.world_corners_m())
    pixels,ahead,in_image = camera.project(camera_points)
    if rng.random() < config.frame_dropout_probability:
        return WindowObservation(window.model.window_id,timestamp_s,
                                 np.full((4,2),np.nan),np.zeros(4,dtype=bool),"frame_dropout")
    if config.pixel_noise_std:
        pixels[ahead] += rng.normal(0,config.pixel_noise_std,(int(np.count_nonzero(ahead)),2))
    # Clipping is determined from the measured image coordinates after noise.
    visible = (ahead & in_image & (pixels[:,0]>=0) & (pixels[:,0]<camera.width_px)
               & (pixels[:,1]>=0) & (pixels[:,1]<camera.height_px))
    visible &= rng.random(4) >= config.corner_dropout_probability
    pixels[~visible] = np.nan
    if np.count_nonzero(visible) == 4:
        reason = None
    elif not np.any(ahead):
        reason = "behind_camera"
    elif not np.any(in_image):
        reason = "outside_image"
    elif not np.any(visible):
        reason = "no_visible_corners"
    else:
        reason = "partial_visibility"
    return WindowObservation(window.model.window_id,timestamp_s,pixels,visible,reason)
