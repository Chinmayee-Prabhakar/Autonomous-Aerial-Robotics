"""M12 pixel-only color adaptation for lit Blender RGB images.

The accepted M9/M10/M11 detectors are imported unchanged for paired scoring.
These masks use only the declared target hue and image pixel ratios; no depth,
object ID, projected truth corner, or scene transform enters detection.
"""

from __future__ import annotations

import numpy as np

from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m10_vision import PixelArmDetector


class AdaptiveColorWindowDetector:
    def __init__(self, colors_by_id):
        self.colors_by_id = dict(colors_by_id)
        self._fitter = {key: ColorWindowDetector({key: (245, 0, 0)})
                        for key in self.colors_by_id}

    def color_mask(self, image_rgb, target_id):
        a = np.asarray(image_rgb, dtype=np.float32)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("expected RGB image")
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        target = np.asarray(self.colors_by_id[target_id])
        if target[1] > target[0] and target[2] > target[0]:
            # Cyan delta: require both cool channels to dominate red.
            return ((g > 105) & (b > 100) & (g > 1.38 * (r + 1))
                    & (b > 1.30 * (r + 1)) & (abs(g - b) < 85))
        dominant = int(np.argmax(target))
        channel = (r, g, b)[dominant]
        others = [(r, g, b)[i] for i in range(3) if i != dominant]
        return ((channel > 105) & (channel > 1.35 * (others[0] + 1))
                & (channel > 1.35 * (others[1] + 1)))

    def detect(self, image_rgb, camera, target_id, expected_side="positive_normal"):
        if target_id not in self.colors_by_id:
            return ColorWindowDetector({}).detect(image_rgb, camera, target_id,
                                                   expected_side)
        image = np.asarray(image_rgb)
        if image.shape != (camera.height_px, camera.width_px, 3) or image.dtype != np.uint8:
            return self._fitter[target_id].detect(image_rgb, camera, target_id,
                                                   expected_side)
        mask = self.color_mask(image, target_id)
        isolated = np.zeros_like(image)
        isolated[mask, 0] = 245
        return self._fitter[target_id].detect(isolated, camera, target_id,
                                               expected_side)


class AdaptivePixelArmDetector:
    def __init__(self):
        self._accepted_fitter = PixelArmDetector()

    def detect(self, image_rgb, fix, timestamp_s):
        a = np.asarray(image_rgb, dtype=np.float32)
        if a.ndim != 3 or a.shape[2] != 3:
            return self._accepted_fitter.detect(image_rgb, fix, timestamp_s)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        shaft = ((r > 115) & (g > 95) & (b < .70 * r)
                 & (b < .76 * g) & (r > 1.10 * g))
        hub = ((r > 110) & (b > 100) & (g < .66 * r)
               & (g < .72 * b))
        isolated = np.zeros_like(np.asarray(image_rgb, dtype=np.uint8))
        isolated[shaft] = [248, 204, 33]
        isolated[hub] = [246, 39, 233]
        return self._accepted_fitter.detect(isolated, fix, timestamp_s)
