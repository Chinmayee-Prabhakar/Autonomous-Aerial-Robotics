"""Known-color isolation before the accepted M9 pixel-frame fitting method.

The fourth frame uses cyan, beyond M9's three distinct dominant channels.
Only RGB pixels and the declared target color enter this adapter.
"""

import numpy as np

from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector


class TargetColorDetector:
    def __init__(self, colors_by_id):
        if len(colors_by_id) != 1:
            raise ValueError("one active known-color target is required")
        self.colors_by_id = dict(colors_by_id)
        self._fitter = ColorWindowDetector({next(iter(colors_by_id)): (245, 0, 0)})

    def detect(self, image_rgb, camera, window_id, expected_side="positive_normal"):
        if window_id not in self.colors_by_id:
            return self._fitter.detect(image_rgb, camera, window_id, expected_side)
        image = np.asarray(image_rgb)
        if image.shape != (camera.height_px, camera.width_px, 3) or image.dtype != np.uint8:
            return self._fitter.detect(image_rgb, camera, window_id, expected_side)
        declared = np.asarray(self.colors_by_id[window_id], dtype=np.int16)
        mask = np.max(np.abs(image.astype(np.int16)-declared), axis=2) <= 30
        isolated = np.zeros_like(image)
        isolated[mask, 0] = 245
        return self._fitter.detect(isolated, camera, window_id, expected_side)
