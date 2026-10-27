"""Pixel-only OpenCV interface for MATLAB's independent planar simulator.

JSON return values deliberately avoid MATLAB release-specific conversion of
Python dataclasses or nested NumPy arrays. Images enter as uint8 MATLAB matrices
and are copied into contiguous NumPy storage before OpenCV sees them. Neither
detector function accepts spacecraft or asteroid truth states.
"""
from __future__ import annotations

import json
import numpy as np


def runtime_info_json(require_opencv5: bool = True) -> str:
    import cv2
    import sys

    major = int(cv2.__version__.split(".")[0])
    if require_opencv5 and major != 5:
        raise RuntimeError(
            f"MATLAB selected OpenCV {cv2.__version__}. Select the Python "
            "environment containing OpenCV 5 with pyenv before running, or "
            "explicitly use RequireOpenCV5=false to test the compatibility path."
        )
    return json.dumps({"opencv_version": cv2.__version__,
                       "numpy_version": np.__version__,
                       "python_executable": sys.executable,
                       "python_version": sys.version.split()[0]})


def detect_json(image) -> str:
    """Return image detections, using zero-based OpenCV pixel coordinates."""
    from .vision import AsteroidDetector

    pixels = np.asarray(image)
    # imread in MATLAB uses RGB(A); OpenCV's public detector uses BGR(A).
    if pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        channels = [2, 1, 0] if pixels.shape[2] == 3 else [2, 1, 0, 3]
        pixels = pixels[:, :, channels]
    pixels = np.ascontiguousarray(pixels)
    candidates = AsteroidDetector().detect(pixels)
    return json.dumps([
        {"centroid_px": np.asarray(c.centroid_px, dtype=float).tolist(),
         "radius_px": float(c.radius_px),
         "covariance_px2": np.asarray(c.covariance_px2, dtype=float).tolist(),
         "area_px2": float(c.area_px2),
         "score": float(c.score),
         "touches_border": bool(c.touches_border)}
        for c in candidates
    ], allow_nan=False)

