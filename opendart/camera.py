"""Ideal geometric camera and detector interface. No OpenCV or image reader."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    fx: float = 2628.326 / 0.013  # mm / binned-pixel pitch in mm
    fy: float = 2628.326 / 0.013
    cx: float = 511.5  # Synthetic centered principal point, NOT calibrated DRACO value
    cy: float = 511.5
    width: int = 1024
    height: int = 1024

    def project(self, relative_position, R_CI):
        """OpenCV convention: camera x right, y down, z forward; R_CI maps I to C."""
        x, y, z = np.asarray(R_CI) @ relative_position
        if z <= 0:
            return None
        return np.array([self.fx*x/z + self.cx, self.fy*y/z + self.cy])

    def jacobian(self, relative_position, R_CI):
        x, y, z = np.asarray(R_CI) @ relative_position
        if z <= 0:
            raise ValueError("Target behind camera")
        h = np.zeros((2, 6))
        h[:, :3] = np.array([[self.fx/z, 0, -self.fx*x/z**2],
                             [0, self.fy/z, -self.fy*y/z**2]]) @ R_CI
        return h

    def contains(self, uv):
        return (uv is not None and 0 <= uv[0] < self.width
                and 0 <= uv[1] < self.height)


@dataclass
class Detection:
    timestamp_s: float
    uv_px: np.ndarray
    covariance_px2: np.ndarray
    target_id: int = 1
    valid: bool = True


def pointing_rotation(estimated_los):
    """Instantaneous ideal pointing from the ESTIMATE; no hidden truth pointing."""
    z = np.asarray(estimated_los) / np.linalg.norm(estimated_los)
    up = np.array([0., 1., 0.])
    if abs(z @ up) > 0.95:
        up = np.array([1., 0., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def crop_intrinsics(camera, x0, y0, width, height):
    """Crop coordinates in the already oriented full image; no binning change."""
    return Camera(camera.fx, camera.fy, camera.cx-x0, camera.cy-y0, width, height)


def calibrated_valid_mask(array):
    """Mask Ernst 2024 §6.8 sentinels and nonfinite values; retain negative radiance."""
    return np.isfinite(array) & (np.abs(array) < 1e8)
