"""Image-only OpenCV blob detection and prediction-gated asteroid association.

The detector has no access to dynamics truth. It reports optical centroids, not
centres of mass. The tracker needs either a target-pixel prior or a resolved
two-body acquisition hypothesis. A single unknown bright blob is ambiguous.
API calls used here are shared by OpenCV 4 and OpenCV 5.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


def require_opencv():
    """Import lazily so the original dynamics package stays usable without cv2."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Image simulation requires OpenCV; install the project's vision dependencies.") from exc
    return cv2


@dataclass
class Candidate:
    centroid_px: np.ndarray
    radius_px: float
    covariance_px2: np.ndarray
    area_px2: float
    bbox_xywh: tuple
    score: float
    touches_border: bool = False
    ellipse_axes_px: Optional[tuple] = None
    ellipse_angle_deg: Optional[float] = None

    def to_dict(self):
        return {
            "centroid_px": self.centroid_px.tolist(),
            "radius_px": float(self.radius_px),
            "covariance_px2": self.covariance_px2.tolist(),
            "area_px2": float(self.area_px2),
            "bbox_xywh": list(self.bbox_xywh),
            "score": float(self.score),
            "touches_border": bool(self.touches_border),
            "ellipse_axes_px": self.ellipse_axes_px,
            "ellipse_angle_deg": self.ellipse_angle_deg,
        }


class AsteroidDetector:
    """Contrast-normalize, denoise, segment and measure disconnected objects.

    Calibrated arrays retain negative radiance until robust normalization.
    PDS nonfinite/sentinel values (|I| >= 1e8) and a supplied validity mask are
    excluded. The threshold is deliberately permissive for the small secondary;
    opening and area filtering suppress isolated hot pixels. Component size is
    an optical extent estimate and depends on illumination, blur and threshold.
    """
    def __init__(self, min_area_px2=6., threshold_u8=16., noise_sigma=6.,
                 centroid_sigma_px=1.5, relative_bias_sigma=0.03):
        if min_area_px2 <= 0 or centroid_sigma_px <= 0:
            raise ValueError("Detector area and centroid uncertainty must be positive")
        self.min_area_px2 = float(min_area_px2)
        self.threshold_u8 = float(threshold_u8)
        self.noise_sigma = float(noise_sigma)
        self.centroid_sigma_px = float(centroid_sigma_px)
        self.relative_bias_sigma = float(relative_bias_sigma)
        self.last_normalized = None
        self.last_binary = None
        self.last_valid_mask = None

    def normalize(self, image, valid_mask=None):
        cv = require_opencv()
        source = np.asarray(image)
        if source.ndim == 3:
            if source.shape[2] not in (3, 4):
                raise ValueError("Color images must have three BGR or four BGRA channels")
            # Float conversion supports calibrated float64 color data too.
            source = cv.cvtColor(source[..., :3].astype(np.float32), cv.COLOR_BGR2GRAY)
        if source.ndim != 2 or source.size == 0:
            raise ValueError("Expected a nonempty two-dimensional image")
        finite = np.isfinite(source) & (np.abs(source) < 1e8)
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=bool)
            if supplied.shape != source.shape:
                raise ValueError("Validity mask must have the image shape")
            finite &= supplied
        normalized = np.zeros(source.shape, dtype=np.uint8)
        values = source[finite].astype(float)
        if values.size:
            low = float(np.percentile(values, 1.))
            high = float(np.percentile(values, 99.98))
            # Sparse small targets can occupy less than 0.02% of the image.
            if high <= low + np.finfo(float).eps * max(1., abs(low)):
                high = float(values.max())
            if high > low:
                normalized[finite] = np.clip(255. * (values-low)/(high-low), 0, 255).astype(np.uint8)
        return normalized, finite

    def detect(self, image, valid_mask=None):
        cv = require_opencv()
        normalized, valid = self.normalize(image, valid_mask)
        self.last_normalized = normalized
        self.last_valid_mask = valid
        if not np.any(valid) or not np.any(normalized):
            self.last_binary = np.zeros_like(normalized)
            return []
        smooth = cv.GaussianBlur(normalized, (3, 3), 0.65)
        background = smooth[valid]
        median = float(np.median(background))
        sigma = 1.4826 * float(np.median(np.abs(background.astype(float)-median)))
        threshold = max(self.threshold_u8, median + self.noise_sigma * max(sigma, 0.25))
        binary = ((smooth > threshold) & valid).astype(np.uint8) * 255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        binary[~valid] = 0
        self.last_binary = binary
        count, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        height, width = normalized.shape
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < self.min_area_px2:
                continue
            region = (labels[y:y+h, x:x+w] == label)
            yy, xx = np.nonzero(region)
            weights = smooth[y:y+h, x:x+w][region].astype(float) - median
            weights = np.maximum(weights, 0.01)
            center = np.array([x + np.average(xx, weights=weights),
                               y + np.average(yy, weights=weights)])
            radius = float(np.sqrt(area/np.pi))
            # An explicit unresolved/illumination floor avoids fictitious subpixel
            # precision from a bright but biased resolved-body photocentre.
            variance = self.centroid_sigma_px**2 + (self.relative_bias_sigma*radius)**2
            covariance = np.eye(2)*variance
            component_mask = region.astype(np.uint8)*255
            contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv.contourArea)
            perimeter = cv.arcLength(contour, True)
            circularity = min(1., 4*np.pi*area/max(perimeter**2, 1.))
            axes = angle = None
            if len(contour) >= 5:
                _, ellipse_axes, ellipse_angle = cv.fitEllipse(contour)
                axes = tuple(map(float, ellipse_axes))
                angle = float(ellipse_angle)
            # Invalid-data boundaries are clipping boundaries too.
            local_valid = valid[max(0,y-1):min(height,y+h+1), max(0,x-1):min(width,x+w+1)]
            touches = (x == 0 or y == 0 or x+w == width or y+h == height
                       or not np.all(local_valid))
            score = float(circularity*np.mean(weights)/255.)
            candidates.append(Candidate(center, radius, covariance, float(area),
                                        (x,y,w,h), score, touches, axes, angle))
        return sorted(candidates, key=lambda c: c.area_px2, reverse=True)


class AsteroidTracker:
    """Constant-velocity pixel Kalman association with explicit lost/coast modes.

    External predicted_uv/covariance may come from the navigation filter. Without
    a prior the initial hypothesis uses a secondary smaller than a resolved
    primary. This is a demo association prior, not semantic body recognition.
    Acquired tracks then use position gates and size continuity, so primary
    brightness cannot silently substitute for the target after a dropout.
    """
    def __init__(self, max_missed=5, gate_chi2=13.8155, initial_sigma_px=30.,
                 acceleration_sigma_px_s2=2., allow_single=False,
                 max_area_ratio=6., primary_area_ratio=2.5):
        self.max_missed = int(max_missed)
        self.gate_chi2 = float(gate_chi2)
        self.initial_sigma_px = float(initial_sigma_px)
        self.acceleration_sigma_px_s2 = float(acceleration_sigma_px_s2)
        self.allow_single = bool(allow_single)
        self.max_area_ratio = float(max_area_ratio)
        self.primary_area_ratio = float(primary_area_ratio)
        self.x = None
        self.P = None
        self.time_s = None
        self.last_candidate = None
        self.missed_frames = 0
        self.status = "SEARCHING"
        self.mode = "search"
        self.last_innovation_chi2 = float("nan")

    def _acquire_without_prior(self, candidates):
        ordered = sorted(candidates, key=lambda c:c.area_px2, reverse=True)
        if len(ordered) == 1:
            return ordered[0] if self.allow_single else None
        if len(ordered) < 2:
            return None
        primary = ordered[0]
        plausible = [c for c in ordered[1:]
                     if primary.area_px2 >= self.primary_area_ratio*c.area_px2
                     and np.linalg.norm(primary.centroid_px-c.centroid_px)
                     > primary.radius_px+c.radius_px]
        return plausible[0] if plausible else None

    def update(self, candidates, timestamp_s, predicted_uv=None,
               prediction_covariance_px2=None):
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("Timestamp must be finite")
        if self.time_s is not None and timestamp_s < self.time_s-1e-10:
            raise ValueError("Tracker timestamps must be nondecreasing")
        dt = 0. if self.time_s is None else max(0., timestamp_s-self.time_s)
        self.time_s = timestamp_s
        if self.x is not None and dt:
            transition = np.eye(4)
            transition[:2,2:] = dt*np.eye(2)
            noise_map = np.vstack([.5*dt**2*np.eye(2), dt*np.eye(2)])
            self.x = transition@self.x
            self.P = transition@self.P@transition.T + self.acceleration_sigma_px_s2**2*(noise_map@noise_map.T)
        prior = None
        covariance = None
        if predicted_uv is not None:
            prior = np.asarray(predicted_uv, dtype=float)
            if prior.shape != (2,) or not np.all(np.isfinite(prior)):
                raise ValueError("Prediction must contain two finite pixel coordinates")
            covariance = (self.P[:2,:2].copy() if self.P is not None
                          else np.eye(2)*self.initial_sigma_px**2)
            if prediction_covariance_px2 is not None:
                covariance = np.asarray(prediction_covariance_px2, dtype=float)
                if (covariance.shape != (2,2) or not np.all(np.isfinite(covariance))
                        or not np.allclose(covariance,covariance.T)
                        or np.min(np.linalg.eigvalsh(covariance)) < 0):
                    raise ValueError("Prediction covariance must be symmetric positive semidefinite")
        elif self.x is not None:
            prior, covariance = self.x[:2], self.P[:2,:2]
        selected = None
        if prior is None:
            selected = self._acquire_without_prior(candidates)
        else:
            eligible = []
            for candidate in candidates:
                ratio = 1. if self.last_candidate is None else candidate.area_px2/self.last_candidate.area_px2
                if self.last_candidate is not None and not 1/self.max_area_ratio <= ratio <= self.max_area_ratio:
                    continue
                delta = candidate.centroid_px-prior
                innovation_covariance = covariance+candidate.covariance_px2
                chi2 = float(delta @ np.linalg.solve(innovation_covariance,delta))
                if chi2 <= self.gate_chi2:
                    eligible.append((chi2+0.5*np.log(ratio)**2, chi2, candidate))
            if eligible:
                _, self.last_innovation_chi2, selected = min(eligible,key=lambda item:item[0])
        if selected is None:
            self.missed_frames += 1
            if self.x is None:
                self.status, self.mode = "SEARCHING", "search"
            elif self.missed_frames > self.max_missed:
                self.status, self.mode = "LOST", "lost"
            else:
                self.status, self.mode = "COAST", "coast"
            return None
        first = self.x is None
        had_gap = self.missed_frames > 0
        if first:
            self.x = np.r_[selected.centroid_px, 0., 0.]
            self.P = np.diag([*np.diag(selected.covariance_px2),
                              self.initial_sigma_px**2, self.initial_sigma_px**2])
        else:
            h = np.c_[np.eye(2), np.zeros((2,2))]
            gain = np.linalg.solve(h@self.P@h.T+selected.covariance_px2, h@self.P).T
            self.x += gain@(selected.centroid_px-h@self.x)
            residual_map = np.eye(4)-gain@h
            self.P = residual_map@self.P@residual_map.T+gain@selected.covariance_px2@gain.T
            self.P = (self.P+self.P.T)/2
        self.last_candidate = selected
        self.missed_frames = 0
        self.status = "ACQUIRED" if first else "REACQUIRED" if had_gap else "TRACKED"
        self.mode = "track"
        return selected


def annotate_candidates(image, candidates, target=None):
    """Return a BGR diagnostic overlay; the overlay never feeds the detector."""
    cv = require_opencv()
    gray, _ = AsteroidDetector().normalize(image)
    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    label_positions = []
    for index, candidate in enumerate(candidates):
        selected = candidate is target
        color = (80,230,80) if selected else (255,190,60)
        x,y,w,h = candidate.bbox_xywh
        cv.rectangle(canvas, (x,y), (x+w-1,y+h-1), color, 1)
        center = tuple(np.rint(candidate.centroid_px).astype(int))
        cv.drawMarker(canvas, center, color, cv.MARKER_CROSS, 12, 1)
        text = f"{'target hypothesis' if selected else 'blob '+str(index)} r={candidate.radius_px:.1f}px"
        label_y = max(15,y-8)
        while any(abs(label_y-old)<16 for old in label_positions) and label_y>30:
            label_y -= 17
        label_positions.append(label_y)
        cv.putText(canvas, text, (max(0,x-15),label_y), cv.FONT_HERSHEY_SIMPLEX,.43,color,1,cv.LINE_AA)
    return canvas
