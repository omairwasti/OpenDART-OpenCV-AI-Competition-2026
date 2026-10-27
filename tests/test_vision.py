"""Pixel measurements, association and failure modes; no image truth input."""
import os
from pathlib import Path

import numpy as np
import pytest

cv = pytest.importorskip("cv2")
from opendart.vision import AsteroidDetector, AsteroidTracker, Candidate


def scene(secondary=(175,145), primary=(65,120), radius=8):
    image = np.zeros((256,256), dtype=np.uint8)
    cv.circle(image, primary, 30, 220, -1, cv.LINE_AA)
    if secondary is not None:
        cv.circle(image, secondary, radius, 140, -1, cv.LINE_AA)
    return image


def blob(u,v,radius):
    return Candidate(np.array([u,v],dtype=float),float(radius),np.eye(2)*2.25,
                     float(np.pi*radius**2),(int(u-radius),int(v-radius),2*radius,2*radius),.8)


def test_image_detector_measures_both_bodies_and_ignores_hot_pixels():
    image = scene()
    image[20,20] = 255
    image[75,210] = 255
    detections = AsteroidDetector().detect(image)
    assert len(detections) == 2
    assert np.linalg.norm(detections[0].centroid_px-[65,120]) < .3
    assert np.linalg.norm(detections[1].centroid_px-[175,145]) < .3
    assert 7 < detections[1].radius_px < 10
    assert np.linalg.eigvalsh(detections[1].covariance_px2).min() >= 2.25
    target = AsteroidTracker().update(detections,0.)
    assert target is detections[1]


def test_calibrated_valid_mask_excludes_sentinel_and_masked_bright_patch():
    image = scene().astype(float)*1e-4-0.001
    image[:10,:10] = 1e30
    image[10:20,:10] = np.nan
    image[200:225,190:215] = .025
    valid = np.ones(image.shape,dtype=bool)
    valid[198:228,188:218] = False
    detector = AsteroidDetector()
    candidates = detector.detect(image,valid)
    assert len(candidates) == 2
    assert not detector.last_binary[:20,:10].any()
    assert not detector.last_binary[198:228,188:218].any()
    assert not detector.last_valid_mask[:20,:10].any()
    assert detector.detect(np.full((16,16),np.nan)) == []
    with pytest.raises(ValueError,match="Validity mask"):
        detector.detect(image,np.ones((2,2),dtype=bool))


def test_single_unknown_blob_does_not_establish_dimorphos_identity():
    tracker = AsteroidTracker()
    assert tracker.update([blob(40,50,30)],0.) is None
    assert tracker.status == "SEARCHING"
    prior_target = blob(40,50,8)
    assert tracker.update([prior_target],1.,predicted_uv=np.array([41,49])) is prior_target
    assert tracker.status == "ACQUIRED"


def test_tracking_prediction_rejects_primary_distractor_and_reacquires():
    tracker = AsteroidTracker(max_missed=1)
    first = blob(175,145,8)
    assert tracker.update([blob(65,120,30),first],0.) is first
    second = blob(178,145,8)
    assert tracker.update([blob(65,120,30),second,blob(215,200,8)],1.,
                          predicted_uv=[178,145],prediction_covariance_px2=np.eye(2)) is second
    assert tracker.status == "TRACKED"
    # A much brighter primary at almost the same location must not steal identity.
    assert tracker.update([blob(181,145,30)],2.,predicted_uv=[181,145]) is None
    assert tracker.status == "COAST"
    assert tracker.update([],3.,predicted_uv=[184,145]) is None
    assert tracker.status == "LOST"
    recovered = blob(187,145,9)
    assert tracker.update([blob(65,120,30),recovered],4.,predicted_uv=[187,145],
                          prediction_covariance_px2=np.eye(2)*4) is recovered
    assert tracker.status == "REACQUIRED"
    assert tracker.missed_frames == 0
    assert np.linalg.eigvalsh(tracker.P).min() > 0
    with pytest.raises(ValueError,match="nondecreasing"):
        tracker.update([],3.)


def test_edge_clipped_blobs_are_marked_and_bgr_supported():
    image = scene(secondary=(254,145),radius=9)
    candidates = AsteroidDetector().detect(cv.cvtColor(image,cv.COLOR_GRAY2BGR))
    assert len(candidates) == 2
    assert any(c.touches_border for c in candidates)
    assert not candidates[0].touches_border


def test_supplied_draco_preview_when_available():
    bundled = Path(__file__).resolve().parents[1]/"assets"/"dart_0401929788_46871_01_iof.png"
    supplied = Path(os.environ.get("OPENDART_TEST_IMAGE", str(bundled)))
    if not supplied.exists():
        pytest.skip("Set OPENDART_TEST_IMAGE to run the supplied DRACO preview regression")
    candidates = AsteroidDetector().detect(cv.imread(str(supplied),cv.IMREAD_UNCHANGED))
    target = AsteroidTracker().update(candidates,0.)
    assert target is not None
    assert np.linalg.norm(target.centroid_px-[662,669]) < 5.
    assert 7 < target.radius_px < 14
    assert np.linalg.norm(candidates[0].centroid_px-[562,581]) < 8.
    assert candidates[0].area_px2 > 10*target.area_px2
