"""Independent camera derivatives and causal pixel-feedback outcome checks."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
pytest.importorskip("cv2")
from opendart.planar import (PlanarConfig, CameraRenderer, simulate_planar,
                            measurement_model, camera_axes)
from opendart.model import Binary
from opendart.vision import AsteroidDetector


def compatible_config(**kwargs):
    # Tests report actual runtime; the production default still requires OpenCV 5.
    return PlanarConfig(require_opencv5=False, **kwargs)


def test_camera_derivatives_and_scale_ambiguity():
    x = np.array([300., 10000., -.2, -20.])
    theta = 1.53
    prediction, h = measurement_model(x, theta, 700., 88., 255.5)
    eps = .01
    numerical = np.column_stack([
        (measurement_model(x+eps*np.eye(4)[k], theta, 700., 88., 255.5)[0]
         - measurement_model(x-eps*np.eye(4)[k], theta, 700., 88., 255.5)[0])/(2*eps)
        for k in range(4)])
    assert_allclose(h, numerical, atol=1e-8)
    doubled = measurement_model(2*x, theta, 700., 88., 255.5)[0]
    assert_allclose(doubled[0], prediction[0])
    assert_allclose(doubled[1]-prediction[1], -np.log(2))
    right, forward = camera_axes(np.pi/2)
    assert_allclose(right, [1, 0], atol=1e-15)
    assert_allclose(forward, [0, 1], atol=1e-15)


def test_rendered_pixels_change_with_observer_motion():
    cfg = compatible_config(noise_dn=0.)
    b = Binary(period_s=cfg.period_s)
    renderer = CameraRenderer(cfg)
    pos = b.states(0.)[0][1, :2]-[0., 12000.]
    first = renderer.render(0., pos, np.pi/2, b)
    second = renderer.render(0., pos+[200., 0.], np.pi/2, b)
    assert np.count_nonzero(first != second) > 100
    detector = AsteroidDetector()
    # Select secondary from separated viable components by optical area only.
    a = sorted(detector.detect(first), key=lambda c:c.area_px2, reverse=True)[1]
    c = sorted(detector.detect(second), key=lambda c:c.area_px2, reverse=True)[1]
    assert c.centroid_px[0] < a.centroid_px[0]-8.


def test_reference_image_is_used_as_appearance_not_state():
    reference = Path(__file__).resolve().parents[1]/"assets"/"dart_0401929788_46871_01_iof.png"
    cfg = compatible_config(noise_dn=0.)
    b = Binary(period_s=cfg.period_s)
    observer = b.states(cfg.epoch_s)[0][1, :2]-[0., 12000.]
    analytic = CameraRenderer(cfg).render(cfg.epoch_s, observer, np.pi/2, b)
    renderer = CameraRenderer(replace(cfg, reference_image=str(reference)))
    textured = renderer.render(cfg.epoch_s, observer, np.pi/2, b)
    assert renderer.templates is not None and len(renderer.templates) == 2
    assert np.count_nonzero(textured != analytic) > 100
    assert len(AsteroidDetector().detect(textured)) >= 2


@pytest.fixture(scope="module")
def controlled():
    return simulate_planar(compatible_config())


def test_pixels_control_actual_contact(controlled):
    summary = controlled["summary"]
    assert summary["status"] == "TARGET_IMPACT"
    assert summary["accepted_updates"] > 400
    assert summary["terminal_relative_speed_mps"] > 15.
    assert 85 < summary["terminal_center_distance_m"] < 90
    assert summary["delta_v_mps"] > .1
    assert 0 < summary["propellant_used_kg"] < 3


def test_target_loss_holds_thrust_and_reacquires(controlled):
    rows = controlled["telemetry"]
    held = [r for r in rows if r["mode"] == "NAV_HOLD"]
    assert len(held) >= 10
    assert all(r["force_n"] == 0 for r in held)
    assert any(r["tracker"] == "REACQUIRED" and r["elapsed_s"] >= 220 for r in rows)
    assert all(not r["accepted"] for r in rows if 200 <= r["elapsed_s"] < 220)


def test_absent_pixels_do_not_use_truth():
    run = simulate_planar(compatible_config(measurements=False))
    assert run["summary"]["status"] == "MISS"
    assert run["summary"]["accepted_updates"] == 0
    assert run["summary"]["delta_v_mps"] == 0
    assert all(r["force_n"] == 0 for r in run["telemetry"])


def test_ballistic_comparison_misses():
    run = simulate_planar(compatible_config(control=False))
    assert run["summary"]["status"] == "MISS"
    assert run["summary"]["delta_v_mps"] == 0
    assert run["summary"]["terminal_center_distance_m"] > 300
