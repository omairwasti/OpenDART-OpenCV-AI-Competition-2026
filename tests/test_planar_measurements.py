"""Measurement calibration, update accounting and estimator evidence contracts."""
from dataclasses import replace
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

cv = pytest.importorskip("cv2")
from opendart.calibrate_size import CalibrationConfig, calibrate, summarize_calibration
from opendart.model import Environment
from opendart.planar import (PlanarConfig, PlanarEKF, UPDATE_STATUSES,
                             measurement_model, simulate_planar)
from opendart.vision import Candidate


@pytest.mark.parametrize("kwargs", [
    {"thrust_n": float("nan")}, {"thrust_n": float("inf")}, {"thrust_n": -1.},
    {"estimate_error": [0., 0., float("nan"), 0.]},
    {"prior_sigma": [1., float("nan"), 1., 1.]},
    {"prior_sigma": [1., 1., 1., float("inf")]}, {"prior_sigma": [1., 1., 0., 1.]},
    {"prior_sigma": [[1.], [1.], [1.], [1.]]},
    {"acceleration_psd": -1.}, {"acceleration_psd": float("nan")},
    {"acceleration_psd": float("inf")}, {"noise_dn": float("nan")},
    {"noise_dn": -1.}, {"camera_f_px": float("inf")}, {"camera_f_px": 0.},
    {"camera_rate_radps": float("nan")}, {"camera_width": 512.5},
    {"camera_width": True}, {"camera_height": float("nan")}, {"camera_height": 63},
    {"size_fractional_sigma": 0.}, {"size_fractional_sigma": float("nan")},
    {"size_radius_bias_px": -1.}, {"size_radius_bias_px": float("nan")},
    {"maximum_fractional_range_sigma": -1.}, {"maximum_fractional_range_sigma": float("inf")},
    {"initial_cross_m": float("inf")}, {"cross_velocity_mps": float("nan")},
    {"phase_rad": float("nan")}, {"epoch_s": float("inf")},
    {"blackout_s": [[0., float("inf")]]}, {"blackout_s": [[float("nan"), 20.]]},
    {"blackout_s": [[20., 10.]]}, {"blackout_s": [[0.]]},
    {"coast_tgo_s": float("nan")}, {"dropout_probability": float("nan")},
    {"seed": -1}, {"seed": 7.5}, {"overview_frames": -1}, {"overview_frames": 1.5},
    {"control": "false"}, {"mass_kg": 479.}, {"duration_s": 0.},
])
def test_invalid_planar_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        PlanarConfig(**kwargs).validate()


def test_zero_noise_force_and_legacy_bias_are_valid():
    PlanarConfig(thrust_n=0., acceleration_psd=0., noise_dn=0., size_radius_bias_px=0.,
                 coast_tgo_s=0., overview_frames=0, blackout_s=[], epoch_s=-1.).validate()


def make_filter():
    cfg = PlanarConfig()
    return PlanarEKF([0., 12000., 0., -20.], np.diag([1., 1., .1, .1]), cfg,
                     Environment(solar_tide=False, srp=False))


def matching_candidate():
    return Candidate(np.array([255.5, 255.5]), 700.*88./12000.+PlanarConfig().size_radius_bias_px,
                     np.eye(2)*2.25, 120., (249, 249, 13, 13), .8)


@pytest.mark.parametrize("reason", UPDATE_STATUSES)
def test_each_attempt_has_one_reason_and_correct_counter(reason):
    nav = make_filter()
    candidate = matching_candidate()
    count = 1
    if reason in ("NO_CANDIDATE", "ASSOCIATION_FAILED"):
        candidate = None
        count = 0 if reason == "NO_CANDIDATE" else 3
    elif reason == "CLIPPED":
        candidate = replace(candidate, touches_border=True)
    elif reason == "TOO_SMALL":
        candidate = replace(candidate, radius_px=1.)
    elif reason == "INVALID_GEOMETRY":
        nav.x[1] = -12000.
    elif reason == "INVALID_MEASUREMENT":
        candidate = replace(candidate, covariance_px2=-np.eye(2))
    elif reason == "NIS_REJECTED":
        candidate = replace(candidate, centroid_px=np.array([400., 255.5]))
    before = nav.x.copy()
    accepted = nav.update(candidate, np.pi/2, 0., candidate_count=count)
    assert accepted == (reason == "ACCEPTED")
    assert nav.last_diagnostics["update_status"] == reason
    assert nav.last_diagnostics["rejection_reason"] == ("" if accepted else reason)
    assert sum(nav.update_counts.values()) == 1
    assert nav.update_counts[reason] == 1
    assert nav.accepted == int(accepted)
    assert nav.rejected == int(reason not in ("ACCEPTED", "NO_CANDIDATE", "ASSOCIATION_FAILED"))
    if not accepted:
        assert_allclose(nav.x, before)
    if reason == "NIS_REJECTED":
        assert nav.nis > 9.21034
        assert nav.last_diagnostics["innovation_u_px"] > 100.


def test_missing_frame_does_not_reuse_previous_nis_or_innovation():
    nav = make_filter()
    assert nav.update(matching_candidate(), np.pi/2, 0.)
    assert np.isfinite(nav.nis)
    assert not nav.update(None, np.pi/2, 1.)
    assert math.isnan(nav.nis)
    assert math.isnan(nav.last_diagnostics["innovation_u_px"])
    assert nav.last_update_s == 0.
    assert nav.update_counts["NO_CANDIDATE"] == 1


def test_biased_measurement_jacobian_and_likelihood():
    x = np.array([300., 10000., -.2, -20.])
    bias = 1.07
    prediction, h = measurement_model(x, 1.53, 700., 88., 255.5, radius_bias_px=bias)
    ideal, _ = measurement_model(x, 1.53, 700., 88., 255.5)
    assert_allclose(np.exp(prediction[1]), np.exp(ideal[1])+bias)
    eps = .01
    numerical = np.column_stack([
        (measurement_model(x+eps*np.eye(4)[k], 1.53, 700., 88., 255.5, radius_bias_px=bias)[0]
         - measurement_model(x-eps*np.eye(4)[k], 1.53, 700., 88., 255.5, radius_bias_px=bias)[0])/(2*eps)
        for k in range(4)])
    assert_allclose(h, numerical, atol=1e-8)
    nav = make_filter()
    assert nav.update(matching_candidate(), np.pi/2, 0.)
    assert_allclose(nav.last_diagnostics["innovation_log_radius"], 0., atol=1e-14)
    assert np.linalg.eigvalsh(nav.P).min() > 0


def test_calibration_fit_never_uses_validation_residuals():
    rows = []
    for split, excess in [("train", 1.2), ("validation", 9.)]:
        for radius in [5., 20.]:
            rows.append(dict(appearance="analytic", status="MEASURED", split=split,
                             truth_radius_px=radius, measured_radius_px=radius+excess,
                             radius_residual_px=excess, log_radius_residual=np.log1p(excess/radius),
                             centroid_u_residual_px=0., centroid_v_residual_px=0.))
    summary = summarize_calibration(rows)
    assert summary["pooled_model"]["rounded_radius_bias_px"] == 1.2
    assert summary["pooled_model"]["training_samples"] == 2
    assert summary["pooled_model"]["validation"]["analytic"]["log_likelihood_residual"]["rmse"] > .1


@pytest.mark.skipif(int(cv.__version__.split(".")[0]) != 5, reason="Calibration requires actual OpenCV 5")
def test_offline_calibration_is_deterministic_and_reduces_sampled_bias():
    config = CalibrationConfig(radii_px=[5., 20.], orbit_fractions=[0., .125], noise_dn=[1.],
                               train_offsets_px=[0.], validation_offsets_px=[.25])
    rows, summary = calibrate(config)
    again, repeated_summary = calibrate(config)
    assert rows == again and summary == repeated_summary
    assert len(rows) == 8
    assert all(r["status"] == "MEASURED" for r in rows)
    stats = summary["appearance_results"]["analytic"]["splits"]["validation"]
    assert stats["corrected_log_radius_residual"]["rmse"] < stats["raw_log_radius_residual"]["rmse"]/3


def test_visual_telemetry_distinguishes_missing_estimate_and_evaluation():
    run = simulate_planar(PlanarConfig(max_time_s=2., measurements=False))
    summary = run["summary"]
    assert summary["attempted_updates"] == summary["missing_updates"] == 2
    assert summary["rejected_updates"] == summary["nis_rejected_updates"] == 0
    assert summary["update_reason_counts"]["NO_CANDIDATE"] == 2
    assert sum(summary["update_reason_counts"].values()) == 2
    assert summary["delta_v_mps"] == 0
    for row in run["telemetry"]:
        assert row["update_status"] == "NO_CANDIDATE"
        assert row["nav_stale"] == 1 and row["nav_hold_reason"] == "STALE"
        assert row["measurement_rejected"] == 0
        assert row["eval_true_range_m"] == row["true_range_m"]
        assert row["eval_relative_x_m"] == row["relative_x_m"]
        assert np.isfinite(row["eval_relative_vx_mps"]) and np.isfinite(row["eval_relative_vy_mps"])
        assert row["estimate_range_sigma_m"] > 0
        covariance = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                covariance[i, j] = covariance[j, i] = row[f"estimate_cov_{i}{j}"]
        assert np.linalg.eigvalsh(covariance).min() > 0
        assert_allclose(row["eval_range_error_sigma"], row["eval_range_error_m"]/row["estimate_range_sigma_m"])


def test_accepted_visual_telemetry_retains_prior_residuals():
    run = simulate_planar(PlanarConfig(max_time_s=2., dropout_probability=0., blackout_s=[]))
    for row in run["telemetry"]:
        assert row["update_status"] == "ACCEPTED"
        assert np.isfinite(row["nis"]) and np.isfinite(row["selected_score"])
        assert_allclose(row["innovation_log_radius"], np.log(row["measured_radius_px"]/row["predicted_radius_px"]))
        assert_allclose(row["innovation_u_px"], row["measured_u_px"]-row["predicted_u_px"])
        residual = np.array([row["innovation_u_px"], row["innovation_log_radius"]])
        s = np.array([[row["innovation_var_u_px2"], row["innovation_cov_u_log_radius"]],
                      [row["innovation_cov_u_log_radius"], row["innovation_var_log_radius"]]])
        assert_allclose(row["nis"], residual@np.linalg.solve(s, residual))
