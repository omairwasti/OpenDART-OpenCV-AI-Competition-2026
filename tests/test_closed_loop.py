from dataclasses import replace
import numpy as np
from opendart.simulation import Scenario, simulate


def test_controlled_hit_versus_ballistic_miss():
    base = Scenario()
    controlled = simulate(base)["summary"]
    ballistic = simulate(replace(base, guidance="none"))["summary"]
    assert controlled["status"] == "TARGET_IMPACT"
    assert controlled["terminal_relative_speed_mps"] > 6000
    assert controlled["propellant_used_kg"] < 2
    assert abs(controlled["surface_event_residual"]) < 1e-6
    assert ballistic["status"] == "MISS"
    assert ballistic["terminal_center_distance_m"] > 1000
    assert ballistic["delta_v_mps"] == 0


def test_control_update_and_integration_refinement():
    coarse = simulate(Scenario())["summary"]
    fine = simulate(Scenario(control_dt_s=.5, max_step_s=.25, rtol=1e-12,
                             atol_position_m=1e-7))["summary"]
    assert coarse["success"] and fine["success"]
    assert abs(coarse["terminal_time_s"]-fine["terminal_time_s"]) < .01
    assert np.linalg.norm(np.array(coarse["terminal_state_relative"][:3])
                          - np.array(fine["terminal_state_relative"][:3])) < .1


def test_insufficient_authority_is_a_failure_not_a_claimed_hit():
    result = simulate(Scenario(max_force_n=.01))["summary"]
    assert not result["success"]
    assert result["status"] == "MISS"
    assert result["saturation_time_s"] > 100


def test_tuned_pd_tracks_a_closing_reference_to_contact():
    result = simulate(Scenario(guidance="pd"))["summary"]
    assert result["success"]
    assert result["terminal_relative_speed_mps"] > 6000
    assert result["saturation_time_s"] > 0


def test_absent_bearings_do_not_fall_back_to_truth_guidance():
    run = simulate(Scenario(navigation="bearing", dropout_probability=1.))
    assert not run["summary"]["success"]
    assert run["summary"]["accepted_measurements"] == 0
    assert run["summary"]["delta_v_mps"] == 0
    assert all(row["force_n"] == 0 for row in run["telemetry"])
