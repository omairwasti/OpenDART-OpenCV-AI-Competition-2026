"""Frozen evaluator generation, scientific scoring and artifact protection."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import binomtest

pytest.importorskip("cv2")
from opendart.planar import PlanarConfig, CameraRenderer, PlanarEKF
from opendart.model import Binary, Environment
from opendart.vision import Candidate
from opendart.visual_evaluate import (ROOT, RunSpec, campaign_plan, initialize, aggregate,
    digest, verify_frozen, run_one, aggregate_groups, wilson, distribution, episodes)

REFERENCE=ROOT/"assets/dart_0401929788_46871_01_iof.png"


def test_plan_reproducible_counts_and_distinct_directories():
    base=PlanarConfig()
    a=campaign_plan(base,REFERENCE);b=campaign_plan(base,REFERENCE)
    assert a==b and len(a)==110
    assert len({str(s.relative_path()) for s in a})==110
    assert sum(s.group=="guided" for s in a)==96
    assert sum(s.group=="control" for s in a)==4
    assert sum(s.group=="radius_sensitivity" for s in a)==10
    assert base.estimate_error==[60.,-250.,.1,-.3]


@pytest.mark.parametrize("appearance",["ANALYTIC","REFERENCE"])
def test_exact_conditions_and_prior_only_changes(appearance):
    base=PlanarConfig()
    specs=[s for s in campaign_plan(base,REFERENCE) if s.appearance==appearance and s.group=="guided"]
    for s in specs:
        cfg=s.config
        assert cfg["size_radius_bias_px"]==1.07 and cfg["navigation_radius_m"]==88.
        assert cfg["prior_sigma"]==base.prior_sigma
        assert cfg["initial_range_m"]==12000. and cfg["initial_cross_m"]==base.initial_cross_m
        assert cfg["speed_mps"]==base.speed_mps and cfg["control"] and cfg["measurements"]
        assert bool(cfg["reference_image"])==(appearance=="REFERENCE")
        expected=asdict(replace(base,seed=s.seed,reference_image=str(REFERENCE) if appearance=="REFERENCE" else None))
        if s.condition=="IMAGE_DEGRADED":
            expected.update(noise_dn=3.,dropout_probability=.10)
        elif s.condition=="EXTENDED_BLACKOUT":
            expected["blackout_s"]=[[200.,240.]]
        elif s.condition=="RANGE_PRIOR_ERROR":
            expected["estimate_error"][1]=1200. if s.seed%2==0 else -1200.
        assert cfg==expected


def test_negative_controls_and_radius_diagnostics_are_separate():
    specs=campaign_plan(PlanarConfig(),REFERENCE)
    for s in specs:
        if s.group=="control":
            assert s.seed==7 and s.config["control"]==(s.condition!="BALLISTIC")
            assert s.config["measurements"]==(s.condition!="NO_MEASUREMENT")
        elif s.group=="radius_sensitivity":
            assert s.seed==7 and s.radius_factor in [.9,.95,1.,1.05,1.1]
            assert s.config["navigation_radius_m"]==88*s.radius_factor
            assert s.config["size_radius_bias_px"]==1.07
            assert s.relative_path().parts[0]=="radius_sensitivity"


def test_assumed_radius_changes_measurement_prediction_but_never_renderer():
    base=PlanarConfig(noise_dn=1.)
    changed=replace(base,navigation_radius_m=79.2)
    binary=Binary(period_s=base.period_s)
    sc=binary.states(base.epoch_s)[0][1,:2]-[0.,12000.]
    first=CameraRenderer(base).render(base.epoch_s,sc,np.pi/2,binary)
    second=CameraRenderer(changed).render(base.epoch_s,sc,np.pi/2,binary)
    assert np.array_equal(first,second)
    predictions=[]
    candidate=Candidate(np.array([255.5,255.5]),700*88/12000+1.07,np.eye(2)*2.25,120.,(249,249,13,13),.8)
    for cfg in [base,changed]:
        nav=PlanarEKF([0.,12000.,0.,-20.],np.eye(4),cfg,Environment(solar_tide=False,srp=False))
        nav.update(candidate,np.pi/2,0.)
        predictions.append(nav.last_diagnostics["predicted_radius_px"])
    assert_allclose(predictions,[700*88/12000+1.07,700*79.2/12000+1.07])


@pytest.mark.parametrize("value",[0.,-88.,float("nan"),float("inf")])
def test_invalid_navigation_radius(value):
    with pytest.raises(ValueError):PlanarConfig(navigation_radius_m=value).validate()


@pytest.mark.parametrize("k,n",[(0,12),(6,12),(12,12),(2,3)])
def test_wilson_matches_independent_scipy_implementation(k,n):
    ci=binomtest(k,n).proportion_ci(confidence_level=.95,method="wilson")
    assert_allclose(wilson(k,n),[ci.low,ci.high],atol=1e-12)


def test_distribution_and_episode_duration_use_actual_time():
    stats=distribution([1.,2.,3.,float("nan"),None])
    assert stats["n"]==3 and stats["median"]==2.
    assert distribution([])["mean"] is None
    rows=[dict(elapsed_s=float(i),lost=i in [1,2,4]) for i in range(5)]
    es=episodes(rows,lambda r:r["lost"],4.4)
    assert es[0]==dict(start_s=1.,end_s=3.,duration_s=2.,recovered=True)
    assert_allclose(es[1]["duration_s"],.4)
    assert es[1]["recovered"] is False


def test_new_campaign_refuses_existing_path_and_early_aggregation(tmp_path):
    path=tmp_path/"campaign"
    initialize(path,ROOT/"configs/visual_2d.json",REFERENCE)
    manifest_before=digest(path/"manifest.json")
    with pytest.raises(FileExistsError):initialize(path,ROOT/"configs/visual_2d.json",REFERENCE)
    assert digest(path/"manifest.json")==manifest_before
    with pytest.raises(RuntimeError,match="every run is complete"):aggregate(path)
    assert not (path/"aggregate").exists()


def test_frozen_source_change_stops_execution(tmp_path):
    path=tmp_path/"source.txt";path.write_text("original")
    manifest={"frozen_sha256":{str(path):digest(path)}}
    verify_frozen(manifest)
    path.write_text("modified")
    with pytest.raises(RuntimeError,match="Frozen source/input changed"):verify_frozen(manifest)


def test_worker_preserves_zero_thrust_control_and_aggregation_excludes_ablations(tmp_path):
    (tmp_path/"runs").mkdir()
    cfg=PlanarConfig(max_time_s=2.,measurements=False)
    spec=RunSpec("probe","control","ANALYTIC","NO_MEASUREMENT",7,1.,asdict(cfg))
    result=run_one(tmp_path,asdict(spec));path=tmp_path/"runs/probe"
    assert result["status"]=="TIMEOUT"
    complete=json.loads((path/"complete.json").read_text())
    assert all(digest(path/name)==h for name,h in complete["sha256"].items())
    metrics=json.loads((path/"metrics.json").read_text())
    assert metrics["accepted_updates"]==0 and metrics["delta_v_mps"]==0 and metrics["max_force_n"]==0
    assert metrics["missing_updates"]==2 and metrics["nis"]["n"]==0
    with pytest.raises(FileExistsError):run_one(tmp_path,asdict(spec))
    assert digest(path/"config.json")==complete["sha256"]["config.json"]
    guided=[]
    for i,status in enumerate(["TARGET_IMPACT","TARGET_IMPACT","MISS"]):
        m=deepcopy(metrics);m.update(run_id=f"guided_{i}",group="guided",condition="NOMINAL",status=status)
        guided.append(m)
    groups=aggregate_groups(guided+[metrics])
    assert len(groups)==1 and groups[0]["n"]==3
    assert groups[0]["outcome_counts"]==dict(TARGET_IMPACT=2,MISS=1,WRONG_BODY_IMPACT=0,TIMEOUT=0)
    assert_allclose(groups[0]["success_fraction"],2/3)
    assert_allclose(groups[0]["wilson_95_interval"],wilson(2,3))
    invalid=deepcopy(guided[0]);invalid["status"]="EXECUTION_ERROR"
    with pytest.raises(ValueError):aggregate_groups([invalid])
