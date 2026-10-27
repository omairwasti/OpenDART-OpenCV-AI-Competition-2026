"""Deployment boundary tests; scientific behavior is checked by the existing suite."""
from dataclasses import asdict
import json
import subprocess
import threading
import time
import uuid

import pytest
from opendart.web import Busy, JobManager, create_app
from opendart.web_runner import PRESETS, ROOT, clean, preset_config, write_json
from opendart.planar import PlanarConfig


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_only_apply_approved_overrides(preset):
    base = PlanarConfig(**json.loads((ROOT / "configs/visual_2d.json").read_text()))
    expected = asdict(base)
    if preset == "BLACKOUT":
        expected["blackout_s"] = [[200., 240.]]
    elif preset == "NO_MEASUREMENT":
        expected["measurements"] = False
    elif preset == "BALLISTIC":
        expected["control"] = False
    assert asdict(preset_config(preset)) == expected
    assert expected["seed"] == 7 and expected["size_radius_bias_px"] == 1.07
    with pytest.raises(ValueError):
        preset_config("--out /tmp/arbitrary")


@pytest.fixture
def client(tmp_path):
    return create_app(dict(TESTING=True, JOB_ROOT=str(tmp_path / "jobs"))).test_client()


def test_browser_routes_and_origin_boundary(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"Empirical undercoverage" in page.data
    assert page.data.count(b'type="radio"') == 4
    assert page.data.count(b'id="run-experiment"') == 1
    assert b"NO_MEASUREMENT" in page.data
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
    assert client.get("/healthz").json["status"] == "ok"
    assert client.post("/api/jobs", json={"preset":"NOMINAL"}, headers={"Origin":"https://evil.invalid"}).status_code == 403
    assert client.post("/api/jobs", json={"preset":"NOMINAL"}, headers={"Sec-Fetch-Site":"cross-site"}).status_code == 403


@pytest.mark.parametrize("payload", [{}, {"preset":"custom"}, {"preset":"NOMINAL","seed":8}, {"preset":[]}, ["NOMINAL"]])
def test_public_inputs_are_strictly_allowlisted(client, payload):
    assert client.post("/api/jobs", json=payload).status_code == 400


def test_global_lock_prevents_second_manager_and_releases_after_completion(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def fake_run(command, **kwargs):
        assert command[1:3] == ["-m", "opendart.web_runner"]
        assert kwargs["timeout"] == 180 and kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        entered.set()
        assert release.wait(5)
        from pathlib import Path
        write_json(Path(command[-1]) / "result.json", {"summary":{"status":"TARGET_IMPACT"}})
    monkeypatch.setattr(subprocess, "run", fake_run)
    first, second = JobManager(tmp_path), JobManager(tmp_path)
    status = first.submit("NOMINAL")
    assert entered.wait(5)
    try:
        with pytest.raises(Busy):
            second.submit("BALLISTIC")
    finally:
        release.set()
        first.threads[0].join(5)
    assert first.read(status["id"])["state"] == "COMPLETE"
    lock = second.acquire()
    lock.close()


def test_timeout_is_reported_and_lock_is_released(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    manager = JobManager(tmp_path, timeout=.01)
    status = manager.submit("NOMINAL")
    manager.threads[0].join(5)
    assert manager.read(status["id"])["state"] == "TIMEOUT"
    lock = manager.acquire()
    lock.close()


def test_retention_does_not_delete_unrelated_directories(tmp_path):
    manager = JobManager(tmp_path, max_jobs=2)
    unrelated = tmp_path / "keep-me"
    unrelated.mkdir()
    for i in range(3):
        folder = tmp_path / uuid.uuid4().hex
        folder.mkdir()
        (folder / "result.json").write_text("{}")
    manager.prune()
    assert unrelated.exists()
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 2


def test_job_polling_artifact_allowlist_and_busy_response(client, monkeypatch):
    manager = client.application.extensions["jobs"]
    job_id = uuid.uuid4().hex
    directory = manager.root / job_id
    directory.mkdir()
    status = dict(id=job_id,preset="NOMINAL",state="COMPLETE")
    write_json(directory / "status.json", status)
    write_json(directory / "result.json", {"summary":{"status":"TARGET_IMPACT"}})
    (directory / "worker.log").write_text("private diagnostic")
    assert client.get(f"/api/jobs/{job_id}").json["result"]["summary"]["status"] == "TARGET_IMPACT"
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/jobs/{job_id}/artifacts/result.json").status_code == 200
    assert client.get(f"/jobs/{job_id}/artifacts/worker.log").status_code == 404
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/evidence/private.txt").status_code == 404
    def busy(preset):
        raise Busy("Another simulation is running")
    monkeypatch.setattr(manager, "submit", busy)
    assert client.post("/api/jobs", json={"preset":"BALLISTIC"}).status_code == 409


def test_json_exports_are_standard_compliant(tmp_path):
    write_json(tmp_path / "test.json", {"nis":float("nan"),"x":[float("inf"),2.]})
    assert json.loads((tmp_path / "test.json").read_text()) == {"nis":None,"x":[None,2.]}
