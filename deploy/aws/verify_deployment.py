"""Exercise actual HTTP jobs and compare with frozen seed-7 scientific evidence."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()


def verify(endpoint, output, baseline_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(Path(baseline_path).read_text())
    endpoint = endpoint.rstrip("/")
    home = request(endpoint + "/")
    assert b"OpenDART" in home and b"Camera" in home
    (output / "homepage.html").write_bytes(home)
    health = json.loads(request(endpoint + "/healthz"))
    results = {}
    for preset in ("NOMINAL", "BLACKOUT", "NO_MEASUREMENT", "BALLISTIC"):
        for attempt in range(10):
            try:
                submission = json.loads(request(endpoint + "/api/jobs", {"preset":preset}))
                break
            except urllib.error.HTTPError as error:
                if error.code not in (409,429):
                    raise
                time.sleep(15)
        else:
            raise RuntimeError("Endpoint remained busy or rate limited")
        job_id = submission["id"]
        started = time.monotonic()
        while True:
            status = json.loads(request(endpoint + "/api/jobs/" + job_id))
            if status["state"] != "RUNNING":
                break
            if time.monotonic()-started > 200:
                raise RuntimeError("Job polling exceeded its deadline")
            time.sleep(2)
        (output / (preset + ".json")).write_text(json.dumps(status, indent=2), encoding="utf-8")
        assert status["state"] == "COMPLETE", status
        result = status["result"]
        summary = result["summary"]
        assert result["preset"] == preset and result["config"]["seed"] == 7
        assert result["runtime"]["platform"] == "Linux"
        assert result["runtime"]["opencv"] == "5.0.0" and result["runtime"]["opencv_distribution"] == "5.0.0.93"
        assert result["runtime"]["source_sha256"] == health["source_sha256"]
        assert result["runtime"]["release_sha256"] == health["release_sha256"]
        expected = "TARGET_IMPACT" if preset in ("NOMINAL","BLACKOUT") else "MISS"
        assert summary["status"] == expected
        if preset in ("NO_MEASUREMENT", "BALLISTIC"):
            assert summary["delta_v_mps"] == 0 and result["max_force_n"] == 0
        if preset == "NO_MEASUREMENT":
            assert summary["accepted_updates"] == 0
        else:
            assert summary["accepted_updates"] > 0
        if preset == "BLACKOUT":
            assert result["recovery"]["accepted_after_blackout"] > 0
        comparisons = {}
        for key in ("status","elapsed_s","delta_v_mps","accepted_updates","missing_updates","nav_hold_s","final_estimation_error_m"):
            local, cloud = baseline[preset][key], summary[key]
            match = math.isclose(local,cloud,rel_tol=1e-7,abs_tol=1e-6) if isinstance(local,float) else local == cloud
            comparisons[key] = dict(local=local,cloud=cloud,match=match)
        results[preset] = dict(job_id=job_id,runtime=result["runtime"],comparison=comparisons)
        (output / "comparison.json").write_text(json.dumps(dict(endpoint=endpoint,health=health,results=results),indent=2))
        if not all(c["match"] for c in comparisons.values()):
            raise RuntimeError("Cloud metrics differ from frozen local evidence; inspect before proceeding")
        assert b"evaluation ground truth" in request(endpoint + "/jobs/" + job_id)
        for frame in result["frames"]:
            image = request(endpoint + f"/jobs/{job_id}/artifacts/{frame['file']}")
            assert image.startswith(b"\x89PNG")
            (output / (preset + "_" + frame["file"])).write_bytes(image)
        print(json.dumps(dict(preset=preset,job_id=job_id,outcome=summary["status"],runtime=result["runtime"],all_metrics_match=True)),flush=True)
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--baseline",required=True)
    args=parser.parse_args()
    verify(args.endpoint,args.out,args.baseline)
