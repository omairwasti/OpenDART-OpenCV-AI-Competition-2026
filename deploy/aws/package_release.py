"""Allowlisted release builder. Never traverse the workspace or raw results."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/visual_campaign_2026-10-25/aggregate"


def build(output, draco_assets=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for pattern in ("opendart/*.py", "opendart/templates/*.html", "opendart/static/*.js", "opendart/static/*.css",
                    "tests/test_*.py", "configs/*.json", "deploy/aws/*", "docs/AWS_DEPLOYMENT.md"):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("pyproject.toml", "README.md", "assets/dart_0401929788_46871_01_iof.png", "opendart/static/flight_hero.png"):
        files[name] = (ROOT / name).read_bytes()
    for name in ("CAMPAIGN_REPORT.md", "groups.json", "success_rates.png", "uncertainty_diagnostics.png",
                 "loss_recovery.png", "representative_camera_sheet.png"):
        source = ROOT / 'docs/VISUAL_RESULTS.md' if name == 'CAMPAIGN_REPORT.md' else EVIDENCE / name
        payload = source.read_bytes()
        if name.endswith((".md", ".json")):
            text = payload.decode("utf-8").replace(str(ROOT), "[local package]").replace(ROOT.as_posix(), "[local package]")
            payload = text.encode("utf-8")
        files["evidence/" + name] = payload
    frozen = json.loads((ROOT / "results/visual_campaign_2026-10-25/manifest.json").read_text())
    preserved = {}
    for path, expected in frozen["frozen_sha256"].items():
        local = Path(path)
        if local.name == "pyproject.toml":
            continue  # Explicitly approved web/package metadata change.
        actual = hashlib.sha256(local.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen scientific input changed: {local.name}")
        preserved[local.relative_to(ROOT).as_posix()] = actual
    files["scientific_manifest.json"] = json.dumps(preserved, indent=2).encode()
    baseline = {}
    for preset, run_id in {"NOMINAL":"analytic_nominal_s07", "BLACKOUT":"analytic_extended_blackout_s07",
                           "NO_MEASUREMENT":"analytic_no_measurement_s07", "BALLISTIC":"analytic_ballistic_s07"}.items():
        baseline[preset] = json.loads((EVIDENCE.parent / "runs" / run_id / "summary.json").read_text())
    files["evidence/frozen_seed7.json"] = json.dumps(baseline, indent=2).encode()
    if draco_assets is not None:
        curated=Path(draco_assets)
        catalog=json.loads((curated/'manifest.json').read_text())
        if len(catalog['images'])!=12:raise RuntimeError('Expected exactly twelve curated DRACO originals')
        names={'manifest.json','benchmark_metrics.json','replay.mp4','heldout_benchmark.png','comparison_compact.png','comparison_resolved.png','checkpoints/expected.json'}
        for item in catalog['images']:
            names.update({'images/'+item['filename'],'checkpoints/'+item['id']+'.npz'})
            if hashlib.sha256((curated/'images'/item['filename']).read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Curated original checksum mismatch')
            if hashlib.sha256((curated/'checkpoints'/(item['id']+'.npz')).read_bytes()).hexdigest()!=item['checkpoint_sha256']:raise RuntimeError('Checkpoint checksum mismatch')
        if sum((curated/name).stat().st_size for name in names)>50*1024**2:raise RuntimeError('Curated DRACO release exceeds budget')
        for name in sorted(names):
            if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Invalid curated path')
            files['draco_assets/'+name]=(curated/name).read_bytes()
        for name,expected_sha in [('tools/draco_perception.py',catalog['method_sha256']),('tools/draco_perception_config_v1.json',catalog['configuration_sha256'])]:
            payload=(ROOT/name).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=expected_sha:raise RuntimeError('Frozen DRACO source changed')
            files[name]=payload
    manifest = {name:hashlib.sha256(payload).hexdigest() for name,payload in sorted(files.items())}
    files["release_manifest.json"] = json.dumps(manifest, indent=2).encode()
    archive = output / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            tar.addfile(info, io.BytesIO(payload))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    named = archive.with_name(sha + ".tar.gz")
    archive.rename(named)
    from opendart.web_runner import source_digest
    metadata = dict(archive=str(named.resolve()), sha256=sha, key=f"releases/{sha}.tar.gz",
                    source_sha256=source_digest(), size_bytes=named.stat().st_size, files=len(files),
                    scientific_files_verified=len(preserved))
    (output / "release.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument('--draco-assets',type=Path)
    args=parser.parse_args()
    build(args.out,args.draco_assets)
