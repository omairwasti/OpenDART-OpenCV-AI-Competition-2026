# Reproduction

## Install

Use Python 3.12 and one OpenCV wheel variant. Tested local versions are Python 3.12.15, NumPy 2.4.6, SciPy 1.18.1, Matplotlib 3.11.2, OpenCV 5.0.0 (`opencv-python==5.0.0.93`), Flask 3.1.3 and pytest 9.1.1. The AWS runtime uses Ubuntu 24.04 x86_64 and Python 3.12.3. All local package pins are in `requirements-local-py312.txt`; the Linux lock is in `deploy/aws/requirements-linux-py312.txt`.

From this directory:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-local-py312.txt
python -m pip install --no-deps -e .
python -m pip check
python -c "import cv2; print(cv2.__version__); assert cv2.__version__ == '5.0.0'"
```

On Linux, activate with `source .venv/bin/activate` and use the Linux lock before the no-dependency editable install. Gunicorn is Linux-only. Do not install a second OpenCV wheel variant.

## Test and run

From the repository root:

```powershell
python -m pytest -q -ra --basetemp=results/NEW_TEST_TMP
python -m unittest discover -s tools/tests -v
python -m opendart.visual --config configs/visual_2d.json --out results/NEW_VISUAL
python -m flask --app opendart.web:create_app run --host 127.0.0.1 --port 8000
```

Use a fresh output directory. Open the generated HTML viewer or MP4. The core simulation needs no DRACO download. The legacy replay default uses the bundled asset; optional DRACO asset rebuilding locates ffmpeg on PATH. The local web app serves `/` and `/draco` using the bundled curated assets. The Flask development server is for local use; AWS uses Gunicorn behind Nginx.

The core suite has 143 tests. The 61 additional historical protocol tests are validated on Windows; preserved freeze-record keys use Windows separators, so that ancillary suite is not claimed portable to Linux unchanged. These tests use synthetic fixtures and unreviewed sampling metadata; they do not repeat actual held-out scoring. If Windows TEMP is inaccessible, set TEMP and TMP to a newly created workspace directory for that shell. Verification summaries are under `verification/`.

## Read the evaluation

`evidence/task1_comparison.json`, `campaign_groups.json`, `campaign_controls.json`, `campaign_numeric_results.md` and `draco_metrics.json` contain the reported summaries. Controls and known-radius diagnostics are separate from the 96 guided cases. Per-cell confidence intervals, unavailable targets, exclusions and covariance limitations must accompany positive results.

For a separately requested fresh campaign, the existing commands are:

```powershell
python -m opendart.visual_evaluate init --out results/NEW_CAMPAIGN
python -m opendart.visual_evaluate run --out results/NEW_CAMPAIGN --workers 2
python -m opendart.visual_evaluate aggregate --out results/NEW_CAMPAIGN
```

Never overwrite existing evidence or retune against the consumed held-out set.

## Recorded replay and image analysis

From the repository root, `python -m http.server 8890 --directory draco_assets` serves `http://127.0.0.1:8890/replay.mp4`. It presents all 4,000 historical frames at 60 fps, accelerated to approximately 66.667 s. It does not run guidance.

From the repository root, run one actual frozen autonomous continuation:

```powershell
python -c "from pathlib import Path; Path('results/NEW_DRACO').mkdir(parents=True,exist_ok=False)"
python -m opendart.draco_web_runner --assets draco_assets --image-id f3999 --request-id local_demo --out results/NEW_DRACO/result.json
```

The JSON contains fresh inference and its encoded overlay. A validated checkpoint is restored strictly before the selected original image; no current-frame label or cached overlay supplies the answer. Twelve fixed images are included. Complete prefix reconstruction needs the original PNG/XML corpus and development inputs; independent annotation review needs the original review inputs. Those are outside this compact package. Do not rerun `evaluate_draco_heldout.py --phase evaluate` as an unseen test.

## Report and diagrams

The LaTeX project is self-contained in `report_source/`, with standard TeX fonts, native equations and an inline bibliography. Run `tectonic --keep-logs main.tex` from that directory, or run `xelatex -interaction=nonstopmode -halt-on-error main.tex` twice. The first Tectonic build may fetch standard TeX packages. The final report has 16 pages.

Recreate the architecture diagrams from this directory with `python diagrams/build_diagrams.py --out diagrams`. Numeric evidence figures are reused from the recorded evaluations; no new measurements are inferred from presentation crops.

## AWS update, first deployment and cleanup

The existing single-instance architecture and operational limits are described in [AWS_DEMO.md](AWS_DEMO.md). A portable reproduction variant is included under `deployment/`; see [its description](deployment/README.md). Its legacy replay default image path is portable, and it has not been deployed. Resolve its local archive path first:

```powershell
python deployment/resolve_release.py --out deployment/release.local.json
```

For an explicitly authorised application-only update of the existing deployment:

```powershell
python deploy/aws/update_existing_release.py --release deployment/release.local.json --evidence results/NEW_UPDATE
```

This reuses the existing private S3 bucket and sends an SSM activation command. Require Success, not merely dispatch. Activation checks hashes, runs Linux tests before switching, keeps rollback and verifies health. Do not bypass an inventory/hash/test failure. The original release builder requires the author's full frozen campaign metadata; this compact package supports a separately authorised deployment of the supplied portable reproduction variant.

Only for a separately authorised empty/recreated deployment:

```powershell
./deploy/aws/deploy.ps1 -Phase Prepare -ReleaseMetadata deployment/release.local.json -EvidenceDirectory results/NEW_AWS
# Wait for successful preparation.
./deploy/aws/deploy.ps1 -Phase Compute -ReleaseMetadata deployment/release.local.json -EvidenceDirectory results/NEW_AWS
```

Do not create another instance alongside the current one. CloudFormation bootstrap parameters retain an earlier release, so recreation requires applying the desired application update again. For a separately authorised teardown after judging/evidence export:

```powershell
./deploy/aws/cleanup.ps1 -StackName opendart-task3a -Region ap-southeast-2
```

Cleanup deletes the owned deployment resources and release bucket. Keep the running endpoint available during judging. Do not include credentials in code or shared artifacts.

Source line endings are preserved in Git because the original scientific freeze records use byte hashes. The repository contains metadata-only sampling/freeze records under data/ and results/, not the full DRACO image corpus or raw campaign outputs.
