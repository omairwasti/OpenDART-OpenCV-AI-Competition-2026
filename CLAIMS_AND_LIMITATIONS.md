# Claims, exact evidence and scope

The figures and numerical summaries are frozen. The technical report distinguishes supporting 3D experiments, the primary planar visual demonstrator and recorded DRACO perception.

| Claim | Evidence | Scope | Limitation |
|---|---|---|---|
| Image feedback changes downstream actions and later observations | `tests/test_planar.py`; image-to-estimate-to-force execution; `evidence/cloud_synthetic_regression.json`; `figures/campaign_causal_controls.png` | Implemented synthetic causal loop, matched seed-7 ablations | Known size/prior/ephemeris and simplified camera/actuation; no flight claim |
| 96/96 guided tested cases reached TARGET_IMPACT | `evidence/campaign_groups.json`, `campaign_numeric_results.md`, `figures/campaign_success.png` | 2 appearances × 12 seeds × 4 fixed conditions | Each 12/12 cell Wilson 95% [0.758,1.000]; paired conditions; not general reliability |
| Four negative controls missed | `evidence/campaign_controls.json` | BALLISTIC and NO_MEASUREMENT, two appearances, seed 7 | Expected ablation outcomes; 615.52 m closest centre approach, not a guided residual miss |
| All tested 40 s blackout cases recovered an accepted update at 240 s and reached contact | `figures/campaign_blackout.png`, `evidence/campaign_groups.json` | Scheduled [200,240) s synthetic blackout | Not arbitrary outage tolerance; missing and rejected updates differ |
| Task 1 calibration improved one nominal navigation run | `evidence/task1_comparison.json`, `figures/task1_calibration.png` | 59.18→9.23 m final; 2097→375 m peak; 18.11→2.62 maximum normalised range error | +1.07 px synthetic renderer/detector extent prediction; not DRACO instrument calibration or covariance inflation |
| DRACO autonomous gives 6/8 point matches and 0.802 median visible-region IoU | `evidence/draco_metrics.json`, `figures/draco_heldout_benchmark.png` | Small manually annotated held-out benchmark: 18 reviewed of 20 sampled | Two early autonomous points unavailable; two unreviewed boundary cases excluded; no reviewed final-ten-second labels |
| Operator-assisted tracking yields median IoU 0.871 | Same frozen metrics; recorded replay | Six eligible later regions after one fixed development designation | Autonomous has the SAME median on that same subset; no superiority claim |
| AWS executes real OpenCV 5 work | `evidence/cloud_synthetic_regression.json`, `cloud_draco_verification.json`, live endpoint, `verification/runtime.json` | Linux OpenCV 5.0.0; four full synthetic jobs and fresh curated-frame inference previously verified | Public HTTP, fixed inputs, single-instance capacity; application update and public checks are documented in AWS_DEMO.md |

## Estimator and scale limitations

Do not claim statistical EKF consistency. Nominal full-state 95% ellipsoid empirical frame coverage had medians 27.2% analytic / 20.6% reference. Position/range performance can remain useful while full-state covariance undercovers. Correlated frames are not independent trials. At nominal matched epochs 500/580 s no full-state ellipsoid contained truth across 12 seeds per appearance.

Known physical radius supplies monocular scale. Separate ±5%/±10% diagnostics all reached contact while showing large transient bias. For reference +10%, final error was 4.33 m despite about 1615 m peak error and maximum |range error/σ| 9.46. Do not select a radius using favourable contact error or hide transient behaviour. No size/guidance parameter is retuned here.

## Scientific and responsible-use boundary

- Main scene: bounded planar synthetic geometry, prescribed binary ephemeris, known camera calibration/attitude knowledge and simplified actuation. The historical 6.1 km/s 3D case is different.
- Prerecorded DRACO frames cannot respond to new actions. They support image-processing/tracking evidence, not guidance validation, physical-centre/range recovery or 4,000 verified semantic identifications.
- Two UNREVIEWED boundary-clipped frames are absent from metrics, not assigned zero IoU. The final partial frame shows only received content; missing full extent is unknown.
- No flight readiness, reconstruction of the flown autonomy stack, crater/ejecta/deflection model or post-contact propagation is claimed.
- Optional one-time operator designation is restricted to recorded replay. Public live inference is autonomous; judges may select fixed public presets but cannot upload arbitrary inputs or retune science.

See [the report](OpenDART_Technical_Report_Final.pdf) and [AWS operational limits](AWS_DEMO.md). Methods and evaluation inputs are fixed for these reported results.
