# Frozen synthetic visual campaign: numeric results

96 guided cases; 4 negative controls; 10 separate radius-assumption cases.
Primary Wilson intervals use independent seeds within each appearance/condition. Matched seeds across appearances/conditions are paired; do not pool all cases as independent trials.

| Appearance | Condition | Impact / n | MISS | Wrong body | Timeout | Wilson 95% |
|---|---|---:|---:|---:|---:|---|
| ANALYTIC | EXTENDED_BLACKOUT | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| ANALYTIC | IMAGE_DEGRADED | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| ANALYTIC | NOMINAL | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| ANALYTIC | RANGE_PRIOR_ERROR | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| REFERENCE | EXTENDED_BLACKOUT | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| REFERENCE | IMAGE_DEGRADED | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| REFERENCE | NOMINAL | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |
| REFERENCE | RANGE_PRIOR_ERROR | 12/12 | 0 | 0 | 0 | [0.7575, 1.0000] |

## control

| Run | Outcome | Final position error m | Delta-v m/s | Max standardized range error |
|---|---|---:|---:|---:|
| analytic_ballistic_s07 | MISS | 9.479 | 0.000 | 3.322 |
| analytic_no_measurement_s07 | MISS | 446.438 | 0.000 | 0.361 |
| reference_ballistic_s07 | MISS | 10.501 | 0.000 | 3.502 |
| reference_no_measurement_s07 | MISS | 446.438 | 0.000 | 0.361 |

## radius_sensitivity

| Run | Outcome | Final position error m | Delta-v m/s | Max standardized range error |
|---|---|---:|---:|---:|
| analytic_radius_090_s07 | TARGET_IMPACT | 13.377 | 1.749 | 10.847 |
| analytic_radius_095_s07 | TARGET_IMPACT | 2.035 | 1.675 | 6.547 |
| analytic_radius_100_s07 | TARGET_IMPACT | 9.228 | 1.604 | 2.617 |
| analytic_radius_105_s07 | TARGET_IMPACT | 18.822 | 1.548 | 3.263 |
| analytic_radius_110_s07 | TARGET_IMPACT | 28.072 | 1.575 | 6.401 |
| reference_radius_090_s07 | TARGET_IMPACT | 28.854 | 1.635 | 8.112 |
| reference_radius_095_s07 | TARGET_IMPACT | 19.664 | 1.564 | 3.873 |
| reference_radius_100_s07 | TARGET_IMPACT | 11.329 | 1.526 | 2.626 |
| reference_radius_105_s07 | TARGET_IMPACT | 3.558 | 1.571 | 6.108 |
| reference_radius_110_s07 | TARGET_IMPACT | 4.327 | 1.637 | 9.457 |

Run metrics, robust spreads, NIS/NEES and coverage: `run_metrics.json`, `groups.json`, `matched_epoch_coverage.json`, `runs.csv`.
Frame-level diagnostics are descriptive; full covariance/measurement assumptions are not validated by a nominal confidence threshold.
Physical-radius diagnostic and controls are excluded from guided success statistics.
