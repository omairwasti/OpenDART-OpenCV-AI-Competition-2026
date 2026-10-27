# Synthetic visual evaluation

The evaluated OpenDART system uses camera pixels, OpenCV 5, association, PlanarEKF navigation, estimated-state ZEM guidance and bounded actuation. Applied force changes the next synthetic image.

Two appearances, 12 seeds and four conditions produced 96 guided cases. All reached TARGET_IMPACT under the tested synthetic conditions. Each 12/12 condition cell has Wilson 95% interval [0.758, 1.000]; paired conditions are not pooled as independent trials. This is not general reliability or flight validation.

Four matched negative controls missed: two BALLISTIC cases retained image navigation without thrust; two NO_MEASUREMENT cases had no usable updates and zero thrust. Zero-thrust closest approach was about 615.52 m. The extended blackout lasted [200,240) s; all tested cases accepted an update again at 240 s and reached contact.

The +1.07 px synthetic detector-radius calibration reduced one nominal final position error from 59.18 to 9.23 m and sampled peak error from 2096.93 to 375.07 m. It is renderer/detector calibration, not DRACO instrument calibration or covariance inflation.

Known target radius supplies scale. Full-state covariance showed substantial undercoverage: nominal median per-run full-state 95% ellipsoid coverage was 27.2% analytic and 20.6% reference. Contact does not establish statistical EKF consistency. Size-prior diagnostics expose transient range bias even when contact succeeds.

Numeric groups: [groups.json](groups.json). Figures: [outcomes](success_rates.png), [uncertainty](uncertainty_diagnostics.png), [recovery](loss_recovery.png). The final technical report gives methodology, controls and limitations.
