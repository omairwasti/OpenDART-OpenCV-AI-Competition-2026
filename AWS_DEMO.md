# AWS demonstration

**[Run the simulation](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/) · [Explore real DRACO imagery](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/draco)**

The application runs on the existing single CloudFormation-managed c7i-flex.large in ap-southeast-2. The runtime is Ubuntu 24.04 x86_64, Python 3.12.3 and OpenCV 5.0.0 (opencv-python 5.0.0.93), served by Nginx → Gunicorn → Flask. Private encrypted S3 stores releases and selected evidence; SSM provides administration.

## Judge walkthrough

1. Choose **Run the simulation**, leave **Nominal** selected and press **Run this experiment**. Allow about 15 seconds. A completed run shows **Target contact**, estimation error, visual updates, navigation hold, correction delta-v and compute time.
2. Inspect the estimated trajectory against the clearly labelled evaluation truth and the camera frames. Camera blackout, No visual updates and Ballistic remain available. The latter two are negative controls with expected misses.
3. Open **Real DRACO imagery**, follow the historical gallery and choose **Analyse a real DRACO frame**. Select **About 10 s before impact**, then **Analyse this frame**. **Resolved target** describes a visible terrain region; it does not establish a physical centre or range.
4. Select **Final partial transmission**. **Partial transmission** reports only the received strip; the missing body extent is unknown. Exact product ID, UTC, hashes and technical states are expandable.
5. Play **Recorded DRACO replay** and read **Real-image held-out benchmark**. These are historical images, not a closed-loop control environment. Eighteen of 20 sampled frames were reviewed, with two excluded boundary cases and no reviewed labels in the final ten seconds. Operator-assisted tracking and autonomous tracking have the same median IoU on the identical eligible six-region subset.

Live analysis verifies hashes and restores state strictly before the selected original image, then runs perception afresh. A current-frame label or stored overlay does not supply the answer. The frozen method, checkpoints and exact state names remain in technical provenance.

## Release and verification

The presentation release was activated successfully through the existing application updater. Before activation, all 143 Linux tests passed, package/scientific hashes were verified and pip check found no broken requirements. Rollback was retained. The same instance and all five stack resources were confirmed before and after activation.

A fresh production nominal run returned TARGET_IMPACT. All seven regression fields matched the recorded baseline: outcome, simulated event time, delta-v, accepted updates, missing updates, navigation hold and final estimation error. The run showed 9.23 m final estimation error, 557 accepted / 40 missing updates, 16 s navigation hold, 1.60 m/s correction and about 14.15 s compute time. This is estimation error, not residual miss distance.

Fresh resolved and partial DRACO requests matched every recorded scientific checkpoint-output field. All 12 public original images matched their source hashes. Replay playback and HTTP byte ranges were verified. Current summaries are in verification/; earlier numerical evidence remains in evidence/.

- Serving release SHA-256: 44685f7cf710d79a857f68124c4b5dde5930a055486ab6a167052577d24888a2.
- Scientific application source digest: f3cc6eaf1813646e4ae8cc447d042904ad079d5484dbf81a13a3d2f653c9fbfa.
- DRACO method SHA-256: 6afb56362fe841f2158c291600d63760929ea9948940d984dc05d9f58796aba9.
- DRACO configuration SHA-256: 84554612ed68c6b699ac2d4ccc013a38dcf317d38eb95b4d9740f3e2ea7558d5.
- [Health](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/healthz), [image inventory](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/draco/media/manifest.json), [aggregate benchmark metrics](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/draco/media/benchmark_metrics.json) and [recorded replay](http://ec2-54-253-67-121.ap-southeast-2.compute.amazonaws.com/draco/media/replay.mp4).

A separately labelled portable reproduction archive, relative-path metadata and operational manifest are in deployment/. Its only source change from the serving release is the legacy replay default image path. It has not been deployed. See [REPRODUCTION.md](REPRODUCTION.md) for application update and cleanup commands. CloudFormation bootstrap parameters retain an earlier release; recreation requires activating the desired audited application release again.

## Operational limits

Only fixed preset/image IDs are accepted. There are no uploads, arbitrary paths or URLs, or user-tunable scientific parameters. A shared computation lock admits one job at a time. DRACO requests have a six/minute/IP rate limit with burst two; the existing shared application cap is 20 admitted computations/hour. The counter resets on restart. Synthetic execution has a 180 s bound and DRACO 15 s. The service uses one OpenCV/BLAS thread and a 3 GiB systemd memory ceiling.

The endpoint uses HTTP without authentication/TLS and accepts only public fixed inputs. It is demonstration infrastructure. Keep the existing instance running during judging; its public address can change after a stop/start. Future resource teardown requires separate authorisation.
