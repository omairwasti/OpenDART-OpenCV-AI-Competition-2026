# Portable reproduction archive

This is a portable reproduction variant of the previously deployed OpenDART release. It has not been deployed. The serving AWS application is unchanged.

The legacy replay module now defaults to the image already bundled under assets/, resolved relative to the installed package. This is the only executable source change inside this archive. The separate asset-builder source in tools/ now locates ffmpeg through PATH.

Scientific methods, controller/estimator logic, thresholds, calibration, data, checkpoints and numerical results are unchanged. Operational manifests inside this reproduction archive describe its own files. Source provenance retained under verification/ describes the previously deployed baseline; its legacy replay-file hash therefore differs from this portable path-only copy.

release.json identifies this reproduction variant and its original deployed-release identity. resolve_release.py prepares a local archive path without contacting AWS. Any future deployment requires separate authorisation and the existing hash/Linux-test/health gates. The portable variant has received static syntax and path checks only; no scientific evaluation or deployment was run during this cleanup.
