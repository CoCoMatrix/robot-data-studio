# Historical sidecar contract (superseded)

This v1 note is retained as historical evidence. Current denominator and
handoff rules live in `quality-checks.md`, `workflow.md`, and `v3-handoff.md`.

The pilot writes `episodes/<episode-fingerprint>.json` and `manifest.json` beneath `sidecar_root`. Episode sidecars contain schema/content fingerprints, lineage, normalized language, start/contact/end references, findings, four independent dispositions, and the original frame/timestamp/validity resampling map. They also retain per-metric `evaluated`/`not_evaluated` evidence, separate `qwen_core_score` and `extension_score`, core coverage, and final decision. Missing metrics are N/A and lower coverage; they are never imputed as zero or one.

The manifest contains stable hashes, reasons, lineage, task/source/duration/motion coverage and sampler advice derived from configured weights. Reruns reuse episode fingerprints. JSON is the prototype output; compact Parquet is an extension point. RGB/MP4 bytes never enter sidecars.
