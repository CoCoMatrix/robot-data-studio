# End-to-end workflow

1. Resolve only paths that match `train_allowlist`; apply case-insensitive deny patterns for `val`, `validation`, `test`, and `eval` first.
2. Read README, schema, and metadata before bounded episode data. Record immutable source hashes, episode boundaries, timestamps, FPS, media references, fields, units, frames, and validity authority.
3. Run T0/T1 cheap checks by shard/episode. Do not copy source RGB/MP4 or accumulate a whole corpus in memory.
4. Select checks independently from the current v2 profile. Build prerequisite intermediates as a DAG and reuse caches keyed by content, model, prompt/config, and sampled frames. For C1/VLM, keep the output schema invariant and version a Human- or Robot-specific prompt profile under [prompt-policy.md](prompt-policy.md).
5. Preserve four independent outcomes: hard reject, review, window mask, and soft weight. Promote only eligible episode IDs; retain raw and rejection evidence.
6. Publish promoted episodes with `/mnt/workspace/spy/data_pipeline` as official LeRobot V3, reload with the official loader, and validate offsets, timestamps, video, feature dtypes/shapes, and statistics.
7. Publish compact JSON/Parquet sidecars containing hashes, lineage, reasons, dispositions, task/source/duration/motion coverage, resume keys, and selected/evaluated denominators.

Run `validate`, then `dry-run`, then a positive bounded `pilot`. Pass every real episode explicitly; never discover or scan beyond the approved list. External VLM/SAM providers require separate explicit authorization and strict budgets.

For an authorized local Robot C2 audit, use the dedicated `c2-real-pilot` command so no C1/VLM provider is constructed. Pin the profile, episode IDs, official V3 root, local checkpoint, and cache explicitly; treat official-SAM3 evidence as provisional until a versioned train-only manual-mask calibration exists.
