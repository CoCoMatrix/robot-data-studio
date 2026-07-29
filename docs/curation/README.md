# Embodied Data Curation

This workflow prepares immutable Human and Robot source data for training. It is primarily driven by Codex/CLI; the legacy web filters are not the authority for the current `qwen-cleaning-v2` contract.

## What the pipeline does

1. Define an explicit train-only source contract and bounded pilot.
2. Prefilter broken files, schemas, timestamps, episode boundaries, non-finite values and duplicates.
3. Preserve or derive authoritative per-frame absolute Human hand / Robot EEF poses and Camera0 poses with validity.
4. Run independently selected S1/S2/S3/C1/C2/C3 checks plus Human extensions.
5. Keep hard reject, employee review, window mask and soft weight independent.
6. Promote eligible episode IDs without deleting raw or rejected data.
7. Publish official Hugging Face LeRobot V3 through `/mnt/workspace/spy/data_pipeline` and reload it with the official loader.
8. Build horizon-dependent Camera0-t0-relative actions online after loading; never store one fixed chunk horizon in V3.

Robot V3 keeps source-native state/joint/action fields and absolute EEF poses. Human V3 keeps measured absolute hand poses and validity without fabricated Robot joints.

## Quick start with Codex

Start Codex from the repository root and ask it to use `$curate-embodied-data`. Provide the raw train root, Human/Robot embodiment, task, explicit pilot episode IDs, desired output root and any authoritative URDF/camera/pose information.

Example request:

```text
Use $curate-embodied-data for a bounded train-only pilot.
Raw root: /path/to/raw
Embodiment: Robot, GR3
Task/source: block stacking, supplier-a
Allowed episodes: 12, 31, 48
Do not access validation/test/eval or discover more episodes.
Run intake, cheap prefilter, pose checks, selected qwen-cleaning-v2 checks,
employee review-queue generation, official LeRobot V3 publication and loader round-trip.
Stop and report before any full scan or external provider call.
```

The skill is stored at `.codex/skills/curate-embodied-data/`. Its references define source onboarding, pose semantics, VLM prompt versioning, quality checks, employee review and V3 handoff.

## SAM3 policy

The current official SAM3 integration is usable for prioritizing Robot frames for employee review. It never deletes data automatically. A C2 run writes ranked `review_queue.json` and `review_queue.csv`, overlays and contact sheets. Employees inspect them and record a separate decision.

Only a later versioned manual-mask calibration may authorize a hard C2 threshold. Until then, low overlap means “review first,” not “bad data.”

## VLM prompt policy

Keep the output JSON schema stable, but version prompts per source and task. Human and Robot prompts must differ. A prompt profile records task vocabulary, camera assumptions, actors, objects, temporal samples, ambiguity rules and negative examples. Draft and calibrate prompts only on a small explicit train allowlist; never tune them on validation/test/eval.

## Outputs

- immutable raw data remains in place;
- compact manifests and sidecars contain lineage, hashes, evidence and dispositions;
- review queues contain employee-facing evidence paths;
- eligible manifests select publishable episodes;
- official LeRobot V3 contains truthful native fields plus absolute pose/validity;
- round-trip reports prove episode, timestamp, video and schema integrity.

Run `PYTHONPATH=packages python -m robot_data_studio.curation --help` for the current CLI commands. Always run `validate`, `dry-run`, a synthetic smoke and a bounded real pilot before requesting a full run.
