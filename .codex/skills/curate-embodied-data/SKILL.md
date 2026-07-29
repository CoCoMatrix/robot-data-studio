---
name: curate-embodied-data
description: Inspect, clean, pose, review, and publish Human or Robot embodied training data without modifying raw media. Use for new-source onboarding, train-only integrity audits, Human/Robot pose preparation, Qwen-v2 quality checks, source-specific VLM prompts, GR3 URDF/SAM3 review, eligible-manifest promotion, official Hugging Face LeRobot V3 conversion, or employee review-queue generation.
---

# Curate Embodied Training Data

Operate from the repository root. Read `AGENTS.md` first. Keep raw media immutable and treat held-out-looking paths as inaccessible even when an allowlist is broad.

Read [workflow.md](references/workflow.md) for the end-to-end order before starting a real source.

## Non-negotiable outcomes

- Preserve raw files. Express pass, reject, review, window mask, and soft weight in manifests/sidecars.
- Open only an explicit train allowlist. Apply case-insensitive `val|validation|test|eval` denies first.
- Start with metadata and a positive episode limit. Never discover or scan a full real dataset by default.
- Keep Human and Robot schemas truthful: Human has no fabricated joints; Robot retains native joints/actions and authoritative absolute EEF poses.
- Store per-frame absolute poses. Construct horizon-dependent Camera0-t0-relative actions only after loading.
- Treat official SAM3 C2 as review-only until manual-mask calibration. Never delete data from provisional SAM3 scores.
- Publish through the official LeRobot V3 writer and prove a round-trip through the official loader.

## Procedure

1. **Onboard the source.** Follow [source-onboarding.md](references/source-onboarding.md). Record the raw root, explicit train episodes, deny rules, source/task IDs, embodiment, fields, units, FPS, video binding, coordinate frames, pose/FK authority, validity and expected outputs. Stop if any required authority would change label meaning.
2. **Create a profile.** Copy `profiles/curation/current_v2.json` or a source-specific profile. Enable checks independently; do not count disabled, retired or unavailable checks. Validate before reading episode payloads:

   ```bash
   python .codex/skills/curate-embodied-data/scripts/validate_profile.py PROFILE.json
   PYTHONPATH=packages python -m robot_data_studio.curation dry-run --profile PROFILE.json --input TRAIN_ROOT --episode ID --max-episodes 1
   ```

3. **Run cheap prefiltering.** Inventory metadata first; then stream explicit episodes. Check file/video readability, schema, finite values, episode boundaries, timestamps, FPS/PTS, field lengths, duplicates and lineage. Retain every disposition and reason.
4. **Prepare poses and validity.** Follow [pose-contract.md](references/pose-contract.md). Verify FK/name mapping and transform directions. Keep each Human hand independent. Never fill missing Human joints, interpolate invalid gaps or pre-materialize window-relative actions.
5. **Run selected deep checks.** Follow [quality-checks.md](references/quality-checks.md). Reuse content-addressed intermediates. For C1, follow [prompt-policy.md](references/prompt-policy.md), version/hash a Human- or Robot-specific prompt, and calibrate only on a small explicit train pilot. For Robot C2, require authoritative URDF/Camera0/exact-frame joints and a real local SAM3 checkpoint.
6. **Create decisions and employee review.** Keep hard rejection separate from review. Follow [review-queue.md](references/review-queue.md). Provisional C2 writes ranked JSON/CSV plus overlays, but `automatic_exclusion_allowed=false`; an employee records the final decision separately.
7. **Promote without deletion.** Produce an eligible manifest. Leave failed/review data in raw storage. Publish only approved IDs through `/mnt/workspace/spy/data_pipeline` as official LeRobot V3; follow [v3-handoff.md](references/v3-handoff.md).
8. **Prove the handoff.** Reload with the official loader and audit offsets, timestamps, video PTS, fields, validity, statistics, hashes, lineage, selected/evaluated denominators and episode isolation. Keep reports, caches and review artifacts outside raw roots.

## Execution order

Run `validate` → `dry-run` → synthetic smoke → bounded real pilot → employee/scientific review → explicit full-run approval. A successful pilot does not authorize a full scan.

For an isolated authorized Robot C2 audit:

```bash
PYTHONPATH=packages python -m robot_data_studio.curation c2-real-pilot \
  --profile PROFILE.json \
  --dataset-root OFFICIAL_V3_ROOT \
  --repo-id LOCAL_REPO_ID \
  --episode 0 --episode 1 --episode 2 \
  --max-episodes 3 \
  --source-video-sha256 SHA256 \
  --sam3-model-dir LOCAL_OFFICIAL_SAM3_DIR \
  --sam3-cache-root CACHE_ROOT \
  --output OUTPUT_ROOT
```

Do not copy the example episode IDs or paths to another source. Bind them in that source's reviewed profile.

## Required handoff summary

Report: opened roots/episodes, held-out count (must be zero), selected and evaluated checks, pass/review/reject counts, masks/weights, VLM/SAM model and prompt hashes, API cost, review-queue paths, eligible manifest, official V3 round-trip, unresolved N/A items, and the exact command required for any separately authorized next phase.

Use `scripts/make_synthetic_train_fixture.py` for smoke fixtures. Use `scripts/audit_dual_action_pilot.py`, `scripts/audit_official_lerobot_v3.py`, and `scripts/audit_live_quality_pilot.py` for deterministic handoff checks. Never pass credentials on a CLI or count mock evidence as a real result.
