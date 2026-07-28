---
name: align-robot-kinematics
description: Use when Codex needs to reconcile a robot dataset with a URDF through FK/EEF consistency, especially when link or joint names differ, pose fields are unknown, base or TCP frames are offset, joint signs or zero offsets may be wrong, or a dynamic robot-data import needs a reviewable SE(3) correction proposal.
---

# Align Robot Kinematics

Discover mappings from URDF structure and held-out SE(3) geometry. Treat names only as tie-breakers. Never invent vendor link names, state indices, or correction constants.

## Workflow

1. Read the repository instructions and identify its Python environment. Keep the dataset and URDF read-only.
2. Run the bundled analyzer with explicit repository, dataset, URDF, sample budget, and output paths:

```bash
python skills/align-robot-kinematics/scripts/analyze.py \
  --repository /path/to/robot-data-studio \
  --dataset /path/to/dataset \
  --format auto \
  --urdf /path/to/robot.urdf \
  --sample-budget 64 \
  --output /path/to/proposal.json
```

3. Inspect the JSON contract:
   - For `status: proposed`, review the selected chains, joint/state mappings, pose representation, base/TCP transforms, joint signs/offsets, confidence, warnings, candidate margin, and held-out before/after position metres and orientation radians.
   - For `reason: ambiguous`, `insufficient_excitation`, `no_candidates`, or `fit_failed`, report the evidence and stop. Request a more informative trajectory or explicit mapping; do not lower safety thresholds silently.
4. Preview the proposal in Data Studio or compare its held-out residuals. Check every arm independently for multi-arm robots.
5. Ask for authorization before accepting a proposal. Acceptance may update project metadata and derived exports, but must not overwrite source observations.
6. Verify the proposal fingerprint before applying or exporting. Re-run discovery if the dataset, URDF, or mapping changed.

## Quick reference

| Situation | Action |
|---|---|
| Unknown dataset field names | Let geometric discovery rank mappings |
| Multiple near-equal candidates | Refuse and show candidate-margin evidence |
| Nearly static joints | Request an excitation trajectory |
| Only XYZ is logged | Use the legacy position-only compatibility path; do not claim full SE(3) |
| Accepted correction | Apply only to derived preview/export fields and preserve provenance |
| New robot or vendor | Re-run the analyzer; never add robot-specific constants |

## Common mistakes

- Do not select `tool0`, `eef`, gripper links, or joint columns by spelling alone.
- Do not combine position and orientation into one unlabeled score.
- Do not fit and validate on the same frames; use the analyzer's held-out metrics.
- Do not treat base, TCP, and joint-zero corrections as uniquely identifiable when warnings say otherwise.
- Do not accept stale proposals or write corrections into source files.
