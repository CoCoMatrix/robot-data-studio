# Employee review queue

Automatic evidence and employee decisions are separate layers. Never rewrite raw data or turn provisional evidence into a rejection.

## SAM3 / Robot C2

Official SAM3 without a versioned manual-mask calibration is a reviewer. Rank weak overlap first and emit:

- `review_queue.json` for machines and audit;
- `review_queue.csv` for employees;
- per-frame overlay with original, URDF silhouette, SAM3 mask and overlap/error;
- per-episode contact sheet;
- episode/frame/timestamp, IoU, precision, recall, coverage, SAM confidence, temporal-offset diagnostic and evidence paths.

Every queue and row must say `disposition=review` and `automatic_exclusion_allowed=false`. Ranking is triage only; do not invent a pass/fail threshold from the current dataset.

## Employee procedure

Review lowest IoU first. Check that the decoded frame, timestamp, visible Robot pose, left/right topology, occlusion, URDF boundary and SAM mask refer to the same instant. Record one of:

- `accept`: evidence is a benign boundary/model difference;
- `mask_window`: a bounded bad interval should not supervise training;
- `review_episode`: more context or source-owner input is required;
- `exclude_episode`: a human-confirmed episode-level defect, with reason;
- `not_decidable`: missing authority or ambiguous visibility.

Store employee ID, decision time, reason code, free-text note and reviewed artifact hashes in a separate append-only decision file. Do not edit the generated queue. Episode exclusion must never physically delete raw source data.

## Calibration upgrade

Only upgrade C2 to a hard gate after producing versioned train-only manual full-Robot masks, measuring the SAM3 and URDF error separately, freezing per-camera thresholds and verifying them without accessing formal validation/test/eval data.
