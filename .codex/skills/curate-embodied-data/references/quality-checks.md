# Current quality checks

The superseding contract is `qwen-cleaning-v2`. Active Qwen checks are independently selectable: S1 sudden change, S2 state/action trend alignment, S3 extreme value, C1 structured instruction/video/action-phase consistency, C2 Robot full-silhouette video-state consistency, and C3 video quality. S4 and S5 are retired historical checks; omit them from score, coverage, gates, and confidence.

Disabled checks are omitted. Missing prerequisites fail closed as `not_evaluated` with a reason. Compute overall coverage from the actual selected Core plus Human-extension checks, and also report Core/extension coverage separately. Non-selectable legacy evidence does not inflate the denominator. Deduplicate shared evidence and aggregate a shared rejection once.

C2 renders the visible full URDF visual mesh using exact-frame joints, trusted mappings, Camera0/display transforms, z-buffer visibility, and stored resolution. Compare to a cached SAM3 mask using IoU, precision, recall, coverage, clip aggregation, and local temporal offsets. Official SAM3 is provisional; only a versioned calibrated fine-tuned checkpoint may enable calibrated thresholds. Before calibration C2 is review-only. The old support-disc metric is historical evidence, not C2.

Human extensions keep each hand independent. Identity continuity runs only where both hands are valid. Task-active absence near contact/manipulation/release masks supervision or reviews; inactive-hand absence is acceptable. Run hand-object interaction only for eligible C1 manipulation candidates, reuse cached masks/tracks, and report approach, contact/grasp, manipulate/carry, release, or uncertain/occluded. Missing masks/tracks are N/A or review, never imputed zero or an uncalibrated hard reject.
