# Current cleaning contract: `qwen-cleaning-v2`

This contract supersedes, but does not delete, the historical eight-item scorecard.

- Active and independently selectable: S1, S2, S3, C1, C2, C3.
- Retired and outside score, coverage, gates and confidence: S4, S5.
- Disabled active checks remain visible as disabled but are omitted from every denominator.
- Selected checks with missing prerequisites fail closed as not evaluated and reduce coverage.
- Coverage is evaluated selected checks divided by the actual selected Core plus Human-extension denominator; Core and extension coverage are also reported separately. Non-selectable legacy extension evidence does not inflate it.
- Shared decode, masks, tracking, pose and language intermediates are keyed once by content, model, prompt, configuration and sampled frames.
- Integrity gates run before scores. Findings remain separate; identical disposition reasons and windows aggregate once.

Robot C2 now means a visible full-URDF-visual silhouette rendered with z-buffer visibility at the exact recorded state and Camera0 image geometry. Official SAM3 is provisional because the paper's fine-tuned checkpoint, configuration and calibrated per-camera thresholds are unavailable. Before calibration C2 is review evidence only. The older EEF support-disc result remains historical/provisional evidence and is not current C2.

The renderer fails closed when the authoritative URDF has a movable joint absent from the exact-frame recorded mapping, when a visual mesh is unavailable/unsupported, or when Camera0/display binding is incomplete. Thus the cached 3+3 support-disc result is not silently upgraded to full-silhouette C2.

Human extension validity is per hand and per step. Within a chunk, the first invalid step masks that hand through the chunk end; another hand remains supervised and a later anchor may re-enter. Interaction evidence is task-conditioned, reuses cached masks/tracks and is review-only before calibration.
