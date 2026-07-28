# Historical profile contract (superseded)

This v1 note is retained as historical evidence. For the current versioned
selection and workflow contract, use `workflow.md` and `quality-checks.md`.

`profiles/curation/profile.schema.json` is authoritative. A profile names only train input patterns and explicit deny patterns; deny wins. It declares task text and start/contact/end events, modalities, fixed-time or event-or-motion progress, target point count, source embodiment/fps/latency/validity/resampling/channel bounds, hard/review gates, cheap-check thresholds, mock VLM budget, sidecar root, and configurable source/task weights.

Framework code is task-neutral. Retain raw language and normalize it into verb, object, target and modifiers. Source validity remains first-class; nearest resampling records invalid target points instead of filling gaps.

Real-source profiles also select a bounded adapter, state/action field mapping, validity fields and video key. Geometry configuration is optional and source-specific. Unless URDF, joint names, end-effector link and logged EEF mapping are all trusted, FK stays `not_evaluated`; reprojection additionally requires trusted intrinsics, extrinsics, optical-axis convention and EEF/image binding.
