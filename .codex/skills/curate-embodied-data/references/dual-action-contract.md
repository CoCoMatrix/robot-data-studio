# Historical dual-action contract (superseded)

This note is retained as historical publication evidence. Current pose-only
Human semantics and the official handoff live in `pose-contract.md` and
`v3-handoff.md`.

The final layout is written and reloaded by Hugging Face
`lerobot.datasets.lerobot_dataset.LeRobotDataset` 0.6.0. It uses official V3
chunked Parquet, MP4, `meta/info.json`, tasks, episode metadata and statistics.
The older `lerobot_v3_dual_action_reference_v0.2` JSONL remains read-only
promotion evidence, not the final training dataset.

Each frame has source-native `observation.state` and `action`, absolute
`observation.eef_pose_absolute`, `action.eef_pose_absolute`, absolute Camera0,
and independent validity masks. Official fixed-width Human joint/action
padding is zero only where its mask is false; loaders must use a missing-joint
token. Pose encoding is xyz plus the first two rotation-matrix rows with
Gram-Schmidt row reconstruction. Units and transform directions are recorded
in `meta/curation_contract.json`.

The raw prefilter records every disposition. Only eligible episode IDs are
promoted; source data is never deleted. LeRobot shared-video mapping uses
`episode_time=row.timestamp` and `file_time=from_timestamp+episode_time`.
Decoded PTS is tolerance-only. Only the explicit bounded official publisher
decodes selected pilot frames and encodes the official video field.

Action chunks are downstream views. Freeze Camera0 at t0, then compute exact
FINAL_V2 `p_target-p_anchor` and `R_target@R_anchor.T`. Crossing an episode
fails and invalid targets stay invalid. Robot training labels are recomputed
from chunk t0 state plus each selected recorded action through frozen GR3 FK;
future observation state and lag shifts are forbidden. Human future rows use
measured absolute hand poses as action surrogates.

`quality_hand=1` means complete hand visibility. It is zero-weight source
metadata, never a cross-supplier hard gate. SAM3 is the C2 mask contract:
disabled/missing weights/missing API credentials all yield `not_evaluated`;
mock masks test cache plumbing only and never constitute a real C2 pass.
