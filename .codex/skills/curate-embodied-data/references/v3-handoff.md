# Official LeRobot V3 handoff

Use `/mnt/workspace/spy/data_pipeline`; do not imitate the directory layout. The acceptance criterion is a roundtrip through the installed official `lerobot.datasets.lerobot_dataset.LeRobotDataset` loader.

The official feature contract stores timestamp, episode/frame/task/source lineage, Camera0 absolute pose, left/right absolute EEF pose and validity, and official video fields. Robot adds source-native `observation.state`, `action`, their validity, and same-row action-target FK poses. Human is pose-only and does not publish fabricated Robot joint/action columns. Keep `quality_hand` optional source metadata.

Use `episode_time=row.timestamp` and `file_time=from_timestamp+episode_time`; decoded PTS is only a tolerance check. Chunks and 64-point maps cannot cross episodes and retain original frame, timestamp, and validity.

Run the bounded publisher through the repository `dual-pilot` bridge with explicit episodes, then run `scripts/audit_official_lerobot_v3.py`. Verify official reload, features, shapes/dtypes, statistics, offsets, timestamps, video alignment, manifests, source/config hashes, and lineage. The publisher may encode only the explicitly approved pilot video; never copy or re-encode a full source collection.
