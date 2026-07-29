# Embodied-data curation guardrails

- Work only with synthetic fixtures or inputs explicitly matched by a profile `train_allowlist`. Deny patterns win. Never open real validation/test/eval samples.
- Do not start DLC, training, external VLM calls, full dataset scans, or expensive motive/gradient attribution.
- Stream by shard and episode. Require a positive pilot limit; do not build whole-dataset arrays or use `np.vstack` across the corpus.
- Never modify source RGB/MP4. Only the explicit bounded official LeRobot V3 publisher may decode selected pilot frames and encode its required video field; all other stages write compact sidecars beneath `sidecar_root`.
- LeRobot V3 uses `episode_time=row.timestamp` and `file_time=from_timestamp+episode_time`. Decoded PTS is a tolerance check only; interval heuristics are forbidden.
- Chunks and resampling maps never cross episodes. Each target point retains original frame, timestamp and validity; invalid gaps stay invalid.
- Keep hard reject, review, window mask and soft weight independent. Prefer masking a bad window and retaining valid segments over deleting an episode.
- Keep Human/Robot and source/task normalization configurable. Never hardcode a Human:Robot ratio or a task such as grasp in framework code.
- Run cheap T0/T1 rules before optional T3 mock VLM. Real VLM backends are outside this prototype.
- Final training publication uses the installed official Hugging Face `LeRobotDataset` V3 writer and must round-trip through its loader. Reference-only JSONL remains historical evidence, not the final artifact.

Use `PYTHONPATH=packages python -m robot_data_studio.curation validate|dry-run|pilot --help`. Focused tests live under `tests/curation/` and use synthetic data only.
