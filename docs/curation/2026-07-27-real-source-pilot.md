# Bounded real-source pilot — 2026-07-27

> Authority update (2026-07-28): the frozen AHA EEF, Robot Camera0 B-grade
> gate, and XJGT DISPLAYMATRIX evidence supersede the geometry blockers in the
> original snapshot below. See `2026-07-28-dual-action-v3-pilot.md` for the
> bounded rerun; no already-supplied EEF/camera/orientation evidence is pending.

Only `/mnt/cpfs/egodata/XJGT_20260624` and `/mnt/cpfs/block_dataset` were accessed. Paths containing `val`, `validation`, `test`, or `eval` are denied case-insensitively before opening. No video frames were decoded or copied, and no training, DLC, external VLM, full scan, or commit was run.

## Observed formats

| Source | Episode boundary | Time | State/action | Images and validity |
|---|---|---|---|---|
| XJGT Human | One `hdf5/<asset>_hdf5.hdf5` file per episode | HDF5 has no timestamp field. Adapter uses configured 30 Hz; selected MP4 headers report base rate 30 and frame counts exactly matching HDF5. | State is left/right hand world translation from the documented absolute 4x4 transforms. There is no commanded action, so action checks are skipped. Per-episode English description is retained; timed annotation-event bounds are kept in lineage without inventing a contact event. | Paired `video/<asset>_video.mp4`; validity is both configured hand confidences >=0.5 plus finite state. `quality_hand` is retained as counts and polarity hypotheses but is not treated as ground truth because its 0/1 meaning is undocumented. |
| block Robot | LeRobot V3 `episode_index`; each selected episode maps to exactly one parquet row group | `episode_time=row.timestamp`; `file_time=videos/.../from_timestamp+episode_time`. Nanosecond source/action/camera clocks are available. Decoded PTS was not evaluated because the bounded pilot did not decode video. | `observation.state` has 33 channels, `action` 37; pose variants have 38/42 channels. Finite state/action/timestamp defines validity. Per-episode `expand_task` is retained as raw language while the generic task remains the coverage ID. | RGB AV1 and depth FFV1 references are metadata only; the adapter uses the configured RGB key and never copies media. |

The block adapter reuses `FormatRegistry`/`LeRobotDatasetReader` for V3 metadata and reads only selected parquet row groups in batches of 256. The XJGT adapter requires explicit asset IDs and indexes HDF5 datasets frame-by-frame; it does not use the existing recursive HDF5 glob or whole-array reader.

## Pilot selection and outcomes

- Human: `1006561`, `1006564`, `1006570` — 866, 662, 855 frames; durations 28.83, 22.03, 28.47 s; 7, 8, and 5 timed annotation events. Both hand confidence masks and numeric states were valid. All three produced episode-local 64-point maps and no hard rejects or window masks. Their Qwen Core coverage is 0 because robot-only metrics are N/A and no video/VLM core stage was run; therefore the scorecard final decision is `review`, independently of the cheap-check disposition.
- Robot: episode indexes `0`, `1`, `2` from `task_001287/batch_000001`, whose `info.json` declares `train: 0:21` — 417, 3935, 981 frames; durations 13.87, 131.13, 32.67 s. All numeric rows were valid and all produced 64-point maps. Each remains review-only for `state_action_latency_mismatch` on `left_shoulder_pitch_joint`, with correlation lag 3 frames.
- Robot clock audit: median `state_ts_ns-action_ts_ns` is between -0.07 and -0.02 ms; absolute P95 is 18.83–19.51 ms. The lag-3 finding is therefore treated as possible command-to-motion response semantics, not rewritten as a capture-clock offset.
- Coverage is three episodes per source/task, all in `gte_10s` and `high` motion buckets. No whole-dataset normalization was inferred.

The latest compact episode sidecars and manifests are under `.rds-artifacts/real-pilot-qwen-v4/{human,robot}`. Human manifest hash is `2c13d671b40fd6ef14de740a76d414310be06c1902d2699eab688845a0d7b192` (4.4 KB); Robot is `767cdd34960a2b3154e6d843aeef50b642f69983224713d73ba7f079fc43307b` (4.6 KB). Detailed 64-point maps and evidence stay in per-episode sidecars rather than being duplicated in the manifest. The prior format-inspection reports remain under `.rds-artifacts/real-pilot/reports`.

## Qwen Core contract status

The existing `quality/qwen_filters.py`, `filter_service.py`, `scorer.py`, models, weights and legacy UI meanings remain authoritative. The curation bridge adds explicit evidence and N/A coverage; it does not replace them. In particular, the existing UI `state_action_alignment` stage is time-sync behavior, while paper S2 is the separate trend-alignment implementation.

| ID | Contract / implementation | This bounded pilot |
|---|---|---|
| S1 | Existing sudden-change implementation; requires source/task train-only normalization. | Human N/A; Robot N/A because no reference distribution was fitted. |
| S2 | Existing state-action trend alignment on explicit name/index channel maps. | Human N/A; Robot evaluated on `left_shoulder_pitch_joint`, 3/3 score 1.0. |
| S3 | Existing extreme-value implementation with reference bounds. | N/A; no train reference bounds were fitted. |
| S4 | Existing URDF FK consistency implementation. | Human not applicable; Robot N/A because the logged pose frame and exact EEF link are undocumented. |
| S5 | Existing base-frame/EEF orientation alignment. | N/A; trusted base/EEF orientation mapping is absent. |
| C1 | New structured instruction-evidence provider contract; legacy `task_success` is marked approximate only. | N/A, provider disabled and no external VLM called. |
| C2 | New URDF projection + robot-mask provider + deterministic IoU evidence contract. | N/A: camera binding is now frozen, but no SAM3/mask provider is installed. |
| C3 | Existing video-quality implementation. | N/A: metadata-only pilot decoded zero frames. |

Robot core score is 1.0 over the one evaluated metric, with coverage `1/8=0.125`; missing metrics are never imputed as 0 or 1. Human core score is null with zero coverage. Hard gates execute before any weighted score.

## Human extension status

The three episodes scored 0.869, 0.831 and 0.833 over evaluated extension metrics. Hand confidence spans are exact and no invalid gap is bridged by interpolation. H1 tracking, H2 named bone/topology stability, H4 camera-relative translational/rotational SE(3) derivatives, H6 calibration-canvas FOV, H7 per-hand/bimanual masks and H8 structural text/event checks ran. H3 left/right identity is N/A because forearm confidence is zero throughout the selected samples. H5 retains both `quality_hand` polarity hypotheses and is N/A until label semantics are documented. The 2026-07-28 rerun evaluates H6b using the frozen encoded 720x1280 to display 1280x720 DISPLAYMATRIX binding.

The H6 calibration-canvas in-FOV ratios were 0.998, 0.968 and 0.997. Two episodes received H8 score 0.5 because one annotation event extends outside the configured episode duration; this is evidence for review, not a fabricated semantic conflict.

## UI bridge

The existing web app now incrementally shows dataset-level Qwen Core, Extension and Coverage summaries plus pass/review/mask/reject counts. Episode details group Qwen Core, Human/Robot Extension and Safety Gates, displaying run/N-A status, score, preserved default weight, reason and abnormal frame intervals. C2 and Human evidence panels expose placeholders for RGB/mask/URDF projection/IoU and video/hand projection/confidence/valid spans. Existing sliders, manual review, legacy status and seven-stage behavior are unchanged.

## Geometry and unresolved semantics

- EEF FK now uses the frozen `left_end_effector_link` / `right_end_effector_link` experiment authority with the SHA-pinned GR3 URDF. It is not a deployment TCP claim; `base_height` and `base_pitch` remain outside the URDF chain.
- Human calibration-space projection uses `T_camera_joint=inv(T_world_camera)@T_world_joint`; the frozen container DISPLAYMATRIX maps encoded 720x1280 to display 1280x720, so H6b is evaluated in the 2026-07-28 rerun.
- Robot Camera0 now uses the frozen B-grade gate, dynamic head-pitch FK, stored 832x480 K and explicit mechanical-to-OpenCV axes. C2 alone remains `not_evaluated` because no robot mask provider is available.
- XJGT `quality_hand` polarity, explicit per-frame timestamps, and contact-event semantics remain unverified. Robot action semantics and whether the observed three-frame response lag is expected remain unverified.
- Remaining evidence, excluding the now-frozen EEF/camera/orientation authorities: XJGT `quality_hand` definition; robot mask/SAM3 provider; action-command response semantics; and a train-only normalization reference.
