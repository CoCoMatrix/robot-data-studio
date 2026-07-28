# Bounded dual-action reference pilot

## Scope

The pilot used only the previously verified train episodes:

- Human XJGT: `1006561`, `1006564`, `1006570`
- Robot block dataset: `0`, `1`, `2` from
  `task_001287/batch_000001` (`train: 0:21`)

No validation/test/eval path was opened. No video was decoded, copied or
re-encoded. No VLM/SAM3, DLC, training, full scan or normalization ran.

## Pipeline result

Both sources completed raw manifest → deterministic prefilter → eligible
manifest → `lerobot_v3_dual_action_reference_v0.2` promotion → reload →
existing curation deep clean → final accepted manifest. Each source had 3 raw,
3 eligible, 3 promoted, 3 reloaded and 3 non-hard-rejected final episodes.
The final repeat resumed all 3 promoted episodes per source. Both audit reports
are `passed` with no copied media.

| Evidence | Human | Robot |
|---|---:|---:|
| Frames | 866 + 662 + 855 | 417 + 3935 + 981 |
| Joint observation/target valid | 0 / 0 (explicitly missing) | every frame / every frame |
| Absolute EEF observation/target | two hands per frame | two experiment EEF links per frame |
| Absolute Camera0 | every frame, `T_world_camera0` | every frame, dynamic `T_base_link_camera0` |
| Deep-clean outcome | review, no cheap findings | review, lag finding on each episode |
| Motion coverage | 3 high | 3 high |
| Extension score | 0.869 / 0.831 / 0.833 | 0.998 / 1.000 / 0.999 |
| Qwen Core coverage | 0/8 | 1/8; S2 only |

Human canonical joint fields remain `null` with false validity. The reload
compatibility view derives the old engine's motion state from absolute EEF
translations under the source validity; it does not alter canonical fields or
interpolate gaps.

Robot observation EEF is exact-name FK from the 33D source state. Target EEF
holds anchor-state waist joints and applies only the exact-name arm7 values
from the 37D action. URDF and authority hashes are recorded. The result is an
offline `base_link` experiment EEF, not a deployment TCP claim.

## Round-trip evidence

Promotion records independent hashes for timestamps, joint values/name order,
absolute poses/masks, language and media references. The audit streamed every
canonical JSONL row and checked monotonic timestamps, finite valid payloads,
invalid-pose null payloads, stable names, source frame binding, referenced
media existence and promoted/final lineage.
For shared Robot videos, each reference separately records
`episode_time=row.timestamp` and
`file_time=from_timestamp+episode_time`; the audit recomputes this equality
per row and never uses interval heuristics.

- Human pilot report SHA256:
  `d43c5df8317fe349266d317cc9e6303a60aad33f43b5ce425fc957cf2ae007ce`
- Human audit SHA256:
  `a49a552675eb342183a23f4db2cf6692f552ecd657588f7c0c42ff283795d6e9`
- Robot pilot report SHA256:
  `e9d0d4a6eceebfa0f5fc850c6a49b24374c514e4becaee52caeded7d37e76390`
- Robot audit SHA256:
  `bd7904b03e8be1d61f42036391f3cfdffa0247ae03a87801488c3aed9f3a1e53`

## Status and remaining evidence

Implemented and evaluated: typed dual modalities, missing masks, train-only
prefilter, eligible-only promotion, reference media, reload, existing deep
clean, final manifest, resume, GR3 experiment-EEF FK, Human absolute
hand/camera poses, and synthetic arbitrary 8/64-horizon Camera0 chunks.

Implemented and evaluated on the same 3+3 train-only pilot: all Robot frames
now carry dynamic Camera0 from the SHA-pinned head-pitch FK, OAK sidecar and
B-grade gate; all Human headers retain encoded 720x1280 and display 1280x720,
and H6b is evaluated after the frozen DISPLAYMATRIX. Robot C2 remains N/A
because no mask provider was run. Dynamic chunks follow FINAL_V2 exactly:
`p_target-p_anchor` and `R_target @ R_anchor.T` in Camera0 frozen at t0.

Not official-loader validated: this project-canonical reference layout is not
upstream/HF Parquet+MP4 LeRobot V3. A production consumer must either adopt
this typed loader or add a separate official exporter that can preserve media
references without silently copying them.

Full-run prerequisites are limited to: choose the production consumer/layout;
provide a Robot mask provider only if C2 is required; clarify the observed
Robot response-lag policy and Human `quality_hand` polarity if
they should become gates. A VLM API key is a later optional semantic-provider
input and is not a blocker for this deterministic pipeline.
