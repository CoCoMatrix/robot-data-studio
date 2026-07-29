# Dual-action V3 compatibility note

## Audit outcome

`data_pipeline`'s existing `lerobot_v3_canonical_v0.1` is a useful
project-local JSONL/images format, but it is not the upstream/HF
Parquet+MP4 LeRobot V3 layout consumed by Robot Data Studio. Its current
exporter materializes all episodes and copies or decodes every frame image.
Its conversion CLI also loads all episodes before applying the requested
limit. No project training loader outside `data_pipeline` was found for that
local layout.

The existing exporter/importer stays unchanged. This prototype adds the
honestly named `lerobot_v3_dual_action_reference_v0.2`: compact JSONL with
source media URI/frame/timestamp references and no RGB/video copy. It is not
claimed to pass the official LeRobot loader.

## Additive contract

Each frame has typed joint observation, absolute EEF observation, joint target,
absolute EEF target, absolute camera pose, independent validity masks, names,
units, action type, transform direction, source frame/timestamp, language and
provenance. Human joint fields are explicitly missing rather than zero-filled.
Robot joint fields preserve source-native order.

Dynamic Camera0 chunks are downstream views over absolute poses; no fixed
64-step chunk or training mean/std is written during promotion. The builder
supports arbitrary t0/horizon/stride, exact k0 zero/identity, invalid-target
propagation and a strict no-cross-episode gate.

## GR3 and camera authority

Robot EEF FK uses the existing Qwen URDF chain math with the SHA-pinned
`fourier-robot-descriptions==0.2.7` 6DoF URDF
(`2b87382bbbda834260526b00728e2160529a3cd693267417d3e56683c87dcc9a`).
The AHA-WAM authority record
(`5d81a951e7a9e32d993a6ed79628e3ab7f59ab86dee4a970b7c495057b90096b`)
selects `left_end_effector_link` and `right_end_effector_link`, exact-name
arm targets and anchor-state waist joints. These are offline experiment EEF
frames, not deployment TCP claims.

Robot Camera0 reuses the frozen AHA head-pitch FK and OAK authority: sidecar
SHA `903ca96f...cf54`, B-grade gate SHA `9a82055a...638`, explicit
mechanical-to-OpenCV axes, and the verified stored 832x480 K. This authorizes
current EEF proprio in Camera0 only; future labels still come from target FK,
and C2 remains N/A without a robot-mask provider. Human stores authoritative
`T_world_camera(t)` and `T_world_hand(t)` and applies the frozen MP4
DISPLAYMATRIX from encoded 720x1280 to display 1280x720 before using K.

The FINAL_V2 downstream rotation delta is `R_target @ R_anchor.T`; translation
is `p_target - p_anchor`, both in Camera0 frozen at t0. A non-commuting 3D
regression independently replays the authority so an order reversal cannot be
hidden by identity or yaw-only cases.
