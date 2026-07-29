# Source onboarding contract

Create one versioned profile per source/task contract. Do not reuse a profile merely because two datasets share a file format.

## Required intake

- immutable raw root and source ID;
- explicit train episode allowlist and held-out deny patterns;
- Human or Robot embodiment and task vocabulary;
- metadata, table and media paths plus content hashes;
- episode/frame/timestamp semantics, FPS and shared-video offsets;
- image key, encoded size, display orientation, crop/resize and lens handling;
- state/action names, order, units, command-vs-state meaning and validity;
- pose transform direction, parent/child frame IDs and Rot6D convention;
- for Robot, URDF version/hash, exact logged joint-name mapping, EEF links and Camera0 authority;
- requested checks, external providers, pilot budget, output root and final V3 target.

Inspect README/schema/metadata before media. If a source lacks optional authority, mark the dependent check `not_evaluated`; do not infer a passing value. Stop when missing information would change pose labels, action semantics, episode boundaries or held-out separation.

## Pilot design

Choose a small explicit train set that covers distinct tasks, durations, cameras and motion regimes without adjacent-frame duplication. Record the episode IDs in the profile. A pilot limit is a safety boundary, not a representative distribution claim.

After the pilot, report what was proven, what remains N/A, estimated full-run throughput/storage/API cost and the exact additional authorization required. Never silently expand the allowlist.
