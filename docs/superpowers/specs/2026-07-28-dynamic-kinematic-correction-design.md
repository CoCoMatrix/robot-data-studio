# Dynamic Kinematic Consistency and Correction Design

## Goal

Upgrade kinematic consistency from an XYZ-only configured check into a dataset-adaptive calibration workflow that:

1. measures full SE(3) agreement between URDF forward kinematics and logged end-effector poses;
2. discovers robot chains and state mappings without requiring fixed joint or link names;
3. proposes explainable base, TCP, joint-convention, pose-representation, and per-arm frame corrections;
4. previews and applies an accepted correction without modifying source data; and
5. exposes the same workflow through Data Studio and a reusable Codex skill.

The first validation target will be synthetic fixtures. A future Astribot S1 URDF and dataset sample will be an external validation target, not a hard-coded profile.

## Safety and Source-of-Truth Rules

- Never overwrite imported source data or the uploaded URDF.
- Treat inferred mappings and transforms as proposals until a user accepts them.
- Persist the proposal, evidence, confidence, and acceptance state in project artifacts.
- Apply accepted corrections to derived previews, quality evaluation, and exports.
- Preserve uncorrected values and correction provenance in every export report.
- Do not silently choose a candidate when the evidence is ambiguous or the trajectory lacks sufficient excitation.

## Architecture

### Deterministic Kinematics Engine

Create a focused `robot_data_studio.quality.kinematics` module as the single implementation used by the product service and Qwen-style CLI. It owns:

- URDF model and frame discovery through Pinocchio;
- pose representation decoding and validation;
- multi-chain joint and logged-pose mappings;
- SE(3) residual computation;
- correction hypothesis generation, fitting, and ranking;
- application of an accepted correction to derived frames; and
- serializable diagnostics and proposals.

Remove the behavioral split where `qwen_filters.py` uses a custom FK parser while `filter_service.py` uses Pinocchio. Compatibility functions may remain, but they must delegate to the shared engine.

### Project Integration

Extend the kinematics configuration from one implicit chain into a profile containing one or more named chains. A chain contains:

- URDF end-effector frame;
- ordered joint-to-state mappings;
- logged position and orientation indices;
- logged orientation representation;
- base correction transform;
- TCP correction transform; and
- joint sign and zero-offset corrections.

For backward compatibility, the existing single-chain fields are migrated into one `arm` chain when loaded.

The filter detail response includes residual series, candidate proposals, before/after metrics, confidence, warnings, and the active profile. Data Studio provides discovery, proposal preview, accept/reject, and reset controls.

### Codex Skill

Create `align-robot-kinematics` as a reusable Codex skill. Keep its canonical source in the repository and install a discoverable copy in the Codex skills directory.

The skill must invoke the deterministic analyzer rather than guess link names or directly manufacture transforms. It guides Codex to:

1. inspect dataset metadata and URDF topology;
2. run dynamic discovery on representative episodes;
3. read candidate evidence and ambiguity warnings;
4. preview the best proposal against alternatives;
5. accept only with user authorization; and
6. retain the generated manifest for future reproduction.

The skill contains a concise `SKILL.md`, UI metadata, and a small wrapper script only when the repository CLI alone is insufficient. It does not duplicate FK or fitting logic.

## SE(3) Model

For frame `t` and chain `c`, compute:

```text
T_pred(t, c) = T_base(c) * FK_URDF(q_corrected(t, c), eef_frame(c)) * T_tcp(c)
q_corrected[j] = sign[j] * q_logged[state_index[j]] + zero_offset[j]
T_error(t, c) = inverse(T_pred(t, c)) * T_logged(t, c)
```

Report separate residuals:

- position error: `norm(translation(T_error))` in metres;
- orientation error: `norm(log_SO3(rotation(T_error)))` in radians/degrees; and
- robust summaries: median, p95, maximum, inlier rate, and flagged frame indices.

Position and orientation thresholds remain separate. A frame fails if either configured threshold is exceeded. Do not collapse metres and radians into an unexplained scalar quality metric.

Supported logged orientation representations are quaternion XYZW, quaternion WXYZ, 6D continuous rotation, 3x3 rotation matrix, rotation vector/axis-angle, and RPY. Decoders reject non-finite or geometrically invalid values and return evidence instead of silently normalizing grossly invalid data.

## Dynamic Discovery

### Candidate Generation

Generate candidates from structure and observed data, using names only as a prior:

- Enumerate Pinocchio operational and terminal frames as possible end effectors.
- Enumerate actuated joint paths leading to each frame.
- Normalize metadata tokens for joint/state name similarity when names exist.
- Fall back to dimensions, joint limits, temporal variation, and chain order when names do not match.
- Infer pose field candidates from metadata groups, contiguous dimensions, value ranges, quaternion norms, and rotation orthogonality.
- Evaluate both quaternion orders and every supported representation compatible with the field width.
- Select representative frames that cover joint and Cartesian trajectory extremes.

A URDF using `link_1`, `tool_flange`, or arbitrary vendor names must work without adding those names to source code.

### Hypothesis Search

Rank complete, jointly evaluated hypotheses rather than locking each decision independently:

1. end-effector frame and joint/state mapping;
2. orientation representation;
3. per-joint sign convention;
4. joint zero offsets;
5. fixed world/base transform;
6. fixed local TCP transform; and
7. independent per-arm transforms for bimanual or shoulder-relative logging.

Use a bounded beam for discrete choices and robust nonlinear least squares for continuous SE(3) and joint-offset parameters. Implement the optimizer with NumPy and Pinocchio Jacobians/log maps so the project does not add a general optimization dependency. Fit on one frame subset and score on held-out frames to avoid accepting an overfit transform.

### Identifiability and Confidence

Before proposing a correction, verify that the sampled trajectory excites the relevant degrees of freedom. Examples:

- constant orientation cannot distinguish a world-frame translation from a local TCP translation;
- an almost-static joint cannot establish its sign or zero offset;
- symmetric frame candidates can be indistinguishable on a narrow trajectory.

Confidence combines held-out residual improvement, inlier rate, candidate margin, parameter stability across episode subsets, and excitation checks. Return `insufficient_evidence` or `ambiguous` instead of a usable proposal when a required condition fails.

## Correction Proposal Contract

Each proposal includes:

- stable proposal ID and project/dataset fingerprint;
- chain mappings and selected URDF frames;
- pose indices and representation;
- joint signs and zero offsets;
- base and TCP transforms as 4x4 matrices;
- before/after train and held-out residual summaries;
- confidence and candidate margin;
- evidence messages and ambiguity warnings;
- sampled episode/frame provenance; and
- status: `proposed`, `accepted`, `rejected`, or `stale`.

Changing the dataset, URDF, or relevant mapping invalidates the proposal. Accepting a proposal stores a copy in the active profile; it never mutates the original URDF.

## Preview and Export Behavior

- Preview overlays corrected and uncorrected EEF trajectories and charts both residuals.
- Quality scoring uses the accepted profile. An unaccepted proposal does not change scores.
- Export applies accepted state-pose corrections while retaining raw joint values unless joint correction is explicitly requested by the target mapping.
- Export reports include the full correction manifest, source fingerprint, and before/after metrics.
- Rejecting or resetting a proposal restores the uncorrected derived view without touching source data.

## Error Handling

- Missing Pinocchio, unreadable URDF, invalid pose fields, or no viable chain produce structured skipped/error states.
- Mismatched dimensions identify the offending mapping and candidate rather than raising an opaque FK error.
- Search budgets are bounded by candidate and sampled-frame limits and return partial diagnostics on timeout.
- No correction is auto-accepted solely because it lowers training residual.
- Existing manually configured projects continue to run through the compatibility migration path.

## Verification

### Unit Fixtures

- Exact translation and SO(3) errors for known transforms.
- All supported orientation decoders, including invalid inputs.
- A non-standard URDF whose links and joints do not contain `arm`, `eef`, `tool`, or numbered canonical names.
- Known world/base and rotating local TCP offsets, proving they are distinguished when orientation is excited.
- Joint sign and zero-offset recovery with a held-out trajectory.
- Ambiguous static trajectories that correctly refuse a correction.
- Two independent arms logged in shoulder-relative frames.

### Integration Fixtures

- Discovery through the API creates a persisted proposal with evidence.
- Preview shows before/after position and orientation residuals.
- Accepting a proposal changes derived evaluation and export output, not source files.
- Dataset or URDF changes mark an accepted proposal stale.
- The Qwen CLI and product service return consistent metrics from the shared engine.
- Existing robomimic Panda auto-configuration remains compatible.

### Skill Verification

- Establish the baseline failure mode before writing the skill: a fresh agent given a non-standard URDF must not be able to produce a reproducible profile without the analyzer workflow.
- Validate the skill package structure with the official skill validator.
- Run the skill against the non-standard synthetic fixture and verify it invokes discovery, reports ambiguity, and does not invent fixed link names.
- Run the wrapper script directly and compare its proposal JSON with the product API result.

Subagent-based forward testing requires explicit authorization under the current collaboration policy. Until authorized, deterministic script tests and skill package validation are required, and forward-test status must be reported separately rather than implied.

## Acceptance Criteria

The first two stages are complete only when:

1. Data Studio computes and exposes position plus orientation residuals from one shared Pinocchio engine.
2. Dynamic discovery succeeds on synthetic URDF/data fixtures with arbitrary names.
3. Base, TCP, joint sign/offset, pose representation, and per-arm frame corrections are represented and evaluated as proposals.
4. Low-excitation and ambiguous cases refuse to produce an applicable proposal.
5. An accepted proposal affects preview, quality evaluation, and export while source files remain byte-identical.
6. The reusable Codex skill is installed, structurally validated, and drives the deterministic discovery workflow.
7. Targeted backend, API, CLI, frontend, export, and skill tests pass in a valid project environment.

The future Astribot S1 artifact is an additional real-world validation gate after these acceptance criteria, not a prerequisite for implementing them.
