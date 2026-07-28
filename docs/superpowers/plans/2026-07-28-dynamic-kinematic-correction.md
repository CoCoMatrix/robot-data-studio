# Dynamic Kinematic Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent execution is not authorized for this workspace.

**Goal:** Add full SE(3) kinematic residuals, dataset-adaptive correction proposals, safe accepted-correction application, and a reusable Codex skill that drives deterministic discovery instead of guessing URDF names.

**Architecture:** A new `quality.kinematics` package becomes the only Pinocchio FK and calibration engine. Product filters, project persistence, exports, the Qwen CLI, and the skill all consume its typed profile/proposal contract. Corrections remain immutable project artifacts and are applied through a dataset-adapter wrapper, leaving source files unchanged.

**Tech Stack:** Python 3.11, NumPy, Pinocchio (`pin`), Pydantic 2, FastAPI, React/TypeScript, Vitest, pytest, Codex Agent Skills.

## Global Constraints

- Never overwrite imported dataset files or uploaded URDF files.
- Do not hard-code vendor joint or link names; names are ranking evidence only.
- Position and orientation residuals retain metres and radians as separate metrics.
- Refuse applicable proposals when excitation or candidate margin is insufficient.
- Preserve compatibility with the current single-chain Panda configuration.
- Add no SciPy dependency; use NumPy and Pinocchio Jacobian/log/exp operations.
- Follow test-first RED-GREEN-REFACTOR for every production behavior.

---

### Task 1: Typed SE(3) Pose and Residual Core

**Files:**
- Create: `packages/robot_data_studio/quality/kinematics/__init__.py`
- Create: `packages/robot_data_studio/quality/kinematics/models.py`
- Create: `packages/robot_data_studio/quality/kinematics/pose.py`
- Test: `tests/quality/kinematics/test_pose.py`

**Interfaces:**
- Produces `OrientationRepresentation`, `PoseFieldMapping`, `ResidualSummary`, `SE3ResidualResult`.
- Produces `decode_rotations(values, representation) -> np.ndarray` with shape `(T, 3, 3)`.
- Produces `compute_se3_residuals(predicted, logged) -> SE3ResidualResult` for `(T, 4, 4)` transforms.

- [ ] **Step 1: Write failing orientation-decoder tests**

Cover literal expected matrices for XYZW/WXYZ quaternions, 6D rotation, matrix, rotvec, and RPY; include non-finite, wrong-width, zero-quaternion, and non-orthogonal matrix failures.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/quality/kinematics/test_pose.py`

Expected: collection/import failure because `robot_data_studio.quality.kinematics` does not exist.

- [ ] **Step 3: Implement minimal typed decoders**

Use SVD projection only for small numerical drift in 6D/matrix representations. Reject grossly invalid rotation matrices and zero quaternions with `InvalidPoseRepresentation`.

- [ ] **Step 4: Write failing SE(3) residual tests**

Use hand-derived transforms: 3-4-5 centimetre translation must report `0.05`; a 90-degree Z rotation must report `pi/2`. Assert median, p95, max, and per-frame arrays separately.

- [ ] **Step 5: Implement residuals and verify GREEN**

Run: `pytest -q tests/quality/kinematics/test_pose.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/robot_data_studio/quality/kinematics tests/quality/kinematics/test_pose.py
git commit -m "feat: add SE3 kinematic residual core"
```

### Task 2: Multi-Chain Profile and Pinocchio FK

**Files:**
- Modify: `packages/robot_data_studio/quality/kinematics/models.py`
- Create: `packages/robot_data_studio/quality/kinematics/engine.py`
- Modify: `packages/robot_data_studio/quality/kinematics/__init__.py`
- Test: `tests/quality/kinematics/test_engine.py`
- Test fixture: `tests/quality/kinematics/fixtures/nonstandard_robot.urdf`

**Interfaces:**
- Produces `JointFieldMapping(urdf_joint, state_index, sign, zero_offset)`.
- Produces `KinematicChainProfile(id, end_effector_frame, joints, logged_pose, base_transform, tcp_transform)`.
- Produces `KinematicProfile(urdf_path, chains, fingerprint)`.
- Produces `PinocchioKinematics.evaluate(states, profile) -> dict[str, SE3ResidualResult]`.

- [ ] **Step 1: Add a fixture with arbitrary names**

Use names such as `alpha_axis`, `beta_axis`, `wrist_shell`, and `camera_mount`; do not include `arm`, `link1`, `eef`, `tcp`, `tool`, or `end`.

- [ ] **Step 2: Write failing FK tests**

Test one and two chains, base left-transform, rotating TCP right-transform, joint sign, and joint zero offsets. Expected transforms are literal analytic values for the two-joint fixture.

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/quality/kinematics/test_engine.py`

Expected: profile and engine imports are missing.

- [ ] **Step 4: Implement profile validation and FK**

Validate every state index, joint/frame lookup, 4x4 rigid transform, and orientation field width before iterating frames. Compute:

```python
q_value = mapping.sign * state[mapping.state_index] + mapping.zero_offset
predicted = base_transform @ data.oMf[frame_id].homogeneous @ tcp_transform
```

- [ ] **Step 5: Verify GREEN and compatibility**

Run: `pytest -q tests/quality/kinematics/test_engine.py tests/api/test_filter_details.py::test_kinematic_filter_autoconfigures_robomimic_panda_dataset`

- [ ] **Step 6: Commit**

```bash
git add packages/robot_data_studio/quality/kinematics tests/quality/kinematics
git commit -m "feat: evaluate multi-chain Pinocchio profiles"
```

### Task 3: Structure-First Dynamic Discovery

**Files:**
- Create: `packages/robot_data_studio/quality/kinematics/discovery.py`
- Modify: `packages/robot_data_studio/quality/kinematics/models.py`
- Modify: `packages/robot_data_studio/quality/kinematics/__init__.py`
- Test: `tests/quality/kinematics/test_discovery.py`

**Interfaces:**
- Produces `DiscoveryRequest(states, feature_names, max_candidates, sample_count)`.
- Produces `ProfileCandidate(profile, train_metrics, held_out_metrics, evidence)`.
- Produces `discover_profile(urdf_path, request) -> list[ProfileCandidate]`.

- [ ] **Step 1: Write failing arbitrary-name discovery test**

Provide metadata names unrelated to URDF names and a state vector containing two joints plus a 3D position and quaternion. Require discovery to select the correct terminal frame and indices from residual evidence, not string equality.

- [ ] **Step 2: Write failing rotation-representation tests**

Give equivalent trajectories encoded as XYZW, WXYZ, 6D, matrix, rotvec, and RPY. Require the correct representation to rank first on held-out frames.

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/quality/kinematics/test_discovery.py`

- [ ] **Step 4: Implement bounded candidate generation**

Enumerate terminal/operational frames and their actuated paths; derive candidate state dimensions from metadata groups, finite-value checks, dimensional width, joint-limit plausibility, temporal variation, and rotation validity. Use normalized names as a tie-breaking prior only.

- [ ] **Step 5: Implement representative sampling and held-out scoring**

Select joint-space extremes plus uniformly spaced frames, split deterministically by frame index, and rank by held-out median/p95 residual with complexity/name priors used only after geometric error.

- [ ] **Step 6: Verify GREEN and mutation cases**

Run the tests after renaming every URDF frame and metadata field in the fixture; discovery must still succeed.

- [ ] **Step 7: Commit**

```bash
git add packages/robot_data_studio/quality/kinematics tests/quality/kinematics/test_discovery.py
git commit -m "feat: discover kinematic profiles from structure"
```

### Task 4: Explainable Correction Fitting and Ambiguity Gates

**Files:**
- Create: `packages/robot_data_studio/quality/kinematics/correction.py`
- Modify: `packages/robot_data_studio/quality/kinematics/models.py`
- Modify: `packages/robot_data_studio/quality/kinematics/discovery.py`
- Test: `tests/quality/kinematics/test_correction.py`

**Interfaces:**
- Produces `CorrectionProposal` with ID, fingerprint, profile, before/after train and held-out metrics, confidence, margin, evidence, warnings, and status.
- Produces `fit_correction(states, candidates, options) -> CorrectionProposal | DiscoveryRefusal`.

- [ ] **Step 1: Write failing base-vs-TCP identifiability tests**

Use an orientation-excited trajectory where a local TCP translation rotates in world coordinates and a distinct fixed base transform does not. Require both transforms to be recovered within `1e-4` and validated on held-out frames.

- [ ] **Step 2: Write failing joint convention tests**

Generate logged poses from known joint signs and zero offsets. Require bounded beam search plus numeric refinement to recover signs and offsets without relying on joint names.

- [ ] **Step 3: Write failing refusal tests**

For static orientation, static joints, symmetric frame candidates, and negligible candidate margin, require `reason` values `insufficient_excitation` or `ambiguous`; no applicable proposal may be returned.

- [ ] **Step 4: Write failing bimanual frame test**

Build two chains with separate shoulder-relative left transforms and require independent per-chain base corrections.

- [ ] **Step 5: Run RED**

Run: `pytest -q tests/quality/kinematics/test_correction.py`

- [ ] **Step 6: Implement robust fitting**

Alternate base/TCP SE(3) updates using Pinocchio `log6`/`exp6`, Huber weights from median absolute deviation, and finite-difference or Pinocchio joint Jacobians for zero offsets. Use a bounded sign beam and stop on explicit iteration/tolerance limits.

- [ ] **Step 7: Implement confidence and evidence**

Combine held-out improvement, inlier rate, candidate margin, split stability, and excitation checks. Store human-readable evidence naming the selected frames/indices and why rejected candidates lost.

- [ ] **Step 8: Verify GREEN**

Run: `pytest -q tests/quality/kinematics/test_correction.py tests/quality/kinematics`

- [ ] **Step 9: Commit**

```bash
git add packages/robot_data_studio/quality/kinematics tests/quality/kinematics
git commit -m "feat: fit explainable kinematic corrections"
```

### Task 5: Persist, Preview, Accept, and Invalidate Proposals

**Files:**
- Modify: `packages/robot_data_studio/quality/filter_models.py`
- Modify: `packages/robot_data_studio/projects/service.py`
- Modify: `packages/robot_data_studio/quality/filter_service.py`
- Modify: `packages/robot_data_studio/quality/__init__.py`
- Modify: `apps/api/main.py`
- Test: `tests/api/test_kinematic_corrections.py`
- Modify: `tests/api/test_filter_details.py`

**Interfaces:**
- Adds project methods `discover_kinematic_corrections`, `kinematic_proposals`, and `decide_kinematic_proposal`.
- Adds `POST /api/projects/{id}/filters/kinematics/discover`.
- Adds `GET /api/projects/{id}/filters/kinematics/proposals`.
- Adds `POST /api/projects/{id}/filters/kinematics/proposals/{proposal_id}/decision`.

- [ ] **Step 1: Write failing API lifecycle test**

Upload an arbitrary-name URDF, run discovery, assert the persisted proposal contains before/after position and orientation metrics, accept it, reopen the project config, reject/reset it, and verify statuses.

- [ ] **Step 2: Write failing invalidation test**

Replace the URDF or change a relevant mapping and require the old proposal to become `stale` through a fingerprint mismatch.

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/api/test_kinematic_corrections.py`

- [ ] **Step 4: Add Pydantic request/response models and artifact store**

Store `kinematic_proposals.json` under the project artifact directory. Migrate legacy `FilterKinematicsConfig` fields into one chain at read time and keep serializing compatibility fields for existing clients during this release.

- [ ] **Step 5: Replace product FK helper with shared engine**

Delete `_pinocchio_fk_detail`; product detail must expose `position_error`, `orientation_error`, separate thresholds, selected chain, and proposal evidence from `PinocchioKinematics`.

- [ ] **Step 6: Implement routes and verify GREEN**

Run: `pytest -q tests/api/test_kinematic_corrections.py tests/api/test_filter_details.py`

- [ ] **Step 7: Commit**

```bash
git add packages/robot_data_studio/quality apps/api/main.py tests/api
git commit -m "feat: add kinematic correction proposal lifecycle"
```

### Task 6: Apply Accepted Corrections to Derived Frames and Export

**Files:**
- Create: `packages/robot_data_studio/quality/kinematics/adapter.py`
- Modify: `packages/robot_data_studio/projects/service.py`
- Modify: `packages/robot_data_studio/formats/models.py`
- Modify: `packages/robot_data_studio/formats/registry.py`
- Test: `tests/formats/test_kinematic_correction_export.py`

**Interfaces:**
- Produces `CorrectedDatasetAdapter(source, accepted_profile)` implementing `DatasetAdapter`.
- Extends `conversion_report(..., kinematic_correction=...)`.

- [ ] **Step 1: Write failing source-integrity/export test**

Hash every source file, accept a known correction, export LeRobot JSONL plus one HDF5 target, and assert corrected derived EEF fields, unchanged source hashes, and a complete correction manifest in `conversion_report.json`.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/formats/test_kinematic_correction_export.py`

- [ ] **Step 3: Implement adapter wrapper**

Copy each Pydantic `EpisodeFrame`; update only mapped derived position/orientation dimensions using the accepted profile. Retain joint values unless the export request explicitly selects canonical corrected joint conventions.

- [ ] **Step 4: Wire export and provenance**

Wrap the project reader only when an accepted, non-stale profile exists. Add profile, proposal ID, fingerprint, and before/after metrics to conversion reports.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/formats/test_kinematic_correction_export.py tests/formats`

- [ ] **Step 6: Commit**

```bash
git add packages/robot_data_studio/quality/kinematics/adapter.py packages/robot_data_studio/projects/service.py packages/robot_data_studio/formats tests/formats/test_kinematic_correction_export.py
git commit -m "feat: apply accepted kinematic corrections on export"
```

### Task 7: Data Studio Correction UI

**Files:**
- Modify: `apps/web/src/api.ts`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/src/i18n.ts`
- Modify: `apps/web/src/App.test.tsx`

**Interfaces:**
- Adds TypeScript proposal/profile types and discover/decision requests.
- Adds proposal cards with before/after position/orientation metrics, confidence, evidence, warnings, preview, accept, reject, and reset actions.

- [ ] **Step 1: Write failing UI tests**

Cover discover loading/error, ambiguous refusal, proposal evidence, metric units, accept confirmation, reset, and disabled accept for stale/ambiguous proposals.

- [ ] **Step 2: Run RED**

Run: `pnpm --filter @robot-data-studio/web test -- --run -t "kinematic correction"`

- [ ] **Step 3: Implement API client and UI**

Keep manual configuration available as an advanced fallback. Render both residual charts and explicitly label metres, radians, and degrees.

- [ ] **Step 4: Verify GREEN and build**

Run:

```bash
pnpm --filter @robot-data-studio/web test -- --run
pnpm --filter @robot-data-studio/web build
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/src
git commit -m "feat: add kinematic correction review UI"
```

### Task 8: Unify Qwen CLI with the Shared Engine

**Files:**
- Modify: `packages/robot_data_studio/quality/qwen_filters.py`
- Modify: `scripts/qwen_filter_dataset.py`
- Modify: `tests/quality/test_qwen_filters.py`
- Modify: `tests/scripts/test_qwen_filter_dataset.py`

**Interfaces:**
- Adds CLI `--discover-kinematics`, `--proposal-output`, and `--profile` options.
- Keeps legacy explicit mapping options but delegates evaluation to the shared engine.

- [ ] **Step 1: Write failing CLI parity test**

Run the CLI and product engine on the same synthetic data and assert identical position/orientation residual summaries and proposal JSON.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/quality/test_qwen_filters.py tests/scripts/test_qwen_filter_dataset.py`

- [ ] **Step 3: Remove custom FK behavior and delegate**

Keep public dataclass compatibility only where tests or callers require it. Do not retain a second URDF parser.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/quality/test_qwen_filters.py tests/scripts/test_qwen_filter_dataset.py`

- [ ] **Step 5: Commit**

```bash
git add packages/robot_data_studio/quality/qwen_filters.py scripts/qwen_filter_dataset.py tests/quality/test_qwen_filters.py tests/scripts/test_qwen_filter_dataset.py
git commit -m "refactor: unify Qwen kinematics with Pinocchio engine"
```

### Task 9: Create and Install `align-robot-kinematics` Skill

**Files:**
- Create: `skills/align-robot-kinematics/SKILL.md`
- Create: `skills/align-robot-kinematics/agents/openai.yaml`
- Create: `skills/align-robot-kinematics/scripts/analyze.py`
- Test: `tests/skills/test_align_robot_kinematics.py`
- Install copy: `~/.codex/skills/align-robot-kinematics/`

**Interfaces:**
- Wrapper accepts repository, dataset, URDF, sample budget, and output path; emits the same serialized proposal contract as the API.
- Skill triggers for mismatched URDF/dataset naming, FK/EEF inconsistency, TCP/base/joint convention calibration, or dynamic robot data import.

- [ ] **Step 1: Record the RED constraint**

Because subagents are not authorized, record forward-agent baseline as `not_run_policy`. Write a failing deterministic test showing no installed skill/wrapper can currently analyze the arbitrary-name fixture. Do not describe this as a successful behavioral forward test.

- [ ] **Step 2: Initialize with the official generator**

Run:

```bash
python /root/.codex/skills/.system/skill-creator/scripts/init_skill.py align-robot-kinematics \
  --path skills \
  --resources scripts \
  --interface display_name="Align Robot Kinematics" \
  --interface short_description="Discover and review robot kinematic corrections" \
  --interface default_prompt="Use $align-robot-kinematics to analyze this dataset and URDF before applying any correction."
```

- [ ] **Step 3: Implement the deterministic wrapper**

Import the repository package, open the dataset through `FormatRegistry`, sample requested episodes, call `discover_profile` and `fit_correction`, and write JSON. The wrapper must never contain vendor link names or correction constants.

- [ ] **Step 4: Write the minimal skill**

Instruct Codex to run the wrapper, compare candidate evidence, refuse ambiguous cases, preview proposals, and request authorization before acceptance. Keep workflow details in the body and triggering conditions only in the description.

- [ ] **Step 5: Verify package and wrapper**

Run:

```bash
python /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/align-robot-kinematics
pytest -q tests/skills/test_align_robot_kinematics.py
```

- [ ] **Step 6: Install without diverging source**

Replace the discoverable personal copy with the verified repository contents, then repeat `quick_validate.py` on `~/.codex/skills/align-robot-kinematics`.

- [ ] **Step 7: Commit**

```bash
git add skills/align-robot-kinematics tests/skills/test_align_robot_kinematics.py
git commit -m "feat: add dynamic robot kinematics skill"
```

### Task 10: Full Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/USER_GUIDE.md`

**Interfaces:**
- Documents proposal lifecycle, units, source-data guarantee, CLI/skill invocation, and Astribot S1 validation procedure.

- [ ] **Step 1: Update documentation**

Explain discovery, ambiguity, preview, acceptance, export provenance, and manual fallback without implying the future S1 artifact has already passed.

- [ ] **Step 2: Run backend verification**

```bash
pytest -q tests/quality/kinematics tests/api/test_kinematic_corrections.py tests/api/test_filter_details.py tests/formats/test_kinematic_correction_export.py tests/quality/test_qwen_filters.py tests/scripts/test_qwen_filter_dataset.py tests/skills/test_align_robot_kinematics.py
ruff check packages/robot_data_studio/quality/kinematics packages/robot_data_studio/quality/filter_models.py packages/robot_data_studio/quality/filter_service.py packages/robot_data_studio/projects/service.py scripts/qwen_filter_dataset.py tests/quality/kinematics tests/api/test_kinematic_corrections.py tests/formats/test_kinematic_correction_export.py tests/skills/test_align_robot_kinematics.py
```

- [ ] **Step 3: Run frontend verification**

```bash
pnpm --filter @robot-data-studio/web test -- --run
pnpm --filter @robot-data-studio/web build
```

- [ ] **Step 4: Run source-integrity audit**

Repeat the integration export test with pre/post SHA-256 hashes and inspect the generated correction manifest and proposal evidence manually.

- [ ] **Step 5: Audit every acceptance criterion**

Match fresh command output and artifacts against all seven criteria in the design spec. Report any missing forward-agent test separately; do not call the goal complete if required deterministic behavior or package validation is missing.

- [ ] **Step 6: Commit**

```bash
git add README.md README.zh-CN.md docs/USER_GUIDE.md
git commit -m "docs: document dynamic kinematic correction"
```
