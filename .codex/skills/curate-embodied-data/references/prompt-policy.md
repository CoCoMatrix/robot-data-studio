# VLM prompt policy

Keep one invariant core JSON schema and place dataset knowledge in a versioned source/task-specific prompt profile. Never tune prompts on validation, test, or eval data.

## Invariant core JSON schema

Require strict JSON with no prose outside the object:

```json
{
  "schema_version": "embodied-c1-evidence-v1",
  "source_id": "string",
  "task_profile_id": "string",
  "instruction": {"verb": "string", "object": "string", "target": "string|null", "modifiers": ["string"], "raw_text": "string"},
  "phases": [{"name": "start|approach|contact_or_grasp|manipulate_or_carry|release|end", "state": "consistent|review|conflict|not_observed", "frame_indexes": [0], "timestamps_seconds": [0.0], "reason": "string"}],
  "objects": [{"name": "string", "state": "consistent|review|conflict|not_observed", "evidence_frames": [0]}],
  "success": {"state": "consistent|review|conflict|not_observed", "confidence": 0.0, "reason": "string"},
  "ambiguities": ["string"],
  "overall_state": "consistent|review|conflict",
  "confidence": 0.0
}
```

Validate enums, required fields, finite timestamps, confidence in `[0,1]`, evidence-frame membership, and phase order. Reject malformed output; do not repair it into a pass.

## Source/task-specific prompt profile

Store canonical JSON containing `profile_id`, `version`, `source_id`, `task_id`, and embodiment-specific instructions. Include:

- source-specific task vocabulary: allowed verbs, objects, targets, modifiers, aliases, and forbidden semantic substitutions;
- camera/view assumptions: first/third person, encoded/display orientation, occlusion expectations, and sampled-image binding;
- expected actors and objects, including which hand/robot arm may be active;
- temporal sampling policy: bounded start/contact/end or uniformly spaced keyframes, exact frame indexes, and timestamps;
- ambiguity rules and negative examples for incomplete, contradictory, idle, occluded, or wrong-object executions;
- the strict structured output contract above.

Write distinct Human and Robot prompt text. Human prompts reason about per-hand visibility, measured hand motion, bimanual roles, and task-conditioned contact without inventing Robot joints. Robot prompts reason about named arms, recorded action/state evidence, Camera0, and commanded phases without treating future observation state as a label.

## Train-only calibration

1. Inspect only a small explicit train allowlist after held-out deny rules; never discover extra episodes.
2. Draft the profile, canonicalize it, assign a version, and record its SHA-256 before inference.
3. Run a bounded train-only calibration pilot only when external/local VLM execution is separately authorized. Cache by video content hash, sampled frames, model version, invariant schema version, and prompt-profile hash.
4. Record evaluated coverage, API calls, prompt/output tokens, estimated and actual cost, cache hits, schema failures, provider failures, conflicts, and review cases.
5. Revise by creating a new version/hash. Never inspect or use validation/test/eval outcomes to select vocabulary, examples, sampling, thresholds, or prompt wording.

Missing evidence remains `review` or `not_observed`; mock output and dry-run cost estimates never count as real coverage.
