from __future__ import annotations

import json
import os
from pathlib import Path

from robot_data_studio.curation.cli import _parser, main
from robot_data_studio.curation.profile import CurationProfile, QualityContractProfile


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "profiles" / "curation"
SKILL_ROOT = REPO_ROOT / ".codex" / "skills" / "curate-embodied-data"


def write_synthetic_episode(path: Path) -> None:
    payload = {
        "episode_id": "synthetic-1",
        "source_id": "robot-example",
        "task_id": "move-object",
        "split": "train",
        "from_timestamp": 7.0,
        "expected_duration": 0.3,
        "event_frames": {"start": 0, "contact": 2, "end": 3},
        "lineage": {"fixture": True},
        "frames": [
            {
                "episode_id": "synthetic-1",
                "frame_index": index,
                "timestamp": index / 10,
                "decoded_pts": 7.0 + index / 10,
                "valid": True,
                "state": [float(index)],
                "action": [float(index) / 10],
            }
            for index in range(4)
        ],
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_generic_and_grasp_profiles_validate_without_task_hardcoding() -> None:
    generic = CurationProfile.model_validate_json(
        (PROFILE_ROOT / "generic.json").read_text(encoding="utf-8")
    )
    grasp = CurationProfile.model_validate_json(
        (PROFILE_ROOT / "grasp.json").read_text(encoding="utf-8")
    )
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "packages" / "robot_data_studio" / "curation").glob("*.py")
    ).lower()

    assert generic.progress.mode == "fixed_time"
    assert grasp.progress.mode == "event_or_motion"
    assert 'task_id == "grasp"' not in package_source
    assert "task_id == 'grasp'" not in package_source
    assert {source.embodiment for source in generic.sources} == {"human", "robot"}


def test_checked_in_profile_schema_matches_model() -> None:
    checked_in = json.loads((PROFILE_ROOT / "profile.schema.json").read_text(encoding="utf-8"))

    assert checked_in == CurationProfile.model_json_schema()


def test_current_v2_profile_and_contract_schema_are_explicit() -> None:
    current = CurationProfile.model_validate_json(
        (PROFILE_ROOT / "current_v2.json").read_text(encoding="utf-8")
    )
    contract_schema = json.loads(
        (PROFILE_ROOT / "quality-contract-v2.schema.json").read_text(encoding="utf-8")
    )
    assert current.profile_version == "2"
    assert current.quality_contract is not None
    assert set(current.quality_contract.qwen) == {"S1", "S2", "S3", "C1", "C2", "C3"}
    assert contract_schema == QualityContractProfile.model_json_schema()


def test_block_robot_full_silhouette_v2_profile_is_bounded_and_authoritative() -> None:
    profile = CurationProfile.model_validate_json(
        (PROFILE_ROOT / "block_robot_c2_full_silhouette_v2.json").read_text(
            encoding="utf-8"
        )
    )
    source = profile.sources[0]
    c2 = source.geometry.c2_full_silhouette
    reprojection = source.geometry.reprojection

    assert profile.profile_version == "2"
    assert profile.selection.real_episode_allowlist == ["0", "1", "2"]
    assert c2.enabled is True
    assert c2.keyframes_per_episode == 3
    assert c2.temporal_offsets_frames == [-2, -1, 0, 1, 2]
    assert c2.projection_scope == "full_urdf_visual_silhouette"
    assert c2.calibrated is False
    assert c2.hard_reject_allowed is False
    assert reprojection.stored_video_undistorted is True
    assert reprojection.projection_lens_contract == "train_only_stored_k_override_v2"
    assert (
        reprojection.projection_override_evidence_sha256
        == "01a155630dfb1b6234702f81698826627e96d118a123a301956b849e83d406d8"
    )
    assert len(reprojection.distortion_coefficients or ()) == 14
    assert reprojection.display_transform == (
        (0.65, 0.0, 0.0),
        (0.0, 0.65, -20.0),
        (0.0, 0.0, 1.0),
    )
    assert source.geometry.kinematics.explicit_unrecorded_joint_positions
    assert source.geometry.kinematics.visual_scope == "pose_authorized_upper_body"
    assert len(source.geometry.kinematics.excluded_visual_joint_roots) == 12


def test_skill_routes_vlm_prompt_drafting_to_versioned_train_only_policy() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    policy_path = SKILL_ROOT / "references" / "prompt-policy.md"
    policy = policy_path.read_text(encoding="utf-8")

    assert "prompt-policy.md" in skill
    for required in (
        "invariant core JSON schema",
        "source/task-specific prompt profile",
        "Human",
        "Robot",
        "train allowlist",
        "version",
        "SHA-256",
        "coverage",
        "cost",
        "negative examples",
        "validation/test/eval",
    ):
        assert required.casefold() in policy.casefold()


def test_c2_real_pilot_cli_is_separate_from_qwen_live_quality() -> None:
    args = _parser().parse_args(
        [
            "c2-real-pilot",
            "--profile", "profiles/curation/block_robot_c2_full_silhouette_v2.json",
            "--source-root", "/mnt/cpfs/block_dataset",
            "--official-root", ".rds-artifacts/official-v3-pilot/robot/official_lerobot_v3",
            "--output", ".rds-artifacts/c2-real-v2",
            "--repo-id", "local/block_robot_train_pilot",
            "--episode", "0", "--episode", "1", "--episode", "2",
            "--max-episodes", "3",
            "--sam3-model-dir", "/cache/facebook--sam3/revision",
            "--sam3-cache-root", "/cache/c2",
        ]
    )
    assert args.command == "c2-real-pilot"
    assert args.episode == ["0", "1", "2"]
    assert not hasattr(args, "c1_model")


def test_validate_dry_run_and_pilot_cli_use_bounded_synthetic_stream(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "train" / "synthetic.jsonl"
    profile_path = tmp_path / "profile.json"
    sidecar_root = tmp_path / "sidecars"
    write_synthetic_episode(input_path)
    profile = CurationProfile.model_validate_json(
        (PROFILE_ROOT / "generic.json").read_text(encoding="utf-8")
    )
    profile = profile.model_copy(
        update={
            "selection": profile.selection.model_copy(
                update={
                "train_allowlist": [str(input_path)],
                    "deny_patterns": ["*validation*", "*eval*"],
                }
            ),
            "sidecar_root": str(sidecar_root),
        }
    )
    profile_path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")

    assert main(["validate", "--profile", str(profile_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True

    assert main(
        [
            "dry-run",
            "--profile",
            str(profile_path),
            "--input",
            str(input_path),
            "--max-episodes",
            "1",
        ]
    ) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["episodes"] == 1
    assert not sidecar_root.exists()

    assert main(
        [
            "pilot",
            "--profile",
            str(profile_path),
            "--input",
            str(input_path),
            "--max-episodes",
            "1",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["processed"] == 1
    assert (sidecar_root / "manifest.json").is_file()


def test_repo_skill_has_valid_references_and_deterministic_helpers() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    openai = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "name: curate-embodied-data" in skill
    assert "Inspect, clean, pose, review" in skill
    assert "Prepare poses and validity" in skill
    assert "official Hugging Face LeRobot V3" in skill
    assert "references/workflow.md" in skill
    assert "references/source-onboarding.md" in skill
    assert "references/pose-contract.md" in skill
    assert "references/quality-checks.md" in skill
    assert "references/prompt-policy.md" in skill
    assert "references/review-queue.md" in skill
    assert "references/v3-handoff.md" in skill
    assert "scripts/validate_profile.py" in skill
    assert "scripts/make_synthetic_train_fixture.py" in skill
    assert "/mnt/workspace/spy/data_pipeline" in skill
    assert "current_v2.json" in skill
    assert "automatic_exclusion_allowed=false" in skill
    assert "$curate-embodied-data" in openai
    assert "raw" in openai.lower()
    assert "LeRobot" in openai
    assert "validation/test/eval" in agents
    assert "episode_time=row.timestamp" in agents
    for script in (SKILL_ROOT / "scripts").glob("*.py"):
        assert os.access(script, os.X_OK)

    downloader = (SKILL_ROOT / "scripts" / "resume_sam3_checkpoint.py").read_text()
    assert "Git-LFS pointer" in downloader
    assert "Xet CAS" in downloader
    assert "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a" in downloader


def test_live_quality_cli_is_explicit_bounded_and_has_no_secret_argument() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "live-quality",
            "--profile",
            "profile.json",
            "--source-root",
            "/train/source",
            "--official-root",
            "/pilot/official",
            "--prior-root",
            "/pilot/deep-clean",
            "--output",
            "/pilot/live",
            "--repo-id",
            "local/pilot",
            "--episode",
            "episode-1",
            "--max-episodes",
            "1",
            "--dry-run",
        ]
    )
    assert args.command == "live-quality"
    assert args.episode == ["episode-1"]
    assert args.max_episodes == 1
    assert not hasattr(args, "api_key")
