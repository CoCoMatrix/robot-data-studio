from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_data_studio.curation.checks import check_action_artifacts
from robot_data_studio.curation.cli import main
from robot_data_studio.curation.pilot import run_pilot
from robot_data_studio.curation.profile import CurationProfile, is_source_allowed
from robot_data_studio.curation.sources import (
    XJGT_DISPLAY_MATRIX_ROTATE_90,
    display_oriented_size,
    open_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def profile_for(root: Path, *, adapter: str, embodiment: str) -> CurationProfile:
    return CurationProfile.model_validate(
        {
            "name": f"synthetic-{adapter}",
            "selection": {
                "train_allowlist": [str(root)],
                "deny_patterns": ["*val*", "*validation*", "*eval*"],
            },
            "task": {"text": "move object to target", "events": ["start", "end"]},
            "modalities": [
                {"name": "state"},
                {"name": "action", "required": embodiment == "robot"},
                {"name": "video", "required": False},
            ],
            "progress": {"mode": "fixed_time", "target_points": 4},
            "sources": [
                {
                    "source_id": "source-a",
                    "embodiment": embodiment,
                    "adapter": adapter,
                    "fps": 30.0,
                    "state_fields": ["transforms/leftHand", "transforms/rightHand"],
                    "state_key": "observation.state",
                    "action_key": None if embodiment == "human" else "action",
                    "validity_fields": ["confidences/leftHand", "confidences/rightHand"],
                    "video_key": "observation.images.top" if embodiment == "robot" else None,
                    "human_extension": {
                        "confidence_threshold": 0.5,
                        "short_hole_max_frames": 1,
                        "calibration_image_size": [1280, 720],
                        "stored_video_size": [720, 1280],
                        "video_binding_verified": False,
                    },
                }
            ],
            "vlm": {"backend": "disabled"},
            "sidecar_root": ".rds-artifacts/test",
        }
    )


def write_xjgt_fixture(root: Path) -> None:
    (root / "hdf5").mkdir(parents=True)
    (root / "video").mkdir()
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    matrices[:, 0, 3] = [0.0, 0.1, 0.2]
    with h5py.File(root / "hdf5" / "100_hdf5.hdf5", "w") as file:
        file.create_dataset("camera/intrinsic", data=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
        file.create_dataset("transforms/camera", data=np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0))
        file.create_dataset("transforms/leftHand", data=matrices)
        file.create_dataset("transforms/rightHand", data=matrices)
        file.create_dataset("transforms/leftForearm", data=matrices)
        file.create_dataset("transforms/rightForearm", data=matrices)
        file.create_dataset("confidences/leftHand", data=[1.0, 0.0, 1.0])
        file.create_dataset("confidences/rightHand", data=[1.0, 1.0, 1.0])
        file.create_dataset("confidences/leftForearm", data=[1.0, 1.0, 1.0])
        file.create_dataset("confidences/rightForearm", data=[1.0, 1.0, 1.0])
        file.create_dataset("label/quality_hand", data=np.zeros((3, 2), dtype=np.float32))
        file.create_dataset(
            "label/text_label",
            data=json.dumps(
                {
                    "data_en": {
                        "task": "move blocks",
                        "description": "Move blocks to a target.",
                        "events": [
                            {
                                "start_time": "00:00.00",
                                "end_time": "00:00.10",
                                "caption": "Move one block.",
                            }
                        ],
                    }
                }
            ),
            dtype=h5py.string_dtype(),
        )
    (root / "video" / "100_video.mp4").touch()


def write_lerobot_v3_fixture(root: Path) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic",
        "total_episodes": 2,
        "total_frames": 4,
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["j0", "j1"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["j0", "j1"],
            },
            "observation.images.top": {"dtype": "video"},
        },
        "splits": {"train": "0:2"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episode_table = pa.table(
        {
            "episode_index": [0, 1],
            "tasks": [["move block"], ["place block"]],
            "length": [2, 2],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "videos/observation.images.top/chunk_index": [0, 0],
            "videos/observation.images.top/file_index": [0, 0],
            "videos/observation.images.top/from_timestamp": [0.0, 3.0],
            "videos/observation.images.top/to_timestamp": [0.1, 3.1],
            "expand_task": ["Move the first block.", "Place the selected block on target."],
        }
    )
    pq.write_table(episode_table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data = pa.table(
        {
            "observation.state": [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]],
            "action": [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]],
            "timestamp": [0.0, 1 / 30, 0.0, 1 / 30],
            "frame_index": [0, 1, 0, 1],
            "episode_index": [0, 0, 1, 1],
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet", row_group_size=2)


def test_xjgt_adapter_streams_explicit_episode_and_preserves_confidence_validity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "xjgt_train"
    write_xjgt_fixture(root)
    profile = profile_for(root, adapter="xjgt_hdf5", embodiment="human")

    adapter = open_source(profile, root, ["100"])
    summaries = adapter.summaries()
    records = list(adapter.iter_episodes())

    assert [item["episode_id"] for item in summaries] == ["100"]
    assert len(records) == 1
    assert [frame.valid for frame in records[0].frames] == [True, False, True]
    assert records[0].frames[1].state == pytest.approx([0.1, 0.0, 0.0] * 2)
    assert all(frame.action == [] for frame in records[0].frames)
    assert records[0].lineage["invalid_frame_count"] == 1
    assert records[0].lineage["quality_hand_value_counts"] == {"0.0": 6}
    assert records[0].frames[0].metadata["quality_hand"] == [0.0, 0.0]
    assert records[0].lineage["annotation_event_count"] == 1
    extension = {metric.metric_id: metric for metric in records[0].extension_metrics}
    assert extension["H1_tracking_confidence"].status == "evaluated"
    assert extension["H5_quality_hand_crosscheck"].status == "evaluated"
    assert extension["H5_quality_hand_crosscheck"].weight == 0.0
    assert extension["H6_camera_projection"].status == "evaluated"
    assert extension["H6b_stored_video_projection"].status == "not_evaluated"
    assert records[0].lineage["human_extension"]["no_interpolation"] is True
    dual = records[0].frames[0].dual_action
    assert dual is not None
    assert dual.observation_joint_position.valid is False
    assert dual.observation_joint_position.values is None
    assert dual.observation_eef_pose["left_hand"].transform_direction == "T_world_left_hand"
    assert dual.camera_pose["camera0"].transform_direction == "T_world_camera0"
    assert records[0].frames[1].dual_action.observation_eef_pose["left_hand"].valid is False
    assert records[0].frames[1].dual_action.observation_eef_pose["left_hand"].matrix is None
    assert check_action_artifacts(records[0].frames, profile.sources[0], profile.checks) == []


def test_xjgt_encoded_and_display_dimensions_are_not_swapped() -> None:
    assert display_oriented_size(
        (720, 1280), XJGT_DISPLAY_MATRIX_ROTATE_90
    ) == (1280, 720)
    assert display_oriented_size((720, 1280), None) == (720, 1280)
    with pytest.raises(ValueError, match="display matrix"):
        display_oriented_size((720, 1280), (1, 2, 3, 4, 5, 6, 7, 8, 9))


def test_lerobot_v3_adapter_reads_only_selected_episode_row_group(tmp_path: Path) -> None:
    root = tmp_path / "robot_train"
    write_lerobot_v3_fixture(root)
    profile = profile_for(root, adapter="lerobot_v3", embodiment="robot")

    adapter = open_source(profile, root, ["1"])
    record = list(adapter.iter_episodes())[0]

    assert record.episode_id == "1"
    assert [frame.frame_index for frame in record.frames] == [0, 1]
    assert record.frames[0].timestamp == 0.0
    assert record.from_timestamp == 3.0
    assert record.frames[0].video_uri.endswith("file-000.mp4")
    assert record.lineage["parquet_row_groups"] == [1]
    assert record.lineage["invalid_frame_count"] == 0
    assert record.lineage["validity_basis"] == "finite_state_action_timestamp"
    assert record.raw_task_text == "Place the selected block on target."
    assert record.lineage["task_text_source"] == "meta.episodes.expand_task"
    dual = record.frames[0].dual_action
    assert dual is not None
    assert dual.observation_joint_position.names == ["j0", "j1"]
    assert dual.observation_joint_position.values == [1.0, 1.0]
    assert dual.action_joint_target.names == ["j0", "j1"]
    assert dual.observation_eef_pose["left_eef"].valid is False
    assert "authority" in dual.observation_eef_pose["left_eef"].missing_reason


def test_real_source_gate_is_case_insensitive_and_requires_explicit_episodes(tmp_path: Path) -> None:
    root = tmp_path / "human_train"
    write_xjgt_fixture(root)
    profile = profile_for(root, adapter="xjgt_hdf5", embodiment="human")

    assert not is_source_allowed(profile, str(root / "EVAL_episode.hdf5"), split="train")
    with pytest.raises(ValueError, match="explicit episode"):
        open_source(profile, root, [])


def test_lerobot_metadata_paths_cannot_escape_authorized_root(tmp_path: Path) -> None:
    root = tmp_path / "robot_train"
    write_lerobot_v3_fixture(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["data_path"] = "../outside-{file_index:03d}.parquet"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    profile = profile_for(root, adapter="lerobot_v3", embodiment="robot")

    adapter = open_source(profile, root, ["0"])
    with pytest.raises(PermissionError, match="authorized root"):
        next(adapter.iter_episodes())


def test_missing_geometry_evidence_is_structured_not_evaluated(tmp_path: Path) -> None:
    root = tmp_path / "xjgt_train"
    write_xjgt_fixture(root)
    profile = profile_for(root, adapter="xjgt_hdf5", embodiment="human")
    record = next(open_source(profile, root, ["100"]).iter_episodes())

    result = run_pilot(
        profile,
        iter([record]),
        source_path=str(root),
        sidecar_root=tmp_path / "sidecars",
        max_episodes=1,
    ).episodes[0]

    assert result.kinematics.state == "not_evaluated"
    assert "urdf" in result.kinematics.reason
    assert result.reprojection.state == "not_evaluated"
    assert "intrinsics" in result.reprojection.reason
    assert [item.label for item in result.evidence] == ["start", "end"]


def test_cli_dry_run_and_pilot_require_bounded_explicit_real_episode(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "xjgt_train"
    write_xjgt_fixture(root)
    profile = profile_for(root, adapter="xjgt_hdf5", embodiment="human")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    sidecars = tmp_path / "sidecars"

    assert main(
        [
            "dry-run",
            "--profile",
            str(profile_path),
            "--input",
            str(root),
            "--episode",
            "100",
            "--max-episodes",
            "1",
        ]
    ) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["episodes"] == 1
    assert dry_run["selection"][0]["episode_id"] == "100"
    assert not sidecars.exists()

    assert main(
        [
            "pilot",
            "--profile",
            str(profile_path),
            "--input",
            str(root),
            "--episode",
            "100",
            "--max-episodes",
            "1",
            "--sidecar-root",
            str(sidecars),
        ]
    ) == 0
    pilot = json.loads(capsys.readouterr().out)
    assert pilot["processed"] == 1
    assert (sidecars / "manifest.json").is_file()


def test_real_source_profiles_are_exactly_scoped_and_vlm_disabled() -> None:
    human = CurationProfile.model_validate_json(
        (REPO_ROOT / "profiles" / "curation" / "xjgt_human.json").read_text(encoding="utf-8")
    )
    robot = CurationProfile.model_validate_json(
        (REPO_ROOT / "profiles" / "curation" / "block_robot.json").read_text(encoding="utf-8")
    )

    assert human.sources[0].adapter == "xjgt_hdf5"
    assert robot.sources[0].adapter == "lerobot_v3"
    assert human.vlm.backend == robot.vlm.backend == "disabled"
    assert human.selection.train_allowlist == ["/mnt/cpfs/egodata/XJGT_20260624"]
    assert robot.selection.train_allowlist == ["/mnt/cpfs/block_dataset/task_*/batch_*"]
    assert human.sources[0].human_extension.encoded_video_size == (720, 1280)
    assert human.sources[0].human_extension.stored_video_size == (1280, 720)
    assert human.sources[0].human_extension.display_matrix == XJGT_DISPLAY_MATRIX_ROTATE_90
    assert human.sources[0].human_extension.video_binding_verified is True
    assert robot.sources[0].geometry.reprojection.binding_verified is True
    for profile in (human, robot):
        assert {"*val*", "*validation*", "*test*", "*eval*"} <= set(
            profile.selection.deny_patterns
        )
