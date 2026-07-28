from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_data_studio.curation.dual_action_bridge import (
    DualActionPipeline,
    prefilter_episode,
)
from robot_data_studio.curation.models import (
    EpisodeRecord,
    FrameSample,
    RawDualActionFrame,
    RawNumericFeature,
    RawPose,
)
from robot_data_studio.curation.profile import CurationProfile


DATA_PIPELINE_SRC = Path("/mnt/workspace/spy/data_pipeline/src")


def matrix(x: float) -> list[float]:
    return [
        1.0, 0.0, 0.0, x,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def pose(name: str, x: float, *, valid: bool = True) -> RawPose:
    return RawPose(
        matrix=matrix(x) if valid else None,
        valid=valid,
        reference_frame="world",
        child_frame=name,
        transform_direction=f"T_world_{name}",
        missing_reason=None if valid else "fixture invalid",
    )


def episode(root: Path, episode_id: str = "ep-1", *, static: bool = False) -> EpisodeRecord:
    video = root / "train_video.mp4"
    video.touch()
    (root / "train.parquet").touch()
    frames = []
    for index in range(4):
        x = 0.0 if static else index * 0.1
        frames.append(
            FrameSample(
                episode_id=episode_id,
                frame_index=index,
                timestamp=index / 30,
                valid=True,
                state=[x, x + 1],
                action=[x + 0.2, x + 1.2],
                video_uri=str(video),
                dual_action=RawDualActionFrame(
                    observation_joint_position=RawNumericFeature(
                        values=[x, x + 1],
                        valid=True,
                        valid_mask=[True, True],
                        names=["j0", "j1"],
                        units=["rad", "rad"],
                    ),
                    observation_eef_pose={"left": pose("left", x)},
                    action_joint_target=RawNumericFeature(
                        values=[x + 0.2, x + 1.2],
                        valid=True,
                        valid_mask=[True, True],
                        names=["j0", "j1"],
                        units=["rad", "rad"],
                        action_type="absolute_joint_position",
                    ),
                    action_eef_pose_target={"left": pose("left", x + 0.2)},
                    camera_pose={"camera0": pose("camera0", 0.0)},
                ),
            )
        )
    return EpisodeRecord(
        episode_id=episode_id,
        source_id="robot-source",
        task_id="move",
        split="train",
        from_timestamp=7.0,
        frames=frames,
        raw_task_text="move the block",
        lineage={"data_file": str(root / "train.parquet"), "format": "synthetic"},
    )


def profile(root: Path) -> CurationProfile:
    return CurationProfile.model_validate(
        {
            "name": "dual-action-fixture",
            "selection": {
                "train_allowlist": [str(root)],
                "deny_patterns": ["*val*", "*validation*", "*eval*"],
            },
            "task": {"text": "move the block", "events": ["start", "end"]},
            "modalities": [{"name": "state"}, {"name": "action"}],
            "progress": {"target_points": 4},
            "sources": [
                {
                    "source_id": "robot-source",
                    "embodiment": "robot",
                    "fps": 30,
                    "action_key": "action",
                }
            ],
            "vlm": {"backend": "disabled"},
            "sidecar_root": str(root / "sidecars"),
        }
    )


def test_prefilter_keeps_window_invalidity_but_rejects_static_or_denied_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authorized_train"
    root.mkdir()
    record = episode(root)
    record.frames[2].valid = False

    result = prefilter_episode(record, source_root=root)
    assert result.eligible is True
    assert result.valid_frame_count == 3
    assert result.disposition["window_mask"] == [{"start_frame": 2, "end_frame": 2, "reason": "source_invalid"}]

    static_result = prefilter_episode(episode(root, "static", static=True), source_root=root)
    assert static_result.eligible is False
    assert "all_static" in static_result.reason_codes

    denied = episode(root, "denied")
    denied.lineage["data_file"] = str(root / "Validation" / "episode.parquet")
    denied_result = prefilter_episode(denied, source_root=root)
    assert denied_result.eligible is False
    assert "denied_source_path" in denied_result.reason_codes


def test_pipeline_promotes_only_eligible_reloads_deep_cleans_and_resumes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_train"
    source.mkdir()
    records = [episode(source, "good"), episode(source, "static", static=True)]
    output = tmp_path / "pilot"
    runner = DualActionPipeline(
        profile(source),
        source_root=source,
        output_root=output,
        data_pipeline_src=DATA_PIPELINE_SRC,
        max_episodes=2,
    )

    report = runner.run(iter(records))
    resumed = runner.run(iter(records))

    assert report["raw_episode_count"] == 2
    assert report["eligible_episode_count"] == 1
    assert report["promoted_episode_count"] == 1
    assert report["reloaded_episode_count"] == 1
    assert report["final_accepted_count"] == 1
    assert resumed["promotion_resumed_count"] == 1
    assert not list(output.rglob("*.mp4"))
    raw = json.loads((output / "raw" / "source_manifest.json").read_text())
    eligible = json.loads((output / "raw" / "eligible_manifest.json").read_text())
    final = json.loads((output / "lerobot_v3" / "final_accepted_manifest.json").read_text())
    assert [row["episode_id"] for row in raw["episodes"]] == ["good", "static"]
    assert [row["episode_id"] for row in eligible["episodes"]] == ["good"]
    assert [row["episode_id"] for row in final["episodes"]] == ["good"]
    canonical = output / "lerobot_v3" / "episodes" / "good" / "data.jsonl"
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    assert rows[0]["format"] == "lerobot_v3_dual_action_reference_v0.2"
    assert rows[1]["observation"]["media_references"]["camera0"]["uri"].endswith(
        "train_video.mp4"
    )
    assert rows[1]["observation"]["media_references"]["camera0"]["episode_time"] == 0.0
    assert rows[1]["observation"]["media_references"]["camera0"]["file_time"] == 7.0


def test_pipeline_can_publish_and_reload_official_lerobot_v3(tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = tmp_path / "source_train"
    source.mkdir()
    record = episode(source, "official")
    output = tmp_path / "pilot"
    report = DualActionPipeline(
        profile(source),
        source_root=source,
        output_root=output,
        data_pipeline_src=DATA_PIPELINE_SRC,
        max_episodes=1,
        publish_official=True,
        official_repo_id="local/rds_official_test",
        official_image_shape=(3, 32, 48),
        official_image_provider=lambda frame: np.full(
            (32, 48, 3), frame.frame_index * 30, dtype=np.uint8
        ),
    ).run(iter([record]))

    root = output / "official_lerobot_v3"
    loaded = LeRobotDataset(
        "local/rds_official_test", root=root, download_videos=False, video_backend="pyav"
    )
    assert len(loaded) == 4
    assert report["official_v3"]["loader_reload"] == "verified"
    assert report["official_v3"]["episode_count"] == 1
    assert (root / "meta" / "curation_contract.json").is_file()
    assert list(root.rglob("*.parquet"))
    assert list(root.rglob("*.mp4"))


def test_pipeline_requires_positive_bound_and_explicit_data_pipeline_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_train"
    source.mkdir()
    with pytest.raises(ValueError, match="positive"):
        DualActionPipeline(
            profile(source),
            source_root=source,
            output_root=tmp_path / "pilot",
            data_pipeline_src=DATA_PIPELINE_SRC,
            max_episodes=0,
        )
    with pytest.raises(FileNotFoundError, match="data_pipeline"):
        DualActionPipeline(
            profile(source),
            source_root=source,
            output_root=tmp_path / "pilot",
            data_pipeline_src=tmp_path / "missing",
            max_episodes=1,
        )


def test_human_reload_uses_absolute_eef_pose_for_legacy_motion_view(
    tmp_path: Path,
) -> None:
    source = tmp_path / "human_train"
    source.mkdir()
    record = episode(source, "human")
    record.source_id = "human-source"
    for frame in record.frames:
        frame.action = []
        assert frame.dual_action is not None
        frame.dual_action.observation_joint_position = RawNumericFeature.missing(
            "Human joints unavailable"
        )
        frame.dual_action.action_joint_target = RawNumericFeature.missing(
            "Human joint action not applicable"
        )
    payload = profile(source).model_dump(mode="json")
    payload["sources"][0].update(
        {
            "source_id": "human-source",
            "embodiment": "human",
            "action_key": None,
        }
    )
    runner = DualActionPipeline(
        CurationProfile.model_validate(payload),
        source_root=source,
        output_root=tmp_path / "pilot",
        data_pipeline_src=DATA_PIPELINE_SRC,
        max_episodes=1,
    )

    runner.run(iter([record]))

    deep = json.loads(
        (tmp_path / "pilot" / "lerobot_v3" / "deep_clean" / "manifest.json").read_text()
    )
    assert deep["coverage"]["by_motion"] == {"low": 1}
    assert "idle_ratio" not in deep["episodes"][0]["reason_codes"]


def test_duplicate_detection_ignores_episode_identifier_and_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source_train"
    source.mkdir()
    first = episode(source, "first")
    second = first.model_copy(deep=True)
    second.episode_id = "second"
    second.lineage = {**second.lineage, "alias": True}
    for frame in second.frames:
        frame.episode_id = "second"
    output = tmp_path / "pilot"

    report = DualActionPipeline(
        profile(source),
        source_root=source,
        output_root=output,
        data_pipeline_src=DATA_PIPELINE_SRC,
        max_episodes=2,
    ).run(iter([first, second]))

    assert report["eligible_episode_count"] == 1
    raw = json.loads((output / "raw" / "source_manifest.json").read_text())
    assert raw["episodes"][1]["reason_codes"] == ["duplicate_content"]
