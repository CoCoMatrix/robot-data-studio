from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robot_data_studio.curation.c2_mesh import MeshRenderResult
from robot_data_studio.curation.c2_projection import matrix_to_xyz_rot6d
from robot_data_studio.curation.c2_real_pilot import run_full_silhouette_c2_dataset
from robot_data_studio.curation.models import RawPose
from robot_data_studio.curation.profile import load_profile
from robot_data_studio.curation.sam3 import Sam3Config, Sam3Result


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Episodes:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class _Meta:
    def __init__(self, rows: list[dict], state_names: list[str]) -> None:
        self.episodes = _Episodes(rows)
        self.features = {"observation.state": {"names": state_names}}
        self.fps = 30


class _Dataset:
    def __init__(self, state_names: list[str]) -> None:
        self.num_episodes = 3
        self.meta = _Meta(
            [
                {
                    "episode_index": episode,
                    "dataset_from_index": episode * 11,
                    "dataset_to_index": (episode + 1) * 11,
                    "tasks": [f"task {episode}"],
                }
                for episode in range(3)
            ],
            state_names,
        )
        self.rows = []
        for episode in range(3):
            for local in range(11):
                self.rows.append(
                    {
                        "episode_index": np.asarray([episode]),
                        "frame_index": np.asarray([local]),
                        "timestamp": np.asarray([local / 30], dtype=np.float32),
                        "source.frame_index": np.asarray([local]),
                        "source.timestamp": np.asarray([local / 30], dtype=np.float64),
                        "observation.state": np.zeros(len(state_names), dtype=np.float32),
                        "observation.state.validity": np.ones(len(state_names), dtype=bool),
                        "observation.camera0_pose_absolute": matrix_to_xyz_rot6d(np.eye(4)),
                        "observation.camera0_pose_absolute.validity": np.asarray([True]),
                    }
                )

    def get_raw_item(self, index: int) -> dict:
        return self.rows[index]

    def __getitem__(self, index: int) -> dict:
        return {
            "observation.images.camera0": np.full((3, 480, 832), index, dtype=np.uint8)
        }


class _Provider:
    config = Sam3Config(backend="local", model_version="official-fixture", prompt="robot")

    def segment_batch(self, *, source_video_sha256, frame_indexes, images):
        output = []
        for index, image in zip(frame_indexes, images, strict=True):
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[100:200, 200:400] = True
            output.append(
                Sam3Result(
                    status="evaluated",
                    reason="local official fixture",
                    confidence=0.8,
                    cache_key=f"{index:064x}",
                    cache_hit=True,
                    mask=mask,
                )
            )
        return output


def _render(**kwargs) -> MeshRenderResult:
    assert len(kwargs["state"]) == len(kwargs["state_names"])
    assert kwargs["renderer"] == "opengl_zbuffer"
    assert kwargs.get("included_links")
    mask = np.zeros((480, 832), dtype=bool)
    mask[100:200, 200:400] = True
    return MeshRenderResult(
        mask=mask,
        depth=np.where(mask, 1.0, np.inf),
        link_index=np.full(mask.shape, -1, dtype=np.int32),
        visual_count=58,
        renderer="opengl_triangle_zbuffer_v1",
        triangle_count=942720,
        rendered_links=("base_link", "left_upper_arm_pitch_link"),
    )


def _camera(**kwargs):
    return (
        RawPose(
            matrix=np.eye(4).reshape(-1).tolist(),
            valid=True,
            reference_frame="base_link",
            child_frame="camera0",
            transform_direction="T_base_link_camera0",
        ),
        {"state": "evaluated", "fixture": True},
    )


def test_bounded_real_adapter_uses_exact_v3_time_and_full_mesh_only(tmp_path) -> None:
    profile = load_profile(
        REPO_ROOT / "profiles/curation/block_robot_c2_full_silhouette_v2.json"
    )
    state_names = profile.sources[0].geometry.kinematics.joint_names + [
        "base_height",
        "base_pitch",
    ]
    report = run_full_silhouette_c2_dataset(
        _Dataset(state_names),
        profile=profile,
        output_root=tmp_path / "evidence",
        source_episode_ids=["0", "1", "2"],
        max_episodes=3,
        source_video_sha256="a" * 64,
        sam3_provider=_Provider(),
        render_fn=_render,
        camera_pose_builder=_camera,
    )

    assert report["processed_episode_count"] == 3
    assert report["evaluated_frame_count"] == 9
    assert report["projection_scope"] == "full URDF visual silhouette"
    assert report["historical_support_disc_used"] is False
    assert report["hard_reject_allowed"] is False
    assert report["employee_review_queue"]["item_count"] == 9
    assert report["employee_review_queue"]["automatic_exclusion_allowed"] is False
    serialized = json.dumps(report).casefold()
    assert "support disc" not in serialized
    for episode in report["episodes"]:
        evidence = json.loads((tmp_path / "evidence" / episode["evidence_path"]).read_text())
        assert len(evidence["frames"]) == 3
        assert evidence["visual_scope"] == "pose_authorized_upper_body"
        assert evidence["excluded_out_of_embodiment_link_count"] == 12
        for frame in evidence["frames"]:
            assert frame["episode_time"] == pytest.approx(frame["source_timestamp"], abs=1e-6)
            assert set(frame["temporal_offset_diagnostic"]["iou_by_offset_frames"]) == {
                "-2", "-1", "0", "1", "2"
            }
            assert Path(tmp_path / "evidence" / frame["overlay_path"]).is_file()

    queue = json.loads((tmp_path / "evidence" / "review_queue.json").read_text())
    assert queue["disposition"] == "review_only"
    assert queue["automatic_exclusion_allowed"] is False
    assert [item["review_rank"] for item in queue["items"]] == list(range(1, 10))
    assert [item["iou"] for item in queue["items"]] == sorted(
        item["iou"] for item in queue["items"]
    )
    assert all(item["disposition"] == "review" for item in queue["items"])
    assert (tmp_path / "evidence" / "review_queue.csv").is_file()


def test_bounded_real_adapter_rejects_any_episode_outside_profile_allowlist(tmp_path) -> None:
    profile = load_profile(
        REPO_ROOT / "profiles/curation/block_robot_c2_full_silhouette_v2.json"
    )
    names = profile.sources[0].geometry.kinematics.joint_names + ["base_height", "base_pitch"]
    with pytest.raises(PermissionError, match="allowlist"):
        run_full_silhouette_c2_dataset(
            _Dataset(names),
            profile=profile,
            output_root=tmp_path,
            source_episode_ids=["0", "1", "3"],
            max_episodes=3,
            source_video_sha256="a" * 64,
            sam3_provider=_Provider(),
            render_fn=_render,
            camera_pose_builder=_camera,
        )
