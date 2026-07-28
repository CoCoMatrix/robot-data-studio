import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from robot_data_studio.formats.registry import FormatRegistry
from robot_data_studio.quality.kinematics import (
    CorrectedDatasetAdapter,
    JointFieldMapping,
    KinematicChainProfile,
    KinematicProfile,
    PoseFieldMapping,
)


URDF = Path("tests/quality/kinematics/fixtures/nonstandard_robot.urdf").resolve()


def test_corrected_exports_preserve_source_and_include_manifest(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "episode_000000.hdf5")
    source_hash = _sha256(source)
    registry = FormatRegistry.default()
    raw = registry.open_dataset(source, format_hint="act_hdf5")
    profile = KinematicProfile(
        urdf_path=str(URDF),
        fingerprint="source-fingerprint",
        chains=[
            KinematicChainProfile(
                id="arm",
                end_effector_frame="vector_mount",
                joints=[
                    JointFieldMapping(urdf_joint="alpha_axis", state_index=0),
                    JointFieldMapping(urdf_joint="epsilon_axis", state_index=1),
                ],
                logged_pose=PoseFieldMapping(
                    position_indices=(2, 3, 4),
                    orientation_indices=(5, 6, 7, 8),
                    orientation_representation="quaternion_xyzw",
                ),
                base_transform=[
                    [1.0, 0.0, 0.0, 0.12],
                    [0.0, 1.0, 0.0, -0.08],
                    [0.0, 0.0, 1.0, 0.21],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        ],
    )
    manifest = {
        "proposal_id": "proposal-123",
        "fingerprint": profile.fingerprint,
        "profile": profile.model_dump(mode="json"),
        "before_held_out": {"position": {"p95": 0.4}, "orientation": {"p95": 0.2}},
        "after_held_out": {"position": {"p95": 0.001}, "orientation": {"p95": 0.002}},
    }
    corrected = CorrectedDatasetAdapter(raw, profile, manifest)

    corrected_frames = corrected.read_episode_frames(0)
    assert corrected_frames[0].observation_state[2:9] != raw.read_episode_frames(0)[0].observation_state[2:9]
    assert corrected_frames[0].observation_state[:2] == raw.read_episode_frames(0)[0].observation_state[:2]
    assert raw.read_episode_frames(0)[0].observation_state[2:9] == [0.0] * 7

    lerobot = registry.export_dataset(
        adapter=corrected,
        target_format="lerobot_v3",
        episode_indexes=[0],
        output_root=tmp_path / "lerobot-export",
    )
    act = registry.export_dataset(
        adapter=corrected,
        target_format="act_hdf5",
        episode_indexes=[0],
        output_root=tmp_path / "act-export",
    )

    first_jsonl = json.loads((lerobot.output_path / "frames.jsonl").read_text().splitlines()[0])
    with h5py.File(act.output_path, "r") as file:
        first_hdf5 = np.asarray(file["observations/qpos"][0], dtype=np.float64)
    expected = np.asarray(corrected_frames[0].observation_state, dtype=np.float64)
    assert np.allclose(first_jsonl["observation.state"], expected)
    assert np.allclose(first_hdf5, expected)
    assert _sha256(source) == source_hash

    for report_path in (lerobot.report_path, act.report_path):
        report = json.loads(report_path.read_text())
        correction = report["kinematic_correction"]
        assert correction["proposal_id"] == "proposal-123"
        assert correction["fingerprint"] == "source-fingerprint"
        assert correction["profile"]["chains"][0]["end_effector_frame"] == "vector_mount"
        assert correction["before_held_out"]["position"]["p95"] == 0.4
        assert correction["after_held_out"]["orientation"]["p95"] == 0.002


def _write_source(path: Path) -> Path:
    states = np.zeros((4, 9), dtype=np.float64)
    states[:, 0] = [-0.6, -0.2, 0.2, 0.6]
    states[:, 1] = [0.3, -0.1, 0.2, -0.4]
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=np.zeros((len(states), 2), dtype=np.float64))
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=states)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
