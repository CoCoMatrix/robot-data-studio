from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from robot_data_studio.curation.gr3_fk import (
    build_gr3_camera_pose,
    build_gr3_pose_features,
)
from robot_data_studio.curation.profile import KinematicsProfile, ReprojectionProfile


URDF = """<?xml version="1.0"?>
<robot name="fixture">
  <link name="base_link"/>
  <link name="head_pitch_link"/>
  <link name="left_link"/>
  <link name="left_eef"/>
  <link name="right_link"/>
  <link name="right_eef"/>
  <joint name="head_pitch_joint" type="revolute">
    <parent link="base_link"/><child link="head_pitch_link"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="left_joint" type="revolute">
    <parent link="base_link"/><child link="left_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="left_tip" type="fixed">
    <parent link="left_link"/><child link="left_eef"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="right_joint" type="revolute">
    <parent link="base_link"/><child link="right_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="right_tip" type="fixed">
    <parent link="right_link"/><child link="right_eef"/>
    <origin xyz="-1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def config(path: Path) -> KinematicsProfile:
    return KinematicsProfile(
        urdf_path=str(path),
        urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        root_link="base_link",
        end_effector_links={"left": "left_eef", "right": "right_eef"},
        joint_names=["left_joint", "right_joint"],
        joint_state_indices=[0, 1],
        action_target_policy="anchor_state_plus_named_action",
        action_target_joint_names=["left_joint", "right_joint"],
        authority_evidence="synthetic fixture authority",
    )


def test_gr3_fk_uses_exact_names_and_builds_absolute_base_poses(tmp_path: Path) -> None:
    urdf = tmp_path / "fixture.urdf"
    urdf.write_text(URDF)

    observation, target, provenance = build_gr3_pose_features(
        state=[0.0, 0.0],
        action=[1.5707963267948966, 0.0],
        state_names=["left_joint", "right_joint"],
        action_names=["left_joint", "right_joint"],
        config=config(urdf),
    )

    assert observation["left_eef"].matrix[3] == pytest.approx(1.0)
    assert target["left_eef"].matrix[3] == pytest.approx(0.0, abs=1e-8)
    assert target["left_eef"].matrix[7] == pytest.approx(1.0)
    assert provenance["urdf_sha256"] == hashlib.sha256(urdf.read_bytes()).hexdigest()
    assert provenance["deployment_tcp_claimed"] is False


def test_gr3_fk_fails_closed_on_name_or_hash_mismatch(tmp_path: Path) -> None:
    urdf = tmp_path / "fixture.urdf"
    urdf.write_text(URDF)
    bad = config(urdf)
    bad.urdf_sha256 = "0" * 64
    observation, target, provenance = build_gr3_pose_features(
        state=[0.0, 0.0],
        action=[0.0, 0.0],
        state_names=["left_joint", "right_joint"],
        action_names=["left_joint", "right_joint"],
        config=bad,
    )
    assert all(not pose.valid for pose in observation.values())
    assert all(not pose.valid for pose in target.values())
    assert provenance["state"] == "not_evaluated"
    assert "hash" in provenance["reason"]

    missing_names = config(urdf)
    observation, _, provenance = build_gr3_pose_features(
        state=[0.0],
        action=[0.0],
        state_names=["wrong"],
        action_names=["wrong"],
        config=missing_names,
    )
    assert observation["left_eef"].matrix is None
    assert "joint names" in provenance["reason"]


def test_gr3_camera_pose_replays_frozen_head_sidecar_composition(tmp_path: Path) -> None:
    urdf = tmp_path / "fixture.urdf"
    urdf.write_text(URDF)
    sidecar = tmp_path / "camera.json"
    sidecar.write_text(
        """{
          "schema": "robot_camera0_extrinsic_v1",
          "source_document": {"sha256": "e0d078fe17bec983e37157993b9d2413feff98498519ab230920a2c9c7e4d1ca"},
          "mechanical_frame": {"head_pitch_from_camera_translation_m": [0.1, 0.0, 0.05]},
          "mechanical_to_rgb_opencv_optical": [[0,-1,0],[0,0,-1],[1,0,0]],
          "intrinsics": {
            "raw_K_1280x800": [[568.9395751953125,0,617.2298583984375],[0,568.5732421875,392.6118469238281],[0,0,1]],
            "stored_K_832x480_candidate": [[369.8107238769531,0,401.1994079589844],[0,369.572607421875,235.1977005004883],[0,0,1]]
          }
        }"""
    )
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    gate = tmp_path / "gate.json"
    gate.write_text(
        """{
          "bindings": {"camera_fk_schema": "fastwam_final_robot_camera0_anchor_fk_v1", "sidecar": {"sha256": "%s"}, "pdf": {"sha256": "e0d078fe17bec983e37157993b9d2413feff98498519ab230920a2c9c7e4d1ca"}, "intrinsic": "stored_K_832x480_candidate"},
          "decision": {"camera0_current_ee_proprio_authorized": true, "camera0_future_action_label_authorized": false, "validation_test_eval_opened": false}
        }""" % sidecar_sha
    )
    reprojection = ReprojectionProfile(
        calibration_sidecar_path=str(sidecar),
        calibration_sidecar_sha256=sidecar_sha,
        authority_evidence_path=str(gate),
        authority_evidence_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
        camera_link="head_pitch_link",
        stored_image_size=(832, 480),
        binding_verified=True,
    )

    pose, provenance = build_gr3_camera_pose(
        state=[0.0],
        state_names=["head_pitch_joint"],
        kinematics=config(urdf),
        reprojection=reprojection,
    )

    assert pose.valid is True
    matrix = np.asarray(pose.matrix).reshape(4, 4)
    assert matrix[:3, 3] == pytest.approx([0.1, 0.0, 0.65])
    assert provenance["state"] == "evaluated"
    assert provenance["future_action_label_authorized"] is False
