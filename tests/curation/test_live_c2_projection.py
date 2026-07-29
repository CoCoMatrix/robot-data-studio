from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.curation.c2_projection import (
    evaluate_c2_keyframes,
    matrix_to_xyz_rot6d,
    project_eef_support_mask,
)
from robot_data_studio.curation.c2_mesh import (
    C2Calibration,
    _joint_link_transforms,
    _project_camera_points,
    evaluate_full_mesh_c2_keyframes,
    render_urdf_robot_silhouette,
)
import xml.etree.ElementTree as ET
from robot_data_studio.curation.sam3 import Sam3Result


def test_eef_support_projection_uses_absolute_camera_and_separate_hand_validity() -> None:
    camera = matrix_to_xyz_rot6d(np.eye(4))
    left = np.eye(4)
    left[:3, 3] = [0.0, 0.0, 1.0]
    right = np.eye(4)
    right[:3, 3] = [0.2, 0.0, 1.0]
    eef = np.concatenate([matrix_to_xyz_rot6d(left), matrix_to_xyz_rot6d(right)])

    mask, evidence = project_eef_support_mask(
        eef_pose_absolute=eef,
        eef_validity=[True, False],
        camera_pose_absolute=camera,
        camera_valid=True,
        intrinsic=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        image_shape=(80, 100),
        support_radius_pixels=3,
    )

    assert mask[40, 50]
    assert not mask[40, 70]
    assert evidence["projected_hands"] == ["left"]
    assert evidence["projection_scope"] == "absolute EEF support discs"


def test_c2_keyframes_aggregate_real_sam_masks_and_report_scope() -> None:
    projected = np.zeros((20, 30), dtype=bool)
    projected[5:10, 8:14] = True
    sam = Sam3Result(
        status="evaluated",
        reason="local SAM3",
        confidence=0.9,
        cache_key="a" * 64,
        mask=projected.copy(),
    )
    metric = evaluate_c2_keyframes(
        projected_masks=[projected],
        projection_evidence=[{"projected_hands": ["left"]}],
        sam_results=[sam],
        frame_indexes=[7],
        projection_kind="robot_fk_eef_support",
    )

    assert metric.metric_id == "historical_support_disc_projection"
    assert metric.group == "robot_extension"
    assert metric.status == "evaluated"
    assert metric.fidelity == "historical"
    assert metric.score == pytest.approx(1.0)
    assert metric.evidence["provider_mode"] == "real_local"
    assert metric.evidence["frame_evidence"][0]["iou"] == pytest.approx(1.0)


def test_full_urdf_visual_mesh_uses_joint_state_camera_and_z_buffer(tmp_path) -> None:
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(
        """<robot name="tiny">
        <link name="base"><visual><origin xyz="0 0 3"/><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
        <link name="arm"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
        <joint name="slide" type="prismatic">
          <parent link="base"/><child link="arm"/><origin xyz="0 0 2" rpy="0 0 0"/>
          <axis xyz="1 0 0"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    intrinsic = np.asarray([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])

    at_zero = render_urdf_robot_silhouette(
        urdf_path=urdf,
        root_link="base",
        joint_state={"slide": 0.0},
        camera_from_root=np.eye(4),
        intrinsic=intrinsic,
        image_shape=(48, 64),
    )
    shifted = render_urdf_robot_silhouette(
        urdf_path=urdf,
        root_link="base",
        joint_state={"slide": 0.25},
        camera_from_root=np.eye(4),
        intrinsic=intrinsic,
        image_shape=(48, 64),
    )

    assert at_zero.mask.any()
    assert np.isfinite(at_zero.depth[at_zero.mask]).all()
    assert at_zero.visual_count == shifted.visual_count == 2
    assert np.mean(np.argwhere(shifted.mask)[:, 1]) > np.mean(np.argwhere(at_zero.mask)[:, 1])
    assert at_zero.renderer == "cpu_triangle_zbuffer_v1"


def test_full_urdf_resolves_mimic_and_requires_explicit_unrecorded_joint_values() -> None:
    root = ET.fromstring(
        """<robot name="mimic">
        <link name="base"/><link name="proximal"/><link name="distal"/><link name="leg"/>
        <joint name="finger" type="revolute"><parent link="base"/><child link="proximal"/><axis xyz="0 0 1"/></joint>
        <joint name="finger_mimic" type="revolute"><parent link="proximal"/><child link="distal"/><axis xyz="0 0 1"/><mimic joint="finger" multiplier="2" offset="0.1"/></joint>
        <joint name="unrecorded_leg" type="revolute"><parent link="base"/><child link="leg"/><axis xyz="0 1 0"/></joint>
        </robot>"""
    )

    with pytest.raises(ValueError, match="unrecorded_leg"):
        _joint_link_transforms(root, "base", {"finger": 0.2})

    transforms = _joint_link_transforms(
        root,
        "base",
        {"finger": 0.2},
        explicit_unrecorded_joint_positions={"unrecorded_leg": 0.0},
    )
    expected = np.asarray(
        [
            [np.cos(0.7), -np.sin(0.7), 0.0],
            [np.sin(0.7), np.cos(0.7), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert transforms["distal"][:3, :3] == pytest.approx(expected)


def test_opencv_14d_projection_applies_distortion_before_display_transform() -> None:
    points = np.asarray([[0.4, 0.2, 1.0], [-0.25, 0.1, 1.5]], dtype=np.float64)
    raw_k = np.asarray([[500.0, 0.0, 640.0], [0.0, 500.0, 400.0], [0.0, 0.0, 1.0]])
    display = np.asarray([[0.65, 0.0, 0.0], [0.0, 0.65, -20.0], [0.0, 0.0, 1.0]])
    distortion = np.asarray([0.2, -0.05, 0.01, -0.02, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    actual = _project_camera_points(
        points,
        intrinsic=raw_k,
        distortion_coefficients=distortion,
        display_transform=display,
    )

    import cv2

    expected_raw, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        raw_k,
        distortion,
    )
    expected_h = np.concatenate(
        [expected_raw.reshape(-1, 2), np.ones((len(points), 1))], axis=1
    )
    expected = (display @ expected_h.T).T[:, :2]
    assert actual == pytest.approx(expected, abs=1e-9)

    pinhole = _project_camera_points(
        points,
        intrinsic=display @ raw_k,
        distortion_coefficients=None,
        display_transform=None,
    )
    assert not np.allclose(actual, pinhole)


def test_opengl_full_mesh_renderer_preserves_raw_triangles_and_depth(tmp_path) -> None:
    pytest.importorskip("moderngl")
    urdf = tmp_path / "depth.urdf"
    urdf.write_text(
        """<robot name="depth">
        <link name="base">
          <visual><origin xyz="0 0 2"/><geometry><box size="0.5 0.5 0.1"/></geometry></visual>
          <visual><origin xyz="0.15 0 3"/><geometry><box size="0.5 0.5 0.1"/></geometry></visual>
        </link>
        </robot>""",
        encoding="utf-8",
    )
    result = render_urdf_robot_silhouette(
        urdf_path=urdf,
        root_link="base",
        joint_state={},
        camera_from_root=np.eye(4),
        intrinsic=np.asarray(
            [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]]
        ),
        image_shape=(48, 64),
        renderer="opengl_zbuffer",
    )

    assert result.renderer == "opengl_triangle_zbuffer_v1"
    assert result.triangle_count == 24
    assert result.visual_count == 2
    assert result.mask.any()
    assert np.isfinite(result.depth[result.mask]).all()


def test_full_mesh_c2_reports_all_mask_metrics_and_timing_diagnostic_review_only() -> None:
    observed = np.zeros((12, 16), dtype=bool)
    observed[3:8, 6:11] = True
    exact = np.zeros_like(observed)
    exact[3:8, 5:9] = True
    offset = observed.copy()
    sam = Sam3Result(
        status="evaluated",
        reason="cached official SAM3",
        confidence=0.8,
        cache_key="b" * 64,
        cache_hit=True,
        mask=observed,
    )
    metric = evaluate_full_mesh_c2_keyframes(
        projected_masks=[exact],
        temporal_projected_masks=[{-1: exact, 0: exact, 1: offset}],
        sam_results=[sam],
        frame_indexes=[9],
        calibration=C2Calibration(
            version="camera0-gr3-provisional-v1",
            camera_id="camera0",
            embodiment="gr3",
            mask_checkpoint_kind="official_sam3",
            calibrated=False,
            iou_review_threshold=0.5,
        ),
    )

    assert metric.status == "evaluated"
    assert metric.fidelity == "provisional"
    assert metric.evidence["projection_scope"] == "full URDF visual silhouette"
    assert metric.evidence["hard_reject_allowed"] is False
    frame = metric.evidence["frame_evidence"][0]
    assert set(frame) >= {"iou", "precision", "recall", "coverage"}
    assert frame["precision"] == pytest.approx(0.75)
    assert frame["recall"] == pytest.approx(0.6)
    assert frame["temporal_offset_diagnostic"]["best_offset_frames"] == 1


def test_full_mesh_c2_exact_claim_requires_fine_tuned_checkpoint_and_calibration() -> None:
    with pytest.raises(ValueError, match="fine-tuned"):
        C2Calibration(
            version="bad",
            camera_id="camera0",
            embodiment="gr3",
            mask_checkpoint_kind="official_sam3",
            calibrated=True,
            iou_review_threshold=0.5,
        )
