from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.qwen_filter_dataset import (
    build_config,
    build_parser,
    discover_kinematic_proposal,
    parse_int_list,
    parse_str_list,
)
from robot_data_studio.quality.kinematics import (
    CorrectionOptions,
    DiscoveryRequest,
    discover_profile,
    fit_correction,
)


def test_parse_cli_lists() -> None:
    assert parse_int_list("1, 3,5") == [1, 3, 5]
    assert parse_int_list("") == []
    assert parse_str_list("joint1, joint2") == ["joint1", "joint2"]


def test_build_config_accepts_urdf_options(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='empty'/>", encoding="utf-8")

    config = build_config(
        gripper_indices="6,13",
        urdf=urdf,
        end_effector_link="tool0",
        joint_names="waist,shoulder",
        joint_state_indices="0,1",
        eef_position_indices="7,8,9",
        max_lag=4,
    )

    assert config.gripper_indices == [6, 13]
    assert config.extreme_values.gripper_indices == [6, 13]
    assert config.sudden_change.residual_min_threshold == 0.02
    assert config.sudden_change.acceleration_min_threshold == 0.02
    assert config.sudden_change.jerk_min_threshold == 0.02
    assert config.state_action_alignment.max_lag == 4
    assert config.kinematics.urdf_path == urdf
    assert config.kinematics.end_effector_link == "tool0"
    assert config.kinematics.joint_names == ["waist", "shoulder"]
    assert config.kinematics.joint_state_indices == [0, 1]
    assert config.kinematics.eef_position_indices == [7, 8, 9]


def test_parser_exposes_dynamic_kinematic_options() -> None:
    actions = build_parser().parse_args(
        [
            "/tmp/dataset",
            "--discover-kinematics",
            "--proposal-output",
            "/tmp/proposal.json",
            "--profile",
            "/tmp/profile.json",
        ]
    )

    assert actions.discover_kinematics is True
    assert actions.proposal_output == Path("/tmp/proposal.json")
    assert actions.profile == Path("/tmp/profile.json")


def test_cli_discovery_proposal_is_identical_to_shared_engine() -> None:
    urdf = Path("tests/quality/kinematics/fixtures/nonstandard_robot.urdf").resolve()
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    frame_id = model.getFrameId("prism_mount")
    rows = []
    for angle in np.linspace(-0.8, 0.9, 17):
        q = np.asarray(pin.neutral(model), dtype=np.float64)
        q[0] = angle
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        transform = data.oMf[frame_id]
        rows.append([angle, *transform.translation, *pin.Quaternion(transform.rotation).coeffs()])
    states = np.asarray(rows, dtype=np.float64)
    names = [f"channel_{index}" for index in range(states.shape[1])]
    options = CorrectionOptions(min_candidate_margin=0.0)

    cli = discover_kinematic_proposal(
        states,
        names,
        urdf,
        max_candidates=8,
        sample_count=17,
        options=options,
    )
    candidates = discover_profile(
        urdf,
        DiscoveryRequest(
            states=states,
            feature_names=names,
            max_candidates=8,
            sample_count=17,
        ),
    )
    shared = fit_correction(states, candidates, options)

    assert cli.model_dump(mode="json") == shared.model_dump(mode="json")
