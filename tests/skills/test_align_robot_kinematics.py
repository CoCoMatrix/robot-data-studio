import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pinocchio as pin


REPOSITORY = Path(__file__).resolve().parents[2]
SKILL = REPOSITORY / "skills" / "align-robot-kinematics"
URDF = REPOSITORY / "tests" / "quality" / "kinematics" / "fixtures" / "nonstandard_robot.urdf"
FORWARD_AGENT_STATUS = "not_run_policy"


def test_forward_agent_validation_is_recorded_as_not_run_by_policy() -> None:
    assert FORWARD_AGENT_STATUS == "not_run_policy"


def test_skill_has_required_package_files() -> None:
    assert (SKILL / "SKILL.md").is_file()
    assert (SKILL / "agents" / "openai.yaml").is_file()
    assert (SKILL / "scripts" / "analyze.py").is_file()


def test_wrapper_discovers_arbitrary_names_without_modifying_source(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "episode_000000.hdf5")
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    output = tmp_path / "proposal.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "analyze.py"),
            "--repository",
            str(REPOSITORY),
            "--dataset",
            str(dataset),
            "--format",
            "act_hdf5",
            "--urdf",
            str(URDF),
            "--sample-budget",
            "17",
            "--max-candidates",
            "8",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "proposed"
    assert payload["profile"]["chains"][0]["end_effector_frame"] == "prism_mount"
    joint = payload["profile"]["chains"][0]["joints"][0]
    assert joint["urdf_joint"] == "alpha_axis"
    assert joint["state_index"] == 0
    assert joint["sign"] == 1
    assert abs(joint["zero_offset"]) < 1e-5
    assert payload["after_held_out"]["position"]["p95"] < 1e-5
    assert payload["after_held_out"]["orientation"]["p95"] < 1e-5
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == before


def test_wrapper_contains_no_vendor_specific_link_constants() -> None:
    source = (SKILL / "scripts" / "analyze.py").read_text(encoding="utf-8").lower()
    for forbidden in ("panda_hand", "panda_grasptarget", "tool0", "left_gripper"):
        assert forbidden not in source


def _write_dataset(path: Path) -> Path:
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    frame_id = model.getFrameId("prism_mount")
    rows = []
    for angle in np.linspace(-0.8, 0.9, 17):
        q = np.asarray(pin.neutral(model), dtype=np.float64)
        q[0] = angle
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        transform = data.oMf[frame_id]
        rows.append(
            [
                angle,
                *transform.translation,
                *pin.Quaternion(transform.rotation).coeffs(),
            ]
        )
    states = np.asarray(rows, dtype=np.float64)
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=np.zeros((len(states), 1), dtype=np.float64))
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=states)
    return path
