import json
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from fastapi.testclient import TestClient

from apps.api.main import create_app


URDF = Path("tests/quality/kinematics/fixtures/nonstandard_robot.urdf").resolve()


def test_discover_accept_preview_and_stale_kinematic_correction(tmp_path: Path) -> None:
    dataset = _write_se3_dataset(tmp_path / "episode_000000.hdf5")
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))
    project = client.post(
        "/api/projects",
        json={"path": str(dataset), "format_hint": "act_hdf5"},
    ).json()
    project_id = project["id"]

    upload = client.post(
        f"/api/projects/{project_id}/filters/kinematics/urdf?filename=astral.urdf",
        content=URDF.read_bytes(),
        headers={"content-type": "application/octet-stream"},
    )
    assert upload.status_code == 201

    discovery = client.post(
        f"/api/projects/{project_id}/filters/kinematics/discover",
        json={
            "max_candidates": 8,
            "sample_count": 24,
            "allow_base": True,
            "allow_tcp": False,
            "allow_joint_sign": False,
            "allow_joint_offset": False,
            "min_candidate_margin": 0.0,
        },
    )

    assert discovery.status_code == 201, discovery.text
    proposal = discovery.json()
    assert proposal["status"] == "proposed"
    assert proposal["profile"]["chains"][0]["end_effector_frame"] == "vector_mount"
    assert proposal["after_held_out"]["position"]["p95"] < 1e-4
    assert proposal["after_held_out"]["orientation"]["p95"] < 1e-4

    proposals = client.get(
        f"/api/projects/{project_id}/filters/kinematics/proposals"
    )
    assert proposals.status_code == 200
    assert [item["id"] for item in proposals.json()] == [proposal["id"]]

    accepted = client.post(
        f"/api/projects/{project_id}/filters/kinematics/proposals/{proposal['id']}/decision",
        json={"decision": "accept"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"

    detail = client.get(
        f"/api/projects/{project_id}/filters/kinematic_consistency/episodes/0"
    )
    assert detail.status_code == 200
    assert detail.json()["status"] == "passed"
    assert max(detail.json()["series"]["position_error"]) < 1e-4
    assert max(detail.json()["series"]["orientation_error"]) < 1e-4
    assert detail.json()["parameters"]["accepted_proposal_id"] == proposal["id"]
    assert detail.json()["parameters"]["accepted_proposal"]["evidence"]

    exported = client.post(
        f"/api/projects/{project_id}/exports",
        json={"episode_indexes": [0], "format": "act_hdf5", "options": {}},
    )
    assert exported.status_code == 201
    report = json.loads(Path(exported.json()["report_path"]).read_text())
    assert report["kinematic_correction"]["proposal_id"] == proposal["id"]
    assert report["kinematic_correction"]["profile"] == proposal["profile"]

    with h5py.File(dataset, "r+") as file:
        file["observations/qpos"][0, 0] += 0.05
    stale_detail = client.get(
        f"/api/projects/{project_id}/filters/kinematic_consistency/episodes/0"
    )
    assert stale_detail.status_code == 200
    assert stale_detail.json()["status"] == "skipped"
    assert stale_detail.json()["skipped_reason"] == "eef_pose_missing"
    stale_accept = client.post(
        f"/api/projects/{project_id}/filters/kinematics/proposals/{proposal['id']}/decision",
        json={"decision": "accept"},
    )
    assert stale_accept.status_code == 409

    replacement = client.post(
        f"/api/projects/{project_id}/filters/kinematics/urdf?filename=replacement.urdf",
        content=URDF.read_bytes(),
        headers={"content-type": "application/octet-stream"},
    )
    assert replacement.status_code == 201
    stale = client.get(f"/api/projects/{project_id}/filters/kinematics/proposals").json()
    assert stale[0]["status"] == "stale"


def _write_se3_dataset(path: Path) -> Path:
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    frame_id = model.getFrameId("vector_mount")
    base = pin.SE3(
        pin.rpy.rpyToMatrix(np.asarray([0.18, -0.12, 0.09])),
        np.asarray([0.12, -0.08, 0.21]),
    )
    rows = []
    for index in range(32):
        q = np.asarray(
            [
                -0.8 + 1.6 * index / 31,
                0.45 * np.sin(index * 0.31),
                0.3 * np.cos(index * 0.17),
            ],
            dtype=np.float64,
        )
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        logged = base * data.oMf[frame_id]
        quaternion_xyzw = pin.Quaternion(logged.rotation).coeffs()
        rows.append([q[0], q[1], *logged.translation, *quaternion_xyzw])
    states = np.asarray(rows, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=np.zeros((len(states), 3), dtype=np.float64))
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=states)
    return path
