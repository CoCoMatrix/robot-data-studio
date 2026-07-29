from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import pyarrow.parquet as pq

from robot_data_studio.formats.registry import FormatRegistry

from .gr3_fk import build_gr3_camera_pose, build_gr3_pose_features
from .models import (
    EpisodeRecord,
    FrameSample,
    RawDualActionFrame,
    RawNumericFeature,
    RawPose,
)
from .human_checks import analyze_human_arrays
from .profile import CurationProfile, SourceProfile, is_source_allowed


class CurationSourceAdapter(Protocol):
    def summaries(self) -> list[dict[str, object]]: ...

    def iter_episodes(self) -> Iterator[EpisodeRecord]: ...


_DENIED_PATH_TOKENS = ("val", "validation", "test", "eval")
XJGT_DISPLAY_MATRIX_ROTATE_90 = (
    0,
    -65536,
    0,
    65536,
    0,
    0,
    0,
    0,
    1073741824,
)


def display_oriented_size(
    encoded_size: tuple[int, int],
    display_matrix: Sequence[int] | None,
) -> tuple[int, int]:
    """Resolve encoded width/height to the calibration-facing display canvas."""

    if display_matrix is None:
        return encoded_size
    if tuple(int(value) for value in display_matrix) != XJGT_DISPLAY_MATRIX_ROTATE_90:
        raise ValueError("unsupported container display matrix")
    width, height = encoded_size
    return height, width


def _safe_episode_token(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"unsafe episode identifier: {value!r}")
    if any(token in value.casefold() for token in _DENIED_PATH_TOKENS):
        raise PermissionError(f"episode identifier is denied: {value!r}")
    return value


def _authorized_path(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"metadata path escapes authorized root: {path}") from error
    if any(token in part.casefold() for part in relative.parts for token in _DENIED_PATH_TOKENS):
        raise PermissionError(f"metadata path is denied: {path}")
    return resolved


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _task_payload(file: h5py.File) -> dict:
    if "label/text_label" not in file:
        return {}
    try:
        payload = json.loads(_decode_text(file["label/text_label"][()]))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    english = payload.get("data_en")
    return english if isinstance(english, dict) else {}


def _state_row(dataset: h5py.Dataset, index: int) -> list[float]:
    value = np.asarray(dataset[index], dtype=np.float64)
    if value.shape == (4, 4):
        value = value[:3, 3]
    return [float(item) for item in value.reshape(-1)]


def _event_seconds(value: object) -> float | None:
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip()
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _video_header(
    path: Path,
    *,
    display_matrix: Sequence[int] | None = None,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"state": "not_evaluated", "reason": "video file missing or empty"}
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            encoded_size = (int(stream.width), int(stream.height))
            display_size = display_oriented_size(encoded_size, display_matrix)
            return {
                "state": "evaluated",
                "width": display_size[0],
                "height": display_size[1],
                "encoded_width": encoded_size[0],
                "encoded_height": encoded_size[1],
                "display_width": display_size[0],
                "display_height": display_size[1],
                "display_matrix": list(display_matrix) if display_matrix is not None else None,
                "orientation_source": (
                    "frozen_profile_displaymatrix" if display_matrix is not None else "identity"
                ),
                "frames": int(stream.frames),
                "average_rate": str(stream.average_rate),
                "decoded_frames": 0,
            }
    except (ImportError, OSError, ValueError) as error:
        return {"state": "not_evaluated", "reason": f"video header unavailable: {error}"}


def _spans_from_bool(mask: Sequence[bool]) -> list[list[int]]:
    indexes = [index for index, valid in enumerate(mask) if valid]
    if not indexes:
        return []
    output: list[list[int]] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        output.append([start, previous])
        start = previous = index
    output.append([start, previous])
    return output


def _raw_pose(
    matrix: np.ndarray,
    *,
    valid: bool,
    reference_frame: str,
    child_frame: str,
    direction: str,
    missing_reason: str,
) -> RawPose:
    finite = matrix.shape == (4, 4) and bool(np.isfinite(matrix).all())
    pose_valid = valid and finite
    return RawPose(
        matrix=matrix.astype(np.float64).reshape(-1).tolist() if pose_valid else None,
        valid=pose_valid,
        reference_frame=reference_frame,
        child_frame=child_frame,
        transform_direction=direction,
        missing_reason=None if pose_valid else missing_reason,
    )


class XjgtHdf5SourceAdapter:
    def __init__(self, root: Path, source: SourceProfile, episode_refs: Sequence[str]) -> None:
        self.root = root
        self.source = source
        self.episode_refs = [_safe_episode_token(value) for value in episode_refs]
        self.paths = [
            _authorized_path(root, root / "hdf5" / f"{value}_hdf5.hdf5")
            for value in self.episode_refs
        ]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def summaries(self) -> list[dict[str, object]]:
        summaries = []
        for episode_id, path in zip(self.episode_refs, self.paths, strict=True):
            with h5py.File(path, "r") as file:
                if not self.source.state_fields:
                    raise ValueError("xjgt_hdf5 requires state_fields")
                missing = [key for key in self.source.state_fields if key not in file]
                missing.extend(key for key in self.source.validity_fields if key not in file)
                if missing:
                    raise ValueError(f"missing XJGT fields: {sorted(set(missing))}")
                length = int(file[self.source.state_fields[0]].shape[0])
                task = _task_payload(file)
                summaries.append(
                    {
                        "episode_id": episode_id,
                        "length": length,
                        "duration_seconds": max(0.0, (length - 1) / self.source.fps),
                        "task": str(task.get("task") or task.get("description") or ""),
                        "data_file": str(path),
                        "video_file": str(self.root / "video" / f"{episode_id}_video.mp4"),
                        "timestamp_source": "configured_fps",
                    }
                )
        return summaries

    def iter_episodes(self) -> Iterator[EpisodeRecord]:
        for summary, path in zip(self.summaries(), self.paths, strict=True):
            episode_id = str(summary["episode_id"])
            video_path = self.root / "video" / f"{episode_id}_video.mp4"
            with h5py.File(path, "r") as file:
                task = _task_payload(file)
                length = int(summary["length"])
                states = [file[key] for key in self.source.state_fields]
                validity = [file[key] for key in self.source.validity_fields]
                frames = []
                intrinsic = (
                    np.asarray(file["camera/intrinsic"], dtype=np.float64)
                    if "camera/intrinsic" in file
                    else None
                )
                for index in range(length):
                    state = [value for dataset in states for value in _state_row(dataset, index)]
                    confidence_values = [float(dataset[index]) for dataset in validity]
                    confidence_valid = all(
                        value >= self.source.human_extension.confidence_threshold
                        for value in confidence_values
                    )
                    valid = confidence_valid and all(math.isfinite(value) for value in state)
                    hand_poses: dict[str, RawPose] = {}
                    for source_name, canonical_name, validity_key in (
                        ("leftHand", "left_hand", "confidences/leftHand"),
                        ("rightHand", "right_hand", "confidences/rightHand"),
                    ):
                        matrix = np.asarray(
                            file[f"transforms/{source_name}"][index], dtype=np.float64
                        )
                        hand_valid = (
                            validity_key in file
                            and float(file[validity_key][index])
                            >= self.source.human_extension.confidence_threshold
                        )
                        hand_poses[canonical_name] = _raw_pose(
                            matrix,
                            valid=hand_valid,
                            reference_frame="world",
                            child_frame=canonical_name,
                            direction=f"T_world_{canonical_name}",
                            missing_reason="source hand confidence invalid or transform non-finite",
                        )
                    camera_matrix = (
                        np.asarray(file["transforms/camera"][index], dtype=np.float64)
                        if "transforms/camera" in file
                        else np.empty((0, 0), dtype=np.float64)
                    )
                    camera_pose = _raw_pose(
                        camera_matrix,
                        valid=camera_matrix.shape == (4, 4),
                        reference_frame="world",
                        child_frame="camera0",
                        direction="T_world_camera0",
                        missing_reason="source camera transform unavailable or non-finite",
                    )
                    calibration_hash = (
                        hashlib.sha256(intrinsic.tobytes()).hexdigest()
                        if intrinsic is not None
                        else None
                    )
                    frames.append(
                        FrameSample(
                            episode_id=episode_id,
                            frame_index=index,
                            timestamp=index / self.source.fps,
                            valid=valid,
                            state=state,
                            action=[],
                            video_uri=str(video_path) if video_path.is_file() else None,
                            metadata={
                                "quality_hand": (
                                    np.asarray(file["label/quality_hand"][index]).reshape(-1).tolist()
                                    if "label/quality_hand" in file
                                    else None
                                )
                            },
                            dual_action=RawDualActionFrame(
                                observation_joint_position=RawNumericFeature.missing(
                                    "Human source has no robot joint observation"
                                ),
                                observation_eef_pose=hand_poses,
                                action_joint_target=RawNumericFeature.missing(
                                    "Human source has no commanded robot joint target"
                                ),
                                action_eef_pose_target={
                                    name: pose.model_copy(deep=True)
                                    for name, pose in hand_poses.items()
                                },
                                camera_pose={"camera0": camera_pose},
                                provenance={
                                    "pose_authority": "XJGT HDF5 transforms/*",
                                    "action_type": "observed_hand_pose_imitation_target",
                                    "calibration_sha256": calibration_hash,
                                    "camera_transform_direction": "T_world_camera",
                                    "hand_transform_direction": "T_world_joint",
                                    "interpolation": "none",
                                },
                            ),
                        )
                    )
                quality_counts: dict[str, int] = {}
                if "label/quality_hand" in file:
                    values, counts = np.unique(np.asarray(file["label/quality_hand"]), return_counts=True)
                    quality_counts = {
                        str(float(value)): int(count)
                        for value, count in zip(values, counts, strict=True)
                    }
                annotated_events = task.get("events")
                if not isinstance(annotated_events, list):
                    annotated_events = []
                normalized_events = [
                    {"start_time": start, "end_time": end}
                    for event in annotated_events
                    if isinstance(event, dict)
                    and (start := _event_seconds(event.get("start_time"))) is not None
                    and (end := _event_seconds(event.get("end_time"))) is not None
                ]
                video_header = _video_header(
                    video_path,
                    display_matrix=self.source.human_extension.display_matrix,
                )
                encoded_size_matches = (
                    video_header.get("state") == "evaluated"
                    and self.source.human_extension.encoded_video_size is not None
                    and tuple(
                        int(video_header[key])
                        for key in ("encoded_width", "encoded_height")
                    )
                    == self.source.human_extension.encoded_video_size
                )
                extension_metrics = []
                human_extension_lineage: dict[str, object]
                required_human_fields = {
                    "camera/intrinsic",
                    "transforms/camera",
                    "transforms/leftHand",
                    "transforms/rightHand",
                    "confidences/leftHand",
                    "confidences/rightHand",
                }
                if all(field in file for field in required_human_fields):
                    transform_names = sorted(
                        name.removeprefix("transforms/")
                        for name in file["transforms"]
                        if file[f"transforms/{name}"].shape == (length, 4, 4)
                    )
                    confidence_names = sorted(
                        name.removeprefix("confidences/")
                        for name in file["confidences"]
                        if file[f"confidences/{name}"].shape == (length,)
                    )
                    human_result = analyze_human_arrays(
                        transforms={
                            name: np.asarray(file[f"transforms/{name}"], dtype=np.float64)
                            for name in transform_names
                        },
                        confidences={
                            name: np.asarray(file[f"confidences/{name}"], dtype=np.float64)
                            for name in confidence_names
                        },
                        intrinsic=np.asarray(file["camera/intrinsic"], dtype=np.float64),
                        fps=self.source.fps,
                        config=self.source.human_extension,
                        calibration_image_size=self.source.human_extension.calibration_image_size,
                        stored_video_size=(
                            tuple(video_header[axis] for axis in ("width", "height"))
                            if video_header.get("state") == "evaluated"
                            else self.source.human_extension.stored_video_size
                        ),
                        video_frame_count=(
                            int(video_header["frames"])
                            if video_header.get("state") == "evaluated"
                            else None
                        ),
                        expected_frame_count=length,
                        raw_text=str(task.get("description") or task.get("task") or ""),
                        events=normalized_events,
                        quality_hand=(
                            np.asarray(file["label/quality_hand"], dtype=np.float64)
                            if "label/quality_hand" in file
                            else None
                        ),
                        video_binding_verified=(
                            self.source.human_extension.video_binding_verified
                            and encoded_size_matches
                        ),
                    )
                    extension_metrics = human_result.metrics
                    human_extension_lineage = {
                        "state": "evaluated",
                        "transform_track_count": len(transform_names),
                        "confidence_track_count": len(confidence_names),
                        "per_joint_confidence_spans": {
                            name: summary.model_dump(exclude={"valid_mask"})
                            for name, summary in human_result.confidence_tracks.items()
                        },
                        "per_hand_valid_spans": {
                            hand: _spans_from_bool(mask)
                            for hand, mask in human_result.per_hand_valid_masks.items()
                        },
                        "bimanual_valid_spans": _spans_from_bool(human_result.bimanual_valid_mask),
                        "either_hand_valid_spans": _spans_from_bool(human_result.either_hand_valid_mask),
                        "no_interpolation": True,
                    }
                else:
                    human_extension_lineage = {
                        "state": "not_evaluated",
                        "reason": "missing intrinsic/camera/hand evidence",
                    }
                yield EpisodeRecord(
                    episode_id=episode_id,
                    source_id=self.source.source_id,
                    task_id=str(task.get("task") or self.source.source_id),
                    split="train",
                    expected_duration=max(0.0, (length - 1) / self.source.fps),
                    frames=frames,
                    event_frames={"start": 0, "end": max(0, length - 1)},
                    raw_task_text=str(task.get("description") or task.get("task") or ""),
                    extension_metrics=extension_metrics,
                    lineage={
                        "format": "xjgt_hdf5",
                        "data_file": str(path),
                        "video_file": str(video_path) if video_path.is_file() else None,
                        "timestamp_source": "configured_fps",
                        "validity_fields": self.source.validity_fields,
                        "invalid_frame_count": sum(not frame.valid for frame in frames),
                        "validity_basis": "all_configured_confidences_positive_and_state_finite",
                        "quality_hand": "source_auxiliary_complete_visibility_1_is_complete_not_hard_gate"
                        if "label/quality_hand" in file
                        else "missing",
                        "quality_hand_value_counts": quality_counts,
                        "annotation_event_count": len(annotated_events),
                        "annotation_event_bounds": [
                            {
                                "start_time": event.get("start_time"),
                                "end_time": event.get("end_time"),
                            }
                            for event in annotated_events
                            if isinstance(event, dict)
                        ],
                        "video_header": video_header,
                        "human_extension": human_extension_lineage,
                    },
                )


class BlockLeRobotV3SourceAdapter:
    def __init__(self, root: Path, source: SourceProfile, episode_refs: Sequence[str]) -> None:
        self.root = root
        self.source = source
        self.episode_indexes = [int(_safe_episode_token(value)) for value in episode_refs]
        self.reader = FormatRegistry.default().open_dataset(root, "lerobot_v3")
        self.info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        train_indexes = self._train_indexes()
        denied = [index for index in self.episode_indexes if index not in train_indexes]
        if denied:
            raise PermissionError(f"episodes are not in the declared train split: {denied}")
        summaries = self.reader.list_episodes()
        self.by_index = {summary.episode_index: summary for summary in summaries}
        self.expanded_tasks = self._expanded_tasks()
        missing = [index for index in self.episode_indexes if index not in self.by_index]
        if missing:
            raise KeyError(f"unknown episode indexes: {missing}")

    def _train_indexes(self) -> set[int]:
        split = str(self.info.get("splits", {}).get("train", ""))
        match = re.fullmatch(r"(\d+):(\d+)", split)
        if match is None:
            raise PermissionError("LeRobot V3 metadata lacks a simple explicit train split")
        return set(range(int(match.group(1)), int(match.group(2))))

    def _expanded_tasks(self) -> dict[int, str]:
        selected = set(self.episode_indexes)
        output: dict[int, str] = {}
        for path in sorted((self.root / "meta" / "episodes").glob("**/*.parquet")):
            parquet = pq.ParquetFile(path)
            if not {"episode_index", "expand_task"} <= set(parquet.schema_arrow.names):
                continue
            for batch in parquet.iter_batches(
                batch_size=256,
                columns=["episode_index", "expand_task"],
            ):
                for row in batch.to_pylist():
                    index = int(row["episode_index"])
                    text = str(row.get("expand_task") or "").strip()
                    if index in selected and text:
                        output[index] = text
        return output

    def _from_timestamp(self, summary: object) -> float:
        starts = summary.video_start_seconds
        if self.source.video_key and self.source.video_key in starts:
            return float(starts[self.source.video_key])
        return float(next(iter(starts.values()), 0.0))

    def summaries(self) -> list[dict[str, object]]:
        output = []
        for index in self.episode_indexes:
            summary = self.by_index[index]
            output.append(
                {
                    "episode_id": str(index),
                    "length": summary.length,
                    "duration_seconds": summary.duration_seconds,
                    "task": summary.tasks[0] if summary.tasks else "",
                    "data_file": str(self.root / summary.data_file),
                    "video_files": summary.video_files,
                    "from_timestamp": self._from_timestamp(summary),
                    "timestamp_source": "row.timestamp",
                }
            )
        return output

    @staticmethod
    def _row_groups(parquet: pq.ParquetFile, episode_index: int) -> list[int]:
        field_index = parquet.schema_arrow.names.index("episode_index")
        groups = []
        for index in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(index).column(field_index).statistics
            if statistics and statistics.min <= episode_index <= statistics.max:
                groups.append(index)
        if not groups:
            raise ValueError(f"no parquet row group for episode {episode_index}")
        return groups

    def iter_episodes(self) -> Iterator[EpisodeRecord]:
        for episode_index in self.episode_indexes:
            summary = self.by_index[episode_index]
            parquet_path = _authorized_path(self.root, self.root / summary.data_file)
            parquet = pq.ParquetFile(parquet_path)
            row_groups = self._row_groups(parquet, episode_index)
            columns = [
                self.source.state_key,
                "timestamp",
                "frame_index",
                "episode_index",
            ]
            if self.source.action_key:
                columns.append(self.source.action_key)
            video_path = None
            if self.source.video_key and self.source.video_key in summary.video_files:
                video_path = _authorized_path(
                    self.root,
                    self.root / summary.video_files[self.source.video_key],
                )
            frames = []
            state_feature = self.info.get("features", {}).get(self.source.state_key, {})
            action_feature = self.info.get("features", {}).get(
                self.source.action_key or "", {}
            )
            configured_state_names = list(state_feature.get("names") or [])
            configured_action_names = list(action_feature.get("names") or [])
            for batch in parquet.iter_batches(
                batch_size=256,
                row_groups=row_groups,
                columns=columns,
            ):
                for row in batch.to_pylist():
                    if int(row["episode_index"]) != episode_index:
                        continue
                    state = [float(value) for value in (row.get(self.source.state_key) or [])]
                    action = [
                        float(value)
                        for value in (row.get(self.source.action_key) or [])
                    ] if self.source.action_key else []
                    valid = all(math.isfinite(value) for value in [*state, *action, row["timestamp"]])
                    state_names = (
                        configured_state_names
                        if len(configured_state_names) == len(state)
                        else [f"state_{index}" for index in range(len(state))]
                    )
                    action_names = (
                        configured_action_names
                        if len(configured_action_names) == len(action)
                        else [f"action_{index}" for index in range(len(action))]
                    )
                    observed_eef, target_eef, fk_provenance = build_gr3_pose_features(
                        state=state,
                        action=action,
                        state_names=state_names,
                        action_names=action_names,
                        config=self.source.geometry.kinematics,
                    )
                    camera_pose, camera_provenance = build_gr3_camera_pose(
                        state=state,
                        state_names=state_names,
                        kinematics=self.source.geometry.kinematics,
                        reprojection=self.source.geometry.reprojection,
                    )
                    source_units_state = list(state_feature.get("units") or [])
                    source_units_action = list(action_feature.get("units") or [])
                    if len(source_units_state) != len(state):
                        source_units_state = ["source_native_unit_unspecified"] * len(state)
                    if len(source_units_action) != len(action):
                        source_units_action = ["source_native_unit_unspecified"] * len(action)
                    frames.append(
                        FrameSample(
                            episode_id=str(episode_index),
                            frame_index=int(row["frame_index"]),
                            timestamp=float(row["timestamp"]),
                            valid=valid,
                            state=state,
                            action=action,
                            video_uri=str(video_path) if video_path is not None else None,
                            dual_action=RawDualActionFrame(
                                observation_joint_position=RawNumericFeature(
                                    values=state,
                                    valid=all(math.isfinite(value) for value in state),
                                    valid_mask=[math.isfinite(value) for value in state],
                                    names=state_names,
                                    units=source_units_state,
                                ),
                                observation_eef_pose=observed_eef,
                                action_joint_target=RawNumericFeature(
                                    values=action,
                                    valid=all(math.isfinite(value) for value in action),
                                    valid_mask=[math.isfinite(value) for value in action],
                                    names=action_names,
                                    units=source_units_action,
                                    action_type="source_native_action_target",
                                ),
                                action_eef_pose_target=target_eef,
                                camera_pose={"camera0": camera_pose},
                                provenance={
                                    "fk": fk_provenance,
                                    "camera": camera_provenance,
                                    "calibration_sidecar": (
                                        self.source.geometry.reprojection.calibration_sidecar_path
                                    ),
                                },
                            ),
                        )
                    )
            frames.sort(key=lambda frame: frame.frame_index)
            generic_task = summary.tasks[0] if summary.tasks else self.source.source_id
            raw_task = self.expanded_tasks.get(episode_index, generic_task)
            yield EpisodeRecord(
                episode_id=str(episode_index),
                source_id=self.source.source_id,
                task_id=generic_task,
                split="train",
                from_timestamp=self._from_timestamp(summary),
                expected_duration=frames[-1].timestamp if frames else 0.0,
                frames=frames,
                event_frames={"start": 0, "end": frames[-1].frame_index if frames else 0},
                raw_task_text=raw_task,
                lineage={
                    "format": "lerobot_v3",
                    "data_file": str(parquet_path),
                    "parquet_row_groups": row_groups,
                    "timestamp_source": "row.timestamp",
                    "decoded_pts": "not_evaluated_no_video_decode",
                    "video_file": str(video_path) if video_path is not None else None,
                    "invalid_frame_count": sum(not frame.valid for frame in frames),
                    "validity_basis": "finite_state_action_timestamp",
                    "pose_fields_available": sorted(
                        key for key in self.info.get("features", {}) if key.endswith(".pose")
                    ),
                    "time_fields_available": sorted(
                        key
                        for key in self.info.get("features", {})
                        if key == "timestamp" or key.endswith("_ts_ns")
                    ),
                    "task_text_source": (
                        "meta.episodes.expand_task"
                        if episode_index in self.expanded_tasks
                        else "meta.episodes.tasks"
                    ),
                },
            )


def open_source(
    profile: CurationProfile,
    input_root: str | Path,
    episode_refs: Sequence[str],
) -> CurationSourceAdapter:
    if not episode_refs:
        raise ValueError("real source adapters require an explicit episode selection")
    root = Path(input_root).expanduser().resolve()
    if not is_source_allowed(profile, str(root), split="train"):
        raise PermissionError(f"source is not in the explicit train allowlist: {root}")
    if len(profile.sources) != 1:
        raise ValueError("real source profiles must declare exactly one source")
    source = profile.sources[0]
    if source.adapter == "xjgt_hdf5":
        return XjgtHdf5SourceAdapter(root, source, episode_refs)
    if source.adapter == "lerobot_v3":
        return BlockLeRobotV3SourceAdapter(root, source, episode_refs)
    raise ValueError(f"profile source adapter is not a real source adapter: {source.adapter}")
