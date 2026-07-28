from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pydantic import BaseModel, Field

from .models import (
    EpisodeRecord,
    FrameSample,
    RawDualActionFrame,
    RawNumericFeature,
    RawPose,
)
from .pilot import run_pilot
from .profile import CurationProfile, is_source_allowed


_DENIED = ("val", "validation", "test", "eval")


class PrefilterResult(BaseModel):
    episode_id: str
    eligible: bool
    reason_codes: list[str] = Field(default_factory=list)
    valid_frame_count: int = 0
    frame_count: int = 0
    content_hash: str
    source_fingerprint: str
    disposition: dict[str, Any]


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    stable = dict(payload)
    digest = _canonical_hash(stable)
    target = {**stable, "manifest_hash": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(target, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return digest


def _episode_content_hash(episode: EpisodeRecord) -> str:
    payload = episode.model_dump(mode="json", exclude={"episode_id", "lineage"})
    for frame in payload.get("frames", []):
        frame.pop("episode_id", None)
    return _canonical_hash(payload)


def _denied_path(path: str | Path, root: Path) -> bool:
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    return any(token in part.casefold() for part in relative.parts for token in _DENIED)


def _invalid_spans(frames: list[FrameSample]) -> list[dict[str, Any]]:
    invalid = [frame.frame_index for frame in frames if not frame.valid]
    if not invalid:
        return []
    spans = []
    start = previous = invalid[0]
    for index in invalid[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append(
            {"start_frame": start, "end_frame": previous, "reason": "source_invalid"}
        )
        start = previous = index
    spans.append({"start_frame": start, "end_frame": previous, "reason": "source_invalid"})
    return spans


def _dual_values_finite(frame: FrameSample) -> bool:
    if frame.dual_action is None:
        return False
    numeric = (
        frame.dual_action.observation_joint_position,
        frame.dual_action.action_joint_target,
    )
    for feature in numeric:
        if feature.values is not None and not all(math.isfinite(value) for value in feature.values):
            return False
    for poses in (
        frame.dual_action.observation_eef_pose,
        frame.dual_action.action_eef_pose_target,
        frame.dual_action.camera_pose,
    ):
        for pose in poses.values():
            if pose.matrix is not None and not all(math.isfinite(value) for value in pose.matrix):
                return False
    return True


def prefilter_episode(
    episode: EpisodeRecord,
    *,
    source_root: str | Path,
    embodiment: str | None = None,
    static_epsilon: float = 1e-9,
) -> PrefilterResult:
    """Cheap deterministic integrity gate; no decoding and no interpolation."""

    root = Path(source_root).resolve()
    hard: list[str] = []
    review: list[str] = []
    if episode.split != "train":
        hard.append("split_not_train")
    for key in ("data_file", "video_file"):
        value = episode.lineage.get(key)
        if value and _denied_path(str(value), root):
            hard.append("denied_source_path")
        elif key == "data_file" and value and not Path(str(value)).is_file():
            hard.append("episode_file_unreadable")
    for frame in episode.frames:
        if frame.video_uri and _denied_path(frame.video_uri, root):
            hard.append("denied_source_path")
    if not episode.frames:
        hard.append("empty_episode")
    elif len(episode.frames) < 2:
        hard.append("episode_too_short")
    if episode.frames:
        indexes = [frame.frame_index for frame in episode.frames]
        if indexes != list(range(len(episode.frames))):
            hard.append("frame_index_non_contiguous")
        timestamps = [frame.timestamp for frame in episode.frames]
        if any(now <= previous for previous, now in zip(timestamps, timestamps[1:])):
            hard.append("timestamp_non_monotonic")
        state_dims = {len(frame.state) for frame in episode.frames}
        action_dims = {len(frame.action) for frame in episode.frames}
        if len(state_dims) != 1 or len(action_dims) != 1:
            hard.append("dimension_or_name_order_mismatch")
        typed_name_orders = {
            (
                tuple(frame.dual_action.observation_joint_position.names),
                tuple(frame.dual_action.action_joint_target.names),
            )
            for frame in episode.frames
            if frame.dual_action is not None
        }
        if len(typed_name_orders) > 1:
            hard.append("dimension_or_name_order_mismatch")
        if not all(
            math.isfinite(value)
            for frame in episode.frames
            for value in [frame.timestamp, *frame.state, *frame.action]
        ):
            hard.append("non_finite")
        if any(frame.dual_action is None for frame in episode.frames):
            hard.append("typed_dual_action_missing")
        elif not all(_dual_values_finite(frame) for frame in episode.frames):
            hard.append("dual_action_non_finite")
        media = sorted({frame.video_uri for frame in episode.frames if frame.video_uri})
        if not media:
            review.append("language_or_media_reference_missing" if not episode.raw_task_text else "media_reference_missing")
        elif any(not Path(uri).is_file() for uri in media):
            hard.append("media_reference_unreadable")
        valid_states = [frame.state for frame in episode.frames if frame.valid and frame.state]
        motion = sum(
            sum(abs(a - b) for a, b in zip(previous, current, strict=True))
            for previous, current in zip(valid_states, valid_states[1:])
            if len(previous) == len(current)
        )
        if len(valid_states) > 1 and motion <= static_epsilon:
            hard.append("all_static")
        if not valid_states:
            hard.append("no_valid_segment")
        if embodiment == "human":
            valid_hands = any(
                frame.dual_action
                and any(pose.valid for pose in frame.dual_action.observation_eef_pose.values())
                for frame in episode.frames
            )
            if not valid_hands:
                hard.append("no_valid_hand_segment")
    if not episode.raw_task_text:
        review.append("language_explicit_missing")
    hard = sorted(set(hard))
    review = sorted(set(review))
    disposition = {
        "hard_reject": bool(hard),
        "hard_reasons": hard,
        "review": bool(review),
        "review_reasons": review,
        "window_mask": _invalid_spans(episode.frames),
        "soft_weight": 1.0,
    }
    return PrefilterResult(
        episode_id=episode.episode_id,
        eligible=not hard,
        reason_codes=[*hard, *review],
        valid_frame_count=sum(frame.valid for frame in episode.frames),
        frame_count=len(episode.frames),
        content_hash=_episode_content_hash(episode),
        source_fingerprint=_canonical_hash(episode.model_dump(mode="json")),
        disposition=disposition,
    )


def _load_data_pipeline(data_pipeline_src: Path) -> dict[str, Any]:
    package = data_pipeline_src / "data_pipeline"
    if not package.is_dir():
        raise FileNotFoundError(f"data_pipeline source package not found: {package}")
    source_text = str(data_pipeline_src.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    schema = importlib.import_module("data_pipeline.core.dual_action_schema")
    loaded = {
        name: getattr(schema, name)
        for name in (
            "AbsolutePose",
            "DualActionEpisode",
            "DualActionFrame",
            "MediaReference",
            "NumericModality",
            "PoseModality",
            "read_dual_action_episode",
            "write_dual_action_episode",
        )
    }
    official = importlib.import_module("data_pipeline.exporters.official_lerobot_v3")
    loaded.update(
        {
            "OfficialLeRobotV3Publisher": official.OfficialLeRobotV3Publisher,
            "OfficialV3FeatureContract": official.OfficialV3FeatureContract,
        }
    )
    return loaded


def _numeric(feature: RawNumericFeature, dp: dict[str, Any]) -> Any:
    return dp["NumericModality"](**feature.model_dump(mode="python"))


def _pose_modality(poses: dict[str, RawPose], dp: dict[str, Any]) -> Any:
    return dp["PoseModality"](
        poses={
            name: dp["AbsolutePose"](**pose.model_dump(mode="python"))
            for name, pose in poses.items()
        }
    )


def _episode_to_dual(
    episode: EpisodeRecord,
    *,
    embodiment: str,
    dp: dict[str, Any],
    source_hash: str,
) -> Any:
    frames = []
    for frame in episode.frames:
        if frame.dual_action is None:
            raise ValueError("typed dual-action frame missing after prefilter")
        media = {}
        if frame.video_uri:
            media["camera0"] = dp["MediaReference"](
                uri=frame.video_uri,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                episode_time=frame.timestamp,
                file_time=episode.from_timestamp + frame.timestamp,
            )
        raw = frame.dual_action
        frames.append(
            dp["DualActionFrame"](
                episode_id=episode.episode_id,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                fps=(1.0 / (episode.frames[1].timestamp - episode.frames[0].timestamp)),
                observation_joint_position=_numeric(raw.observation_joint_position, dp),
                observation_eef_pose=_pose_modality(raw.observation_eef_pose, dp),
                action_joint_target=_numeric(raw.action_joint_target, dp),
                action_eef_pose_target=_pose_modality(raw.action_eef_pose_target, dp),
                camera_pose=_pose_modality(raw.camera_pose, dp),
                media_references=media,
                language=episode.raw_task_text,
                language_missing=not bool(episode.raw_task_text),
                source_frame_index=frame.frame_index,
                source_timestamp=frame.timestamp,
                metadata={
                    "source_valid": frame.valid,
                    "quality_hand": frame.metadata.get("quality_hand"),
                },
            )
        )
    first = episode.frames[0].dual_action
    assert first is not None
    feature_contract = {
        "observation.joint_position": {
            "names": first.observation_joint_position.names,
            "units": first.observation_joint_position.units,
        },
        "observation.eef_pose": {
            "names": sorted(first.observation_eef_pose),
            "reference_frame": "absolute source authority per pose record",
            "translation_unit": "m",
            "rotation_convention": "matrix_3x3_row_major",
        },
        "action.joint_target": {
            "names": first.action_joint_target.names,
            "units": first.action_joint_target.units,
            "action_type": first.action_joint_target.action_type,
        },
        "action.eef_pose_target": {
            "names": sorted(first.action_eef_pose_target),
            "reference_frame": "absolute source authority per pose record",
            "translation_unit": "m",
            "rotation_convention": "matrix_3x3_row_major",
        },
    }
    converter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {
        "source_episode_sha256": source_hash,
        "converter_sha256": converter_sha,
        "raw_lineage_sha256": _canonical_hash(episode.lineage),
        "raw_lineage": episode.lineage,
        "event_frames": episode.event_frames,
        "expected_duration": episode.expected_duration,
        "from_timestamp": episode.from_timestamp,
        "extension_metrics": [
            metric.model_dump(mode="json") for metric in episode.extension_metrics
        ],
        "normalization_applied": False,
        "media_policy": "reference_only_no_copy",
        "frame_payload_provenance": [
            frame.dual_action.provenance for frame in episode.frames if frame.dual_action
        ][:1],
    }
    return dp["DualActionEpisode"](
        episode_id=episode.episode_id,
        source_id=episode.source_id,
        embodiment=embodiment,
        split="train",
        task_id=episode.task_id,
        frames=frames,
        feature_contract=feature_contract,
        provenance=provenance,
    )


def _dual_to_episode(dual: Any) -> EpisodeRecord:
    frames = []
    for frame in dual.frames:
        observation = frame.observation_joint_position
        action = frame.action_joint_target
        dual_raw = RawDualActionFrame(
            observation_joint_position=RawNumericFeature(**vars(observation)),
            observation_eef_pose={
                name: RawPose(**vars(pose))
                for name, pose in frame.observation_eef_pose.poses.items()
            },
            action_joint_target=RawNumericFeature(**vars(action)),
            action_eef_pose_target={
                name: RawPose(**vars(pose))
                for name, pose in frame.action_eef_pose_target.poses.items()
            },
            camera_pose={
                name: RawPose(**vars(pose))
                for name, pose in frame.camera_pose.poses.items()
            },
            provenance={"canonical_reload": True},
        )
        media = next(iter(frame.media_references.values()), None)
        valid = bool(frame.metadata.get("source_valid", True))
        state = list(observation.values or [])
        if observation.values is None:
            # Compatibility view for the existing cheap/deep-clean engine only.
            # Canonical joints remain null+invalid; pose invalidity stays explicit.
            state = []
            for name in sorted(frame.observation_eef_pose.poses):
                pose = frame.observation_eef_pose.poses[name]
                state.extend(
                    [pose.matrix[3], pose.matrix[7], pose.matrix[11]]
                    if pose.valid and pose.matrix is not None
                    else [0.0, 0.0, 0.0]
                )
        frames.append(
            FrameSample(
                episode_id=dual.episode_id,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                valid=valid,
                state=state,
                action=list(action.values or []),
                video_uri=media.uri if media else None,
                dual_action=dual_raw,
            )
        )
    return EpisodeRecord(
        episode_id=dual.episode_id,
        source_id=dual.source_id,
        task_id=dual.task_id or dual.source_id,
        split="train",
        from_timestamp=float(dual.provenance.get("from_timestamp", 0.0)),
        frames=frames,
        raw_task_text=frames and dual.frames[0].language or None,
        event_frames={
            str(key): int(value)
            for key, value in dual.provenance.get("event_frames", {}).items()
        },
        expected_duration=dual.provenance.get("expected_duration"),
        extension_metrics=dual.provenance.get("extension_metrics", []),
        lineage={
            "format": "lerobot_v3_dual_action_reference_v0.2",
            "canonical_provenance": dual.provenance,
        },
    )


def _round_trip_hashes(dual: Any) -> dict[str, str]:
    records = [frame.to_record() for frame in dual.frames]
    return {
        "timestamps_sha256": _canonical_hash(
            [record["timestamp"] for record in records]
        ),
        "joint_values_and_order_sha256": _canonical_hash(
            [
                {
                    "observation": record["observation"]["joint_position"],
                    "action": record["action"]["joint_target"],
                }
                for record in records
            ]
        ),
        "absolute_pose_and_masks_sha256": _canonical_hash(
            [
                {
                    "eef": record["observation"]["eef_pose"],
                    "camera": record["observation"]["camera_pose"],
                    "target": record["action"]["eef_pose_target"],
                }
                for record in records
            ]
        ),
        "language_sha256": _canonical_hash(
            [(record["language"], record["language_missing"]) for record in records]
        ),
        "media_references_sha256": _canonical_hash(
            [record["observation"]["media_references"] for record in records]
        ),
    }


def _dual_semantic_hash(dual: Any) -> str:
    return _canonical_hash(
        {
            "header": dual.header_record(),
            "frames": [frame.to_record() for frame in dual.frames],
        }
    )


class DualActionPipeline:
    def __init__(
        self,
        profile: CurationProfile,
        *,
        source_root: str | Path,
        output_root: str | Path,
        data_pipeline_src: str | Path,
        max_episodes: int,
        publish_official: bool = False,
        official_repo_id: str | None = None,
        official_image_shape: tuple[int, int, int] | None = None,
        official_image_provider: Callable[[Any], np.ndarray] | None = None,
    ) -> None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        self.profile = profile
        self.source_root = Path(source_root).resolve()
        self.output_root = Path(output_root)
        self.max_episodes = max_episodes
        self.publish_official = publish_official
        self.official_repo_id = official_repo_id
        self.official_image_shape = official_image_shape
        self.official_image_provider = official_image_provider
        if publish_official and not official_repo_id:
            raise ValueError("official_repo_id is required for official V3 publication")
        if publish_official and official_image_shape is None:
            raise ValueError("official_image_shape is required for official V3 publication")
        if publish_official and official_image_provider is None:
            raise ValueError("official_image_provider is required for official V3 publication")
        if not is_source_allowed(profile, str(self.source_root), split="train"):
            raise PermissionError("source root is not explicit train allowlist")
        self.dp = _load_data_pipeline(Path(data_pipeline_src))

    def run(self, episodes: Iterable[EpisodeRecord]) -> dict[str, Any]:
        raw_rows = []
        eligible: list[tuple[EpisodeRecord, PrefilterResult]] = []
        seen: dict[str, str] = {}
        for index, episode in enumerate(episodes):
            if index >= self.max_episodes:
                break
            result = prefilter_episode(
                episode,
                source_root=self.source_root,
                embodiment=self.profile.source(episode.source_id).embodiment,
            )
            duplicate = seen.get(result.content_hash)
            if duplicate is not None:
                result.eligible = False
                result.reason_codes.append("duplicate_content")
                result.disposition["review"] = True
                result.disposition["review_reasons"].append(f"duplicate_of:{duplicate}")
            else:
                seen[result.content_hash] = episode.episode_id
            raw_rows.append(result.model_dump(mode="json"))
            if result.eligible:
                eligible.append((episode, result))
        raw_manifest = {
            "schema": "robot-data-studio.raw-source-manifest.v1",
            "source_root": str(self.source_root),
            "max_episodes": self.max_episodes,
            "episodes": raw_rows,
        }
        raw_hash = _write_json(self.output_root / "raw" / "source_manifest.json", raw_manifest)
        eligible_rows = [
            {
                "episode_id": episode.episode_id,
                "content_hash": result.content_hash,
                "source_fingerprint": result.source_fingerprint,
                "raw_manifest_hash": raw_hash,
            }
            for episode, result in eligible
        ]
        eligible_hash = _write_json(
            self.output_root / "raw" / "eligible_manifest.json",
            {
                "schema": "robot-data-studio.eligible-manifest.v1",
                "episodes": eligible_rows,
            },
        )
        reloaded: list[EpisodeRecord] = []
        promoted_duals = []
        promotion_rows = []
        resumed = 0
        for episode, prefilter in eligible:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", episode.episode_id)
            path = self.output_root / "lerobot_v3" / "episodes" / safe_id / "data.jsonl"
            dual = _episode_to_dual(
                episode,
                embodiment=self.profile.source(episode.source_id).embodiment,
                dp=self.dp,
                source_hash=prefilter.source_fingerprint,
            )
            if path.is_file():
                existing = self.dp["read_dual_action_episode"](path)
                if (
                    existing.provenance.get("source_episode_sha256")
                    == prefilter.source_fingerprint
                    and existing.provenance.get("converter_sha256")
                    == dual.provenance.get("converter_sha256")
                ):
                    dual = existing
                    resumed += 1
                else:
                    digest = self.dp["write_dual_action_episode"](path, dual)
            if (
                not path.is_file()
                or dual.provenance.get("source_episode_sha256")
                != prefilter.source_fingerprint
            ):
                digest = self.dp["write_dual_action_episode"](path, dual)
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored = self.dp["read_dual_action_episode"](path)
            if _dual_semantic_hash(restored) != _dual_semantic_hash(dual):
                raise ValueError("typed dual-action serialization round-trip mismatch")
            if len(restored.frames) != len(episode.frames):
                raise ValueError("round-trip frame count mismatch")
            if [frame.timestamp for frame in restored.frames] != [
                frame.timestamp for frame in episode.frames
            ]:
                raise ValueError("round-trip timestamp mismatch")
            reloaded.append(_dual_to_episode(restored))
            promoted_duals.append(restored)
            promotion_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "data_path": str(path.relative_to(self.output_root / "lerobot_v3")),
                    "sha256": digest,
                    "frame_count": len(restored.frames),
                    "round_trip": "verified",
                    "round_trip_hashes": _round_trip_hashes(restored),
                    "media_policy": "reference_only_no_copy",
                }
            )
        _write_json(
            self.output_root / "lerobot_v3" / "meta" / "info.json",
            {
                "format": "lerobot_v3_dual_action_reference_v0.2",
                "official_hf_lerobot_v3_compatible": False,
                "media_policy": "reference_only_no_copy",
                "early_action_chunks": False,
                "training_normalization": False,
                "episode_count": len(promotion_rows),
            },
        )
        promotion_hash = _write_json(
            self.output_root / "lerobot_v3" / "promotion_manifest.json",
            {
                "schema": "robot-data-studio.dual-action-promotion.v1",
                "eligible_manifest_hash": eligible_hash,
                "episodes": promotion_rows,
            },
        )
        official_report: dict[str, Any] = {
            "state": "not_requested",
            "reason": "legacy reference promotion retained for evidence only",
        }
        if self.publish_official:
            official_root = self.output_root / "official_lerobot_v3"
            if official_root.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing official V3 evidence: {official_root}"
                )
            source_state_names = []
            source_action_names = []
            embodiments = {dual.embodiment for dual in promoted_duals}
            if len(embodiments) != 1:
                raise ValueError("one official V3 shard cannot mix Human and Robot feature schemas")
            embodiment = embodiments.pop()
            for dual in promoted_duals:
                for frame in dual.frames:
                    for name in frame.observation_joint_position.names:
                        if name not in source_state_names:
                            source_state_names.append(name)
                    for name in frame.action_joint_target.names:
                        if name not in source_action_names:
                            source_action_names.append(name)
            state_names = source_state_names if embodiment == "robot" else []
            action_names = source_action_names if embodiment == "robot" else []
            contract = self.dp["OfficialV3FeatureContract"](
                embodiment=embodiment,
                state_names=state_names,
                action_names=action_names,
                image_shape=self.official_image_shape,
            )
            official_report = self.dp["OfficialLeRobotV3Publisher"](
                root=official_root,
                repo_id=self.official_repo_id,
                fps=int(round(promoted_duals[0].frames[0].fps)),
                contract=contract,
            ).publish(promoted_duals, image_provider=self.official_image_provider)
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as exc:
                raise RuntimeError("official LeRobot loader unavailable after publication") from exc
            loaded = LeRobotDataset(
                self.official_repo_id,
                root=official_root,
                download_videos=False,
                video_backend="pyav",
            )
            if len(loaded) != sum(len(dual.frames) for dual in promoted_duals):
                raise ValueError("official loader frame-count round-trip mismatch")
            official_report["loader_reload"] = "verified"
            official_report["media_policy"] = "bounded_pilot_official_video_encoding"
        deep = run_pilot(
            self.profile,
            iter(reloaded),
            source_path=str(self.source_root),
            sidecar_root=self.output_root / "lerobot_v3" / "deep_clean",
            max_episodes=self.max_episodes,
        )
        accepted = [
            {
                "episode_id": result.episode_id,
                "final_decision": result.final_decision,
                "hard_reject": result.disposition.hard_reject,
                "window_masks": [
                    mask.model_dump(mode="json") for mask in result.disposition.window_masks
                ],
                "soft_weight": result.disposition.soft_weight,
                "deep_clean_fingerprint": result.fingerprint,
            }
            for result in deep.episodes
            if not result.disposition.hard_reject
        ]
        final_hash = _write_json(
            self.output_root / "lerobot_v3" / "final_accepted_manifest.json",
            {
                "schema": "robot-data-studio.final-accepted-manifest.v1",
                "promotion_manifest_hash": promotion_hash,
                "deep_clean_manifest_hash": deep.manifest_hash,
                "episodes": accepted,
            },
        )
        report = {
            "schema": "robot-data-studio.dual-action-pilot-report.v1",
            "raw_episode_count": len(raw_rows),
            "eligible_episode_count": len(eligible),
            "promoted_episode_count": len(promotion_rows),
            "promotion_resumed_count": resumed,
            "reloaded_episode_count": len(reloaded),
            "final_accepted_count": len(accepted),
            "raw_manifest_hash": raw_hash,
            "eligible_manifest_hash": eligible_hash,
            "promotion_manifest_hash": promotion_hash,
            "final_manifest_hash": final_hash,
            "external_vlm": "disabled",
            "media_copy_count": 0 if not self.publish_official else 1,
            "official_v3": official_report,
        }
        _write_json(self.output_root / "pilot_report.json", report)
        return report
