from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .c2_mesh import MeshRenderResult, _mask_metrics, render_profiled_gr3_silhouette
from .c2_projection import xyz_rot6d_to_matrix
from .gr3_fk import build_gr3_camera_pose
from .profile import CurationProfile
from .sam3 import Sam3Provider, Sam3Result


RenderFn = Callable[..., MeshRenderResult]
CameraPoseBuilder = Callable[..., tuple[Any, dict]]


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_array(value).reshape(-1)[0])


def _uint8_rgb(value: Any) -> np.ndarray:
    image = _array(value)
    if image.ndim != 3:
        raise ValueError("official Camera0 frame must be three-dimensional")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[2] != 3:
        raise ValueError("official Camera0 frame must contain three channels")
    if image.dtype.kind == "f":
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return np.ascontiguousarray(image.astype(np.uint8))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_review_queue(
    output: Path,
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Write a review-only queue ranked by the weakest provisional C2 overlap."""

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        evidence_path = str(episode["evidence_path"])
        evidence = json.loads((output / evidence_path).read_text(encoding="utf-8"))
        contact_sheet_path = str(evidence["contact_sheet_path"])
        for frame in evidence["frames"]:
            temporal = frame["temporal_offset_diagnostic"]
            rows.append(
                {
                    "episode_id": episode_id,
                    "source_frame_index": int(frame["source_frame_index"]),
                    "source_timestamp": float(frame["source_timestamp"]),
                    "iou": float(frame["iou"]),
                    "precision": float(frame["precision"]),
                    "recall": float(frame["recall"]),
                    "coverage": float(frame["coverage"]),
                    "sam3_confidence": float(frame["sam3_confidence"]),
                    "best_temporal_offset_frames": int(temporal["best_offset_frames"]),
                    "exact_frame_is_best": bool(temporal["exact_frame_is_best"]),
                    "overlay_path": str(frame["overlay_path"]),
                    "contact_sheet_path": contact_sheet_path,
                    "evidence_path": evidence_path,
                    "disposition": "review",
                    "automatic_exclusion_allowed": False,
                    "review_reason": (
                        "official SAM3 is provisional; inspect low-overlap geometry and "
                        "possible temporal offset before any employee decision"
                    ),
                }
            )
    rows.sort(key=lambda row: (row["iou"], row["episode_id"], row["source_frame_index"]))
    for rank, row in enumerate(rows, start=1):
        row["review_rank"] = rank

    payload = {
        "schema": "robot-data-studio.c2-human-review-queue.v1",
        "disposition": "review_only",
        "automatic_exclusion_allowed": False,
        "ranking": "ascending provisional C2 IoU; ranking is triage, not a calibrated threshold",
        "employee_instruction": (
            "Open the overlay/contact sheet, compare original, URDF render and SAM3 mask, "
            "then record a separate human decision. Never delete source data from this queue."
        ),
        "item_count": len(rows),
        "items": rows,
    }
    json_hash = _write_json(output / "review_queue.json", payload)
    csv_path = output / "review_queue.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["review_rank", "disposition"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return payload, json_hash


def _episode_keyframes(start: int, stop: int, count: int, offsets: Sequence[int]) -> list[int]:
    if stop <= start:
        raise ValueError("episode has no rows")
    lower = start - min(offsets)
    upper = stop - 1 - max(offsets)
    if upper < lower or upper - lower + 1 < count:
        raise ValueError("episode is too short for bounded C2 keyframes and temporal offsets")
    indexes = np.rint(np.linspace(lower, upper, count)).astype(np.int64).tolist()
    if len(set(indexes)) != count:
        raise ValueError("C2 keyframe selection produced duplicate frames")
    return [int(value) for value in indexes]


def _overlay(
    image: np.ndarray,
    projected: np.ndarray,
    observed: np.ndarray,
    *,
    label: str,
) -> Image.Image:
    if image.shape[:2] != projected.shape or projected.shape != observed.shape:
        raise ValueError("C2 overlay image/render/SAM shapes differ")
    base = np.asarray(image, dtype=np.uint8)

    def tint(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        result = base.copy()
        result[mask] = np.rint(0.45 * result[mask] + 0.55 * np.asarray(color)).astype(
            np.uint8
        )
        return result

    rendered = tint(projected, (0, 255, 0))
    sam = tint(observed, (255, 0, 255))
    error = base.copy()
    true_positive = projected & observed
    render_only = projected & ~observed
    sam_only = observed & ~projected
    error[true_positive] = [0, 220, 0]
    error[render_only] = [255, 0, 0]
    error[sam_only] = [0, 80, 255]
    panels = [base, rendered, sam, error]
    titles = ["original", "full URDF silhouette", "official SAM3", "overlap/error"]
    height, width = projected.shape
    scale = min(1.0, 416.0 / width)
    panel_width, panel_height = int(round(width * scale)), int(round(height * scale))
    title_height = 46
    sheet = Image.new("RGB", (panel_width * 4, panel_height + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, title_height), fill="black")
    draw.text((6, 4), label, fill="white")
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        resized = Image.fromarray(panel).resize((panel_width, panel_height))
        sheet.paste(resized, (index * panel_width, title_height))
        draw.text((index * panel_width + 6, 26), title, fill="white")
    return sheet


def _matrix_from_pose(pose: Any) -> np.ndarray:
    values = getattr(pose, "matrix", None)
    if not getattr(pose, "valid", False) or values is None:
        raise ValueError(getattr(pose, "missing_reason", None) or "Camera0 FK is invalid")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def run_full_silhouette_c2_dataset(
    dataset: Any,
    *,
    profile: CurationProfile,
    output_root: str | Path,
    source_episode_ids: Sequence[str],
    max_episodes: int,
    source_video_sha256: str,
    sam3_provider: Sam3Provider,
    sam3_checkpoint_evidence: dict[str, Any] | None = None,
    render_fn: RenderFn = render_profiled_gr3_silhouette,
    camera_pose_builder: CameraPoseBuilder = build_gr3_camera_pose,
) -> dict[str, Any]:
    """Run only the current full-URDF Robot C2 path on an explicit V3 shard."""

    if max_episodes <= 0 or len(source_episode_ids) != max_episodes or max_episodes > 5:
        raise ValueError("C2 pilot requires an exact positive episode bound no larger than five")
    requested = [str(value) for value in source_episode_ids]
    allowed = profile.selection.real_episode_allowlist
    if not allowed or requested != allowed:
        raise PermissionError("requested Robot episodes differ from the profile real train allowlist")
    if any(token in value.casefold() for value in requested for token in ("val", "validation", "test", "eval")):
        raise PermissionError("held-out episode token denied")
    if len(source_video_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_video_sha256):
        raise ValueError("source video SHA256 must be 64 lowercase hexadecimal characters")
    if getattr(dataset, "num_episodes", None) != len(requested):
        raise ValueError("official V3 shard episode count differs from explicit selection")
    source = profile.sources[0]
    if source.embodiment != "robot" or source.adapter != "lerobot_v3":
        raise ValueError("full-silhouette C2 pilot requires the Robot LeRobot V3 adapter")
    c2 = source.geometry.c2_full_silhouette
    if not c2.enabled or c2.projection_scope != "full_urdf_visual_silhouette":
        raise ValueError("current full-silhouette C2 contract is not enabled")
    if sam3_provider.config.backend != "local":
        raise ValueError("real C2 accepts only the local official SAM3 provider")

    kinematics = source.geometry.kinematics
    reprojection = source.geometry.reprojection
    if kinematics.visual_scope != "pose_authorized_upper_body":
        raise ValueError("Block Robot C2 v2 requires the pose-authorized upper-body scope")
    authority_path = Path(kinematics.visual_scope_authority_path or "")
    if (
        not authority_path.is_file()
        or not kinematics.visual_scope_authority_sha256
        or _sha256(authority_path) != kinematics.visual_scope_authority_sha256
    ):
        raise ValueError("upper-body visual-scope authority path/hash is invalid")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    task_claim = authority.get("task_claim", {})
    if (
        task_claim.get("whole_body_policy") is not False
        or "upper-body" not in str(task_claim.get("name", "")).casefold()
    ):
        raise ValueError("source authority does not establish an upper-body embodiment")
    excluded_joint_roots = set(kinematics.excluded_visual_joint_roots)
    if not excluded_joint_roots or excluded_joint_roots != set(
        kinematics.explicit_unrecorded_joint_positions
    ):
        raise ValueError("excluded visual roots must exactly match unrecorded leg joints")
    if reprojection.projection_lens_contract != "train_only_stored_k_override_v2":
        raise ValueError("Block Robot C2 v2 stored-K override is not selected")
    override_path = Path(reprojection.projection_override_evidence_path or "")
    if (
        reprojection.stored_video_undistorted is not True
        or not override_path.is_file()
        or not reprojection.projection_override_evidence_sha256
        or _sha256(override_path) != reprojection.projection_override_evidence_sha256
    ):
        raise ValueError("stored-K override evidence path/hash is invalid")
    override = json.loads(override_path.read_text(encoding="utf-8"))
    if (
        override.get("selection", {}).get("source_episode_ids") != requested
        or override.get("hard_reject_allowed") is not False
        or not str(override.get("status", "")).startswith("provisional")
    ):
        raise ValueError("stored-K override evidence contract differs from the v2 profile")

    urdf_path = kinematics.urdf_path
    if not urdf_path:
        raise ValueError("full-silhouette C2 requires the SHA-pinned URDF path")
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    all_links = {node.attrib["name"] for node in root.findall("link")}
    children: dict[str, list[str]] = {}
    excluded_link_roots: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.attrib["link"], []).append(child.attrib["link"])
        if joint.attrib.get("name") in excluded_joint_roots:
            excluded_link_roots.add(child.attrib["link"])
    excluded_links: set[str] = set()
    stack = list(excluded_link_roots)
    while stack:
        link = stack.pop()
        if link in excluded_links:
            continue
        excluded_links.add(link)
        stack.extend(children.get(link, ()))
    included_links = all_links - excluded_links
    if len(excluded_links) != 12 or not included_links:
        raise ValueError("URDF upper-body visual scope did not resolve the expected 12 leg links")

    episode_rows = dataset.meta.episodes.to_pandas().to_dict(orient="records")
    if len(episode_rows) != len(requested):
        raise ValueError("official V3 episode metadata count mismatch")
    feature = dataset.meta.features["observation.state"]
    state_names = list(feature["names"] if isinstance(feature, dict) else feature.names)
    state_units = ["rad" if name in source.geometry.kinematics.joint_names else "source_native" for name in state_names]
    offsets = list(c2.temporal_offsets_frames)
    output = Path(output_root)
    overlays_root = output / "overlays"
    episodes_root = output / "episodes"
    all_episode_summaries: list[dict[str, Any]] = []
    all_metrics: list[dict[str, float]] = []
    total_evaluated = 0

    for requested_id, metadata in zip(requested, episode_rows, strict=True):
        metadata_episode = int(metadata["episode_index"])
        if str(metadata_episode) != requested_id:
            raise ValueError("official V3 episode order/index differs from explicit allowlist")
        start = int(metadata["dataset_from_index"])
        stop = int(metadata["dataset_to_index"])
        centers = _episode_keyframes(start, stop, c2.keyframes_per_episode, offsets)
        required = sorted({center + offset for center in centers for offset in offsets})
        if required[0] < start or required[-1] >= stop:
            raise ValueError("C2 temporal mapping crossed an episode boundary")
        raw_rows = {index: dataset.get_raw_item(index) for index in required}
        images = []
        for center in centers:
            image = _uint8_rgb(dataset[center]["observation.images.camera0"])
            expected = source.geometry.reprojection.stored_image_size
            if expected is None or image.shape[:2] != (expected[1], expected[0]):
                raise ValueError("decoded Camera0 shape differs from frozen stored-image authority")
            images.append(image)
        sam_results = sam3_provider.segment_batch(
            source_video_sha256=source_video_sha256,
            frame_indexes=centers,
            images=images,
        )
        if len(sam_results) != len(centers):
            raise RuntimeError("local SAM3 returned an unexpected number of masks")

        rendered: dict[int, MeshRenderResult] = {}
        camera_evidence: dict[int, dict] = {}
        camera_matrices: dict[int, np.ndarray] = {}
        for global_index in required:
            row = raw_rows[global_index]
            if int(round(_scalar(row["episode_index"]))) != metadata_episode:
                raise ValueError("official V3 temporal offset crossed an episode")
            local_frame = int(round(_scalar(row["frame_index"])))
            if local_frame != global_index - start:
                raise ValueError("official V3 frame_index is not episode-local and contiguous")
            source_frame = int(round(_scalar(row["source.frame_index"])))
            if source_frame != local_frame:
                raise ValueError("source frame and official episode frame mapping differ")
            episode_time = _scalar(row["timestamp"])
            source_time = _scalar(row["source.timestamp"])
            if abs(episode_time - source_time) > source.decoded_pts_tolerance_seconds:
                raise ValueError("official V3 episode_time differs from row source timestamp")
            state = _array(row["observation.state"]).reshape(-1)
            validity = _array(row["observation.state.validity"]).astype(bool).reshape(-1)
            if len(state) != len(state_names) or len(validity) != len(state_names):
                raise ValueError("official V3 state/name/validity dimensions differ")
            mapped = source.geometry.kinematics.joint_state_indices
            if not np.all(validity[mapped]) or not np.all(np.isfinite(state[mapped])):
                raise ValueError("exact-frame recorded GR3 joint state is invalid")
            camera_valid = bool(
                _array(row["observation.camera0_pose_absolute.validity"]).reshape(-1)[0]
            )
            if not camera_valid:
                raise ValueError("official V3 Camera0 pose is invalid")
            logged_base_from_camera = xyz_rot6d_to_matrix(
                _array(row["observation.camera0_pose_absolute"]).reshape(-1)
            )
            built_pose, built_evidence = camera_pose_builder(
                state=state,
                state_names=state_names,
                kinematics=source.geometry.kinematics,
                reprojection=source.geometry.reprojection,
            )
            built_base_from_camera = _matrix_from_pose(built_pose)
            if not np.allclose(logged_base_from_camera, built_base_from_camera, atol=2e-5, rtol=0.0):
                raise ValueError("official V3 Camera0 pose differs from exact-frame authoritative head FK")
            camera_from_root = np.linalg.inv(logged_base_from_camera)
            camera_matrices[global_index] = camera_from_root
            camera_evidence[global_index] = built_evidence
            rendered[global_index] = render_fn(
                state=state,
                state_names=state_names,
                state_units=state_units,
                camera_from_root=camera_from_root,
                kinematics=source.geometry.kinematics,
                reprojection=source.geometry.reprojection,
                renderer="opengl_zbuffer",
                included_links=included_links,
            )
            result = rendered[global_index]
            if result.renderer != "opengl_triangle_zbuffer_v1" or result.triangle_count <= 0:
                raise ValueError("full raw-triangle OpenGL z-buffer renderer did not run")

        frames: list[dict[str, Any]] = []
        contact_panels: list[Image.Image] = []
        for center, image, sam in zip(centers, images, sam_results, strict=True):
            row = raw_rows[center]
            source_frame = int(round(_scalar(row["source.frame_index"])))
            source_time = _scalar(row["source.timestamp"])
            episode_time = _scalar(row["timestamp"])
            projected = rendered[center].mask
            if sam.status != "evaluated" or sam.mask is None:
                raise RuntimeError(f"official SAM3 failed closed at episode {requested_id} frame {source_frame}: {sam.reason}")
            if sam.mask.shape != projected.shape:
                raise ValueError("local SAM3 mask shape differs from full-mesh render")
            metrics = _mask_metrics(projected, sam.mask)
            temporal = {
                str(offset): _mask_metrics(rendered[center + offset].mask, sam.mask)["iou"]
                for offset in offsets
            }
            best_offset = max(temporal, key=temporal.get)
            relative_overlay = Path("overlays") / f"episode-{requested_id}-frame-{source_frame:06d}.jpg"
            panel = _overlay(
                image,
                projected,
                sam.mask,
                label=(
                    f"episode={requested_id} frame={source_frame} "
                    f"timestamp={source_time:.9f}s"
                ),
            )
            (output / relative_overlay).parent.mkdir(parents=True, exist_ok=True)
            panel.save(output / relative_overlay, quality=92)
            contact_panels.append(panel)
            frame = {
                "global_frame_index": center,
                "source_frame_index": source_frame,
                "episode_time": episode_time,
                "source_timestamp": source_time,
                "projection_scope": "full URDF visual silhouette",
                "renderer": rendered[center].renderer,
                "raw_triangle_count": rendered[center].triangle_count,
                "visual_count": rendered[center].visual_count,
                "visual_scope": "pose_authorized_upper_body",
                "excluded_out_of_embodiment_link_count": len(excluded_links),
                **metrics,
                "sam3_confidence": sam.confidence,
                "sam3_cache_hit": sam.cache_hit,
                "sam3_cache_key": sam.cache_key,
                "sam3_reason": sam.reason,
                "temporal_offset_diagnostic": {
                    "iou_by_offset_frames": temporal,
                    "best_offset_frames": int(best_offset),
                    "exact_frame_is_best": best_offset == "0",
                },
                "camera_fk": camera_evidence[center],
                "overlay_path": str(relative_overlay),
            }
            frames.append(frame)
            all_metrics.append(metrics)
            total_evaluated += 1

        contact_path = Path("overlays") / f"episode-{requested_id}-contact-sheet.jpg"
        contact = Image.new(
            "RGB",
            (contact_panels[0].width, sum(panel.height for panel in contact_panels)),
            "white",
        )
        y = 0
        for panel in contact_panels:
            contact.paste(panel, (0, y))
            y += panel.height
        contact.save(output / contact_path, quality=90)
        aggregate = {
            name: float(np.mean([frame[name] for frame in frames]))
            for name in ("iou", "precision", "recall", "coverage")
        }
        episode_payload = {
            "schema": "robot-data-studio.c2-full-silhouette-real-evidence.v2",
            "episode_id": requested_id,
            "task": list(metadata.get("tasks", [])),
            "projection_scope": "full URDF visual silhouette",
            "visual_scope": "pose_authorized_upper_body",
            "excluded_out_of_embodiment_links": sorted(excluded_links),
            "excluded_out_of_embodiment_link_count": len(excluded_links),
            "historical_support_disc_used": False,
            "fidelity": "provisional_official_sam3",
            "hard_reject_allowed": False,
            "clip_aggregation": aggregate,
            "contact_sheet_path": str(contact_path),
            "frames": frames,
        }
        relative_evidence = Path("episodes") / f"episode-{requested_id}.json"
        digest = _write_json(output / relative_evidence, episode_payload)
        all_episode_summaries.append(
            {
                "episode_id": requested_id,
                "evidence_path": str(relative_evidence),
                "evidence_sha256": digest,
                "contact_sheet_path": str(contact_path),
                **aggregate,
            }
        )

    aggregate = {
        name: float(np.mean([metric[name] for metric in all_metrics]))
        for name in ("iou", "precision", "recall", "coverage")
    }
    manifest = {
        "schema": "robot-data-studio.c2-full-silhouette-real-manifest.v2",
        "profile_name": profile.name,
        "profile_sha256": hashlib.sha256(
            json.dumps(profile.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "processed_episode_count": len(all_episode_summaries),
        "evaluated_frame_count": total_evaluated,
        "source_episode_ids": requested,
        "source_video_sha256": source_video_sha256,
        "projection_scope": "full URDF visual silhouette",
        "visual_scope": "pose_authorized_upper_body",
        "excluded_out_of_embodiment_links": sorted(excluded_links),
        "excluded_out_of_embodiment_link_count": len(excluded_links),
        "historical_support_disc_used": False,
        "renderer": "opengl_triangle_zbuffer_v1",
        "mesh_decimation": "none",
        "camera_model": "stored_K_832x480_candidate with no additional 14D distortion",
        "projection_override_evidence_path": str(override_path),
        "projection_override_evidence_sha256": reprojection.projection_override_evidence_sha256,
        "sam3_provider": {
            "mode": "local_official",
            "model_version": sam3_provider.config.model_version,
            "prompt": sam3_provider.config.prompt,
            "confidence_threshold": sam3_provider.config.confidence_threshold,
            "cache_root": str(getattr(sam3_provider, "cache_root", "injected")),
            "checkpoint": sam3_checkpoint_evidence or {},
        },
        "fidelity": "provisional_official_sam3_not_paper_finetuned",
        "hard_reject_allowed": False,
        "calibration_required": "train-only manually labeled robot masks plus paper fine-tuned SAM3 checkpoint/config and versioned per-camera thresholds",
        "clip_aggregation": aggregate,
        "episodes": all_episode_summaries,
    }
    review_queue, review_queue_hash = _write_review_queue(output, all_episode_summaries)
    manifest["employee_review_queue"] = {
        "json_path": "review_queue.json",
        "json_sha256": review_queue_hash,
        "csv_path": "review_queue.csv",
        "item_count": review_queue["item_count"],
        "automatic_exclusion_allowed": False,
    }
    manifest_hash = _write_json(output / "manifest.json", manifest)
    return {**manifest, "manifest_sha256": manifest_hash}
