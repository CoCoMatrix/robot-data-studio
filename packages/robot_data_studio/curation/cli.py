from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import ValidationError

from .models import EpisodeRecord
from .c2_real_pilot import run_full_silhouette_c2_dataset
from .dual_action_bridge import DualActionPipeline
from .live_quality import (
    DashScopeC1Provider,
    LiveBudget,
    run_fast_live_quality_dataset,
    run_live_quality_dataset,
)
from .pilot import run_pilot
from .profile import CurationProfile, is_source_allowed, load_profile
from .sources import open_source
from .sam3 import Sam3Config, Sam3Provider
from .vlm import BudgetLedger, MockVlmEvaluator, estimate_vlm_cost
from .video_frames import ExactPtsVideoFrameProvider


def iter_jsonl_episodes(path: str | Path) -> Iterator[EpisodeRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield EpisodeRecord.model_validate_json(stripped)
            except ValidationError as error:
                raise ValueError(f"invalid episode at line {line_number}: {error}") from error


def _authorize(profile: CurationProfile, input_path: str | Path) -> None:
    if not is_source_allowed(profile, str(input_path), split="train"):
        raise PermissionError(f"input is not explicitly allowlisted train data: {input_path}")


def _profile_hash(profile: CurationProfile) -> str:
    payload = json.dumps(
        profile.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


_DENIED_PATH_TOKENS = ("val", "validation", "test", "eval")


def _guard_train_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if any(token in part.casefold() for part in resolved.parts for token in _DENIED_PATH_TOKENS):
        raise PermissionError(f"denied held-out path: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sam3_evidence(model_dir: str | Path | None) -> dict:
    expected_bytes = 3_439_938_512
    expected_sha256 = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
    revision = "3c879f39826c"
    if model_dir is None:
        return {
            "status": "not_evaluated",
            "reason": "SAM3 model directory was not provided",
            "repository": "facebook/sam3",
            "revision": revision,
            "checkpoint_loaded": False,
            "mock_counted_as_real": False,
        }
    root = Path(model_dir).resolve()
    weights = root / "model.safetensors"
    actual_bytes = weights.stat().st_size if weights.is_file() else 0
    partial_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and ("partial" in path.name or "resume" in path.name)
    )
    state_path = root / "download-state.json"
    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state_sha = state.get("sha256") if state.get("status") == "complete" else None
    checkpoint_sha256 = (
        state_sha
        if actual_bytes == expected_bytes and state_sha == expected_sha256
        else (_sha256_file(weights) if actual_bytes == expected_bytes else None)
    )
    verified = actual_bytes == expected_bytes and checkpoint_sha256 == expected_sha256
    return {
        "status": "available" if verified else "not_evaluated",
        "reason": (
            "checkpoint is incomplete; real SAM3 inference did not run"
            if actual_bytes != expected_bytes
            else (
                "official LFS size/SHA verified; inference status is recorded per mask"
                if verified
                else "checkpoint SHA256 differs from the official LFS pointer"
            )
        ),
        "repository": "facebook/sam3",
        "revision": revision,
        "checkpoint_expected_bytes": expected_bytes,
        "checkpoint_bytes": actual_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_expected_sha256": expected_sha256,
        "checkpoint_hash_source": "verified_download_state" if state_sha == expected_sha256 else "file_scan",
        "partial_bytes": partial_bytes,
        "processor_config_available": (root / "processor_config.json").is_file(),
        "checkpoint_loaded": False,
        "mock_counted_as_real": False,
    }


def dry_run(
    profile: CurationProfile,
    input_path: str | Path,
    *,
    max_episodes: int,
) -> dict:
    if max_episodes < 1:
        raise ValueError("max_episodes must be positive; unbounded scans are disabled")
    _authorize(profile, input_path)
    episodes = 0
    sources: set[str] = set()
    tasks: set[str] = set()
    for episode in iter_jsonl_episodes(input_path):
        profile.source(episode.source_id)
        if episode.split != "train":
            raise PermissionError(f"episode {episode.episode_id} is not train split")
        episodes += 1
        sources.add(episode.source_id)
        tasks.add(episode.task_id)
        if episodes >= max_episodes:
            break
    return {
        "profile": profile.name,
        "profile_hash": _profile_hash(profile),
        "input": str(input_path),
        "episodes": episodes,
        "bounded_by": max_episodes,
        "sources": sorted(sources),
        "tasks": sorted(tasks),
        "vlm": estimate_vlm_cost(profile.vlm, episodes),
        "writes_sidecars": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m robot_data_studio.curation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="validate a curation profile")
    validate.add_argument("--profile", required=True)
    for name in ("dry-run", "pilot"):
        command = subcommands.add_parser(name)
        command.add_argument("--profile", required=True)
        command.add_argument("--input", required=True)
        command.add_argument("--max-episodes", type=int, required=True)
        command.add_argument(
            "--episode",
            action="append",
            default=[],
            help="explicit episode id/index; required and repeatable for real source adapters",
        )
        if name == "pilot":
            command.add_argument("--sidecar-root")
    dual = subcommands.add_parser(
        "dual-pilot",
        help="bounded raw→prefilter→official LeRobot V3→deep-clean pilot",
    )
    dual.add_argument("--profile", required=True)
    dual.add_argument("--input", required=True)
    dual.add_argument("--max-episodes", type=int, required=True)
    dual.add_argument("--episode", action="append", required=True)
    dual.add_argument("--output", required=True)
    dual.add_argument("--data-pipeline-src", required=True)
    dual.add_argument(
        "--official-repo-id",
        help="publish with the installed official Hugging Face LeRobot V3 API",
    )
    live = subcommands.add_parser(
        "live-quality",
        help="bounded real C1/C3 enrichment over an existing official V3 pilot",
    )
    live.add_argument("--profile", required=True)
    live.add_argument("--source-root", required=True)
    live.add_argument("--official-root", required=True)
    live.add_argument("--prior-root", required=True)
    live.add_argument("--output", required=True)
    live.add_argument("--repo-id", required=True)
    live.add_argument("--episode", action="append", required=True)
    live.add_argument("--max-episodes", type=int, required=True)
    live.add_argument("--c1-model", default="qwen3.7-plus")
    live.add_argument("--c1-samples", type=int, default=5)
    live.add_argument("--c3-samples", type=int, default=16)
    live.add_argument("--sam3-model-dir")
    live.add_argument("--sam3-cache-root")
    live.add_argument("--sam3-samples", type=int, default=3)
    live.add_argument("--sam3-device", default="cuda:0")
    live.add_argument("--sam3-batch-size", type=int, default=4)
    live.add_argument("--support-radius-pixels", type=int, default=24)
    live.add_argument("--c1-cache-root")
    live.add_argument("--c1-concurrency", type=int, default=2)
    live.add_argument("--source-video-sha256")
    live.add_argument("--fast-path", action="store_true")
    live.add_argument("--dry-run", action="store_true")
    c2_real = subcommands.add_parser(
        "c2-real-pilot",
        help="bounded local-SAM3/full-URDF Robot C2 over an official V3 pilot",
    )
    c2_real.add_argument("--profile", required=True)
    c2_real.add_argument("--source-root", required=True)
    c2_real.add_argument("--official-root", required=True)
    c2_real.add_argument("--output", required=True)
    c2_real.add_argument("--repo-id", required=True)
    c2_real.add_argument("--episode", action="append", required=True)
    c2_real.add_argument("--max-episodes", type=int, required=True)
    c2_real.add_argument("--sam3-model-dir", required=True)
    c2_real.add_argument("--sam3-cache-root", required=True)
    c2_real.add_argument("--sam3-device", default="cuda:0")
    c2_real.add_argument("--sam3-batch-size", type=int, default=3)
    c2_real.add_argument("--source-video-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        if args.command == "validate":
            payload = {
                "valid": True,
                "profile": profile.name,
                "profile_hash": _profile_hash(profile),
                "sources": [source.source_id for source in profile.sources],
            }
        elif args.command == "dry-run":
            if profile.sources[0].adapter == "jsonl":
                payload = dry_run(profile, args.input, max_episodes=args.max_episodes)
            else:
                if len(args.episode) > args.max_episodes:
                    raise ValueError("explicit episode selection exceeds --max-episodes")
                adapter = open_source(profile, args.input, args.episode)
                selection = adapter.summaries()
                payload = {
                    "profile": profile.name,
                    "profile_hash": _profile_hash(profile),
                    "input": str(args.input),
                    "episodes": len(selection),
                    "bounded_by": args.max_episodes,
                    "selection": selection,
                    "vlm": estimate_vlm_cost(profile.vlm, len(selection)),
                    "writes_sidecars": False,
                }
        elif args.command == "pilot":
            _authorize(profile, args.input)
            sidecar_root = Path(args.sidecar_root or profile.sidecar_root)
            evaluator = None
            if profile.vlm.backend == "mock":
                evaluator = MockVlmEvaluator(
                    sidecar_root / "vlm-cache",
                    BudgetLedger.from_profile(profile.vlm),
                )
            if profile.sources[0].adapter == "jsonl":
                episodes = iter_jsonl_episodes(args.input)
            else:
                if len(args.episode) > args.max_episodes:
                    raise ValueError("explicit episode selection exceeds --max-episodes")
                episodes = open_source(profile, args.input, args.episode).iter_episodes()
            payload = run_pilot(
                profile,
                episodes,
                source_path=str(args.input),
                sidecar_root=sidecar_root,
                max_episodes=args.max_episodes,
                evaluator=evaluator,
            ).model_dump(mode="json")
        elif args.command == "dual-pilot":
            if len(args.episode) > args.max_episodes:
                raise ValueError("explicit episode selection exceeds --max-episodes")
            adapter = open_source(profile, args.input, args.episode)
            source = profile.sources[0]
            provider = None
            image_shape = None
            if args.official_repo_id:
                if source.embodiment == "human":
                    size = source.human_extension.stored_video_size
                    display_matrix = source.human_extension.display_matrix
                else:
                    size = source.geometry.reprojection.stored_image_size
                    display_matrix = None
                if size is None:
                    raise ValueError("official video publication requires configured display image size")
                provider = ExactPtsVideoFrameProvider(
                    expected_width=size[0],
                    expected_height=size[1],
                    tolerance_seconds=source.decoded_pts_tolerance_seconds,
                    display_matrix=display_matrix,
                )
                image_shape = (3, size[1], size[0])
            try:
                payload = DualActionPipeline(
                    profile,
                    source_root=args.input,
                    output_root=args.output,
                    data_pipeline_src=args.data_pipeline_src,
                    max_episodes=args.max_episodes,
                    publish_official=bool(args.official_repo_id),
                    official_repo_id=args.official_repo_id,
                    official_image_shape=image_shape,
                    official_image_provider=provider,
                ).run(adapter.iter_episodes())
            finally:
                if provider is not None:
                    provider.close()
        elif args.command == "c2-real-pilot":
            _authorize(profile, args.source_root)
            _guard_train_path(args.source_root)
            official_root = _guard_train_path(args.official_root)
            if len(args.episode) != args.max_episodes or args.max_episodes > 5:
                raise ValueError("C2 pilot requires the exact bounded episode selection")
            if any(
                token in str(episode).casefold()
                for episode in args.episode
                for token in _DENIED_PATH_TOKENS
            ):
                raise PermissionError("denied held-out token in explicit episode id")
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as error:
                raise ImportError("official lerobot loader is required for c2-real-pilot") from error
            dataset = LeRobotDataset(
                args.repo_id,
                root=official_root,
                download_videos=False,
                video_backend="pyav",
                tolerance_s=profile.sources[0].decoded_pts_tolerance_seconds,
            )
            if dataset.num_episodes != len(args.episode):
                raise ValueError("official dataset episode count does not match explicit selection")
            videos = sorted(official_root.rglob("*.mp4"))
            if len(videos) != 1:
                raise ValueError("bounded official pilot must contain exactly one camera0 video")
            source_video_sha256 = (
                args.source_video_sha256.casefold()
                if args.source_video_sha256
                else _sha256_file(videos[0])
            )
            if len(source_video_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in source_video_sha256
            ):
                raise ValueError("--source-video-sha256 must be 64 lowercase hexadecimal characters")
            sam3_evidence = _sam3_evidence(args.sam3_model_dir)
            if sam3_evidence["status"] != "available":
                raise ValueError(f"verified local SAM3 checkpoint required: {sam3_evidence['reason']}")
            source = profile.sources[0]
            base_sam3 = source.geometry.reprojection.sam3.model_copy(
                update={
                    "backend": "local",
                    "model_version": "facebook/sam3@3c879f39826c",
                    "local_weights_path": str(Path(args.sam3_model_dir) / "model.safetensors"),
                    "device": args.sam3_device,
                    "batch_size": args.sam3_batch_size,
                }
            )
            provider = Sam3Provider(base_sam3, cache_root=args.sam3_cache_root)
            payload = run_full_silhouette_c2_dataset(
                dataset,
                profile=profile,
                output_root=args.output,
                source_episode_ids=args.episode,
                max_episodes=args.max_episodes,
                source_video_sha256=source_video_sha256,
                sam3_provider=provider,
                sam3_checkpoint_evidence=sam3_evidence,
            )
        else:
            _authorize(profile, args.source_root)
            official_root = _guard_train_path(args.official_root)
            prior_root = _guard_train_path(args.prior_root)
            _guard_train_path(args.source_root)
            if len(args.episode) > args.max_episodes or len(args.episode) > 5:
                raise ValueError("explicit live episode selection exceeds the bounded maximum")
            if any(
                token in str(episode).casefold()
                for episode in args.episode
                for token in _DENIED_PATH_TOKENS
            ):
                raise PermissionError("denied held-out token in explicit episode id")
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as error:
                raise ImportError("official lerobot loader is required for live-quality") from error
            dataset = LeRobotDataset(
                args.repo_id,
                root=official_root,
                download_videos=False,
                video_backend="pyav",
            )
            if dataset.num_episodes != len(args.episode):
                raise ValueError("official dataset episode count does not match explicit selection")
            videos = sorted(official_root.rglob("*.mp4"))
            if len(videos) != 1:
                raise ValueError("bounded official pilot must contain exactly one camera0 video")
            if args.source_video_sha256:
                source_video_sha256 = args.source_video_sha256.casefold()
                if len(source_video_sha256) != 64 or any(
                    character not in "0123456789abcdef" for character in source_video_sha256
                ):
                    raise ValueError("--source-video-sha256 must be 64 lowercase hexadecimal characters")
            else:
                source_video_sha256 = None if args.dry_run else _sha256_file(videos[0])
            sam3_evidence = _sam3_evidence(args.sam3_model_dir)
            if args.dry_run:
                payload = {
                    "schema": "robot-data-studio.live-quality-dry-run.v1",
                    "profile": profile.name,
                    "source_root": str(Path(args.source_root).resolve()),
                    "official_root": str(official_root),
                    "episodes": [str(value) for value in args.episode],
                    "max_episodes": args.max_episodes,
                    "c1_model": args.c1_model,
                    "c1_calls": len(args.episode),
                    "c1_concurrency": args.c1_concurrency,
                    "c1_evidence_frames": len(args.episode) * args.c1_samples,
                    "c3_decoded_frames": len(args.episode) * args.c3_samples,
                    "sam3_decoded_frames": len(args.episode) * args.sam3_samples,
                    "sam3": sam3_evidence,
                    "fast_path": args.fast_path,
                    "writes_sidecars": False,
                }
            else:
                c1 = DashScopeC1Provider(
                    cache_root=args.c1_cache_root or Path(args.output) / "c1-cache",
                    budget=LiveBudget(
                        max_calls=len(args.episode),
                        max_evidence_frames=len(args.episode) * args.c1_samples,
                    ),
                    model=args.c1_model,
                )
                if args.fast_path:
                    source = profile.sources[0]
                    reprojection = source.geometry.reprojection
                    if not reprojection.binding_verified or reprojection.stored_intrinsic is None:
                        raise ValueError("fast C2 requires a verified stored-image intrinsic")
                    if sam3_evidence["status"] != "available" or not args.sam3_model_dir:
                        raise ValueError("fast C2 requires a size/SHA-verified local SAM3 checkpoint")
                    sam3_cache = Path(args.sam3_cache_root or Path(args.output) / "sam3-cache")
                    weights = str(Path(args.sam3_model_dir) / "model.safetensors")
                    base_sam3 = reprojection.sam3.model_copy(
                        update={
                            "backend": "local",
                            "model_version": "facebook/sam3@3c879f39826c",
                            "local_weights_path": weights,
                            "device": args.sam3_device,
                            "batch_size": args.sam3_batch_size,
                        }
                    )
                    entity = Sam3Provider(base_sam3, cache_root=sam3_cache / "entity")
                    objects = Sam3Provider(
                        base_sam3.model_copy(update={"prompt": "manipulated object"}),
                        cache_root=sam3_cache / "object",
                    )
                    payload = run_fast_live_quality_dataset(
                        dataset,
                        prior_root=prior_root,
                        output_root=args.output,
                        source_episode_ids=args.episode,
                        max_episodes=args.max_episodes,
                        c1_provider=c1,
                        source_video_sha256=source_video_sha256 or _sha256_file(videos[0]),
                        sam3_entity_provider=entity,
                        sam3_object_provider=objects,
                        sam3_checkpoint_evidence=sam3_evidence,
                        intrinsic=reprojection.stored_intrinsic,
                        embodiment=source.embodiment,
                        c1_concurrency=args.c1_concurrency,
                        c3_samples=args.c3_samples,
                        c1_samples=args.c1_samples,
                        sam3_samples=args.sam3_samples,
                        support_radius_pixels=args.support_radius_pixels,
                    )
                else:
                    payload = run_live_quality_dataset(
                        dataset,
                        prior_root=prior_root,
                        output_root=args.output,
                        source_episode_ids=args.episode,
                        max_episodes=args.max_episodes,
                        c1_provider=c1,
                        source_video_sha256=source_video_sha256 or _sha256_file(videos[0]),
                        sam3_evidence=sam3_evidence,
                        c3_samples=args.c3_samples,
                        c1_samples=args.c1_samples,
                        embodiment=profile.sources[0].embodiment,
                    )
    except (ImportError, KeyError, OSError, PermissionError, RuntimeError, ValidationError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
