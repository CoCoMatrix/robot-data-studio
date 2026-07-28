from __future__ import annotations

import json
import hashlib
from pathlib import Path
from uuid import uuid4

import numpy as np

from robot_data_studio.formats import ExportResult, FormatRegistry
from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.quality import (
    CleaningConfig,
    CleaningRun,
    CleaningSummary,
    EpisodeDecisionRequest,
    EpisodeQualityScorer,
    EpisodeQualityResult,
    FilterConfig,
    FilterConfigPatch,
    FilterDetail,
    FilterKinematicsConfig,
    FilterRun,
    FilterSummary,
    KinematicDiscoveryRequest,
    KinematicProposalDecision,
    VlmSettings,
)
from robot_data_studio.quality.kinematics import (
    CorrectionOptions,
    CorrectionProposal,
    DiscoveryRefusal,
    DiscoveryRequest,
    discover_profile,
    fit_correction,
)
from robot_data_studio.quality.filter_service import (
    DatasetFilterService,
    ProgressCallback,
    infer_gripper_indices,
)
from robot_data_studio.quality.dover_backend import (
    VideoTechnicalScore,
    checkpoint_file_info,
    visual_quality_fingerprint,
)
from robot_data_studio.quality.models import utc_now
from robot_data_studio.quality.store import CleaningStateStore, VlmSettingsStore, build_summary
from robot_data_studio.quality.visual_quality import extract_video_frame_jpeg
from robot_data_studio.reports import ReportSignals, build_report_signals

from .models import Project


def _normalize_filesystem_path(path: str) -> str:
    stripped = path.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


class ProjectService:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._projects: dict[str, tuple[Project, DatasetAdapter]] = {}
        self._scorer = EpisodeQualityScorer()
        self._formats = FormatRegistry.default()

    def formats(self):
        return self._formats.list_formats()

    def create(self, path: str, format_hint: str | None = None) -> Project:
        path = self._normalize_user_path(path)
        if not path:
            raise ValueError("Dataset path is required")
        reader = self._formats.open_dataset(path, format_hint=format_hint)
        project = Project(id=uuid4().hex[:12], dataset=reader.metadata())
        self.project_artifact_dir(project.id).mkdir(parents=True, exist_ok=True)
        self._projects[project.id] = (project, reader)
        self._remember_filter_config(project.id, self.filter_config(project.id))
        return self.project(project.id)

    def project(self, project_id: str) -> Project:
        return self._entry(project_id)[0]

    def reader(self, project_id: str) -> DatasetAdapter:
        return self._entry(project_id)[1]

    def report_signals(self, project_id: str, episode_index: int) -> ReportSignals:
        return build_report_signals(self.reader(project_id), episode_index)

    def export_dataset(
        self,
        project_id: str,
        format_id: str,
        episode_indexes: list[int],
        output_dir: str | None = None,
    ) -> ExportResult:
        output_base = self.artifact_root
        if output_dir and output_dir.strip():
            output_base = Path(_normalize_filesystem_path(output_dir)).expanduser().resolve()
            if output_base.exists() and not output_base.is_dir():
                raise ValueError("Export output directory must be a directory")

        output = output_base / f"{project_id}-{format_id}"
        if len(episode_indexes) == 1:
            output = output_base / f"{project_id}-episode-{episode_indexes[0]:06d}-{format_id}"
        return self._formats.export_dataset(
            adapter=self.reader(project_id),
            target_format=format_id,
            episode_indexes=episode_indexes,
            output_root=output,
        )

    def project_artifact_dir(self, project_id: str) -> Path:
        return self.artifact_root / "projects" / project_id

    def cleaning_summary(self, project_id: str) -> CleaningSummary:
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        existing = self._cleaning_store(project_id).load()
        if existing is not None and existing.scorer_version == self._scorer.scorer_version:
            return existing
        summary = build_summary(
            episodes,
            [],
            CleaningConfig(),
            self._scorer.scorer_version,
        )
        if existing is not None:
            return summary.model_copy(
                update={
                    "requires_rerun": True,
                    "previous_scorer_version": existing.scorer_version,
                }
            )
        return summary

    def vlm_settings(self, project_id: str) -> VlmSettings:
        self.project(project_id)
        return self._vlm_settings_store(project_id).load()

    def update_vlm_settings(self, project_id: str, settings: VlmSettings) -> VlmSettings:
        self.project(project_id)
        if "api_key" not in settings.model_fields_set:
            existing = self._vlm_settings_store(project_id).load()
            settings = settings.model_copy(update={"api_key": existing.api_key})
        return self._vlm_settings_store(project_id).save(settings)

    def run_cleaning(
        self,
        project_id: str,
        config: CleaningConfig,
        on_progress: ProgressCallback | None = None,
        filter_summary: FilterSummary | None = None,
    ) -> CleaningRun:
        episode_indexes = getattr(config, "episode_indexes", None)
        scoring_config = CleaningConfig.model_validate(
            config.model_dump(exclude={"episode_indexes"})
        )
        if config.review_threshold > config.pass_threshold:
            raise ValueError("review_threshold must be less than or equal to pass_threshold")
        if "vlm" in config.model_fields_set:
            scoring_config = scoring_config.model_copy(
                update={"vlm": self.update_vlm_settings(project_id, config.vlm)}
            )
        else:
            scoring_config = scoring_config.model_copy(update={"vlm": self.vlm_settings(project_id)})
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        if episode_indexes is not None:
            valid_indexes = {episode.episode_index for episode in episodes}
            invalid_indexes = sorted(set(episode_indexes) - valid_indexes)
            if invalid_indexes:
                raise ValueError(f"Episode indexes not found: {invalid_indexes}")
        previous = self._cleaning_store(project_id).load()
        previous_manual = {
            result.episode_index: result
            for result in (previous.results if previous else [])
            if result.source == "manual"
        }
        if filter_summary is None:
            filter_summary = self.run_filters(
                project_id,
                enabled_filter_stages=scoring_config.enabled_filter_stages,
            ).summary
        results = self._scorer.score_dataset(
            reader,
            scoring_config,
            episode_indexes,
            on_progress=on_progress,
            filter_summary=filter_summary,
        )
        if not config.overwrite_manual:
            results = [
                result.model_copy(
                    update={
                        "status": previous_manual[result.episode_index].status,
                        "source": "manual",
                        "review_note": previous_manual[result.episode_index].review_note,
                        "updated_at": previous_manual[result.episode_index].updated_at,
                    }
                )
                if result.episode_index in previous_manual
                else result
                for result in results
            ]
        if episode_indexes is not None and previous is not None:
            by_index = {result.episode_index: result for result in previous.results}
            by_index.update({result.episode_index: result for result in results})
            results = list(by_index.values())
        summary = build_summary(episodes, results, scoring_config, self._scorer.scorer_version)
        self._cleaning_store(project_id).save(summary)
        return CleaningRun(run_id=uuid4().hex[:12], status="succeeded", summary=summary)

    def run_pipeline(
        self,
        project_id: str,
        config: CleaningConfig,
        cleaning_progress: ProgressCallback | None = None,
        filters_progress: ProgressCallback | None = None,
    ) -> tuple[CleaningRun, FilterRun]:
        episode_indexes = getattr(config, "episode_indexes", None)
        filters = self.run_filters(
            project_id,
            on_progress=filters_progress,
            enabled_filter_stages=config.enabled_filter_stages,
            episode_indexes=episode_indexes,
        )
        cleaning = self.run_cleaning(
            project_id,
            config,
            on_progress=cleaning_progress,
            filter_summary=filters.summary,
        )
        return cleaning, filters

    def update_episode_decision(
        self,
        project_id: str,
        episode_index: int,
        request: EpisodeDecisionRequest,
    ) -> EpisodeQualityResult:
        reader = self.reader(project_id)
        reader.episode(episode_index)
        summary = self.cleaning_summary(project_id)
        by_index = {result.episode_index: result for result in summary.results}
        existing = by_index[episode_index]
        updated = existing.model_copy(
            update={
                "status": request.status,
                "source": "manual",
                "review_note": request.note,
                "updated_at": utc_now(),
            }
        )
        by_index[episode_index] = updated
        new_summary = build_summary(
            reader.list_episodes(),
            list(by_index.values()),
            summary.config,
            summary.scorer_version,
        )
        self._cleaning_store(project_id).save(new_summary)
        return updated

    def run_filters(
        self,
        project_id: str,
        on_progress: ProgressCallback | None = None,
        enabled_filter_stages: list[str] | None = None,
        episode_indexes: list[int] | None = None,
    ) -> FilterRun:
        self.project(project_id)
        filter_config = self.filter_config(project_id)
        if enabled_filter_stages is not None:
            filter_config = filter_config.model_copy(
                update={"enabled_filter_stages": enabled_filter_stages}
            )
        cohort_baseline = (
            self._load_visual_quality_baseline(project_id, filter_config)
            if episode_indexes is not None
            else []
        )
        result = DatasetFilterService(
            self.reader(project_id),
            filter_config,
            cohort_baseline=cohort_baseline,
        ).run(
            on_progress=on_progress,
            episode_indexes=episode_indexes,
        )
        if episode_indexes is None:
            self._filter_summary_store(project_id).write_text(
                result.summary.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if "visual_quality" in filter_config.enabled_filter_stages:
                self._save_visual_quality_artifacts(project_id, filter_config, result)
        return result

    def filter_summary(self, project_id: str) -> FilterSummary:
        self.project(project_id)
        store = self._filter_summary_store(project_id)
        if store.is_file():
            return FilterSummary.model_validate_json(store.read_text())
        return self._filter_service(project_id).summary()

    def filter_detail(self, project_id: str, stage_id: str, episode_index: int) -> FilterDetail:
        self.reader(project_id).episode(episode_index)
        allowed = {
            "visual_quality",
            "sudden_change",
            "state_action_alignment",
            "extreme_value",
            "kinematic_consistency",
            "orientation_alignment",
            "metadata_completeness",
        }
        if stage_id not in allowed:
            raise KeyError(f"Filter stage {stage_id!r} not found")
        if stage_id == "visual_quality":
            artifact = self._load_visual_quality_artifact(project_id, episode_index)
            if artifact is not None:
                return artifact
        if stage_id == "kinematic_consistency":
            self._invalidate_active_kinematics_if_stale(project_id)
        detail = self._filter_service(project_id).detail(stage_id, episode_index)  # type: ignore[arg-type]
        if stage_id == "kinematic_consistency":
            accepted_id = self.filter_config(project_id).kinematics.accepted_proposal_id
            accepted = next(
                (item for item in self.kinematic_proposals(project_id) if item.id == accepted_id),
                None,
            )
            if accepted is not None:
                parameters = {
                    **detail.parameters,
                    "accepted_proposal": {
                        "id": accepted.id,
                        "fingerprint": accepted.fingerprint,
                        "confidence": accepted.confidence,
                        "evidence": accepted.evidence,
                        "warnings": accepted.warnings,
                        "before_held_out": accepted.before_held_out.model_dump(mode="json"),
                        "after_held_out": accepted.after_held_out.model_dump(mode="json"),
                    },
                }
                detail = detail.model_copy(update={"parameters": parameters})
        return detail

    def visual_quality_frame(
        self,
        project_id: str,
        episode_index: int,
        camera: str,
        frame: int,
        width: int,
    ) -> bytes:
        reader = self.reader(project_id)
        episode = reader.episode(episode_index)
        if camera not in episode.video_files:
            raise KeyError(f"Camera {camera!r} not found")
        if frame < 0 or frame >= episode.length:
            raise ValueError(
                f"Frame {frame} is outside episode range 0-{max(episode.length - 1, 0)}"
            )
        video_path = reader.root / episode.video_files[camera]
        if not video_path.is_file():
            raise KeyError(f"Video for camera {camera!r} not found")
        timestamp = float(episode.video_start_seconds.get(camera, 0.0))
        timestamp += frame / float(reader.metadata().fps)
        return extract_video_frame_jpeg(video_path, timestamp, width)

    def filter_config(self, project_id: str) -> FilterConfig:
        self.project(project_id)
        store = self._filter_config_store(project_id)
        if store.is_file():
            return FilterConfig.model_validate_json(store.read_text())
        config = FilterConfig(
            gripper_indices=infer_gripper_indices(self.reader(project_id)),
            kinematics=self._infer_kinematics_config(project_id),
        )
        self._save_filter_config(project_id, config)
        return config

    def update_filter_config(self, project_id: str, patch: FilterConfigPatch) -> FilterConfig:
        config = self.filter_config(project_id)
        update = {}
        mapping_fields = {
            "urdf_path",
            "end_effector_link",
            "joint_names",
            "joint_state_indices",
            "eef_position_indices",
            "eef_orientation_indices",
            "eef_orientation_representation",
            "profile",
        }
        mapping_changed = patch.kinematics is not None and bool(
            mapping_fields.intersection(patch.kinematics.model_fields_set)
        )
        if patch.gripper_indices is not None:
            update["gripper_indices"] = patch.gripper_indices
        if patch.kinematics is not None:
            kinematic_update = patch.kinematics.model_dump(exclude_unset=True)
            if mapping_changed and "profile" not in kinematic_update:
                kinematic_update.update({"profile": None, "accepted_proposal_id": None})
            update["kinematics"] = config.kinematics.model_copy(update=kinematic_update)
        if patch.visual_quality is not None:
            update["visual_quality"] = config.visual_quality.model_copy(
                update=patch.visual_quality.model_dump(exclude_unset=True)
            )
        if patch.time_sync is not None:
            update["time_sync"] = config.time_sync.model_copy(
                update=patch.time_sync.model_dump(exclude_unset=True)
            )
        if patch.metadata_completeness is not None:
            update["metadata_completeness"] = config.metadata_completeness.model_copy(
                update=patch.metadata_completeness.model_dump(exclude_unset=True)
            )
        updated = config.model_copy(update=update)
        self._save_filter_config(project_id, updated)
        self._remember_filter_config(project_id, updated)
        if mapping_changed:
            self._mark_kinematic_proposals_stale(project_id)
        return updated

    def save_filter_urdf(self, project_id: str, filename: str, content: bytes) -> dict[str, str]:
        self.project(project_id)
        safe_name = Path(filename).name or "robot.urdf"
        if not safe_name.endswith(".urdf"):
            raise ValueError("URDF filename must end with .urdf")
        output = self.project_artifact_dir(project_id) / "filters" / "urdf" / safe_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        config = self.filter_config(project_id)
        updated_kinematics = config.kinematics.model_copy(
            update={
                "urdf_path": str(output),
                "profile": None,
                "accepted_proposal_id": None,
            }
        )
        updated = config.model_copy(update={"kinematics": updated_kinematics})
        self._save_filter_config(project_id, updated)
        self._remember_filter_config(project_id, updated)
        self._mark_kinematic_proposals_stale(project_id)
        return {"filename": safe_name, "path": str(output)}

    def discover_kinematic_corrections(
        self,
        project_id: str,
        request: KinematicDiscoveryRequest,
    ) -> CorrectionProposal | DiscoveryRefusal:
        config = self.filter_config(project_id).kinematics
        if not config.urdf_path:
            raise ValueError("Upload a URDF before running kinematic discovery")
        reader = self.reader(project_id)
        available = [episode.episode_index for episode in reader.list_episodes()]
        indexes = request.episode_indexes or available[: min(len(available), 4)]
        if not indexes:
            raise ValueError("The dataset has no episodes to inspect")
        invalid = sorted(set(indexes) - set(available))
        if invalid:
            raise ValueError(f"Unknown episode indexes: {invalid}")
        rows = [
            frame.observation_state
            for episode_index in indexes
            for frame in reader.read_episode_frames(episode_index)
        ]
        states = np.asarray(rows, dtype=np.float64)
        feature_names = self._state_feature_names(project_id, states.shape[1])
        candidates = discover_profile(
            config.urdf_path,
            DiscoveryRequest(
                states=states,
                feature_names=feature_names,
                max_candidates=request.max_candidates,
                sample_count=request.sample_count,
                max_joint_mappings=request.max_joint_mappings,
            ),
        )
        result = fit_correction(
            states,
            candidates,
            CorrectionOptions(
                allow_base=request.allow_base,
                allow_tcp=request.allow_tcp,
                allow_joint_sign=request.allow_joint_sign,
                allow_joint_offset=request.allow_joint_offset,
                min_candidate_margin=request.min_candidate_margin,
                position_tolerance=config.position_tolerance,
                orientation_tolerance=config.orientation_tolerance,
            ),
        )
        if isinstance(result, CorrectionProposal):
            fingerprint = self._kinematic_source_fingerprint(
                Path(config.urdf_path), states, feature_names
            )
            profile = result.profile.model_copy(update={"fingerprint": fingerprint})
            proposal_id = hashlib.sha256(
                (profile.model_dump_json() + fingerprint).encode("utf-8")
            ).hexdigest()[:16]
            result = result.model_copy(
                update={
                    "id": proposal_id,
                    "fingerprint": fingerprint,
                    "profile": profile,
                    "source_episode_indexes": indexes,
                }
            )
            proposals = [item for item in self.kinematic_proposals(project_id) if item.id != result.id]
            proposals.append(result)
            self._save_kinematic_proposals(project_id, proposals)
        return result

    def kinematic_proposals(self, project_id: str) -> list[CorrectionProposal]:
        self.project(project_id)
        path = self._kinematic_proposals_store(project_id)
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [CorrectionProposal.model_validate(item) for item in payload]

    def decide_kinematic_proposal(
        self,
        project_id: str,
        proposal_id: str,
        request: KinematicProposalDecision,
    ) -> CorrectionProposal:
        proposals = self.kinematic_proposals(project_id)
        selected = next((item for item in proposals if item.id == proposal_id), None)
        if selected is None:
            raise KeyError(f"Kinematic proposal {proposal_id!r} not found")
        if not self._kinematic_proposal_is_current(project_id, selected):
            proposals = [
                item.model_copy(update={"status": "stale"})
                if item.id == proposal_id else item
                for item in proposals
            ]
            self._save_kinematic_proposals(project_id, proposals)
            selected = next(item for item in proposals if item.id == proposal_id)
            self._deactivate_kinematic_profile(project_id, proposal_id)
        if selected.status == "stale" and request.decision == "accept":
            raise ValueError("A stale kinematic proposal cannot be accepted; run discovery again")

        if request.decision == "accept":
            proposals = [
                item.model_copy(
                    update={"status": "accepted" if item.id == proposal_id else (
                        "rejected" if item.status == "accepted" else item.status
                    )}
                )
                for item in proposals
            ]
            config = self.filter_config(project_id)
            selected = next(item for item in proposals if item.id == proposal_id)
            kinematics = config.kinematics.model_copy(
                update={
                    "profile": selected.profile,
                    "accepted_proposal_id": selected.id,
                    "urdf_path": selected.profile.urdf_path,
                }
            )
            updated = config.model_copy(update={"kinematics": kinematics})
            self._save_filter_config(project_id, updated)
            self._remember_filter_config(project_id, updated)
        elif request.decision == "reject":
            proposals = [
                item.model_copy(update={"status": "rejected"})
                if item.id == proposal_id else item
                for item in proposals
            ]
        else:
            proposals = [
                item.model_copy(update={"status": "proposed"})
                if item.id == proposal_id else item
                for item in proposals
            ]
            config = self.filter_config(project_id)
            if config.kinematics.accepted_proposal_id == proposal_id:
                kinematics = config.kinematics.model_copy(
                    update={"profile": None, "accepted_proposal_id": None}
                )
                updated = config.model_copy(update={"kinematics": kinematics})
                self._save_filter_config(project_id, updated)
                self._remember_filter_config(project_id, updated)
        self._save_kinematic_proposals(project_id, proposals)
        return next(item for item in proposals if item.id == proposal_id)

    def _entry(self, project_id: str) -> tuple[Project, DatasetAdapter]:
        try:
            return self._projects[project_id]
        except KeyError as error:
            raise KeyError(f"Project {project_id!r} not found") from error

    @staticmethod
    def _normalize_user_path(path: str) -> str:
        normalized = path.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized

    def _cleaning_store(self, project_id: str) -> CleaningStateStore:
        return CleaningStateStore(self.project_artifact_dir(project_id) / "cleaning_state.json")

    def _vlm_settings_store(self, project_id: str) -> VlmSettingsStore:
        return VlmSettingsStore(self.project_artifact_dir(project_id) / "vlm_settings.json")

    def _filter_service(self, project_id: str) -> DatasetFilterService:
        return DatasetFilterService(self.reader(project_id), self.filter_config(project_id))

    def _filter_config_store(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filter_config.json"

    def _filter_summary_store(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filter_summary.json"

    def _kinematic_proposals_store(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filters" / "kinematic_proposals.json"

    def _save_kinematic_proposals(
        self,
        project_id: str,
        proposals: list[CorrectionProposal],
    ) -> None:
        path = self._kinematic_proposals_store(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([item.model_dump(mode="json") for item in proposals], indent=2),
            encoding="utf-8",
        )

    def _mark_kinematic_proposals_stale(self, project_id: str) -> None:
        proposals = self.kinematic_proposals(project_id)
        if not proposals:
            return
        self._save_kinematic_proposals(
            project_id,
            [item.model_copy(update={"status": "stale"}) for item in proposals],
        )

    def _invalidate_active_kinematics_if_stale(self, project_id: str) -> None:
        config = self.filter_config(project_id)
        proposal_id = config.kinematics.accepted_proposal_id
        if not proposal_id:
            return
        proposals = self.kinematic_proposals(project_id)
        proposal = next((item for item in proposals if item.id == proposal_id), None)
        if proposal is not None and self._kinematic_proposal_is_current(project_id, proposal):
            return
        if proposal is not None and proposal.status != "stale":
            self._save_kinematic_proposals(
                project_id,
                [
                    item.model_copy(update={"status": "stale"})
                    if item.id == proposal_id else item
                    for item in proposals
                ],
            )
        self._deactivate_kinematic_profile(project_id, proposal_id)

    def _kinematic_proposal_is_current(
        self,
        project_id: str,
        proposal: CorrectionProposal,
    ) -> bool:
        if proposal.status == "stale":
            return False
        if not proposal.source_episode_indexes:
            return True
        try:
            current_states = np.asarray(
                [
                    frame.observation_state
                    for episode_index in proposal.source_episode_indexes
                    for frame in self.reader(project_id).read_episode_frames(episode_index)
                ],
                dtype=np.float64,
            )
            current_names = self._state_feature_names(project_id, current_states.shape[1])
            current_fingerprint = self._kinematic_source_fingerprint(
                Path(proposal.profile.urdf_path), current_states, current_names
            )
        except (KeyError, OSError, ValueError):
            return False
        return current_fingerprint == proposal.fingerprint

    def _deactivate_kinematic_profile(self, project_id: str, proposal_id: str) -> None:
        config = self.filter_config(project_id)
        if config.kinematics.accepted_proposal_id != proposal_id:
            return
        kinematics = config.kinematics.model_copy(
            update={"profile": None, "accepted_proposal_id": None}
        )
        updated = config.model_copy(update={"kinematics": kinematics})
        self._save_filter_config(project_id, updated)
        self._remember_filter_config(project_id, updated)

    def _state_feature_names(self, project_id: str, width: int) -> list[str]:
        feature = self.project(project_id).dataset.features.get("observation.state", {})
        names = feature.get("names", {}) if isinstance(feature, dict) else {}
        if isinstance(names, dict):
            flattened = [str(name) for group in names.values() if isinstance(group, list) for name in group]
        elif isinstance(names, list):
            flattened = [str(name) for name in names]
        else:
            flattened = []
        return flattened if len(flattened) == width else [f"state_{index}" for index in range(width)]

    @staticmethod
    def _kinematic_source_fingerprint(
        urdf_path: Path,
        states: np.ndarray,
        feature_names: list[str],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(urdf_path.read_bytes())
        contiguous = np.ascontiguousarray(states, dtype=np.float64)
        digest.update(repr(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
        digest.update("\0".join(feature_names).encode("utf-8"))
        return digest.hexdigest()

    def _visual_quality_artifact_dir(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filters" / "visual-quality"

    def _save_visual_quality_artifacts(
        self,
        project_id: str,
        config: FilterConfig,
        result: FilterRun,
    ) -> None:
        output_dir = self._visual_quality_artifact_dir(project_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._visual_quality_fingerprint(project_id, config)
        checkpoint = checkpoint_file_info(config.visual_quality)
        baseline: list[dict[str, object]] = []
        for episode_index, detail in result.visual_quality_details.items():
            visual = detail.visual_quality
            if visual is None:
                continue
            visual = visual.model_copy(update={"artifact_fingerprint": fingerprint})
            persisted = detail.model_copy(update={"visual_quality": visual})
            payload = {
                "fingerprint": fingerprint,
                "checkpoint": checkpoint,
                "detail": persisted.model_dump(mode="json"),
            }
            self._atomic_write_json(
                output_dir / f"episode-{episode_index:06d}.json",
                payload,
            )
            for score in visual.camera_scores:
                if score.technical_score is not None:
                    baseline.append(
                        {
                            "episode_index": episode_index,
                            "camera": score.camera,
                            "technical_score": score.technical_score,
                            "error_code": None,
                        }
                    )
        self._atomic_write_json(
            output_dir / "baseline.json",
            {
                "fingerprint": fingerprint,
                "checkpoint": checkpoint,
                "scores": baseline,
            },
        )
        expected = {
            output_dir / f"episode-{episode_index:06d}.json"
            for episode_index in result.visual_quality_details
        }
        for stale in output_dir.glob("episode-*.json"):
            if stale not in expected:
                stale.unlink()

    def _load_visual_quality_artifact(
        self,
        project_id: str,
        episode_index: int,
    ) -> FilterDetail | None:
        config = self.filter_config(project_id)
        path = self._visual_quality_artifact_dir(project_id) / f"episode-{episode_index:06d}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._visual_quality_fingerprint(project_id, config):
                return None
            return FilterDetail.model_validate(payload["detail"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_visual_quality_baseline(
        self,
        project_id: str,
        config: FilterConfig,
    ) -> list[VideoTechnicalScore]:
        path = self._visual_quality_artifact_dir(project_id) / "baseline.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._visual_quality_fingerprint(project_id, config):
                return []
            return [VideoTechnicalScore(**item) for item in payload.get("scores", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _visual_quality_fingerprint(self, project_id: str, config: FilterConfig) -> str:
        reader = self.reader(project_id)
        episodes = []
        for episode in reader.list_episodes():
            videos = []
            for camera, relative_path in sorted((episode.video_files or {}).items()):
                videos.append(
                    {
                        "camera": camera,
                        "path": relative_path,
                        "start": episode.video_start_seconds.get(camera),
                        "end": episode.video_end_seconds.get(camera),
                        "file": self._source_file_info(reader.root / relative_path),
                    }
                )
            episodes.append(
                {
                    "episode_index": episode.episode_index,
                    "length": episode.length,
                    "duration_seconds": episode.duration_seconds,
                    "data_file": episode.data_file,
                    "data_file_info": self._source_file_info(reader.root / episode.data_file),
                    "videos": videos,
                }
            )
        payload = {
            "visual_config": visual_quality_fingerprint(config.visual_quality),
            "dataset": reader.metadata().model_dump(mode="json"),
            "episodes": episodes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _source_file_info(path: Path) -> dict[str, int | None]:
        if not path.is_file():
            return {"size": None, "mtime_ns": None}
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _infer_kinematics_config(self, project_id: str) -> FilterKinematicsConfig:
        metadata = self.project(project_id).dataset
        state_feature = metadata.features.get("observation.state", {})
        names = state_feature.get("names", {})
        motor_names = names.get("motors", []) if isinstance(names, dict) else []
        if (
            metadata.format == "robomimic_hdf5"
            and metadata.robot_type.lower() == "panda"
            and len(motor_names) >= 10
            and motor_names[:7] == [f"panda_joint{index}" for index in range(1, 8)]
        ):
            urdf_path = Path(__file__).resolve().parents[3] / "data" / "urdf" / "panda_bullet.urdf"
            if urdf_path.is_file():
                return FilterKinematicsConfig(
                    urdf_path=str(urdf_path),
                    end_effector_link="panda_grasptarget",
                    joint_names=[f"panda_joint{index}" for index in range(1, 8)],
                    joint_state_indices=list(range(7)),
                    eef_position_indices=[7, 8, 9],
                    position_tolerance=0.02,
                    resolve_tcp_offset=True,
                )
        return FilterKinematicsConfig()

    def _save_filter_config(self, project_id: str, config: FilterConfig) -> None:
        path = self._filter_config_store(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _remember_filter_config(self, project_id: str, config: FilterConfig) -> None:
        project, reader = self._entry(project_id)
        self._projects[project_id] = (project.model_copy(update={"filter_config": config}), reader)
