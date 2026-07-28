from pathlib import Path

from robot_data_studio.projects.models import Project
from robot_data_studio.projects.service import ProjectService
from robot_data_studio.quality.filter_models import (
    FilterConfig,
    FilterDetail,
    FilterRun,
    FilterSummary,
    VisualCameraScore,
    VisualQualityDetail,
)
from tests.quality.test_filter_scoring import SharedVideoReader


def test_visual_artifact_round_trip_and_fingerprint_invalidation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    video = dataset / "videos" / "shared.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    reader = SharedVideoReader(dataset)
    service = ProjectService(tmp_path / "artifacts")
    service._projects["project"] = (Project(id="project", dataset=reader.metadata()), reader)
    config = FilterConfig()
    service._save_filter_config("project", config)
    detail = FilterDetail(
        stage_id="visual_quality",
        episode_index=0,
        title="Visual quality",
        status="passed",
        visual_quality=VisualQualityDetail(
            sampled_frame_count=4,
            camera_count=1,
            issue_sample_count=0,
            affected_camera_count=0,
            episode_frame_count=101,
            episode_duration_seconds=2.02,
            technical_score=0.8,
            camera_scores=[
                VisualCameraScore(
                    camera="cam_high",
                    technical_score=0.8,
                    percentile=None,
                    cohort_size=1,
                    model_status="scored",
                )
            ],
        ),
    )
    run = FilterRun(
        run_id="run",
        status="succeeded",
        summary=FilterSummary(
            dataset_path=str(reader.root),
            total_episodes=2,
            total_frames=202,
            stages=[],
            episodes=[],
        ),
        visual_quality_details={0: detail},
    )

    stale = service._visual_quality_artifact_dir("project") / "episode-999999.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}")

    service._save_visual_quality_artifacts("project", config, run)

    restored = service._load_visual_quality_artifact("project", 0)
    baseline = service._load_visual_quality_baseline("project", config)
    assert restored is not None
    assert restored.visual_quality is not None
    assert restored.visual_quality.artifact_fingerprint
    assert baseline[0].technical_score == 0.8
    assert not stale.exists()

    video.write_bytes(b"changed-video-content")
    assert service._load_visual_quality_artifact("project", 0) is None
    assert service._load_visual_quality_baseline("project", config) == []

    service._save_visual_quality_artifacts("project", config, run)

    changed = config.model_copy(
        update={"visual_quality": config.visual_quality.model_copy(update={"sample_fps": 3.0})}
    )
    service._save_filter_config("project", changed)
    assert service._load_visual_quality_artifact("project", 0) is None
    assert service._load_visual_quality_baseline("project", changed) == []
