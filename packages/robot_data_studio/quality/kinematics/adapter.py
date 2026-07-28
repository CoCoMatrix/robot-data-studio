from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary

from .engine import PinocchioKinematics
from .models import KinematicProfile
from .pose import encode_rotations


class CorrectedDatasetAdapter:
    """Read-only view that regenerates mapped derived EEF poses from corrected FK."""

    def __init__(
        self,
        source: DatasetAdapter,
        accepted_profile: KinematicProfile,
        correction_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.profile = accepted_profile
        self.root = Path(source.root)
        self._engine = PinocchioKinematics(accepted_profile.urdf_path)
        self._manifest = deepcopy(correction_manifest or {})

    @classmethod
    def probe(cls, root: str | Path) -> bool:
        return False

    def metadata(self) -> DatasetMetadata:
        return self.source.metadata()

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        return self.source.list_episodes(limit=limit)

    def episode(self, episode_index: int) -> EpisodeSummary:
        return self.source.episode(episode_index)

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = self.source.read_episode_frames(episode_index)
        if not frames:
            return []
        states = np.asarray([frame.observation_state for frame in frames], dtype=np.float64)
        corrected = states.copy()
        for chain in self.profile.chains:
            transforms = self._engine.predict_chain(states, chain)
            pose = chain.logged_pose
            corrected[:, pose.position_indices] = transforms[:, :3, 3]
            corrected[:, pose.orientation_indices] = encode_rotations(
                transforms[:, :3, :3],
                pose.orientation_representation,
            )
        return [
            frame.model_copy(update={"observation_state": corrected[index].tolist()})
            for index, frame in enumerate(frames)
        ]

    def kinematic_correction_manifest(self) -> dict[str, Any]:
        return deepcopy(self._manifest)
