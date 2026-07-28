from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from robot_data_studio.quality.qwen_filters import (
    ExtremeValueConfig,
    KinematicsConfig,
    QwenFilterConfig,
    QwenManipDataFilter,
    StateActionAlignmentConfig,
    SuddenChangeConfig,
)
from robot_data_studio.quality.kinematics import (
    CorrectionOptions,
    CorrectionProposal,
    DiscoveryRefusal,
    DiscoveryRequest,
    KinematicProfile,
    discover_profile,
    fit_correction,
)
from robot_data_studio.lerobot.reader import LeRobotDatasetReader

import numpy as np


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_config(
    *,
    gripper_indices: str | None = None,
    urdf: Path | None = None,
    end_effector_link: str | None = None,
    joint_names: str | None = None,
    joint_state_indices: str | None = None,
    eef_position_indices: str | None = None,
    max_lag: int = 8,
    sudden_min_threshold: float = 0.02,
    profile: Path | None = None,
) -> QwenFilterConfig:
    grippers = parse_int_list(gripper_indices)
    return QwenFilterConfig(
        gripper_indices=grippers,
        sudden_change=SuddenChangeConfig(
            residual_min_threshold=sudden_min_threshold,
            acceleration_min_threshold=sudden_min_threshold,
            jerk_min_threshold=sudden_min_threshold,
        ),
        state_action_alignment=StateActionAlignmentConfig(
            max_lag=max_lag,
            ignored_indices=grippers,
        ),
        extreme_values=ExtremeValueConfig(gripper_indices=grippers),
        kinematics=KinematicsConfig(
            urdf_path=urdf,
            end_effector_link=end_effector_link,
            joint_names=parse_str_list(joint_names),
            joint_state_indices=parse_int_list(joint_state_indices),
            eef_position_indices=parse_int_list(eef_position_indices),
            profile=(
                KinematicProfile.model_validate_json(profile.read_text(encoding="utf-8"))
                if profile is not None
                else None
            ),
        ),
    )


def discover_kinematic_proposal(
    states: np.ndarray,
    feature_names: list[str],
    urdf: Path,
    *,
    max_candidates: int = 20,
    sample_count: int = 64,
    options: CorrectionOptions | None = None,
) -> CorrectionProposal | DiscoveryRefusal:
    state_array = np.asarray(states, dtype=np.float64)
    candidates = discover_profile(
        urdf,
        DiscoveryRequest(
            states=state_array,
            feature_names=feature_names,
            max_candidates=max_candidates,
            sample_count=sample_count,
        ),
    )
    return fit_correction(state_array, candidates, options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Qwen-RobotManip S1-S5 state/action data filters on a LeRobot dataset."
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for a quick run.")
    parser.add_argument("--gripper-indices", default="", help="Comma-separated gripper dimensions to exempt.")
    parser.add_argument("--max-lag", type=int, default=8, help="Maximum lag for state-action alignment.")
    parser.add_argument(
        "--sudden-min-threshold",
        type=float,
        default=0.02,
        help="Minimum residual/acceleration/jerk threshold for S1.",
    )
    parser.add_argument("--urdf", type=Path, default=None, help="Optional URDF path for S4 FK checks.")
    parser.add_argument(
        "--discover-kinematics",
        action="store_true",
        help="Discover URDF links, joint/state fields, pose representation, and corrections.",
    )
    parser.add_argument(
        "--proposal-output",
        type=Path,
        default=None,
        help="Write the dynamic kinematic proposal/refusal JSON to this path.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Evaluate S4 with a Data Studio KinematicProfile JSON.",
    )
    parser.add_argument("--end-effector-link", default=None, help="URDF link name used as the EEF frame.")
    parser.add_argument("--joint-names", default="", help="Comma-separated URDF joint names in state order.")
    parser.add_argument(
        "--joint-state-indices",
        default="",
        help="Comma-separated observation.state indices matching --joint-names.",
    )
    parser.add_argument(
        "--eef-position-indices",
        default="",
        help="Comma-separated observation.state indices containing logged EEF xyz.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.discover_kinematics and args.proposal_output is None:
        parser.error("--discover-kinematics requires --proposal-output")
    config = build_config(
        gripper_indices=args.gripper_indices,
        urdf=args.urdf,
        end_effector_link=args.end_effector_link,
        joint_names=args.joint_names,
        joint_state_indices=args.joint_state_indices,
        eef_position_indices=args.eef_position_indices,
        max_lag=args.max_lag,
        sudden_min_threshold=args.sudden_min_threshold,
        profile=args.profile,
    )
    if args.discover_kinematics:
        urdf = args.urdf or (
            Path(config.kinematics.profile.urdf_path)
            if config.kinematics.profile is not None
            else None
        )
        if urdf is None:
            parser.error("--discover-kinematics requires --urdf or --profile")
        states, feature_names = _lerobot_discovery_states(
            args.dataset,
            max_episodes=args.max_episodes,
        )
        proposal = discover_kinematic_proposal(states, feature_names, urdf)
        args.proposal_output.parent.mkdir(parents=True, exist_ok=True)
        args.proposal_output.write_text(
            proposal.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    summary = QwenManipDataFilter(config).evaluate_lerobot_dataset(
        args.dataset,
        max_episodes=args.max_episodes,
    )
    payload = json.dumps(asdict(summary), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _lerobot_discovery_states(
    root: Path,
    *,
    max_episodes: int | None,
) -> tuple[np.ndarray, list[str]]:
    reader = LeRobotDatasetReader(root)
    episodes = reader.list_episodes(limit=max_episodes)
    rows = [
        frame.observation_state
        for episode in episodes
        for frame in reader.read_episode_frames(episode.episode_index)
    ]
    states = np.asarray(rows, dtype=np.float64)
    if states.ndim != 2 or not len(states):
        raise ValueError("The dataset has no observation.state rows for discovery")
    feature = reader.metadata().features.get("observation.state", {})
    names = feature.get("names", {}) if isinstance(feature, dict) else {}
    if isinstance(names, dict):
        flattened = [
            str(name)
            for group in names.values()
            if isinstance(group, list)
            for name in group
        ]
    elif isinstance(names, list):
        flattened = [str(name) for name in names]
    else:
        flattened = []
    if len(flattened) != states.shape[1]:
        flattened = [f"state_{index}" for index in range(states.shape[1])]
    return states, flattened


if __name__ == "__main__":
    raise SystemExit(main())
