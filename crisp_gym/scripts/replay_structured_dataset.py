"""Replay one structured LeRobot episode on a CRISP follower robot.

Structured bilateral recordings store Cartesian actions as home-relative targets
``[x, y, z, rx, ry, rz, gripper]``.  They must go through the same
``CrispCartesianAdapter`` used during recording; passing them to ``env.step``
would incorrectly integrate every target as a per-frame delta.

The command is preflight-only unless ``--execute`` is supplied.  Execution homes
the follower using its selected environment config, captures that pose as the
anchor, and replays targets at the dataset frame rate.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from crisp_gym.bilateral.bilateral_config import make_bilateral_config
from crisp_gym.bilateral.crisp_adapter import CrispCartesianAdapter, pose_to_vec
from crisp_gym.bilateral.runtime import _base_rotation
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)

_CARTESIAN_ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def _local_dataset_root(repo_id: str) -> Path | None:
    """Return the recorder's local dataset directory when it exists."""
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME

    path = Path(HF_LEROBOT_HOME) / repo_id
    return path if path.exists() else None


def _load_episode(
    repo_id: str, episode: int, root: str | None
) -> tuple[np.ndarray, list[str], int]:
    """Load actions, action names and FPS for one local or Hub episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(root).expanduser() if root else _local_dataset_root(repo_id)
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode])
    actions = np.asarray(dataset.hf_dataset.select_columns("action")[:]["action"], dtype=np.float64)
    names = list(dataset.features["action"].get("names") or [])
    return actions, names, int(dataset.fps)


def _validate_actions(
    actions: np.ndarray,
    names: list[str],
    max_start_translation: float,
    max_start_rotation: float,
) -> None:
    """Reject incompatible or discontinuous-at-start Cartesian trajectories."""
    if names != _CARTESIAN_ACTION_NAMES:
        raise ValueError(
            f"Expected structured Cartesian action names {_CARTESIAN_ACTION_NAMES}, got {names}."
        )
    if actions.ndim != 2 or actions.shape[1] != len(_CARTESIAN_ACTION_NAMES):
        raise ValueError(f"Expected action shape (frames, 7), got {actions.shape}.")
    if len(actions) == 0:
        raise ValueError("Episode contains no actions.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Episode actions contain NaN or infinity.")

    start_translation = float(np.linalg.norm(actions[0, :3]))
    start_rotation = float(np.linalg.norm(actions[0, 3:6]))
    if start_translation > max_start_translation or start_rotation > max_start_rotation:
        raise ValueError(
            "Episode does not begin near its recorded home anchor: "
            f"translation={start_translation:.4f}m, rotation={start_rotation:.4f}rad. "
            "Refusing a startup jump."
        )


def _log_summary(actions: np.ndarray, fps: int, speed: float) -> None:
    """Log trajectory duration, extent and largest frame-to-frame change."""
    trans = np.linalg.norm(actions[:, :3], axis=1)
    rot = np.linalg.norm(actions[:, 3:6], axis=1)
    dtrans = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    drot = np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=1)
    logger.info(
        "Frames: %d at %d FPS (%.2fs; replay %.2fs at %.2fx)",
        len(actions),
        fps,
        len(actions) / fps,
        len(actions) / (fps * speed),
        speed,
    )
    logger.info("Target extent: translation %.4fm, rotation %.4frad", trans.max(), rot.max())
    logger.info(
        "Largest frame step: translation %.4fm, rotation %.4frad",
        dtrans.max(initial=0.0),
        drot.max(initial=0.0),
    )
    logger.info("First action: %s", np.array2string(actions[0], precision=5))
    logger.info("Last action:  %s", np.array2string(actions[-1], precision=5))


def _hold_current_pose(robot) -> None:  # noqa: ANN001
    """Clear feed-forward force and latch the follower's current pose."""
    try:
        robot.set_target_wrench()
        robot.set_target(pose=robot.end_effector_pose)
        time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not hold follower during replay shutdown: %s", exc)


def _qualified_topic(namespace: str, topic: str) -> str:
    """Resolve a crisp_py relative topic exactly as its namespaced node does."""
    if topic.startswith("/"):
        return topic
    prefix = namespace.strip("/")
    return f"/{prefix}/{topic}" if prefix else f"/{topic}"


def _execute_replay(args: argparse.Namespace, actions: np.ndarray, fps: int) -> None:
    """Home the follower and stream the validated target sequence."""
    scheme = make_bilateral_config(args.teleop_scheme)
    if scheme.mode != "cartesian":
        raise ValueError(f"Replay currently supports Cartesian schemes, got {scheme.mode!r}.")

    follower_config = args.follower_config or scheme.follower_env_config
    follower_namespace = args.follower_namespace or scheme.follower_namespace
    env = None
    try:
        env = make_env(follower_config, control_type="cartesian", namespace=follower_namespace)
        env.wait_until_ready()
        target_topic = _qualified_topic(follower_namespace, env.robot.config.target_pose_topic)
        publisher_count = env.robot.node.count_publishers(target_topic)
        if publisher_count != 1:
            raise RuntimeError(
                f"Expected replay to be the sole publisher on {target_topic}, found "
                f"{publisher_count}. Stop teleop/recording clients before replay."
            )
        logger.info("Homing follower to '%s' before replay...", follower_config)
        env.home()
        env.reset()
        env.robot.wait_until_ready()

        adapter = CrispCartesianAdapter(
            env.robot,
            pose_to_vec(env.robot.end_effector_pose),
            base_to_common=_base_rotation(scheme.follower_base_to_common_quat),
        )
        env.robot.set_target_wrench()

        period = 1.0 / (fps * args.speed)
        next_tick = time.perf_counter()
        logger.info("Replaying episode now. Ctrl-C holds the current follower pose.")
        for index, action in enumerate(actions):
            adapter.set_target_position(action[:6])
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif index and index % fps == 0:
                logger.warning("Replay is behind schedule by %.1fms", -remaining * 1e3)
        logger.info("Replay complete; holding final pose.")
    finally:
        if env is not None:
            _hold_current_pose(env.robot)
            env.close()


def main() -> None:
    """Validate an episode and optionally replay it on the follower."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--root", default=None, help="Optional local LeRobot dataset root.")
    p.add_argument("--teleop-scheme", default="pf_fr3")
    p.add_argument("--follower-config", default=None)
    p.add_argument("--follower-namespace", default=None)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier; 0.25 is a cautious first hardware test.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Replay only this many frames from the beginning.",
    )
    p.add_argument("--max-start-translation", type=float, default=0.01)
    p.add_argument("--max-start-rotation", type=float, default=0.10)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually command hardware. Without this flag only preflight runs.",
    )
    p.add_argument("--yes", action="store_true", help="Skip the interactive hardware confirmation.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    setup_logging(level=args.log_level)

    if args.speed <= 0.0:
        p.error("--speed must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")

    actions, names, fps = _load_episode(args.repo_id, args.episode, args.root)
    if args.max_frames is not None:
        actions = actions[: args.max_frames]
    _validate_actions(actions, names, args.max_start_translation, args.max_start_rotation)
    _log_summary(actions, fps, args.speed)

    if not args.execute:
        logger.info("Preflight passed. Add --execute to enable robot motion.")
        return
    if not args.yes:
        answer = input(
            "Follower will HOME and replay this trajectory. Clear the workspace, then type REPLAY: "
        )
        if answer.strip() != "REPLAY":
            logger.info("Replay cancelled.")
            return

    try:
        _execute_replay(args, actions, fps)
    except KeyboardInterrupt:
        logger.info("Replay interrupted by operator.")


if __name__ == "__main__":
    main()
