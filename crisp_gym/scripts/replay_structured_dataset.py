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
import json
import logging
import time
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import String

from crisp_gym.bilateral.bilateral_config import make_bilateral_config
from crisp_gym.bilateral.crisp_adapter import CrispCartesianAdapter, pose_to_vec
from crisp_gym.bilateral.runtime import _base_rotation
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)

_CARTESIAN_ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def _next_replay_state(state: str, action: str) -> str:
    """Apply one recording-interface command to the replay state machine."""
    if action == "record":
        return {
            "is_waiting": "recording",
            "recording": "paused",
            "paused": "recording",
        }.get(state, state)
    if action == "exit" and state in {"is_waiting", "paused", "finished"}:
        return "exit"
    return state


class ReplayROSControl:
    """Expose replay through the recorder's status and transition topics."""

    def __init__(
        self,
        repo_id: str,
        episode: int,
        total_frames: int,
        status_topic: str,
        transition_topic: str,
    ) -> None:
        """Create the replay status publisher and transition subscriber."""
        self.repo_id = repo_id
        self.episode = episode
        self.total_frames = total_frames
        self.frames = 0
        self.fps_measured = 0.0
        self.state = "is_waiting"
        self.last_event = "follower homed; ready to replay"
        self.node = rclpy.create_node("replay_manager")
        self.publisher = self.node.create_publisher(String, status_topic, 10)
        self.node.create_subscription(String, transition_topic, self._on_transition, 10)
        self.node.create_timer(0.1, self.publish_status)

    def _on_transition(self, msg: String) -> None:
        previous = self.state
        self.state = _next_replay_state(self.state, msg.data)
        if self.state == previous:
            return
        self.last_event = {
            "recording": "replay started" if previous == "is_waiting" else "replay resumed",
            "paused": "replay paused",
            "exit": "replay stopped",
        }[self.state]

    def publish_status(self) -> None:
        """Publish replay progress using the recorder's JSON schema."""
        msg = String()
        msg.data = json.dumps(
            {
                "mode": "replay",
                "state": self.state,
                "episode": self.episode,
                "num_episodes": 1,
                "fps": round(self.fps_measured, 1),
                "frames": self.frames,
                "total_frames": self.total_frames,
                "repo_id": self.repo_id,
                "last_event": self.last_event,
            }
        )
        self.publisher.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """Process controls and periodic status publication."""
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        """Release the replay control node without shutting down the robot context."""
        self.node.destroy_node()


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


def _run_prompt_replay(adapter, actions: np.ndarray, period: float, fps: int) -> None:  # noqa: ANN001
    """Replay immediately, using Ctrl-C as the only runtime control."""
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


def _wait_for_replay_start(control: ReplayROSControl) -> bool:
    """Wait for the TUI to start replay; return false when it requests exit."""
    logger.info("Follower ready. Press [r] in crisp-record-tui to start replay.")
    control.publish_status()
    while rclpy.ok() and control.state == "is_waiting":
        control.spin_once(timeout_sec=0.05)
    return control.state != "exit"


def _run_ros_replay(
    adapter,  # noqa: ANN001
    robot,  # noqa: ANN001
    actions: np.ndarray,
    period: float,
    control: ReplayROSControl,
) -> None:
    """Replay with start, pause, resume and exit driven by the recording TUI."""
    if not _wait_for_replay_start(control):
        return

    index = 0
    next_tick = time.perf_counter()
    last_command_time: float | None = None
    paused = False
    while rclpy.ok() and index < len(actions) and control.state != "exit":
        control.spin_once(timeout_sec=0.0)
        if control.state == "paused":
            if not paused:
                _hold_current_pose(robot)
                paused = True
                control.fps_measured = 0.0
                control.publish_status()
                logger.info(
                    "Replay paused at frame %d/%d; holding current pose.", index, len(actions)
                )
            control.spin_once(timeout_sec=0.05)
            continue
        if control.state != "recording":
            continue
        if paused:
            paused = False
            next_tick = time.perf_counter()
            last_command_time = None
            logger.info("Replay resumed at frame %d/%d.", index, len(actions))

        adapter.set_target_position(actions[index, :6])
        now = time.perf_counter()
        if last_command_time is not None and now > last_command_time:
            control.fps_measured = 1.0 / (now - last_command_time)
        last_command_time = now
        index += 1
        control.frames = index
        next_tick += period

        while rclpy.ok() and control.state == "recording":
            remaining = next_tick - time.perf_counter()
            if remaining <= 0.0:
                break
            control.spin_once(timeout_sec=min(remaining, 0.02))

    if index == len(actions) and control.state != "exit":
        _hold_current_pose(robot)
        control.state = "finished"
        control.fps_measured = 0.0
        control.last_event = "replay complete; holding final pose"
        control.publish_status()
        logger.info("Replay complete; holding final pose. Press [q] in the TUI to exit.")
        while rclpy.ok() and control.state == "finished":
            control.spin_once(timeout_sec=0.05)


def _execute_replay(args: argparse.Namespace, actions: np.ndarray, fps: int) -> None:
    """Home the follower and stream the validated target sequence."""
    scheme = make_bilateral_config(args.teleop_scheme)
    if scheme.mode != "cartesian":
        raise ValueError(f"Replay currently supports Cartesian schemes, got {scheme.mode!r}.")

    follower_config = args.follower_config or scheme.follower_env_config
    follower_namespace = args.follower_namespace or scheme.follower_namespace
    env = None
    control = None
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
        if args.recording_manager_type == "ros":
            control = ReplayROSControl(
                repo_id=args.repo_id,
                episode=args.episode,
                total_frames=len(actions),
                status_topic=args.status_topic,
                transition_topic=args.transition_topic,
            )
            _run_ros_replay(adapter, env.robot, actions, period, control)
        else:
            _run_prompt_replay(adapter, actions, period, fps)
            logger.info("Replay complete; holding final pose.")
    finally:
        if env is not None:
            _hold_current_pose(env.robot)
        if control is not None:
            control.close()
        if env is not None:
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
        "--recording-manager-type",
        choices=["prompt", "ros"],
        default="prompt",
        help="Use 'ros' to control replay from crisp-record-tui.",
    )
    p.add_argument("--status-topic", default="/record_status")
    p.add_argument("--transition-topic", default="/record_transition")
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
    if args.recording_manager_type == "prompt" and not args.yes:
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
