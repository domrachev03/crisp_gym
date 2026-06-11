"""Episode setup for dataset recording: randomized start pose + post-episode retract.

Replicates the lerobot-panda setup flow (setup_profile + setup_runner) for the
crisp_gym recorder, minus the grasp sequence (this rig is a no-gripper panda-panda
bilateral). Per episode the FOLLOWER (the task arm) is positioned:

  episode 0   : (optional) home, then move to a start pose sampled around a fixed
                center (the configured ``start_center`` or, if unset, the follower's
                home EE pose captured once).
  episode 1..N: retract straight up by ``retract_z_offset`` (~10 cm), then move to a
                fresh sampled start pose -- no full re-home.

After positioning, the bilateral coupling is re-anchored at the new pose so teleop
engages without a jump. Everything is specified via a ``config/setup/<name>.yaml``
selected with ``--setup-config``.

The sampler + config loading are pure (numpy/scipy); the runner drives the robot
through ``crisp_py``'s blocking ``move_to`` / ``home``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class PoseTarget:
    position: list[float]
    quaternion_xyzw: list[float]


@dataclass
class StartJitter:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class EpisodeSetupConfig:
    enabled: bool = False
    home_before_first: bool = True
    move_speed: float = 0.05               # m/s for the blocking moves
    retract_z_offset: float = 0.10         # straight-up retract after each episode (m)
    final_settle_s: float = 0.5
    wait_for_input: bool = False           # pause for Enter before each episode
    seed: int | None = 0
    start_center: PoseTarget | None = None  # None -> capture follower home EE pose once
    start_jitter: StartJitter = field(default_factory=StartJitter)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EpisodeSetupConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        center = data.get("start_center")
        if isinstance(center, dict):
            center = PoseTarget(position=list(center["position"]),
                                quaternion_xyzw=list(center["quaternion_xyzw"]))
        else:
            center = None
        jit = data.get("start_jitter") or {}
        jitter = StartJitter(**{k: jit[k] for k in ("x", "y", "z", "yaw_deg") if k in jit})
        known = {"enabled", "home_before_first", "move_speed", "retract_z_offset",
                 "final_settle_s", "wait_for_input", "seed"}
        kwargs = {k: data[k] for k in known if k in data}
        return cls(start_center=center, start_jitter=jitter, **kwargs)


def make_episode_rng(seed: int | None, episode_index: int) -> np.random.Generator:
    """Per-episode RNG: ``seed + episode_index`` (reproducible), or fresh if seed None."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed + episode_index)


def sample_start_pose(center: PoseTarget, jitter: StartJitter, rng: np.random.Generator) -> PoseTarget:
    """Sample a start pose: center position + uniform xyz jitter, yaw jittered about z."""
    pos = np.asarray(center.position, dtype=float) + np.array([
        rng.uniform(-jitter.x, jitter.x),
        rng.uniform(-jitter.y, jitter.y),
        rng.uniform(-jitter.z, jitter.z),
    ])
    base = Rotation.from_quat(np.asarray(center.quaternion_xyzw, dtype=float))
    yaw = rng.uniform(-jitter.yaw_deg, jitter.yaw_deg)
    rot = Rotation.from_euler("z", yaw, degrees=True) * base
    return PoseTarget(position=pos.tolist(), quaternion_xyzw=rot.as_quat().tolist())


def _default_pose_factory(position, quaternion_xyzw):
    from crisp_py.robot import Pose

    return Pose(np.asarray(position, dtype=float), Rotation.from_quat(quaternion_xyzw))


class EpisodeSetupRunner:
    """Positions the follower for each episode (home/retract + randomized start).

    The fixed center is captured once (after the first home) when ``start_center`` is
    unset, so every episode randomizes around the same point.
    """

    def __init__(self, robot, config: EpisodeSetupConfig, pose_factory=None):
        self.robot = robot
        self.config = config
        self._pose_factory = pose_factory or _default_pose_factory
        self._center: PoseTarget | None = config.start_center

    def _ensure_center(self) -> None:
        if self._center is None:
            p = self.robot.end_effector_pose
            self._center = PoseTarget(position=np.asarray(p.position, dtype=float).tolist(),
                                      quaternion_xyzw=p.orientation.as_quat().tolist())
            logger.info(f"[setup] captured center from follower pose: {self._center.position}")

    def _move(self, pose: PoseTarget, label: str) -> None:
        logger.info(f"[setup] move to {label}: pos={np.round(pose.position, 3).tolist()}")
        self.robot.move_to(pose=self._pose_factory(pose.position, pose.quaternion_xyzw),
                           speed=self.config.move_speed)

    def setup_episode(self, episode_index: int, sleep_fn=time.sleep) -> PoseTarget:
        """Run the setup for ``episode_index`` and return the sampled start pose."""
        cfg = self.config
        if episode_index == 0 and cfg.home_before_first:
            logger.info("[setup] homing follower...")
            self.robot.home()
            # home() leaves the robot on joint_trajectory_controller; move_to needs the
            # cartesian impedance controller to track the streamed target poses.
            switcher = getattr(self.robot, "controller_switcher_client", None)
            if switcher is not None:
                switcher.switch_controller("cartesian_impedance_controller")
        self._ensure_center()

        if episode_index > 0 and cfg.retract_z_offset > 0.0:
            cur = self.robot.end_effector_pose
            retract = PoseTarget(
                position=(np.asarray(cur.position, dtype=float) + [0, 0, cfg.retract_z_offset]).tolist(),
                quaternion_xyzw=cur.orientation.as_quat().tolist(),
            )
            self._move(retract, f"retract +{cfg.retract_z_offset:.2f}m")

        rng = make_episode_rng(cfg.seed, episode_index)
        start = sample_start_pose(self._center, cfg.start_jitter, rng)
        self._move(start, "sampled start")

        if cfg.final_settle_s > 0.0:
            sleep_fn(cfg.final_settle_s)
        if cfg.wait_for_input:
            input("[setup] Press Enter to start episode...")
        return start
