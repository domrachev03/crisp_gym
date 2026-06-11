"""Episode setup: randomized start pose around a fixed center + post-episode retract.

Mirrors lerobot-panda's setup_profile/setup_runner (minus the grasp sequence -- this
rig is a no-gripper panda-panda bilateral). The pure sampler + config loading are
pinned here; the HW runner is exercised against a fake robot (no ROS).
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.record.episode_setup import (
    EpisodeSetupConfig,
    EpisodeSetupRunner,
    PoseTarget,
    StartJitter,
    make_episode_rng,
    sample_start_pose,
)

CONFIG_DIR = Path(__file__).parents[2] / "crisp_gym" / "config" / "setup"


# --- sampler ----------------------------------------------------------------- #
def test_sample_stays_within_jitter_bounds():
    center = PoseTarget([0.5, 0.0, 0.3], [0, 0, 0, 1])
    jit = StartJitter(x=0.05, y=0.04, z=0.02, yaw_deg=15.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_start_pose(center, jit, rng)
        d = np.array(s.position) - np.array(center.position)
        assert abs(d[0]) <= 0.05 + 1e-9 and abs(d[1]) <= 0.04 + 1e-9 and abs(d[2]) <= 0.02 + 1e-9
        # yaw delta within bound
        dyaw = (Rotation.from_quat(s.quaternion_xyzw) * Rotation.from_quat(center.quaternion_xyzw).inv())
        assert abs(dyaw.as_euler("xyz", degrees=True)[2]) <= 15.0 + 1e-6


def test_zero_jitter_returns_center():
    center = PoseTarget([0.5, -0.1, 0.3], [0, 0, 0, 1])
    s = sample_start_pose(center, StartJitter(), np.random.default_rng(1))
    assert np.allclose(s.position, center.position)


def test_rng_is_deterministic_per_episode():
    a = make_episode_rng(seed=7, episode_index=3)
    b = make_episode_rng(seed=7, episode_index=3)
    c = make_episode_rng(seed=7, episode_index=4)
    center, jit = PoseTarget([0.5, 0, 0.3], [0, 0, 0, 1]), StartJitter(x=0.05)
    assert sample_start_pose(center, jit, a).position == sample_start_pose(center, jit, b).position
    assert sample_start_pose(center, jit, a).position != sample_start_pose(center, jit, c).position


def test_config_loads_from_yaml():
    cfg = EpisodeSetupConfig.from_yaml(CONFIG_DIR / "randomized_around_home.yaml")
    assert cfg.enabled is True
    assert cfg.retract_z_offset > 0
    assert cfg.start_jitter.x >= 0


# --- runner (fake robot, no ROS) --------------------------------------------- #
class _FakePose:
    def __init__(self, pos, quat):
        self.position = np.asarray(pos, float)
        self.orientation = Rotation.from_quat(quat)


class _FakeRobot:
    def __init__(self):
        self._pose = _FakePose([0.5, 0.0, 0.4], [0, 0, 0, 1])
        self.calls = []

    @property
    def end_effector_pose(self):
        return self._pose

    def home(self, **kw):
        self.calls.append(("home", None))

    def move_to(self, pose=None, speed=0.05):
        self.calls.append(("move_to", (list(pose.position), speed)))
        self._pose = pose  # robot now at the commanded pose


def _cfg(**kw):
    base = dict(enabled=True, home_before_first=True, move_speed=0.1, retract_z_offset=0.10,
                final_settle_s=0.0, wait_for_input=False, seed=0,
                start_center=None, start_jitter=StartJitter(x=0.03, y=0.03, z=0.0, yaw_deg=0.0))
    base.update(kw)
    return EpisodeSetupConfig(**base)


def test_first_episode_homes_then_moves_to_start():
    robot = _FakeRobot()
    runner = EpisodeSetupRunner(robot, _cfg(), pose_factory=_FakePose)
    runner.setup_episode(episode_index=0, sleep_fn=lambda s: None)
    kinds = [c[0] for c in robot.calls]
    assert kinds[0] == "home"           # homed first
    assert "move_to" in kinds           # then moved to sampled start
    assert ("home", None) not in kinds[1:]  # homed only once


def test_subsequent_episode_retracts_then_moves_no_home():
    robot = _FakeRobot()
    runner = EpisodeSetupRunner(robot, _cfg(), pose_factory=_FakePose)
    runner.setup_episode(0, sleep_fn=lambda s: None)          # establishes center
    z0 = robot._pose.position[2]
    robot.calls.clear()
    runner.setup_episode(1, sleep_fn=lambda s: None)
    kinds = [c[0] for c in robot.calls]
    assert "home" not in kinds                                # no re-home
    # first move_to of a subsequent episode is the +z retract
    first_move = next(c for c in robot.calls if c[0] == "move_to")
    assert first_move[1][0][2] >= z0 + 0.10 - 1e-9            # retracted up ~10cm


def test_center_captured_from_home_pose_when_unset():
    robot = _FakeRobot()
    runner = EpisodeSetupRunner(robot, _cfg(start_center=None, start_jitter=StartJitter()), pose_factory=_FakePose)
    runner.setup_episode(0, sleep_fn=lambda s: None)
    # zero jitter -> start pose == captured center == the home pose
    move = next(c for c in robot.calls if c[0] == "move_to")
    assert np.allclose(move[1][0], [0.5, 0.0, 0.4], atol=1e-9)
