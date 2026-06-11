"""Unit tests for the structured recording schema + state assembly (no ROS)."""
import numpy as np
import pytest

from crisp_gym.record import structured_record as sr


class _Rep:
    value = "euler"


class _RobotCfg:
    def num_joints(self):
        return 7


class _EnvCfg:
    orientation_representation = _Rep()
    robot_config = _RobotCfg()
    control_frequency = 30.0


class _Env:
    config = _EnvCfg()


class _Pose:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def to_array(self, rep):  # noqa: ANN001
        return self._arr


class _LRobot:
    end_effector_pose = _Pose([1, 2, 3, 0.1, 0.2, 0.3])
    joint_values = np.arange(1, 8, dtype=float)
    target_pose = _Pose([4, 5, 6, 0.4, 0.5, 0.6])


class _Gripper:
    value = 0.5


class _Leader:
    robot = _LRobot()
    gripper = _Gripper()


def test_features_shapes_and_keys():
    cams = [sr.CameraSpec("wrist", "sn1"), sr.CameraSpec("front", "sn2", 640, 480, 30)]
    feats, names = sr.build_structured_features(_Env(), hrate_window=140, cameras=cams, include_target=True)
    # state = 6 cart + 7 joints + 1 gripper + 6 target = 20
    assert feats["observation.follower_state"]["shape"] == (20,)
    assert feats["observation.leader_state"]["shape"] == (20,)
    assert feats["observation.follower_ft"]["shape"] == (140, 6)
    assert feats["observation.leader_ft_time"]["shape"] == (140,)
    assert feats["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert feats["observation.images.front"]["dtype"] == "video"
    assert feats["action"]["shape"] == (7,)
    assert len(names) == 20
    assert "observation.state" not in feats  # structured schema omits the flat vector


def test_features_no_target_is_14():
    feats, names = sr.build_structured_features(_Env(), 100, [], include_target=False)
    assert feats["observation.follower_state"]["shape"] == (14,)
    assert len(names) == 14


def test_leader_state_assembly_order():
    s = sr.leader_state(_Leader(), _Env(), include_target=True)
    assert s.dtype == np.float32
    # cart(6) joints(7) gripper(1) target(6)
    np.testing.assert_allclose(s[:6], [1, 2, 3, 0.1, 0.2, 0.3], rtol=1e-5)
    np.testing.assert_allclose(s[6:13], np.arange(1, 8), rtol=1e-5)
    assert s[13] == pytest.approx(0.5)
    np.testing.assert_allclose(s[14:20], [4, 5, 6, 0.4, 0.5, 0.6], rtol=1e-5)


def test_leader_state_tolerates_missing_gripper():
    class _BadGripper:
        @property
        def value(self):
            raise RuntimeError("gripper not initialized")

    leader = _Leader()
    leader.gripper = _BadGripper()
    s = sr.leader_state(leader, _Env(), include_target=True)
    assert s[13] == 0.0  # gripper fell back to 0


def test_follower_state_from_obs():
    obs = {
        "observation.state.cartesian": np.array([1, 2, 3, 0.1, 0.2, 0.3]),
        "observation.state.joints": np.arange(1, 8),
        "observation.state.gripper": np.array([0.9]),
        "observation.state.target": np.array([7, 8, 9, 0.7, 0.8, 0.9]),
    }
    s = sr.follower_state_from_obs(obs, _Env(), include_target=True)
    assert s.shape == (20,)
    assert s[13] == pytest.approx(0.9)
    np.testing.assert_allclose(s[14:20], [7, 8, 9, 0.7, 0.8, 0.9], rtol=1e-5)
