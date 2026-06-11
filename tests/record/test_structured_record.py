"""Unit tests for the structured recording schema + hrate sizing + state assembly (no ROS)."""
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


def test_hrate_window_sizing():
    # fixed, fps-independent: ft gets ft_window; others scale by rate (round(ft_window*rate/FT_RATE))
    assert sr.hrate_window(2100, 150) == 150        # ft anchor
    assert sr.hrate_window(1000, 150) == 71         # joints
    assert sr.hrate_window(250, 150) == 18          # pose/twist
    assert sr.hrate_window(2100, 0) == 1            # never zero
    assert sr.hrate_window(2100, 250) == 250        # honors a larger window


def test_build_hrate_specs():
    specs = sr.build_hrate_specs({"follower": "right", "leader": "left"},
                                 ["ft", "joints"], ft_window=150)
    keys = {s[0] for s in specs}
    assert keys == {"follower_ft", "follower_joints", "leader_ft", "leader_joints"}
    bytopic = {s[0]: s[1] for s in specs}
    assert bytopic["follower_ft"] == "/right/netft_data_unbiased_tcp"
    assert bytopic["leader_joints"] == "/left/joint_states"


def test_features_shapes_and_keys():
    cams = [sr.CameraSpec("wrist", "sn1"), sr.CameraSpec("front", "sn2", 640, 480, 60)]
    feats, names = sr.build_structured_features(
        _Env(), cameras=cams, fps=60, signals=["ft", "joints", "pose", "twist"], include_target=True
    )
    assert feats["observation.follower_state"]["shape"] == (20,)
    assert feats["observation.leader_state"]["shape"] == (20,)
    # fixed ft_window 150 (fps-independent) -> ft 150x6, joints 71x7, pose 18x7, twist 18x6
    assert feats["observation.follower_ft_hrate"]["shape"] == (150, 6)
    assert feats["observation.follower_joints_hrate"]["shape"] == (71, 7)
    assert feats["observation.leader_pose_hrate"]["shape"] == (18, 7)
    assert feats["observation.leader_twist_hrate"]["shape"] == (18, 6)
    assert feats["observation.follower_ft_hrate_time"]["shape"] == (150,)
    assert feats["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert feats["observation.images.front"]["video_info"]["video.fps"] == 60.0
    assert feats["action"]["shape"] == (7,)
    assert len(names) == 20
    assert "observation.state" not in feats


def test_features_subset_signals():
    feats, _ = sr.build_structured_features(_Env(), cameras=[], fps=30, signals=["ft"])
    hrate_keys = [k for k in feats if k.endswith("_hrate")]
    assert set(hrate_keys) == {"observation.follower_ft_hrate", "observation.leader_ft_hrate"}


def test_leader_state_assembly_order():
    s = sr.leader_state(_Leader(), _Env(), include_target=True)
    assert s.dtype == np.float32
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
    assert s[13] == 0.0


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
