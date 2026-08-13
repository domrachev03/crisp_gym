"""HW adapters: present a crisp_py robot to the generic controller in WORLD-frame,
home-relative coordinates.

The control law is axis-generic and frame-agnostic; these adapters carry all the
cartesian semantics (home anchoring, the 1:1 world-axis pose mapping, wrench frame
rotation) and the joint-space passthrough. The mapping is the part that had real
bugs on hardware (startup jump, wrench frame), so it is pinned here against fake
robots — no ROS, no crisp_py (``_vec_to_pose`` is overridden to identity).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.crisp_adapter import (
    CrispCartesianAdapter,
    CrispJointAdapter,
    pose_to_vec,
)
from crisp_gym.bilateral.pose_math import aligned_pose, increment_world, offset_joint, rotate_wrench


# --- fakes ------------------------------------------------------------------- #
class _FakePose:
    def __init__(self, vec):
        self.position = np.asarray(vec[:3], float)
        self.orientation = Rotation.from_quat([vec[4], vec[5], vec[6], vec[3]])


class _FakeTwist:
    def __init__(self, lin, ang):
        self.linear = np.asarray(lin, float)
        self.angular = np.asarray(ang, float)


class _FakeCartRobot:
    def __init__(self, vec):
        self._vec = np.asarray(vec, float)
        self.twist = _FakeTwist(np.zeros(3), np.zeros(3))
        self.target_pose_arg = None
        self.wrench_arg = None

    @property
    def end_effector_pose(self):
        return _FakePose(self._vec)

    @property
    def end_effector_twist(self):
        return self.twist

    def set_target(self, pose=None):
        self.target_pose_arg = pose

    def set_target_wrench(self, force=None, torque=None):
        self.wrench_arg = np.asarray(list(force) + list(torque), float)


class _IdentityCartAdapter(CrispCartesianAdapter):
    """Bypass crisp_py.robot.Pose so set_target records the raw 7-vec."""

    def _vec_to_pose(self, vec):
        return np.asarray(vec, float)


class _FakeJointRobot:
    def __init__(self, q):
        self._q = np.asarray(q, float)
        self.joint_values = self._q
        self.target_joint_arg = None

    def set_target_joint(self, q):
        self.target_joint_arg = np.asarray(q, float)


# --- helpers ----------------------------------------------------------------- #
def _vec(x, y, z, rotvec=(0, 0, 0)):
    q = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
    return np.array([x, y, z, q[3], q[0], q[1], q[2]])


# --- cartesian pose mapping reproduces aligned_pose -------------------------- #
def test_cartesian_position_mapping_equals_aligned_pose():
    leader_home = _vec(0.3, 0.0, 0.5, (0, 0, 0))
    follower_home = _vec(0.4, -0.1, 0.45, (0, 0, np.pi / 2))  # different TCP frame
    leader_now = _vec(0.35, 0.05, 0.55, (0.1, 0, 0))

    leader = _IdentityCartAdapter(_FakeCartRobot(leader_now), leader_home)
    follower = _IdentityCartAdapter(_FakeCartRobot(follower_home), follower_home)

    follower.set_target_position(leader.position)  # delta -> integrate onto follower home
    expected = aligned_pose(follower_home, leader_home, leader_now)
    assert np.allclose(follower.last_target_vec, expected, atol=1e-9)


def test_cartesian_no_startup_jump_at_home():
    leader_home = _vec(0.3, 0.0, 0.5)
    follower_home = _vec(0.4, -0.1, 0.45, (0, 0, np.pi / 2))
    leader = _IdentityCartAdapter(_FakeCartRobot(leader_home), leader_home)  # at home
    follower = _IdentityCartAdapter(_FakeCartRobot(follower_home), follower_home)

    follower.set_target_position(leader.position)  # leader at home -> delta 0
    assert np.allclose(follower.last_target_vec, follower_home, atol=1e-9)


def test_cartesian_wrench_world_roundtrip():
    pose = _vec(0.3, 0.0, 0.5, (0, 0, np.pi / 3))  # non-trivial orientation
    tcp_wrench = np.array([2.0, -1.0, 3.0, 0.1, 0.2, -0.3])
    adapter = CrispCartesianAdapter(_FakeCartRobot(pose), pose, wrench_fn=lambda: tcp_wrench)

    world = adapter.wrench                       # TCP -> world
    assert not np.allclose(world, tcp_wrench)    # orientation actually rotated it
    adapter.set_feedforward_force(world)         # world -> TCP again
    assert np.allclose(adapter.robot.wrench_arg, tcp_wrench, atol=1e-9)


def test_cartesian_wrench_zero_without_source():
    pose = _vec(0.3, 0.0, 0.5)
    adapter = CrispCartesianAdapter(_FakeCartRobot(pose), pose, wrench_fn=None)
    assert np.allclose(adapter.wrench, np.zeros(6))


def test_tilted_base_maps_pose_increment_into_common_frame():
    home = _vec(0.3, 0.0, 0.5)
    now = _vec(0.4, 0.0, 0.5, (0.1, 0.0, 0.0))
    base_to_common = Rotation.from_euler("z", 90, degrees=True)
    adapter = _IdentityCartAdapter(
        _FakeCartRobot(now), home, base_to_common=base_to_common
    )

    assert np.allclose(adapter.position[:3], [0.0, 0.1, 0.0], atol=1e-9)
    assert np.allclose(adapter.position[3:], [0.0, 0.1, 0.0], atol=1e-9)


def test_tilted_base_target_roundtrip():
    home = _vec(0.3, 0.0, 0.5)
    base_to_common = Rotation.from_euler("xyz", [0.6, -0.2, 0.1])
    adapter = _IdentityCartAdapter(
        _FakeCartRobot(home), home, base_to_common=base_to_common
    )
    common_delta = np.array([0.03, -0.02, 0.01, 0.1, -0.04, 0.02])

    adapter.set_target_position(common_delta)
    adapter.robot._vec = adapter.last_target_vec
    assert np.allclose(adapter.position, common_delta, atol=1e-9)


def test_tilted_base_rotates_tcp_local_twist_and_preserves_power():
    pose = _vec(0.3, 0.0, 0.5, (0.2, -0.1, 0.4))
    base_to_common = Rotation.from_euler("xyz", [0.6, -0.2, 0.1])
    wrench_tcp = np.array([2.0, -1.0, 3.0, 0.1, 0.2, -0.3])
    twist_tcp = np.array([0.2, -0.4, 0.1, 0.3, 0.1, -0.2])
    robot = _FakeCartRobot(pose)
    robot.twist = _FakeTwist(twist_tcp[:3], twist_tcp[3:])
    adapter = CrispCartesianAdapter(
        robot, pose, wrench_fn=lambda: wrench_tcp, base_to_common=base_to_common
    )

    assert np.isclose(adapter.wrench @ adapter.velocity, wrench_tcp @ twist_tcp, atol=1e-9)
    adapter.set_feedforward_force(adapter.wrench)
    assert np.allclose(robot.wrench_arg, wrench_tcp, atol=1e-9)


# --- joint passthrough reproduces offset_joint ------------------------------- #
def test_joint_position_mapping_equals_offset_joint():
    leader_home = np.array([0.0, -0.3, 0.1, -2.0, 0.0, 1.6, -0.2])
    follower_home = np.array([0.1, -0.2, 0.0, -2.1, 0.1, 1.5, -0.1])
    leader_now = leader_home + np.array([0.05, 0.0, -0.05, 0.1, 0.0, 0.0, 0.2])

    leader = CrispJointAdapter(_FakeJointRobot(leader_now), leader_home)
    follower = CrispJointAdapter(_FakeJointRobot(follower_home), follower_home)

    follower.set_target_position(leader.position)
    expected = offset_joint(follower_home, leader_home, leader_now)
    assert np.allclose(follower.robot.target_joint_arg, expected, atol=1e-12)


def test_joint_feedforward_is_recorded_noop_by_default():
    home = np.zeros(7)
    adapter = CrispJointAdapter(_FakeJointRobot(home), home)
    tau = np.arange(7.0)
    adapter.set_feedforward_force(tau)  # must not raise; robot has no torque streaming
    assert np.allclose(adapter.last_feedforward, tau)
    assert adapter.robot.target_joint_arg is None  # position target untouched


def test_cartesian_reanchor_no_jump():
    # after the follower is moved (episode setup) and re-anchored, a zero leader
    # delta must command the follower's NEW pose -> no jump at engage.
    new_leader = _vec(0.4, 0.1, 0.6, (0.2, 0, 0))
    new_follower = _vec(0.3, -0.2, 0.5, (0, 0, 0.5))
    leader = _IdentityCartAdapter(_FakeCartRobot(_vec(0, 0, 0)), _vec(0, 0, 0))
    follower = _IdentityCartAdapter(_FakeCartRobot(_vec(0, 0, 0)), _vec(0, 0, 0))
    leader.robot._vec = new_leader
    follower.robot._vec = new_follower
    leader.reanchor()
    follower.reanchor()
    follower.set_target_position(leader.position)  # leader delta == 0 right after reanchor
    assert np.allclose(follower.last_target_vec, new_follower, atol=1e-9)


def test_joint_reanchor_recaptures_home():
    home = np.zeros(7)
    adapter = CrispJointAdapter(_FakeJointRobot(home), home)
    adapter.robot.joint_values = np.arange(7.0)
    adapter.reanchor()
    assert np.allclose(adapter.home, np.arange(7.0))
    adapter.set_target_position(np.zeros(7))  # zero delta -> stays at new home
    assert np.allclose(adapter.robot.target_joint_arg, np.arange(7.0))


def test_pose_to_vec_roundtrip():
    v = _vec(0.1, 0.2, 0.3, (0.2, -0.1, 0.4))
    assert np.allclose(pose_to_vec(_FakePose(v)), v, atol=1e-12)
