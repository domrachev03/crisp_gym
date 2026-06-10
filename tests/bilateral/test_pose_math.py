"""6-DOF pose increment math for real Cartesian teleop (ported from franka_server).

Pose convention: [x, y, z, qw, qx, qy, qz] (w-first). Increments are world-frame
[tx, ty, tz, rvx, rvy, rvz]. The decisive property: integrate_pose(before,
increment_world(before, after)) == after.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.pose_math import (
    increment_world,
    integrate_pose,
    offset_joint,
    offset_pose,
)


def _pose(xyz, rot: Rotation):
    q = rot.as_quat()  # xyzw
    return np.array([*xyz, q[3], q[0], q[1], q[2]])


def _rot_of(pose):
    w, x, y, z = pose[3:7]
    return Rotation.from_quat([x, y, z, w])


def test_pure_translation_increment():
    before = _pose([0, 0, 0], Rotation.identity())
    after = _pose([0.1, -0.2, 0.3], Rotation.identity())
    incr = increment_world(before, after)
    assert np.allclose(incr[:3], [0.1, -0.2, 0.3])
    assert np.allclose(incr[3:], 0.0, atol=1e-9)


def test_zero_increment_is_identity():
    before = _pose([0.4, 0.0, 0.5], Rotation.from_euler("xyz", [0.1, -0.2, 0.3]))
    out = integrate_pose(before, np.zeros(6))
    assert np.allclose(out[:3], before[:3])
    assert np.allclose((_rot_of(out).inv() * _rot_of(before)).magnitude(), 0.0, atol=1e-9)


def test_increment_integrate_roundtrip():
    before = _pose([0.3, 0.1, 0.5], Rotation.from_euler("xyz", [0.2, 0.4, -0.1]))
    after = _pose([0.35, 0.0, 0.55], Rotation.from_euler("xyz", [0.1, 0.5, 0.2]))
    incr = increment_world(before, after)
    recon = integrate_pose(before, incr)
    assert np.allclose(recon[:3], after[:3], atol=1e-9)
    # orientation matches up to quaternion sign -> compare rotations
    ang = (_rot_of(recon).inv() * _rot_of(after)).magnitude()
    assert np.isclose(ang, 0.0, atol=1e-7)


# --- constant frame-offset (anchored absolute coupling) --------------------------
# offset_pose maps a reference (leader) pose through the fixed rigid offset between
# the two end-effector frames at the shared home, so a follower with a *different*
# TCP frame (e.g. fr3_hand_tcp vs panda_hand_tcp, gripper vs no gripper) does not
# jump at engagement.


def test_offset_pose_no_jump_at_engagement():
    # Local (follower) and reference (leader) homes differ in BOTH translation and
    # rotation -> a genuine frame offset. At engagement the target must equal the
    # local home exactly (no jump).
    local_home = _pose([0.31, 0.0, 0.42], Rotation.from_euler("xyz", [0.0, 0.2, 0.0]))
    ref_home = _pose([0.50, -0.04, 0.50], Rotation.from_euler("xyz", [0.1, 0.0, -0.1]))
    out = offset_pose(local_home, ref_home, ref_home)
    assert np.allclose(out[:3], local_home[:3], atol=1e-9)
    ang = (_rot_of(out).inv() * _rot_of(local_home)).magnitude()
    assert np.isclose(ang, 0.0, atol=1e-9)


def test_offset_pose_pure_translation_offset_tracks_one_to_one():
    # No rotation offset (R_off = I): follower displacement == leader displacement.
    local_home = _pose([0.31, 0.0, 0.42], Rotation.identity())
    ref_home = _pose([0.50, -0.04, 0.50], Rotation.identity())
    ref_now = _pose([0.55, -0.02, 0.58], Rotation.identity())  # leader moved [.05,.02,.08]
    out = offset_pose(local_home, ref_home, ref_now)
    assert np.allclose(out[:3], np.array([0.31, 0.0, 0.42]) + np.array([0.05, 0.02, 0.08]), atol=1e-9)
    assert np.isclose(_rot_of(out).magnitude(), 0.0, atol=1e-9)


def test_offset_pose_rotation_offset_rotates_displacement():
    # R_off = Rz(90deg): a leader +x move becomes a follower +y move.
    R_off = Rotation.from_euler("z", 90, degrees=True)
    ref_home = _pose([0.0, 0.0, 0.0], Rotation.identity())
    local_home = _pose([1.0, 2.0, 3.0], R_off)
    ref_now = _pose([0.1, 0.0, 0.0], Rotation.identity())
    out = offset_pose(local_home, ref_home, ref_now)
    assert np.allclose(out[:3], np.array([1.0, 2.0, 3.0]) + np.array([0.0, 0.1, 0.0]), atol=1e-9)


def test_offset_joint_no_jump_and_tracks():
    qh_local = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    qh_ref = np.array([0.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.685])
    assert np.allclose(offset_joint(qh_local, qh_ref, qh_ref), qh_local)  # engagement
    delta = np.array([0.0, 0.1, -0.2, 0.0, 0.05, 0.0, 0.0])
    assert np.allclose(offset_joint(qh_local, qh_ref, qh_ref + delta), qh_local + delta)
