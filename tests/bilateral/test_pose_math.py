"""6-DOF pose increment math for real Cartesian teleop (ported from franka_server).

Pose convention: [x, y, z, qw, qx, qy, qz] (w-first). Increments are world-frame
[tx, ty, tz, rvx, rvy, rvz]. The decisive property: integrate_pose(before,
increment_world(before, after)) == after.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.pose_math import increment_world, integrate_pose


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
