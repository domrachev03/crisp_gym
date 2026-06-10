"""6-DOF pose increment math for real Cartesian bilateral teleop.

Ported from franka_server_standalone increment.py (docs/design/tdpa_reference.md).
Pose convention: ``[x, y, z, qw, qx, qy, qz]`` (w-first). Increments are world-frame
``[tx, ty, tz, rvx, rvy, rvz]`` (rotation as a fixed-frame rotation vector), so they
can be passed through a delayed channel and integrated by the follower.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def _rot(pose: NDArray) -> Rotation:
    w, x, y, z = pose[3], pose[4], pose[5], pose[6]
    return Rotation.from_quat([x, y, z, w])


def _pack(trans: NDArray, rot: Rotation) -> NDArray:
    q = rot.as_quat()  # xyzw
    return np.array([trans[0], trans[1], trans[2], q[3], q[0], q[1], q[2]])


def increment_world(before: NDArray, after: NDArray) -> NDArray:
    """World-frame pose increment from ``before`` to ``after`` -> (6,)."""
    trans = np.asarray(after[:3], dtype=float) - np.asarray(before[:3], dtype=float)
    ra = _rot(after)
    rb = _rot(before)
    rel = rb.inv() * ra  # body-frame relative rotation
    rotvec_world = ra.apply(rel.as_rotvec())  # conjugate to the fixed/world frame
    return np.concatenate([trans, rotvec_world])


def integrate_pose(pose: NDArray, incr: NDArray, scale: float = 1.0) -> NDArray:
    """Apply a (scaled) world-frame increment to a pose -> new pose (7,)."""
    new_trans = np.asarray(pose[:3], dtype=float) + scale * np.asarray(incr[:3], dtype=float)
    delta = Rotation.from_rotvec(scale * np.asarray(incr[3:], dtype=float))
    new_rot = delta * _rot(pose)  # left-multiply: world-frame delta
    return _pack(new_trans, new_rot)
