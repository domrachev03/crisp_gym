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


def offset_pose(local_home: NDArray, ref_home: NDArray, ref_now: NDArray) -> NDArray:
    """Map a reference pose through the constant rigid offset between two EE frames.

    For anchored absolute coupling: the leader (``ref``) and follower (``local``)
    end-effectors sit in *different* frames at the shared home configuration (e.g.
    ``fr3_hand_tcp`` vs ``panda_hand_tcp``, gripper vs no gripper). The fixed
    world-frame offset ``T = local_home ∘ ref_home⁻¹`` captures that difference once;
    applying it to the live reference pose, ``target = T ∘ ref_now``, yields a
    follower target that equals ``local_home`` exactly when ``ref_now == ref_home``
    (no jump at engagement) and mirrors the reference's absolute motion thereafter.
    Recomputed from fixed anchors each call -> drift-free. Poses are
    ``[x, y, z, qw, qx, qy, qz]``.
    """
    r_off = _rot(local_home) * _rot(ref_home).inv()
    t_off = np.asarray(local_home[:3], dtype=float) - r_off.apply(np.asarray(ref_home[:3], dtype=float))
    new_rot = r_off * _rot(ref_now)
    new_trans = r_off.apply(np.asarray(ref_now[:3], dtype=float)) + t_off
    return _pack(new_trans, new_rot)


def offset_joint(local_home: NDArray, ref_home: NDArray, ref_now: NDArray) -> NDArray:
    """Joint-space anchored absolute: ``local_home + (ref_now - ref_home)`` -> (nq,).

    Equals ``local_home`` at engagement (``ref_now == ref_home``); tracks the
    reference's joint displacement through the constant home offset.
    """
    return np.asarray(local_home, dtype=float) + (
        np.asarray(ref_now, dtype=float) - np.asarray(ref_home, dtype=float)
    )
