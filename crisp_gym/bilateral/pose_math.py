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


def aligned_pose(local_home: NDArray, ref_home: NDArray, ref_now: NDArray) -> NDArray:
    """Anchored absolute target with 1:1 world-axis mapping (no inter-frame twist).

    Maps the reference's world-frame translation and rotation *increments* from its
    home directly onto the local home::

        target_pos = local_home_pos + (ref_now_pos - ref_home_pos)
        target_rot = ΔR_world(ref_home -> ref_now) · local_home_rot

    so a +x reference move yields a +x follower move (same axis labels) and a roll
    yields a roll -- independent of any orientation difference between the two TCP
    frames. This is the difference from :func:`offset_pose`, which applies the full
    rigid home offset ``T = local ∘ ref⁻¹`` and therefore *inherits* the leader/
    follower TCP-frame convention difference (here ~90° about z) as a twist on the
    motion. Jump-free at engagement (``ref_now == ref_home`` -> ``local_home``),
    drift-free (recomputed from fixed anchors each call). Poses are
    ``[x, y, z, qw, qx, qy, qz]``.
    """
    return integrate_pose(local_home, increment_world(ref_home, ref_now))


def offset_joint(local_home: NDArray, ref_home: NDArray, ref_now: NDArray) -> NDArray:
    """Joint-space anchored absolute: ``local_home + (ref_now - ref_home)`` -> (nq,).

    Equals ``local_home`` at engagement (``ref_now == ref_home``); tracks the
    reference's joint displacement through the constant home offset.
    """
    return np.asarray(local_home, dtype=float) + (
        np.asarray(ref_now, dtype=float) - np.asarray(ref_home, dtype=float)
    )


def wrench_frame_rotation(leader_home: NDArray, follower_home: NDArray) -> Rotation:
    """Constant rotation mapping a follower-TCP wrench into the leader-TCP frame.

    The follower NetFT wrench is expressed in the follower TCP frame; the leader
    consumes it via ``set_target_wrench`` with ``use_local_jacobian`` (leader TCP
    frame). Assuming the two robot bases are world-aligned (as ``aligned_pose``
    couples the motion), a vector goes follower-TCP -> world -> leader-TCP as
    ``R = R_leaderHome⁻¹ · R_followerHome`` (constant, from the home orientations).
    For a pure yaw offset this rotates the lateral axes and leaves z (table normal)
    unchanged. Homes are ``[x, y, z, qw, qx, qy, qz]``.
    """
    return _rot(leader_home).inv() * _rot(follower_home)


def rotate_wrench(rotation: Rotation, wrench: NDArray) -> NDArray:
    """Rotate the force and torque blocks of a 6-vector wrench by ``rotation`` -> (6,)."""
    out = np.asarray(wrench, dtype=float).copy()
    out[:3] = rotation.apply(out[:3])
    out[3:] = rotation.apply(out[3:])
    return out
