"""Runtime adapters presenting a live ``crisp_py`` robot to ``BilateralController``.

The control law is axis-generic and frame-agnostic: it reads a robot's
*generalized position* and *wrench* and writes a *target position* and a
*feed-forward force*. These adapters supply the cartesian / joint meaning behind
those generic calls so the same controller runs on hardware unchanged.

Frame convention (cartesian): everything the controller sees is in the **world
frame**, expressed **relative to the captured home**.

  - ``position``  -> world-frame pose increment from home, ``(6,)`` ``[tx,ty,tz,rx,ry,rz]``.
  - ``set_target_position(delta)`` -> ``set_target(integrate_pose(home, delta))``,
    which is exactly :func:`aligned_pose` (1:1 world-axis mapping, jump-free at home,
    drift-free) — the HW-validated coupling.
  - ``wrench`` -> the sensed TCP wrench rotated into the world frame by the *live*
    orientation, so reflected force and the 4-channel position spring add in one
    consistent frame.
  - ``set_feedforward_force(world_wrench)`` -> rotate world->TCP (live orientation),
    then ``set_target_wrench`` (crisp uses the local TCP jacobian).

Joint mode is a linear passthrough: ``position`` is the joint offset from home,
``set_target_position`` re-anchors onto the follower home (= :func:`offset_joint`),
and ``set_feedforward_force`` is recorded but NOT applied by default (crisp has no
joint-torque streaming controller — joint force reflection is experimental).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.pose_math import increment_world, integrate_pose, rotate_wrench


def pose_to_vec(pose) -> NDArray:
    """crisp_py ``Pose`` -> ``[x, y, z, qw, qx, qy, qz]`` (w-first)."""
    q = pose.orientation.as_quat()  # xyzw
    p = np.asarray(pose.position, dtype=float)
    return np.array([p[0], p[1], p[2], q[3], q[0], q[1], q[2]])


def _rot_of_vec(vec: NDArray) -> Rotation:
    return Rotation.from_quat([vec[4], vec[5], vec[6], vec[3]])


class CrispCartesianAdapter:
    """World-frame, home-relative ``RobotInterface`` over a crisp_py robot (cartesian)."""

    def __init__(self, robot, home_pose_vec: NDArray, wrench_fn: Callable[[], NDArray] | None = None):
        """Wrap ``robot``; ``home_pose_vec`` is the captured EE home ``(7,)``.

        ``wrench_fn`` returns the live TCP wrench ``(6,)`` (e.g. a NetFT reading);
        ``None`` means this robot reports no force (zeros).
        """
        self.robot = robot
        self.home = np.asarray(home_pose_vec, dtype=float).copy()
        self._wrench_fn = wrench_fn
        self.last_target_vec = self.home.copy()

    @property
    def _now_vec(self) -> NDArray:
        return pose_to_vec(self.robot.end_effector_pose)

    @property
    def position(self) -> NDArray:
        return increment_world(self.home, self._now_vec)

    @property
    def velocity(self) -> NDArray:
        tw = self.robot.end_effector_twist
        return np.concatenate([np.asarray(tw.linear, dtype=float),
                               np.asarray(tw.angular, dtype=float)])

    @property
    def wrench(self) -> NDArray:
        if self._wrench_fn is None:
            return np.zeros(6)
        w_tcp = np.asarray(self._wrench_fn(), dtype=float)
        return rotate_wrench(_rot_of_vec(self._now_vec), w_tcp)  # TCP -> world

    def set_target_position(self, delta: NDArray) -> None:
        self.last_target_vec = integrate_pose(self.home, np.asarray(delta, dtype=float))
        self.robot.set_target(pose=self._vec_to_pose(self.last_target_vec))

    def set_feedforward_force(self, world_wrench: NDArray) -> None:
        w_tcp = rotate_wrench(_rot_of_vec(self._now_vec).inv(), np.asarray(world_wrench, dtype=float))
        self.robot.set_target_wrench(force=w_tcp[:3].tolist(), torque=w_tcp[3:].tolist())

    def _vec_to_pose(self, vec: NDArray):
        from crisp_py.robot import Pose

        return Pose(position=np.asarray(vec[:3], dtype=float),
                    orientation=_rot_of_vec(vec))


class CrispJointAdapter:
    """Joint-space ``RobotInterface`` over a crisp_py robot (1:1 joint coupling).

    EXPERIMENTAL force path: ``set_feedforward_force`` records the reflected joint
    effort but does not command it unless ``apply_feedforward`` is set and the robot
    exposes ``set_target_joint_torque`` — crisp ships no joint-torque streaming
    controller, so by default joint mode is position-coupled with force logged only.
    The adapter must be the SOLE publisher of the follower ``target_joint`` topic
    (multiple publishers are dropped -> follower freeze).
    """

    def __init__(self, robot, home_joints: NDArray, effort_fn: Callable[[], NDArray] | None = None,
                 apply_feedforward: bool = False):
        self.robot = robot
        self.home = np.asarray(home_joints, dtype=float).copy()
        self._effort_fn = effort_fn
        self.apply_feedforward = apply_feedforward
        self.last_target = self.home.copy()
        self.last_feedforward = np.zeros_like(self.home)

    @property
    def position(self) -> NDArray:
        return np.asarray(self.robot.joint_values, dtype=float) - self.home

    @property
    def velocity(self) -> NDArray:
        v = getattr(self.robot, "joint_velocities", None)
        return np.asarray(v, dtype=float) if v is not None else np.zeros_like(self.home)

    @property
    def wrench(self) -> NDArray:
        if self._effort_fn is None:
            return np.zeros_like(self.home)
        return np.asarray(self._effort_fn(), dtype=float)

    def set_target_position(self, delta: NDArray) -> None:
        self.last_target = self.home + np.asarray(delta, dtype=float)
        self.robot.set_target_joint(self.last_target)

    def set_feedforward_force(self, tau: NDArray) -> None:
        self.last_feedforward = np.asarray(tau, dtype=float)
        if self.apply_feedforward and hasattr(self.robot, "set_target_joint_torque"):
            self.robot.set_target_joint_torque(self.last_feedforward.tolist())
