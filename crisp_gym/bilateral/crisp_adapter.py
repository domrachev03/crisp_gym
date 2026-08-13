"""Runtime adapters presenting a live ``crisp_py`` robot to ``BilateralController``.

The control law is axis-generic and frame-agnostic: it reads a robot's
*generalized position* and *wrench* and writes a *target position* and a
*feed-forward force*. These adapters supply the cartesian / joint meaning behind
those generic calls so the same controller runs on hardware unchanged.

Frame convention (cartesian): everything the controller sees is in one explicit
**common teleoperation frame**, expressed **relative to the captured home**.

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


def vec_to_pose(vec: NDArray):
    """``[x, y, z, qw, qx, qy, qz]`` -> crisp_py ``Pose`` (imports crisp_py lazily)."""
    from crisp_py.robot import Pose

    return Pose(position=np.asarray(vec[:3], dtype=float), orientation=_rot_of_vec(vec))


class CrispCartesianAdapter:
    """Common-frame, home-relative ``RobotInterface`` over a Cartesian robot."""

    def __init__(
        self,
        robot,
        home_pose_vec: NDArray,
        wrench_fn: Callable[[], NDArray] | None = None,
        base_to_common: Rotation | NDArray | None = None,
    ):
        """Wrap ``robot``; ``home_pose_vec`` is the captured EE home ``(7,)``.

        ``wrench_fn`` returns the live TCP wrench ``(6,)`` (e.g. a NetFT reading);
        ``None`` means this robot reports no force (zeros). ``base_to_common``
        is :math:`{}^C R_B`: it maps components expressed in this robot's base
        into the shared teleoperation frame. Identity preserves the historical
        aligned-base behavior.
        """
        self.robot = robot
        self.home = np.asarray(home_pose_vec, dtype=float).copy()
        self._wrench_fn = wrench_fn
        if base_to_common is None:
            self.base_to_common = Rotation.identity()
        elif isinstance(base_to_common, Rotation):
            self.base_to_common = base_to_common
        else:
            matrix = np.asarray(base_to_common, dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("base_to_common must be a finite 3x3 rotation matrix")
            self.base_to_common = Rotation.from_matrix(matrix)
        self.last_target_vec = self.home.copy()

    @property
    def _now_vec(self) -> NDArray:
        return pose_to_vec(self.robot.end_effector_pose)

    @property
    def position(self) -> NDArray:
        increment_base = increment_world(self.home, self._now_vec)
        return rotate_wrench(self.base_to_common, increment_base)

    @property
    def velocity(self) -> NDArray:
        tw = self.robot.end_effector_twist
        twist_tcp = np.concatenate([np.asarray(tw.linear, dtype=float),
                                    np.asarray(tw.angular, dtype=float)])
        tcp_to_common = self.base_to_common * _rot_of_vec(self._now_vec)
        return rotate_wrench(tcp_to_common, twist_tcp)

    @property
    def wrench(self) -> NDArray:
        if self._wrench_fn is None:
            return np.zeros(6)
        w_tcp = np.asarray(self._wrench_fn(), dtype=float)
        tcp_to_common = self.base_to_common * _rot_of_vec(self._now_vec)
        return rotate_wrench(tcp_to_common, w_tcp)

    def reanchor(self) -> None:
        """Re-home to the robot's current EE pose (after an episode-setup move)."""
        self.home = pose_to_vec(self.robot.end_effector_pose)
        self.last_target_vec = self.home.copy()

    def set_target_position(self, delta: NDArray) -> None:
        delta_base = rotate_wrench(self.base_to_common.inv(), np.asarray(delta, dtype=float))
        self.last_target_vec = integrate_pose(self.home, delta_base)
        self.robot.set_target(pose=self._vec_to_pose(self.last_target_vec))

    def set_feedforward_force(self, common_wrench: NDArray) -> None:
        tcp_to_common = self.base_to_common * _rot_of_vec(self._now_vec)
        w_tcp = rotate_wrench(tcp_to_common.inv(), np.asarray(common_wrench, dtype=float))
        self.robot.set_target_wrench(force=w_tcp[:3].tolist(), torque=w_tcp[3:].tolist())

    def _vec_to_pose(self, vec: NDArray):
        if hasattr(self.robot, "pose_from_vec"):
            return self.robot.pose_from_vec(vec)
        return vec_to_pose(vec)


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

    def reanchor(self) -> None:
        """Re-home to the robot's current joint values (after an episode-setup move)."""
        self.home = np.asarray(self.robot.joint_values, dtype=float).copy()
        self.last_target = self.home.copy()

    def set_target_position(self, delta: NDArray) -> None:
        self.last_target = self.home + np.asarray(delta, dtype=float)
        self.robot.set_target_joint(self.last_target)

    def set_feedforward_force(self, tau: NDArray) -> None:
        self.last_feedforward = np.asarray(tau, dtype=float)
        if self.apply_feedforward and hasattr(self.robot, "set_target_joint_torque"):
            self.robot.set_target_joint_torque(self.last_feedforward.tolist())
