"""Bilateral leader-follower teleop with selectable coupling, TDPA, and logging.

Integrates the tested control primitives (crisp_gym.bilateral.{pose_math,tdpa,
telemetry} + crisp_gym.sim.DelayedChannel) with real crisp_py robots. Cartesian
and joint modes, absolute/relative coupling, optional TDPA passivation, optional
artificial channel delay (for on-hardware stability testing), and full telemetry.

NOT unit-tested (ROS hardware path) — the control math it calls IS tested in
tests/bilateral/. Verify on hardware (e-stop ready); this is the bulk-deploy entry.

Examples:
    pixi run -e jazzy python crisp_gym/scripts/bilateral_teleop.py \
        --mode cartesian --coupling absolute --tdpa --log /tmp/run.jsonl
    pixi run -e jazzy python crisp_gym/scripts/bilateral_teleop.py \
        --mode joint --coupling absolute --no-force   # joint position teleop, master homed
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import qos_profile_sensor_data

from crisp_py.robot import Pose
from crisp_py.robot.robot_config import make_robot_config
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.pose_math import (
    aligned_pose,
    increment_world,
    integrate_pose,
    offset_joint,
)
from crisp_gym.bilateral.tdpa import MasterOnlyPOPC
from crisp_gym.bilateral.telemetry import TeleopLogger
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import make_env_config
from crisp_gym.sim import DelayedChannel
from crisp_gym.teleop.teleop_robot import TeleopRobot
from crisp_gym.teleop.teleop_robot_config import make_leader_config
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


# --- pose <-> 7-vector [x,y,z,qw,qx,qy,qz] helpers -------------------------------
def pose_to_vec(pose: Pose) -> np.ndarray:
    q = pose.orientation.as_quat()  # xyzw
    return np.array([*pose.position, q[3], q[0], q[1], q[2]])


def vec_to_pose(vec: np.ndarray) -> Pose:
    w, x, y, z = vec[3], vec[4], vec[5], vec[6]
    return Pose(position=np.asarray(vec[:3], dtype=float), orientation=Rotation.from_quat([x, y, z, w]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["cartesian", "joint"], default="cartesian")
    p.add_argument("--coupling", choices=["absolute", "relative"], default="absolute",
                   help="absolute: follower tracks leader pose/joints (no drift). "
                        "relative: follower integrates leader increments (clutchable, drifts).")
    p.add_argument("--tdpa", dest="tdpa", action="store_true", default=True,
                   help="Passivate the reflected wrench with TDPA (default on).")
    p.add_argument("--no-tdpa", dest="tdpa", action="store_false")
    p.add_argument("--force", dest="force", action="store_true", default=True,
                   help="Reflect follower force to the leader (default on).")
    p.add_argument("--no-force", dest="force", action="store_false")
    p.add_argument("--control-frequency", type=float, default=100.0)
    p.add_argument("--delay-steps", type=int, default=0,
                   help="Artificial round-trip channel delay in control steps (HW stability testing).")
    p.add_argument("--feedback-gain", type=float, default=1.0)
    p.add_argument("--feedback-sign", type=float, default=1.0)
    p.add_argument("--feedback-max-force", type=float, default=20.0)
    p.add_argument("--feedback-wrench-topic", type=str, default="/right/netft_data_unbiased_tcp")
    p.add_argument("--contact-threshold-n", type=float, default=1.0)
    p.add_argument("--leader-config", type=str, default="left_no_gripper")
    p.add_argument("--leader-namespace", type=str, default="left")
    p.add_argument("--follower-namespace", type=str, default="right")
    p.add_argument("--env-type", type=str, default="no_cam_franka")
    p.add_argument("--log", type=str, default=None, help="Path to write telemetry JSONL.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    dt = 1.0 / args.control_frequency

    # --- robots -----------------------------------------------------------------
    logger.info("Setting up leader robot...")
    leader_config = make_leader_config(args.leader_config)
    leader_config.use_gripper = False
    leader = TeleopRobot(config=leader_config, namespace=args.leader_namespace)
    leader.wait_until_ready()
    leader.prepare_for_teleop()  # homes leader + gravity-comp cartesian_impedance

    logger.info("Setting up follower environment...")
    env_config = make_env_config(args.env_type)
    env_config.robot_config = make_robot_config("panda")  # real hardware is a Panda
    env_config.joint_control_param_config = None
    env = ManipulatorCartesianEnv(config=env_config, namespace=args.follower_namespace)
    env.robot.home()
    env.reset()

    if args.mode == "joint":
        # 1:1 joint mapping needs identical configs; both already homed above.
        env.robot.controller_switcher_client.switch_controller("joint_impedance_controller")

    # --- constant home anchors (both arms are homed & settled at this point) -----
    # Leader (fr3_hand_tcp, no gripper) and follower (panda_hand_tcp, gripper) share
    # the same home joint config but report DIFFERENT end-effector poses (different
    # TCP frames). Capture both homes once: absolute coupling maps leader_home ->
    # follower_home (constant frame offset), so engagement is jump-free.
    if args.mode == "cartesian":
        leader_home = pose_to_vec(leader.robot.end_effector_pose)
        follower_home = pose_to_vec(env.robot.end_effector_pose)
        env.robot.set_target(pose=vec_to_pose(follower_home))  # hold home until loop commands
    else:
        leader_home = leader.robot.joint_values.copy()
        follower_home = env.robot.joint_values.copy()
        env.robot.set_target_joint(follower_home)
    prev_leader = leader_home.copy()        # relative-coupling increment anchor
    follower_target = follower_home.copy()  # relative-coupling integrated target

    # --- follower wrench subscription (force reflection source) ------------------
    follower_wrench = np.zeros(6)
    if args.force:
        def _wrench_cb(msg: WrenchStamped) -> None:
            follower_wrench[:] = (
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            )

        leader.robot.node.create_subscription(
            WrenchStamped, args.feedback_wrench_topic, _wrench_cb, qos_profile_sensor_data
        )
        logger.info("Capturing wrench bias (keep follower still)...")
        bias_samples = []
        t_end = time.time() + 1.0
        while time.time() < t_end:
            bias_samples.append(follower_wrench.copy())
            time.sleep(0.02)
        wrench_bias = np.mean(bias_samples, axis=0) if bias_samples else np.zeros(6)
    else:
        wrench_bias = np.zeros(6)

    # --- channels (optional artificial delay), TDPA, logger ---------------------
    # Forward-channel payload + priming fill depend on coupling:
    #   absolute -> full leader pose/joints; fill = leader_home, so an un-primed
    #               channel (first delay_steps ticks) yields the home target and the
    #               follower stays put -- no jump.
    #   relative -> per-step increments; fill = zeros.
    if args.coupling == "absolute":
        fwd_dim, fwd_fill = len(leader_home), leader_home
    else:
        fwd_dim, fwd_fill = (6 if args.mode == "cartesian" else 7), None
    ch_fwd = DelayedChannel(args.delay_steps, fwd_dim, fill=fwd_fill)  # leader -> follower
    ch_back = DelayedChannel(args.delay_steps, 6)     # follower -> leader (wrench)
    tdpa = MasterOnlyPOPC(dof=6, contact_threshold_n=args.contact_threshold_n) if args.tdpa else None
    telem = TeleopLogger() if args.log else None

    logger.info(f":rocket: Bilateral teleop: mode={args.mode} coupling={args.coupling} "
                f"tdpa={args.tdpa} force={args.force} delay_steps={args.delay_steps}")

    t0 = time.monotonic()
    step = 0
    try:
        while True:
            t_loop = time.monotonic()

            # ---- forward path: leader -> follower (with optional delay) ----------
            if args.mode == "cartesian":
                leader_vec = pose_to_vec(leader.robot.end_effector_pose)
                if args.coupling == "absolute":
                    ch_fwd.send(leader_vec)
                    leader_delayed = ch_fwd.receive()  # == leader_home until primed (fill)
                    # 1:1 world-axis mapping from matched home (no fr3/panda TCP twist)
                    target_vec = aligned_pose(follower_home, leader_home, leader_delayed)
                    env.robot.set_target(pose=vec_to_pose(target_vec))
                else:  # relative
                    incr = increment_world(prev_leader, leader_vec)
                    prev_leader = leader_vec
                    ch_fwd.send(incr)
                    follower_target = integrate_pose(follower_target, ch_fwd.receive())
                    env.robot.set_target(pose=vec_to_pose(follower_target))
            else:  # joint
                leader_q = leader.robot.joint_values
                if args.coupling == "absolute":
                    ch_fwd.send(leader_q)
                    leader_delayed = ch_fwd.receive()  # == leader_home until primed (fill)
                    tq = offset_joint(follower_home, leader_home, leader_delayed)
                    env.robot.set_target_joint(tq)
                else:
                    incr = leader_q - prev_leader
                    prev_leader = leader_q.copy()
                    ch_fwd.send(incr)
                    follower_target = follower_target + ch_fwd.receive()
                    env.robot.set_target_joint(follower_target)

            # ---- return path: follower wrench -> leader (with optional delay) ----
            reflected = np.zeros(6)
            if args.force:
                fe = args.feedback_sign * args.feedback_gain * (follower_wrench - wrench_bias)
                ch_back.send(fe)
                reflected = ch_back.receive()
                leader_twist = leader.robot.end_effector_twist
                vm = np.concatenate([leader_twist.linear, leader_twist.angular])
                if tdpa is not None:
                    reflected = tdpa.modify(reflected, vm, np.zeros(6), dt)
                # clamp force magnitude
                fmag = float(np.linalg.norm(reflected[:3]))
                if fmag > args.feedback_max_force:
                    reflected[:3] *= args.feedback_max_force / fmag
                leader.robot.set_target_wrench(force=reflected[:3].tolist(), torque=reflected[3:].tolist())

            # ---- telemetry -------------------------------------------------------
            if telem is not None:
                telem.log(
                    step=step, t=time.monotonic() - t0, t_mono=time.monotonic(),
                    leader_pos=leader.robot.end_effector_pose.position,
                    follower_pos=env.robot.end_effector_pose.position,
                    follower_wrench=follower_wrench.copy(),
                    reflected=reflected.copy(),
                )

            step += 1
            time.sleep(max(0.0, dt - (time.monotonic() - t_loop)))
    except KeyboardInterrupt:
        logger.info("Stopping.")
    finally:
        if telem is not None and args.log:
            telem.to_jsonl(args.log)
            logger.info(f"Wrote telemetry: {args.log}")


if __name__ == "__main__":
    main()
