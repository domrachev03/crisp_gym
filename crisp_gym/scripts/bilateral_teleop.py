"""Standalone bilateral leader-follower teleop (no recording).

Drives the shared :class:`crisp_gym.bilateral.controller.BilateralController` from a
named scheme (``position | pf | pf_tdpa | pfpf | pfpf_tdpa | joint_pf``) so this runner
and the structured recorder execute an identical control law. Use it to *feel* a
scheme on hardware before recording. The control math (coupling, reflection, TDPA)
is unit-tested in ``tests/bilateral/``; this entry is the HW path (keep e-stop ready).

Examples:
    pixi run -e jazzy python crisp_gym/scripts/bilateral_teleop.py --teleop-scheme pf
    pixi run -e jazzy python crisp_gym/scripts/bilateral_teleop.py \
        --teleop-scheme pf_tdpa --delay-steps 30 --log /tmp/run.jsonl
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

import rclpy

from crisp_gym.bilateral.bilateral_config import make_bilateral_config
from crisp_gym.bilateral.runtime import build_bilateral_controller
from crisp_gym.bilateral.telemetry import TeleopLogger
from crisp_gym.config.home import HomeConfig
from crisp_gym.config.path import find_config
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


def _hold_leader(leader) -> None:
    """On shutdown, stiffen the leader at its current pose (disable freedrive).

    The leader teleoperates in gravity-compensation (zero task stiffness = free to
    backdrive). On stop, command its current pose as the target and load a stiff
    cartesian-impedance config so it holds position instead of flopping.
    """
    if leader is None:
        return
    try:
        leader.robot.set_target(pose=leader.robot.end_effector_pose)  # target = current (no jump)
        stiff = find_config("control/default_cartesian_impedance.yaml")
        if stiff is None:
            logger.warning("default_cartesian_impedance config not found; leader left backdrivable.")
            return
        leader.robot.cartesian_controller_parameters_client.load_param_config(stiff)
        logger.info("Leader holding current pose (freedrive disabled).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not switch leader to hold: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teleop-scheme", type=str, default="pf",
                   help="position | pf | pf_tdpa | pfpf | pfpf_tdpa | joint_pf")
    p.add_argument("--delay-steps", type=int, default=None,
                   help="Override the scheme's artificial round-trip channel delay (control steps).")
    p.add_argument("--feedback-gain", type=float, default=None,
                   help="Override the scheme's reflected-force gain.")
    p.add_argument("--position-spring-k", type=float, default=None,
                   help="Override the 4-channel return position-spring stiffness (N/m).")
    p.add_argument("--rot-spring-k", type=float, default=None,
                   help="Override the rotational spring stiffness (N*m/rad).")
    p.add_argument("--force-fwd-gain", type=float, default=None,
                   help="Override the 4-channel forward-force gain (leader wrench -> follower ff).")
    p.add_argument("--control-frequency", type=float, default=None,
                   help="Override the scheme's control frequency (Hz).")
    p.add_argument("--leader-config", type=str, default=None,
                   help="Override the scheme's leader config name.")
    p.add_argument("--leader-namespace", type=str, default=None)
    p.add_argument("--follower-namespace", type=str, default=None)
    p.add_argument("--follower-config", type=str, default=None,
                   help="Override the scheme's follower env config name.")
    p.add_argument("--log", type=str, default=None,
                   help="Telemetry JSONL path (default: /tmp/bilateral_<scheme>.jsonl).")
    p.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                   help="Disable telemetry logging.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    overrides = {}
    if args.delay_steps is not None:
        overrides["delay_steps"] = args.delay_steps
    if args.feedback_gain is not None:
        overrides["feedback_gain"] = args.feedback_gain
    if args.position_spring_k is not None:
        overrides["position_spring_k"] = args.position_spring_k
    if args.rot_spring_k is not None:
        overrides["rot_spring_k"] = args.rot_spring_k
    if args.force_fwd_gain is not None:
        overrides["force_fwd_gain"] = args.force_fwd_gain
    if args.control_frequency is not None:
        overrides["control_frequency"] = args.control_frequency
    if args.leader_config is not None:
        overrides["leader_config"] = args.leader_config
    if args.leader_namespace is not None:
        overrides["leader_namespace"] = args.leader_namespace
    if args.follower_namespace is not None:
        overrides["follower_namespace"] = args.follower_namespace
    if args.follower_config is not None:
        overrides["follower_env_config"] = args.follower_config
    config = make_bilateral_config(args.teleop_scheme, **overrides)
    dt = 1.0 / config.control_frequency

    logger.info("Setting up leader robot...")
    leader = make_leader(config.leader_config, namespace=config.leader_namespace)
    leader.wait_until_ready()
    leader.prepare_for_teleop()  # homes leader + gravity-comp cartesian_impedance

    logger.info("Setting up follower environment...")
    control_type = "joint" if config.mode == "joint" else "cartesian"
    env = make_env(config.follower_env_config, control_type=control_type,
                   namespace=config.follower_namespace)
    env.wait_until_ready()
    env.home(home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=0.0))
    env.reset()

    controller = build_bilateral_controller(env, leader, config, dt=dt)
    log_path = None if args.no_log else (args.log or f"/tmp/bilateral_{config.scheme}.jsonl")
    telem = TeleopLogger() if log_path else None

    # Install our own SIGINT handler: rclpy's default handler shuts the ROS context
    # down the instant Ctrl-C lands, which invalidates the node before _hold_leader's
    # service call can run ("rcl node's context is invalid"). Setting a flag instead
    # lets the loop exit with the context still alive, so the leader hold succeeds.
    stop = {"flag": False}

    def _on_sigint(signum, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    logger.info(f":rocket: Bilateral teleop scheme={config.scheme} dt={dt:.4f}s "
                f"log={log_path}. Ctrl-C to stop.")
    t0 = time.monotonic()
    try:
        while not stop["flag"]:
            t_loop = time.monotonic()
            tel = controller.step(dt=dt, human_force=None)  # human pushes the leader physically
            if telem is not None:
                # full telemetry per tick so any scheme is analyzable offline
                telem.log(
                    step=tel.step, t=time.monotonic() - t0, t_mono=time.monotonic(),
                    leader_pos=tel.live_leader_pos, follower_pos=tel.live_follower_pos,
                    commanded_target=tel.commanded_follower_target,
                    leader_feedforward=tel.leader_feedforward, reflected=tel.reflected_wrench,
                    spring_force=tel.spring_force, forward_force=tel.forward_force,
                    follower_wrench=tel.follower_wrench, leader_wrench=tel.leader_wrench,
                    delay_steps=tel.delay_steps, control_dt=tel.control_dt,
                )
            time.sleep(max(0.0, dt - (time.monotonic() - t_loop)))
    finally:
        logger.info("Stopping.")
        _hold_leader(leader)  # ROS context still valid here -> hold actually applies
        if telem is not None and log_path:
            telem.to_jsonl(log_path)
            logger.info(f"Wrote telemetry: {log_path}")
        if rclpy.ok():
            rclpy.try_shutdown()  # end the crisp_py spin thread cleanly after the hold


if __name__ == "__main__":
    main()
