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
import os
import signal
import sys
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


def _apply_realtime(priority: int | None, cpu: int | None) -> None:
    """Best-effort RT setup: pin the control thread to a CPU and/or raise SCHED_FIFO.

    Both are opt-in (need privilege / a free isolated core) and never fatal -- a warning
    is logged and the loop runs at normal priority if they fail.
    """
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {cpu})
            logger.info(f"Pinned control thread to CPU {cpu}.")
        except OSError as e:  # noqa: BLE001
            logger.warning(f"sched_setaffinity(CPU {cpu}) failed: {e}")
    if priority is not None:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
            logger.info(f"Control thread SCHED_FIFO priority {priority}.")
        except (OSError, PermissionError) as e:  # noqa: BLE001
            logger.warning(f"SCHED_FIFO prio {priority} failed (need CAP_SYS_NICE/root): {e}")


def _log_rate_stats(periods: list[float], overruns: int, target_hz: float) -> None:
    """Print the achieved loop-rate distribution so each tuning iteration is measurable."""
    if len(periods) < 3:
        return
    s = sorted(periods)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))] * 1e3  # ms

    mean = sum(periods) / len(periods)
    logger.info(
        "loop rate: %.1f Hz achieved (target %.0f) | period ms p50 %.2f p90 %.2f "
        "p99 %.2f max %.2f | overruns %d/%d (%.1f%%)",
        (1.0 / mean) if mean > 0 else float("inf"), target_hz,
        pct(0.50), pct(0.90), pct(0.99), max(periods) * 1e3,
        overruns, len(periods), 100.0 * overruns / max(1, len(periods)),
    )


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
    p.add_argument("--rt-priority", type=int, default=None,
                   help="Raise the control thread to SCHED_FIFO at this priority "
                        "(needs CAP_SYS_NICE/root; best-effort, non-fatal).")
    p.add_argument("--rt-cpu", type=int, default=None,
                   help="Pin the control thread to this CPU core (best-effort, non-fatal).")
    p.add_argument("--wrench-process", action="store_true", default=False,
                   help="Offload the NetFT wrench subscriptions (~2.1kHz x2) to a separate "
                        "poller process so their callbacks don't starve the control loop's GIL.")
    p.add_argument("--wrench-cores", type=int, nargs="+", default=None,
                   help="CPU cores to pin the wrench poller process to (e.g. --wrench-cores 26 27).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    # The crisp_py per-robot executor threads fire ~10k callbacks/s (NetFT ~2.1kHz x2,
    # joints 1kHz x2, pose/twist 250Hz). With the default 5ms GIL switch interval the
    # spin thread can hold the GIL multi-ms, so our time.sleep wakes late and the loop
    # jitters below target. Cap the switch interval to 1ms so control regains the GIL
    # promptly.
    sys.setswitchinterval(0.001)

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

    controller = build_bilateral_controller(env, leader, config, dt=dt,
                                             wrench_process=args.wrench_process,
                                             wrench_cores=args.wrench_cores)
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

    _apply_realtime(args.rt_priority, args.rt_cpu)
    logger.info(f":rocket: Bilateral teleop scheme={config.scheme} "
                f"target={config.control_frequency:.0f}Hz dt={dt:.4f}s "
                f"log={log_path}. Ctrl-C to stop.")
    t0 = time.monotonic()
    next_t = time.monotonic()       # absolute-time schedule anchor (drift-free pacing)
    prev = None                     # previous tick stamp -> measured period
    meas_periods: list[float] = []
    overruns = 0
    try:
        while not stop["flag"]:
            now = time.monotonic()
            # Feed the REAL measured period to the control law so TDPA energy
            # (power*dt, alpha=E/(V^2*dt)) and any dt-based term match wall time
            # instead of the nominal 1/freq. Clamp pathological gaps (first tick /
            # post-preemption) so a multi-ms stall can't inject a huge energy step.
            meas_dt = dt if prev is None else (now - prev)
            prev = now
            step_dt = min(max(meas_dt, 0.5 * dt), 5.0 * dt)
            tel = controller.step(dt=step_dt, human_force=None)  # human pushes the leader physically
            if telem is not None:
                # full telemetry per tick so any scheme is analyzable offline
                telem.log(
                    step=tel.step, t=now - t0, t_mono=now,
                    leader_pos=tel.live_leader_pos, follower_pos=tel.live_follower_pos,
                    commanded_target=tel.commanded_follower_target,
                    leader_feedforward=tel.leader_feedforward, reflected=tel.reflected_wrench,
                    spring_force=tel.spring_force, forward_force=tel.forward_force,
                    follower_wrench=tel.follower_wrench, leader_wrench=tel.leader_wrench,
                    delay_steps=tel.delay_steps, control_dt=tel.control_dt, loop_dt=meas_dt,
                )
            meas_periods.append(meas_dt)
            # Absolute-time pacing: advance the schedule by one nominal period and sleep
            # to that instant (drift-free vs re-measuring from the loop top each tick).
            # time.sleep releases the GIL so the crisp_py executors can update state while
            # we wait. On overrun (work outran the period) re-anchor rather than burst
            # catch-up, which would otherwise fire several zero-sleep ticks back-to-back.
            next_t += dt
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
                next_t = time.monotonic()
    finally:
        logger.info("Stopping.")
        _hold_leader(leader)  # ROS context still valid here -> hold actually applies
        _log_rate_stats(meas_periods, overruns, config.control_frequency)
        if getattr(controller, "wrench_proc", None) is not None:
            controller.wrench_proc.close()  # stop the poller process + free shared memory
        if telem is not None and log_path:
            telem.to_jsonl(log_path)
            logger.info(f"Wrote telemetry: {log_path}")
        if rclpy.ok():
            rclpy.try_shutdown()  # end the crisp_py spin thread cleanly after the hold


if __name__ == "__main__":
    main()
