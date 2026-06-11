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
import time

from crisp_gym.bilateral.bilateral_config import make_bilateral_config
from crisp_gym.bilateral.runtime import build_bilateral_controller
from crisp_gym.bilateral.telemetry import TeleopLogger
from crisp_gym.config.home import HomeConfig
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teleop-scheme", type=str, default="pf",
                   help="position | pf | pf_tdpa | pfpf | pfpf_tdpa | joint_pf")
    p.add_argument("--delay-steps", type=int, default=None,
                   help="Override the scheme's artificial round-trip channel delay (control steps).")
    p.add_argument("--feedback-gain", type=float, default=None,
                   help="Override the scheme's reflected-force gain.")
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

    logger.info(f":rocket: Bilateral teleop scheme={config.scheme} dt={dt:.4f}s "
                f"log={log_path}. Ctrl-C to stop.")
    t0 = time.monotonic()
    try:
        while True:
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
    except KeyboardInterrupt:
        logger.info("Stopping.")
    finally:
        if telem is not None and log_path:
            telem.to_jsonl(log_path)
            logger.info(f"Wrote telemetry: {log_path}")


if __name__ == "__main__":
    main()
