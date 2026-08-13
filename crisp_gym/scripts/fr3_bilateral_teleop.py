"""Mixed-Humble/Jazzy FR3 bilateral teleoperation over standard ROS messages.

Run endpoint bring-up and the distro-local supervisor on each RT PC first.  This
process never calls controller-manager or Franka services across distributions.
Without ``--arm`` it performs only a graph/frame/rate preflight and exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.bilateral.bilateral_config import BilateralConfig
from crisp_gym.bilateral.controller import BilateralController
from crisp_gym.bilateral.crisp_adapter import CrispCartesianAdapter, pose_to_vec
from crisp_gym.bilateral.standard_endpoint import StandardMessageEndpoint
from crisp_gym.bilateral.telemetry import TeleopLogger

logger = logging.getLogger(__name__)

_FRAME_CHECK_MAX_TRANSLATION_M = 0.010
_FRAME_CHECK_MAX_ROTATION_RAD = 0.010
_FRAME_CHECK_MAX_COMMAND_STEP_M = 0.00025
_FRAME_CHECK_MAX_COMMAND_STEP_RAD = 0.001


def _rotation(values: list[float] | None, name: str) -> Rotation:
    if values is None:
        return Rotation.identity()
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must be four finite XYZW values")
    norm = float(np.linalg.norm(quaternion))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"{name} quaternion norm must be approximately one, got {norm}")
    return Rotation.from_quat(quaternion / norm)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme", choices=("position", "pf"), default="pf",
        help="Only the two reviewed dual-FR3 commissioning profiles are accepted.",
    )
    parser.add_argument("--leader-namespace", default="leader")
    parser.add_argument("--follower-namespace", default="follower")
    parser.add_argument(
        "--leader-domain-id", type=int, default=78,
        help="ROS domain used only by the Jazzy leader endpoint.",
    )
    parser.add_argument(
        "--follower-domain-id", type=int, default=79,
        help="ROS domain used only by the Humble follower endpoint.",
    )
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--tcp-frame", default="fr3_hand_tcp")
    parser.add_argument(
        "--leader-base-to-common-quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"),
        help="Audited ^C R_Bleader quaternion. Mandatory for arming the tilted leader.",
    )
    parser.add_argument(
        "--follower-base-to-common-quat", type=float, nargs=4,
        default=[0.0, 0.0, 0.0, 1.0], metavar=("X", "Y", "Z", "W"),
    )
    parser.add_argument(
        "--transforms-verified", action="store_true",
        help="Confirm that both base mappings passed the documented physical +axis test.",
    )
    parser.add_argument("--arm", action="store_true", help="Enable commands after preflight.")
    parser.add_argument(
        "--candidate-frame-check", action="store_true",
        help=(
            "Permit only a tightly bounded, timed position check of an unverified base "
            "transform. This does not approve the transform for normal teleoperation."
        ),
    )
    parser.add_argument(
        "--frame-check-duration-s", type=float, default=30.0,
        help="Automatic stop time for --candidate-frame-check (default: 30 seconds).",
    )
    parser.add_argument("--feedback-gain", type=float, default=None)
    parser.add_argument("--feedback-ramp-s", type=float, default=3.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.10)
    parser.add_argument("--bias-seconds", type=float, default=1.0)
    parser.add_argument("--bias-max-force", type=float, default=5.0)
    parser.add_argument("--bias-max-torque", type=float, default=0.5)
    parser.add_argument("--ready-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--log", type=Path,
        help="JSONL telemetry path (default when armed: /tmp/fr3_bilateral_<scheme>.jsonl).",
    )
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _validate_arm_mode(args: argparse.Namespace) -> None:
    if args.candidate_frame_check:
        if not args.arm:
            raise SystemExit("Refusing frame check: --candidate-frame-check requires --arm")
        if args.scheme != "position":
            raise SystemExit("Refusing frame check: only --scheme position is permitted")
        if args.leader_base_to_common_quat is None:
            raise SystemExit(
                "Refusing frame check: provide the candidate --leader-base-to-common-quat"
            )
        if not np.isfinite(args.frame_check_duration_s) or args.frame_check_duration_s <= 0.0:
            raise SystemExit("Refusing frame check: --frame-check-duration-s must be positive")
        return
    if args.arm and (not args.transforms_verified or args.leader_base_to_common_quat is None):
        raise SystemExit(
            "Refusing to arm: provide --leader-base-to-common-quat and "
            "--transforms-verified after the physical axis test, or use the bounded "
            "--candidate-frame-check mode"
        )


def _apply_candidate_frame_limits(config: BilateralConfig) -> None:
    config.max_translation_m = _FRAME_CHECK_MAX_TRANSLATION_M
    config.max_rotation_rad = _FRAME_CHECK_MAX_ROTATION_RAD
    config.max_command_step_m = _FRAME_CHECK_MAX_COMMAND_STEP_M
    config.max_command_step_rad = _FRAME_CHECK_MAX_COMMAND_STEP_RAD


def _safe_hold(endpoints: tuple[StandardMessageEndpoint, ...], repeats: int = 20) -> None:
    for _ in range(repeats):
        for endpoint in endpoints:
            try:
                endpoint.publish_hold()
            except Exception as error:  # noqa: BLE001
                logger.error("safe-hold publish failed for %s: %s", endpoint.namespace, error)
        time.sleep(0.01)


def main() -> None:
    """Run fail-closed mixed-version FR3 preflight or bounded teleoperation."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    _validate_arm_mode(args)
    leader_rotation = _rotation(args.leader_base_to_common_quat, "leader base mapping")
    follower_rotation = _rotation(args.follower_base_to_common_quat, "follower base mapping")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config" / "teleop" / "bilateral" / f"{args.scheme}.yaml"
    )
    overrides = {} if args.feedback_gain is None else {"feedback_gain": args.feedback_gain}
    config = BilateralConfig.from_yaml(config_path, **overrides)
    if config.mode != "cartesian":
        raise RuntimeError("the mixed-distro standard-message runner supports Cartesian mode only")
    if args.candidate_frame_check:
        _apply_candidate_frame_limits(config)
        logger.warning(
            "CANDIDATE FRAME CHECK: transform is unverified; limiting motion to %.1f mm, "
            "%.3f rad, and %.2f mm command steps for %.1f seconds",
            config.max_translation_m * 1000.0,
            config.max_rotation_rad,
            config.max_command_step_m * 1000.0,
            args.frame_check_duration_s,
        )

    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    try:
        from rclpy.signals import SignalHandlerOptions
        signal_options = SignalHandlerOptions.NO
    except ImportError:
        signal_options = None

    if args.leader_domain_id == args.follower_domain_id:
        logger.warning(
            "leader and follower share ROS domain %d; mixed Humble/Jazzy discovery may emit "
            "incompatible type-hash/deserialization errors",
            args.leader_domain_id,
        )

    # One ROS context per endpoint keeps incompatible Humble/Jazzy service and
    # discovery schemas in different domains.  Only the stable geometry-message
    # data plane enters this process; no DDS bridge or controller service crosses
    # the distro boundary.
    contexts: list[Context] = []
    nodes = []
    executors: list[SingleThreadedExecutor] = []
    spin_threads: list[threading.Thread] = []
    for role, domain_id in (
        ("leader", args.leader_domain_id),
        ("follower", args.follower_domain_id),
    ):
        context = Context()
        init_kwargs = {"context": context, "domain_id": domain_id}
        if signal_options is not None:
            init_kwargs["signal_handler_options"] = signal_options
        try:
            rclpy.init(**init_kwargs)
        except TypeError:
            # Older rclpy accepts domain_id but not SignalHandlerOptions.NO.
            init_kwargs.pop("signal_handler_options", None)
            rclpy.init(**init_kwargs)
        node = rclpy.create_node(f"fr3_bilateral_{role}_data_plane", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        thread = threading.Thread(
            target=executor.spin, name=f"ros-{role}-data-plane", daemon=True
        )
        thread.start()
        contexts.append(context)
        nodes.append(node)
        executors.append(executor)
        spin_threads.append(thread)

    leader_node, follower_node = nodes
    leader: StandardMessageEndpoint | None = None
    follower: StandardMessageEndpoint | None = None
    telemetry: TeleopLogger | None = None
    log_path: Path | None = None
    try:
        leader = StandardMessageEndpoint(
            leader_node, namespace=args.leader_namespace, base_frame=args.base_frame,
            tcp_frame=args.tcp_frame,
            wrench_topic="netft_data_unbiased_tcp",
        )
        follower = StandardMessageEndpoint(
            follower_node, namespace=args.follower_namespace, base_frame=args.base_frame,
            tcp_frame=args.tcp_frame,
            wrench_topic="netft_data_unbiased_tcp",
        )
        endpoints = (leader, follower)
        for endpoint in endpoints:
            endpoint.wait_ready(args.ready_timeout_s, require_wrench=config.force)
        time.sleep(0.5)  # allow DDS graph publisher counts to converge
        for endpoint in endpoints:
            endpoint.assert_command_ownership()
        logger.info("Both endpoint streams, frame IDs, and command ownership passed preflight")
        if not args.arm:
            logger.info("Preflight-only run complete; no command was published")
            return

        if config.force:
            bias = follower.capture_wrench_bias(args.bias_seconds)
            if (
                np.linalg.norm(bias[:3]) > args.bias_max_force
                or np.linalg.norm(bias[3:]) > args.bias_max_torque
            ):
                raise RuntimeError(
                    f"follower residual bias {bias.tolist()} exceeds the arming threshold; "
                    "review NetFT calibration instead of hiding it"
                )
            logger.info(
                "Follower residual wrench bias accepted: %s",
                np.array2string(bias, precision=4),
            )

        leader_adapter = CrispCartesianAdapter(
            leader, pose_to_vec(leader.end_effector_pose), wrench_fn=lambda: leader.wrench,
            base_to_common=leader_rotation,
        )
        follower_adapter = CrispCartesianAdapter(
            follower, pose_to_vec(follower.end_effector_pose), wrench_fn=lambda: follower.wrench,
            base_to_common=follower_rotation,
        )
        requested_gain = config.feedback_gain
        config.feedback_gain = 0.0
        controller = BilateralController(leader_adapter, follower_adapter, config)
        if not args.no_log:
            log_path = args.log or Path(f"/tmp/fr3_bilateral_{args.scheme}.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry = TeleopLogger()

        _safe_hold(endpoints, repeats=5)
        start = time.monotonic()
        previous = start
        next_tick = start
        period = 1.0 / config.control_frequency
        state_streams = ("pose", "twist", "wrench") if config.force else ("pose", "twist")
        if args.candidate_frame_check:
            logger.warning(
                "CANDIDATE FRAME CHECK ARMED at %.1f Hz for at most %.1f seconds; "
                "move one leader axis only a few millimeters and stop on any wrong direction",
                config.control_frequency,
                args.frame_check_duration_s,
            )
        else:
            logger.info(
                "ARMED: %s at %.1f Hz; reflected gain ramps 0 -> %.3f over %.1f s",
                config.scheme, config.control_frequency, requested_gain, args.feedback_ramp_s,
            )
        while not stop.is_set():
            now = time.monotonic()
            ages = {
                f"{endpoint.namespace}/{stream}": endpoint.stream_age(stream)
                for endpoint in endpoints for stream in state_streams
            }
            stale = {name: age for name, age in ages.items() if age > args.state_timeout_s}
            if stale:
                raise RuntimeError(f"stale endpoint data: {stale}")
            elapsed = now - start
            if args.candidate_frame_check and elapsed >= args.frame_check_duration_s:
                logger.info(
                    "Candidate frame-check time limit reached; the transform remains "
                    "unapproved until the physical direction observation is recorded"
                )
                break
            ramp = 1.0 if args.feedback_ramp_s <= 0.0 else min(1.0, elapsed / args.feedback_ramp_s)
            config.feedback_gain = requested_gain * ramp
            measured_dt = min(max(now - previous, 0.5 * period), 5.0 * period)
            previous = now
            sample = controller.step(dt=measured_dt)
            if telemetry is not None:
                telemetry.log(
                    step=sample.step,
                    t=now - start,
                    control_dt=sample.control_dt,
                    feedback_gain=config.feedback_gain,
                    stream_ages_s=ages,
                    leader_pos=sample.live_leader_pos,
                    follower_pos=sample.live_follower_pos,
                    commanded_target=sample.commanded_follower_target,
                    leader_feedforward=sample.leader_feedforward,
                    reflected_wrench=sample.reflected_wrench,
                    follower_wrench=sample.follower_wrench,
                    leader_wrench=sample.leader_wrench,
                )
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()
    finally:
        if leader is not None and follower is not None and args.arm:
            logger.info("Disarming: zero wrench and current-pose hold on both endpoints")
            _safe_hold((leader, follower))
        if telemetry is not None and log_path is not None:
            try:
                telemetry.to_jsonl(log_path)
                logger.info("Wrote %d telemetry samples to %s", len(telemetry.records), log_path)
            except Exception as error:  # noqa: BLE001
                logger.error("Could not write telemetry to %s: %s", log_path, error)
        for executor in executors:
            executor.shutdown(timeout_sec=2.0)
        for node in nodes:
            node.destroy_node()
        for context in contexts:
            if context.ok():
                rclpy.shutdown(context=context)
        for thread in spin_threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
