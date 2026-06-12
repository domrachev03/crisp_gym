"""Record leader-follower data in the structured schema (both arms + multi-signal hrate + cameras).

Writes the structured features (observation.{follower,leader}_state /
<arm>_<sig>_hrate / images.*) via ``crisp_gym.record.structured_record``.
Configs (panda env, leader, cameras) ship in crisp_gym/config.
"""

import argparse
import logging

import numpy as np
import rclpy

import crisp_gym  # noqa: F401
from crisp_gym.config.home import HomeConfig
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.bilateral.bilateral_config import make_bilateral_config
from crisp_gym.bilateral.runtime import build_bilateral_controller
from crisp_gym.config.path import find_config
from crisp_gym.record.episode_setup import EpisodeSetupConfig, EpisodeSetupRunner
from crisp_gym.record.hrate_process import HrateProcessManager
from crisp_gym.record.rerun_stream import RerunStreamer
from crisp_gym.record.structured_record import (
    DEFAULT_SIGNALS,
    CameraReader,
    HrateManager,
    build_hrate_specs,
    build_structured_features,
    load_camera_specs,
    make_structured_bilateral_fn,
    make_structured_teleop_fn,
)
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging


def _with_rerun(data_fn: callable, rerun: RerunStreamer) -> callable:
    """Wrap a frame producer so each non-skipped obs is also streamed to rerun.

    The wrapper logs only after a real (obs, action) is produced, so the first-frame
    ``(None, None)`` priming the teleop fn is ignored. rerun logging is non-fatal.
    """

    def _fn() -> tuple:
        obs, action = data_fn()
        if obs is not None:
            rerun.log_frame(obs)
        return obs, action

    return _fn


def main():  # noqa: C901
    """Record leader-follower data in the structured (both-arm) LeRobot schema."""
    p = argparse.ArgumentParser(description="Structured leader-follower recording")
    p.add_argument("--repo-id", type=str, default="test/structured")
    p.add_argument("--tasks", type=str, nargs="+", default=["pick the lego block."])
    p.add_argument("--robot-type", type=str, default="panda")
    p.add_argument("--fps", type=int, default=60, help="Record frame rate (RealSense run at this fps).")
    p.add_argument("--camera-fps", type=int, default=None, help="Camera fps (default: --fps).")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--recording-manager-type", type=str, default="keyboard", choices=["keyboard", "ros"])
    p.add_argument("--follower-config", type=str, default="panda_no_cam")
    p.add_argument("--leader-config", type=str, default="left_leader_nogripper")
    p.add_argument("--follower-namespace", type=str, default="right")
    p.add_argument("--leader-namespace", type=str, default="left")
    p.add_argument("--teleop-scheme", type=str, default=None,
                   help="Bilateral scheme (position|pf|pf_tdpa|pfpf|pfpf_tdpa|joint_pf). "
                        "Unset = legacy position-only teleop fn.")
    p.add_argument("--delay-steps", type=int, default=None,
                   help="Override the scheme's artificial round-trip channel delay (control steps).")
    p.add_argument("--feedback-gain", type=float, default=None,
                   help="Override the scheme's reflected-force gain.")
    p.add_argument("--setup-config", type=str, default=None,
                   help="Episode-setup config under config/setup/ (randomized start + retract). "
                        "Unset = no per-episode repositioning.")
    p.add_argument("--signals", type=str, nargs="+", default=DEFAULT_SIGNALS,
                   help=f"High-rate signals to record. Available: {DEFAULT_SIGNALS}.")
    p.add_argument("--hrate-window-ft", type=int, default=150,
                   help="Fixed ft high-rate window in samples (fps-independent, lerobot-panda style); "
                        "other signals scale by their native rate to cover the same time span.")
    p.add_argument("--topic-template", type=str, default="/{ns}/{topic}")
    p.add_argument("--hrate-process", action=argparse.BooleanOptionalAction, default=True,
                   help="Capture hrate in a separate process (own GIL) so F/T + joints keep "
                        "their full ~kHz rate. --no-hrate-process uses the in-process manager.")
    p.add_argument("--hrate-cores", type=int, nargs="+", default=None,
                   help="CPU cores to pin the hrate process to (e.g. --hrate-cores 6 7), so its "
                        "kHz callbacks never contend with the record/encode loop.")
    p.add_argument("--camera-config", type=str, default="realsense_rig",
                   help="Camera config under cameras/, or 'none'.")
    p.add_argument("--camera-read-timeout-ms", type=int, default=600,
                   help="Per-frame RealSense read timeout; a stall beyond this nullifies the episode.")
    p.add_argument("--image-writer-threads", type=int, default=8,
                   help="Async image-writer threads (0 = synchronous; processes deadlock with ROS).")
    p.add_argument("--no-target", dest="include_target", action="store_false", default=True)
    p.add_argument("--rerun", action=argparse.BooleanOptionalAction, default=False,
                   help="Stream each recorded frame to a rerun viewer (optional, non-fatal).")
    p.add_argument("--rerun-mode", type=str, default="spawn", choices=["spawn", "web", "save"],
                   help="rerun sink: 'spawn' native window (needs DISPLAY), 'web' headless "
                        "viewer for the dashboard iframe, 'save' write an .rrd file.")
    p.add_argument("--rerun-web-port", type=int, default=9090, help="rerun web-viewer HTML port (web mode).")
    p.add_argument("--rerun-ws-port", type=int, default=9877, help="rerun websocket data port (web mode).")
    p.add_argument("--rerun-save-path", type=str, default="crisp_record.rrd",
                   help="Output .rrd path (save mode).")
    p.add_argument("--rerun-images-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Log only camera images to rerun (skip F/T + pose); on by default.")
    p.add_argument("--rerun-max-fps", type=float, default=30.0,
                   help="Cap rerun frames/s (0=unlimited). Lower if the viewer lags; rr.log "
                        "is fire-and-forget so a too-fast feed accumulates downstream.")
    p.add_argument("--rerun-jpeg-quality", type=int, default=50,
                   help="JPEG quality for rerun image frames (lower = less bandwidth/latency).")
    p.add_argument("--rerun-memory-limit", type=str, default="10%",
                   help="rerun viewer store cap (drops old data to bound latency).")
    p.add_argument("--rerun-max-dim", type=int, default=320,
                   help="Downscale the rerun PREVIEW so max(H,W)<=this (0=full res). The "
                        "recorded dataset keeps full resolution; only the live preview shrinks.")
    p.add_argument("--home-config-noise", type=float, default=0.0)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = p.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)
    cam_fps = args.camera_fps if args.camera_fps is not None else args.fps
    for arg, value in vars(args).items():
        logger.info(f"{arg:<24}: {value}")

    env = leader = hrate = cams = None
    rerun = RerunStreamer(
        enabled=args.rerun, mode=args.rerun_mode,
        web_port=args.rerun_web_port, ws_port=args.rerun_ws_port,
        images_only=args.rerun_images_only, save_path=args.rerun_save_path,
        max_fps=args.rerun_max_fps, jpeg_quality=args.rerun_jpeg_quality,
        memory_limit=args.rerun_memory_limit, max_dim=args.rerun_max_dim,
    )
    try:
        env = make_env(env_type=args.follower_config, control_type="cartesian",
                       namespace=args.follower_namespace)
        leader = make_leader(args.leader_config, namespace=args.leader_namespace)
        leader.wait_until_ready()
        logger.info("Follower env + leader ready.")

        arms = {"follower": args.follower_namespace, "leader": args.leader_namespace}
        specs = build_hrate_specs(arms, args.signals, args.hrate_window_ft, args.topic_template)
        for key, topic, sig, window in specs:
            logger.info(f"hrate {key:<16} {topic:<36} window={window}")
        if args.hrate_process:
            logger.info(f"hrate capture: separate process (cores={args.hrate_cores})")
            hrate = HrateProcessManager(specs, core_affinity=args.hrate_cores)
        else:
            logger.info("hrate capture: in-process (legacy, GIL-shared)")
            hrate = HrateManager(specs)

        cam_specs = []
        if args.camera_config and args.camera_config.lower() != "none":
            cam_specs = load_camera_specs(args.camera_config)
            for s in cam_specs:
                s.fps = cam_fps
            cams = CameraReader(cam_specs, read_timeout_ms=args.camera_read_timeout_ms)

        # Bilateral scheme (opt-in). When set, the dataset records the controller
        # telemetry + a commanded-target action; dof = 7 joint, 6 cartesian.
        bilateral_config = None
        bilateral_dof = None
        if args.teleop_scheme is not None:
            overrides = {}
            if args.delay_steps is not None:
                overrides["delay_steps"] = args.delay_steps
            if args.feedback_gain is not None:
                overrides["feedback_gain"] = args.feedback_gain
            bilateral_config = make_bilateral_config(args.teleop_scheme, **overrides)
            bilateral_dof = 7 if bilateral_config.mode == "joint" else 6
            if args.home_config_noise > 0.0:
                logger.warning("home-config-noise > 0 with a bilateral scheme: homes are "
                               "anchored once, so keep noise 0 for consistent coupling.")

        # Episode-setup config (loaded early so we can decide whether to pre-home).
        setup_cfg = None
        if args.setup_config:
            setup_path = find_config(f"setup/{args.setup_config}.yaml")
            if setup_path is None:
                raise ValueError(f"Setup config not found: setup/{args.setup_config}.yaml")
            setup_cfg = EpisodeSetupConfig.from_yaml(setup_path)

        features, _ = build_structured_features(
            env, cam_specs, args.fps, args.signals, args.hrate_window_ft, args.include_target,
            bilateral_dof=bilateral_dof,
        )
        logger.info(f"Recording {len(features)} features at {args.fps} fps "
                    f"(scheme={args.teleop_scheme}).")

        rm = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features, repo_id=args.repo_id, robot_type=args.robot_type,
            num_episodes=args.num_episodes, fps=args.fps, resume=args.resume,
            push_to_hub=args.push_to_hub, queue_size=64,
            image_writer_threads=args.image_writer_threads, image_writer_processes=0,
        )
        rm.wait_until_ready()
        logger.info("Recording manager ready.")

        leader.prepare_for_teleop()
        env.wait_until_ready()
        # Skip the pre-loop home when the setup uses the CURRENT follower TCP as the
        # randomization origin (home_before_first=false) -- homing would move it away.
        keep_current = setup_cfg is not None and setup_cfg.enabled and not setup_cfg.home_before_first
        if keep_current:
            logger.info("Setup origin = current follower TCP (skipping pre-loop home).")
        else:
            env.home(home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
        env.reset()

        # Build the bilateral controller once, with both arms at home (anchors the
        # home-relative coupling and subscribes the wrench topics a single time).
        controller = None
        if bilateral_config is not None:
            controller = build_bilateral_controller(env, leader, bilateral_config)

        # Optional per-episode setup: randomized follower start + retract between episodes.
        setup_runner = None
        if setup_cfg is not None:
            if setup_cfg.enabled:
                setup_runner = EpisodeSetupRunner(env.robot, setup_cfg)
                logger.info(f"Episode setup '{args.setup_config}': retract {setup_cfg.retract_z_offset}m, "
                            f"jitter +-[{setup_cfg.start_jitter.x},{setup_cfg.start_jitter.y},"
                            f"{setup_cfg.start_jitter.z}]m yaw+-{setup_cfg.start_jitter.yaw_deg}deg")

        def on_start():
            env.robot.reset_targets()
            env.reset()
            leader.robot.reset_targets()
            leader.robot.cartesian_controller_parameters_client.load_param_config(
                leader.config.gravity_compensation_controller
            )
            leader.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            if leader.gripper is not None:
                leader.gripper.disable_torque()
            if controller is not None:
                # re-anchor the coupling at the current poses (follower at its setup
                # start, leader where the operator is ready) -> jump-free engage
                controller.reanchor()

        def on_end():
            env.robot.reset_targets()
            if setup_runner is None:
                # setup handles repositioning (retract + randomized start) next episode
                env.robot.home(blocking=False,
                               home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
            leader.robot.reset_targets()
            leader.robot.home(blocking=False)
            if env.gripper is not None:
                env.gripper.open()

        tasks = list(args.tasks)
        with rm:
            while not rm.done():
                logger.info(f"→ Episode {rm.episode_count + 1} / {rm.num_episodes}")
                if setup_runner is not None:
                    setup_runner.setup_episode(rm.episode_count)
                if controller is not None:
                    teleop_fn = make_structured_bilateral_fn(
                        env, leader, controller, hrate, cams, args.signals, args.include_target
                    )
                else:
                    teleop_fn = make_structured_teleop_fn(
                        env, leader, hrate, cams, args.signals, args.include_target
                    )
                if rerun.active:
                    teleop_fn = _with_rerun(teleop_fn, rerun)
                task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
                logger.info(f"▷ Task: {task}")
                rm.record_episode(data_fn=teleop_fn, task=task, on_start=on_start, on_end=on_end)

        logger.info("Homing robots.")
        leader.robot.home()
        env.home()
    except TimeoutError as e:
        logger.exception(f"Timeout during recording: {e}. Check robot containers / namespaces.")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Error during recording: {e}.")
    finally:
        rerun.close()
        if cams is not None:
            cams.close()
        if hrate is not None:
            hrate.close()
        if env is not None:
            env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
