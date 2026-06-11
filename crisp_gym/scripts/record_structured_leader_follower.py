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
from crisp_gym.record.structured_record import (
    DEFAULT_SIGNALS,
    CameraReader,
    HrateManager,
    build_hrate_specs,
    build_structured_features,
    load_camera_specs,
    make_structured_teleop_fn,
)
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging


def main():  # noqa: C901
    """Record leader-follower data in the structured (both-arm) LeRobot schema."""
    p = argparse.ArgumentParser(description="Structured leader-follower recording")
    p.add_argument("--repo-id", type=str, default="test/structured")
    p.add_argument("--tasks", type=str, nargs="+", default=["pick the lego block."])
    p.add_argument("--robot-type", type=str, default="panda")
    p.add_argument("--fps", type=int, default=30, help="Record frame rate (cameras can do 60).")
    p.add_argument("--camera-fps", type=int, default=None, help="Camera fps (default: --fps).")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--recording-manager-type", type=str, default="keyboard", choices=["keyboard", "ros"])
    p.add_argument("--follower-config", type=str, default="panda_no_cam")
    p.add_argument("--leader-config", type=str, default="left_leader_nogripper")
    p.add_argument("--follower-namespace", type=str, default="right")
    p.add_argument("--leader-namespace", type=str, default="left")
    p.add_argument("--signals", type=str, nargs="+", default=DEFAULT_SIGNALS,
                   help=f"High-rate signals to record. Available: {DEFAULT_SIGNALS}.")
    p.add_argument("--hrate-overlap", type=float, default=2.0,
                   help="Inter-frame gaps each hrate window covers (>=1 gives overlap).")
    p.add_argument("--topic-template", type=str, default="/{ns}/{topic}")
    p.add_argument("--camera-config", type=str, default="realsense_rig",
                   help="Camera config under cameras/, or 'none'.")
    p.add_argument("--image-writer-threads", type=int, default=8,
                   help="Async image-writer threads (0 = synchronous; processes deadlock with ROS).")
    p.add_argument("--no-target", dest="include_target", action="store_false", default=True)
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
    try:
        env = make_env(env_type=args.follower_config, control_type="cartesian",
                       namespace=args.follower_namespace)
        leader = make_leader(args.leader_config, namespace=args.leader_namespace)
        leader.wait_until_ready()
        logger.info("Follower env + leader ready.")

        arms = {"follower": args.follower_namespace, "leader": args.leader_namespace}
        specs = build_hrate_specs(arms, args.signals, args.fps, args.hrate_overlap, args.topic_template)
        for key, topic, sig, window in specs:
            logger.info(f"hrate {key:<16} {topic:<36} window={window}")
        hrate = HrateManager(specs)

        cam_specs = []
        if args.camera_config and args.camera_config.lower() != "none":
            cam_specs = load_camera_specs(args.camera_config)
            for s in cam_specs:
                s.fps = cam_fps
            cams = CameraReader(cam_specs)

        features, _ = build_structured_features(
            env, cam_specs, args.fps, args.signals, args.hrate_overlap, args.include_target
        )
        logger.info(f"Recording {len(features)} features at {args.fps} fps.")

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
        env.home(home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
        env.reset()

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

        def on_end():
            env.robot.reset_targets()
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
                teleop_fn = make_structured_teleop_fn(
                    env, leader, hrate, cams, args.signals, args.include_target
                )
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
