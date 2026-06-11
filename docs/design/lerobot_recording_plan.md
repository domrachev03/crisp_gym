# lerobot-panda recording adaptation plan — for step 4 (crisp_gym-1.5)

Goal: record crisp_gym-driven datasets at max freq incl cameras + proprio + F/T, hrate sub-step F/T logging.
**Good news: lerobot-panda already has a complete hrate window mechanism** — replicate client-side for ROS2.

## Robot backend — implement `BaseFrankaRobot` (robots/base/robot_base.py), 6 abstract methods:
`_connect_transport`, `_disconnect_transport`, `_fetch_raw_observation()->dict` (non-blocking, cached),
`_send_pose_command(pos, Rotation, gripper_norm)`, `_send_gripper_command(int)`, `_send_home_command(timeout)`.
Optional `_on_connected()`. Raw obs dict keys: `ee_pos(3), ee_quat_xyzw(4), joint_positions(7|None),
joint_velocities, gripper_pos(float 0-1), sensors{name:val}, sensors_hrate{name:{values:(N,K),times:(N,)}},
twist, t_obs`. `observation_features`/`action_features` auto-built from config flags.

New robot files: `robots/crisp_gym/{__init__,config_crisp_gym,robot_crisp_gym}.py`.
`CrispGymRobot` owns a `ManipulatorCartesianEnv(namespace="right")`; `_fetch_raw_observation` maps
`env.robot.end_effector_pose`(.position/.orientation.as_quat), `obs["observation.state.joints/gripper"]`,
sensor `.value`(6) -> raw keys. `_send_pose_command` -> `env.step(concat(pos, R.as_euler('xyz'), [grip]), block=False)`.

## hrate F/T (client-side, no server handshake): per sensor with hrate_window=N, start daemon poller thread:
poll `sensor.value` at ~2x native rate (NetFT ~500Hz on /right/external_wrench or /right/netft_data_unbiased_tcp),
append `(perf_counter, val.copy())` to `deque(maxlen=4N)` on change. In `_fetch_raw_observation`: snapshot latest N,
`times_rel = times - t_obs`, pad zeros at front if <N. Seed `self._confirmed_hrate_windows[name]=N` at connect.
Rest of path identical to CrispWS: base writes `sensor.hrate_<name>.{i}.{j}` + `_t.{i}` scalars into obs frame.
Reconstruct offline via `utils/hrate_reconstruction.py:reconstruct_by_timestamps`.

## Teleoperator — implement `lerobot.teleoperators.Teleoperator` (ref teleoperators/haply/teleop_haply.py):
`connect/disconnect, get_action()->dict, get_teleop_events(), send_feedback, calibrate, configure, is_connected,
is_calibrated, action_features, feedback_features`. New: `teleoperators/crisp_gym_leader/...`. Wrap `make_leader`
(crisp_gym TeleopRobot, namespace="left"); `get_action` returns Haply-schema dict {x,y,z,qw,qx,qy,qz,vx,vy,vz,
button_a/b/c} so existing processor pipeline reuses. is_intervention always True (leader-follower always active).

## Record loop (scripts/record_loop.py:222-348), fps from configs/record.yaml (default 60). Per step:
get_observation -> obs processor -> build_dataset_frame(prefix=observation) -> teleop.get_action -> action processor
-> robot.send_action -> build_dataset_frame(prefix=action) -> dataset.add_frame -> precise_sleep(1/fps-elapsed).
LeRobot v3: parquet frames + video cameras under HF_LEROBOT_HOME.

## Hydra registration: extend `configs.py` build_robot_config/make_robot/build_teleop_config for
`_target_: crisp_gym_ros2` / `crisp_gym_leader`; add config groups `configs/robot/crisp_gym.yaml`,
`configs/teleop/crisp_gym_leader.yaml`, `configs/experiment/crisp_gym_leader_follower_record.yaml`.
Subtasks: 1.5.1 robot, 1.5.2 teleop, 1.5.3 hrate, 1.5.4 record loop.

Key files: robots/base/robot_base.py, robots/crisp_ws/robot_crisp_ws.py (transport+hrate ref),
teleoperators/haply/teleop_haply.py, scripts/record{,_loop}.py, configs.py, utils/hrate_reconstruction.py,
crisp_gym/envs/manipulator_env.py (get_obs/step), record/record_functions.py (make_teleop_fn).
