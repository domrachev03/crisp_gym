"""Hardware glue: build a ``BilateralController`` from a ``BilateralConfig``.

Captures the leader/follower homes, wires the NetFT wrench subscriptions a scheme
needs, builds the cartesian or joint adapters, switches the follower controller for
joint mode, and holds the follower at home until the loop commands it. Both the
standalone runner and the structured recorder call this so they drive an identical
control law.

ROS / crisp_py only — exercised on hardware, not in the dry-run test suite (the law
and the adapter math are unit-tested separately against mocks).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import qos_profile_sensor_data

from crisp_gym.bilateral.bilateral_config import BilateralConfig
from crisp_gym.bilateral.controller import BilateralController
from crisp_gym.bilateral.crisp_adapter import (
    CrispCartesianAdapter,
    CrispJointAdapter,
    pose_to_vec,
    vec_to_pose,
)
from crisp_gym.config.path import find_config

# Follower cartesian impedance loaded for bilateral teleop (lowered rotational PD).
FOLLOWER_CARTESIAN_IMPEDANCE = "control/teleop_cartesian_impedance.yaml"

logger = logging.getLogger(__name__)


def _subscribe_wrench(node, topic: str, bias_seconds: float = 0.0):
    """Subscribe to a NetFT ``WrenchStamped`` topic; return a getter of the live
    (optionally bias-subtracted) TCP wrench ``(6,)``.

    A non-zero ``bias_seconds`` averages the wrench at rest (keep the arm still) and
    subtracts it — used for the follower so the free-space noise floor is nulled.
    """
    latest = np.zeros(6)

    def _cb(msg: WrenchStamped) -> None:
        w = msg.wrench
        latest[:] = (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)

    node.create_subscription(WrenchStamped, topic, _cb, qos_profile_sensor_data)

    bias = np.zeros(6)
    if bias_seconds > 0.0:
        logger.info(f"Capturing wrench bias on {topic} (keep arm still {bias_seconds:.1f}s)...")
        samples = []
        t_end = time.time() + bias_seconds
        while time.time() < t_end:
            samples.append(latest.copy())
            time.sleep(0.02)
        bias = np.mean(samples, axis=0) if samples else np.zeros(6)

    return lambda: latest - bias


def _build_wrench_process(config: BilateralConfig, bias_seconds: float,
                          cores: list[int] | None) -> tuple:
    """Offload the NetFT wrench subscriptions to a separate poller process.

    The in-process subscriptions fire at the sensor's native ~2.1 kHz on the control
    thread's GIL (x2 arms), starving the control loop below its target rate. This
    moves them to a dedicated process (own GIL) writing the latest value into a
    1-deep shared-memory ring per arm; the control loop reads that latest value
    each tick instead of servicing kHz callbacks.

    Returns ``(leader_wrench_fn, follower_wrench_fn, manager)``; ``manager`` is the
    ``HrateProcessManager`` to ``close()`` on shutdown (``None`` if no force channel).
    """
    from crisp_gym.record.hrate_process import HrateProcessManager

    specs = []
    if config.force:
        specs.append(("follower_ft", config.follower_wrench_topic, "ft", 1))
    if config.force_fwd or config.tdpa:
        specs.append(("leader_ft", config.leader_wrench_topic, "ft", 1))
    if not specs:
        return None, None, None

    mgr = HrateProcessManager(specs, core_affinity=cores)

    def _raw(key: str) -> np.ndarray:
        vals, _ = mgr.snapshot(key, 0.0)  # window=1 -> latest sample is the last row
        return np.asarray(vals[-1], dtype=float)

    # Wait for the spawned process to start delivering before relying on the value.
    t_end = time.time() + 5.0
    while time.time() < t_end and any(mgr.count(k) == 0 for k, *_ in specs):
        time.sleep(0.05)
    for k, *_ in specs:
        if mgr.count(k) == 0:
            logger.warning(f"wrench poller '{k}' delivered no samples yet (topic up?).")

    follower_fn = None
    if config.force:
        bias = np.zeros(6)
        if bias_seconds > 0.0:
            logger.info(f"Capturing follower wrench bias from poller (keep arm still "
                        f"{bias_seconds:.1f}s)...")
            samples = []
            t_end = time.time() + bias_seconds
            while time.time() < t_end:
                samples.append(_raw("follower_ft"))
                time.sleep(0.02)
            bias = np.mean(samples, axis=0) if samples else np.zeros(6)
        follower_fn = lambda: _raw("follower_ft") - bias  # noqa: E731

    leader_fn = None
    if config.force_fwd or config.tdpa:
        leader_fn = lambda: _raw("leader_ft")  # already unbiased; ~ human force  # noqa: E731

    return leader_fn, follower_fn, mgr


def build_bilateral_controller(env, leader, config: BilateralConfig, dt: float | None = None,
                               bias_seconds: float = 1.0, wrench_process: bool = False,
                               wrench_cores: list[int] | None = None) -> BilateralController:
    """Construct the controller for ``config`` against the live ``env`` + ``leader``.

    Args:
        env: a ManipulatorEnv whose ``.robot`` is the follower crisp_py robot.
        leader: a TeleopRobot whose ``.robot`` is the leader crisp_py robot.
        config: the bilateral scheme.
        dt: control timestep (defaults to ``1 / config.control_frequency``).
        bias_seconds: follower wrench bias capture duration.
        wrench_process: offload the NetFT wrench subs to a separate poller process
            (own GIL) so the ~2.1 kHz x2 callbacks don't starve the control loop.
        wrench_cores: optional CPU cores to pin that poller process to.
    """
    follower_robot = env.robot
    leader_robot = leader.robot
    dt = dt if dt is not None else 1.0 / config.control_frequency
    wrench_proc = None  # set when wrench_process offloads NetFT to a poller process

    if config.mode == "joint":
        leader_home = np.asarray(leader_robot.joint_values, dtype=float).copy()
        follower_home = np.asarray(follower_robot.joint_values, dtype=float).copy()
        follower_robot.controller_switcher_client.switch_controller("joint_impedance_controller")
        if config.force:
            logger.warning(
                "joint_pf force reflection is recorded only: crisp has no joint-torque "
                "streaming controller, so reflected joint effort is not rendered on the leader."
            )
        leader_adapter = CrispJointAdapter(leader_robot, leader_home)
        follower_adapter = CrispJointAdapter(follower_robot, follower_home)
        follower_robot.set_target_joint(follower_home)  # hold at home
    else:
        leader_home = pose_to_vec(leader_robot.end_effector_pose)
        follower_home = pose_to_vec(follower_robot.end_effector_pose)

        if wrench_process:
            leader_wrench_fn, follower_wrench_fn, wrench_proc = _build_wrench_process(
                config, bias_seconds, wrench_cores)
        else:
            follower_wrench_fn = None
            if config.force:
                follower_wrench_fn = _subscribe_wrench(
                    follower_robot.node, config.follower_wrench_topic, bias_seconds
                )
            leader_wrench_fn = None
            if config.force_fwd or config.tdpa:
                # leader netft (already unbiased) ~ human applied force; no extra bias
                leader_wrench_fn = _subscribe_wrench(leader_robot.node, config.leader_wrench_topic, 0.0)

        # lower the follower's cartesian rotational PD for teleop (stiff/undamped
        # yaw was hard to rotate and diverged); translation gains unchanged.
        cart_cfg = find_config(FOLLOWER_CARTESIAN_IMPEDANCE)
        if cart_cfg is not None:
            try:
                follower_robot.cartesian_controller_parameters_client.load_param_config(cart_cfg)
                logger.info(f"Loaded follower cartesian impedance: {FOLLOWER_CARTESIAN_IMPEDANCE}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load follower cartesian impedance: {e}")

        leader_adapter = CrispCartesianAdapter(leader_robot, leader_home, wrench_fn=leader_wrench_fn)
        follower_adapter = CrispCartesianAdapter(follower_robot, follower_home, wrench_fn=follower_wrench_fn)
        follower_robot.set_target(pose=vec_to_pose(follower_home))  # hold at home

    if config.tdpa and config.delay_steps == 0:
        logger.warning(
            "TDPA enabled with delay_steps=0. TDPA passivates a DELAYED channel; with no "
            "delay the channel is already passive, so TDPA only over-damps and throttles the "
            "reflected force intermittently (laggy/inconsistent feel). Use plain 'pf' here, or "
            "add --delay-steps to actually exercise TDPA."
        )
    logger.info(
        f"Bilateral controller: scheme={config.scheme} mode={config.mode} "
        f"force={config.force} force_fwd={config.force_fwd} pos_spring={config.pos_spring} "
        f"tdpa={config.tdpa} delay_steps={config.delay_steps}"
    )
    ctrl = BilateralController(leader_adapter, follower_adapter, config, dt=dt)
    ctrl.wrench_proc = wrench_proc  # caller closes it on shutdown (None if in-process)
    return ctrl
