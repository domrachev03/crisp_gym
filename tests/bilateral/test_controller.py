"""BilateralController: one control law for every scheme, dry-run against MSD plants.

The controller computes one teleop tick (forward position, return force, forward
force, return position spring, optional TDPA) and commands the two robots; the
test harness advances the MSD plants. Each scheme is pinned here so the shared law
can be refactored without silently changing what a scheme does. Cartesian framing
(quaternions, wrench-frame rotation) lives in the HW adapter, not here.
"""

import numpy as np

from crisp_gym.sim import MSDRobot
from crisp_gym.bilateral.bilateral_config import BilateralConfig
from crisp_gym.bilateral.controller import BilateralController, TeleopTelemetry


# --- harness ----------------------------------------------------------------- #
def _wall(x_wall: float, k_wall: float, dof: int = 1):
    def env(pos):
        pen = pos - x_wall
        f = np.zeros(dof)
        m = pen > 0
        f[m] = -k_wall * pen[m]
        return f
    return env


def _pair(dof=1, wall=True, leader_env=None, leader_x0=None, follower_x0=None,
          leader_stiffness=0.0):
    leader = MSDRobot(dof=dof, mass=1.0, stiffness=leader_stiffness, damping=2.0,
                      x0=leader_x0, environment=leader_env)
    follower = MSDRobot(dof=dof, mass=1.0, stiffness=300.0, damping=8.0, x0=follower_x0,
                        environment=_wall(0.05, 800.0, dof) if wall else None)
    return leader, follower


def _run(ctrl, leader, follower, hf, n, dt):
    leader_pos, follower_pos, tels = [], [], []
    for _ in range(n):
        tel = ctrl.step(dt=dt, human_force=hf)
        leader.step(dt)
        follower.step(dt)
        leader_pos.append(leader.position.copy())
        follower_pos.append(follower.position.copy())
        tels.append(tel)
    return {"leader_pos": np.array(leader_pos), "follower_pos": np.array(follower_pos),
            "tels": tels}


# --- P-F --------------------------------------------------------------------- #
def test_pf_no_delay_is_bounded():
    leader, follower = _pair(wall=True)
    cfg = BilateralConfig(scheme="pf", force=True, tdpa=False, delay_steps=0,
                          contact_threshold_n=1.0, reflect_deadband_n=0.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([5.0]), n=4000, dt=1e-3)
    assert np.isfinite(log["leader_pos"]).all()
    assert np.max(np.abs(log["leader_pos"])) < 0.25


def test_pf_naive_diverges_under_delay():
    leader, follower = _pair(wall=True)
    # unbounded force clamp so this isolates passivity, not the safety clamp
    cfg = BilateralConfig(scheme="pf", force=True, tdpa=False, delay_steps=30,
                          contact_threshold_n=1.0, reflect_deadband_n=0.0,
                          feedback_max_force=1e9)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([5.0]), n=4000, dt=1e-3)
    assert np.max(np.abs(log["leader_pos"])) > 1.0


# --- P-F + TDPA (with the Fh = leader/human force fix) ------------------------ #
def test_pf_tdpa_bounded_under_same_delay():
    leader, follower = _pair(wall=True)
    # same unbounded clamp + same delay as the diverging case: only TDPA differs
    cfg = BilateralConfig(scheme="pf_tdpa", force=True, tdpa=True, delay_steps=30,
                          contact_threshold_n=1.0, reflect_deadband_n=0.0,
                          feedback_max_force=1e9)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([5.0]), n=4000, dt=1e-3)
    # TDPA passivates the delayed reflection that diverges without it
    assert np.isfinite(log["leader_pos"]).all()
    assert np.max(np.abs(log["leader_pos"])) < 1.0


def test_tdpa_master_budget_grows_with_nonzero_fh():
    # The fix: Fh drives e_m_in. With Fh=0 (old bug) the budget stays 0.
    leader, follower = _pair(wall=True)
    cfg = BilateralConfig(scheme="pf_tdpa", force=True, tdpa=True, delay_steps=10,
                          contact_threshold_n=1.0, reflect_deadband_n=0.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    _run(ctrl, leader, follower, hf=np.array([5.0]), n=500, dt=1e-3)
    assert ctrl.tdpa is not None
    assert np.sum(ctrl.tdpa.e_m_in) > 0.0


# --- 4-channel (PF-PF) ------------------------------------------------------- #
def test_pfpf_forward_force_reaches_follower():
    # leader feels a constant external force -> fed forward to the follower
    leader, follower = _pair(wall=False, leader_env=lambda pos: np.array([3.0]))
    cfg = BilateralConfig(scheme="pfpf", force=True, force_fwd=True, pos_spring=True,
                          tdpa=False, delay_steps=0, feedback_gain=1.0,
                          reflect_deadband_n=0.0, position_spring_k=50.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([0.0]), n=50, dt=1e-3)
    # by the last tick the leader wrench (3 N) has propagated to the follower ff
    assert np.allclose(log["tels"][-1].forward_force, np.array([3.0]), atol=1e-6)


def test_pfpf_position_spring_pulls_leader_toward_follower():
    leader, follower = _pair(wall=False, leader_x0=np.array([0.0]),
                             follower_x0=np.array([0.2]))
    cfg = BilateralConfig(scheme="pfpf", force=False, force_fwd=False, pos_spring=True,
                          tdpa=False, delay_steps=0, position_spring_k=500.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([0.0]), n=200, dt=1e-3)
    assert log["tels"][0].spring_force[0] > 0.0          # pulls toward follower (+x)
    assert np.max(log["leader_pos"]) > 0.01              # leader actually moved +x


def test_spring_uses_separate_rotational_gain():
    # 6-vec cartesian world-delta [tx,ty,tz, rx,ry,rz]: translation must use
    # position_spring_k, rotation the (small) rot_spring_k -- not the same huge k
    # (that yaw torque is what diverged ppf on hardware).
    leader, follower = _pair(dof=6, wall=False, leader_x0=np.zeros(6),
                             follower_x0=np.array([0.1, 0, 0, 0.2, 0, 0]))
    cfg = BilateralConfig(scheme="ppf", force=False, pos_spring=True, tdpa=False,
                          delay_steps=0, position_spring_k=100.0, rot_spring_k=5.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    tel = ctrl.step(dt=1e-3, human_force=None)
    assert np.isclose(tel.spring_force[0], 10.0, atol=1e-6)   # 100 N/m * 0.1
    assert np.isclose(tel.spring_force[3], 1.0, atol=1e-6)    # 5 N*m/rad * 0.2


def test_forward_force_is_clamped_to_max():
    # a large leader wrench must NOT inject an unbounded force into the follower
    # (the 64 N spike that diverged pfpf on hardware)
    leader, follower = _pair(wall=False, leader_env=lambda pos: np.array([100.0]))
    cfg = BilateralConfig(scheme="pfpf", force_fwd=True, force_fwd_gain=1.0,
                          feedback_max_force=20.0, delay_steps=0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([0.0]), n=20, dt=1e-3)
    fwd = np.array([abs(t.forward_force[0]) for t in log["tels"]])
    assert fwd.max() <= 20.0 + 1e-6


def test_cartesian_force_and_torque_are_clamped_independently():
    leader, follower = _pair(dof=6, wall=False)
    follower._contact = np.array([30.0, 40.0, 0.0, 3.0, 4.0, 0.0])
    cfg = BilateralConfig(
        scheme="pf", force=True, feedback_max_force=10.0, feedback_max_torque=1.0,
        reflect_deadband_n=0.0, reflect_deadband_nm=0.0,
    )
    tel = BilateralController(leader, follower, cfg, dt=1e-3).step(dt=1e-3)
    assert np.isclose(np.linalg.norm(tel.reflected_wrench[:3]), 10.0)
    assert np.isclose(np.linalg.norm(tel.reflected_wrench[3:]), 1.0)


def test_cartesian_force_and_torque_deadbands_are_independent():
    leader, follower = _pair(dof=6, wall=False)
    follower._contact = np.array([3.0, 0.0, 0.0, 0.4, 0.0, 0.0])
    cfg = BilateralConfig(
        scheme="pf", force=True, feedback_max_force=10.0, feedback_max_torque=10.0,
        reflect_deadband_n=2.0, reflect_deadband_nm=0.5,
    )
    tel = BilateralController(leader, follower, cfg, dt=1e-3).step(dt=1e-3)
    assert np.allclose(tel.reflected_wrench[:3], [1.0, 0.0, 0.0])
    assert np.allclose(tel.reflected_wrench[3:], np.zeros(3))


def test_invalid_control_timestep_is_rejected():
    leader, follower = _pair(wall=False)
    ctrl = BilateralController(leader, follower, BilateralConfig(), dt=1e-3)
    for invalid in (0.0, -0.1, np.nan, np.inf):
        try:
            ctrl.step(dt=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid timestep {invalid} was accepted")


def test_force_fwd_gain_scales_forward_force():
    leader, follower = _pair(wall=False, leader_env=lambda pos: np.array([4.0]))
    cfg = BilateralConfig(scheme="pfpf", force_fwd=True, force_fwd_gain=0.5,
                          feedback_max_force=20.0, delay_steps=0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.array([0.0]), n=20, dt=1e-3)
    assert np.isclose(log["tels"][-1].forward_force[0], 2.0, atol=1e-6)  # 0.5 * 4 N


# --- joint 1:1 --------------------------------------------------------------- #
def test_joint_pf_follower_tracks_leader():
    q_des = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    leader, follower = _pair(dof=7, wall=False, leader_x0=q_des,
                             follower_x0=np.zeros(7))
    cfg = BilateralConfig(scheme="joint_pf", mode="joint", force=True, tdpa=False,
                          delay_steps=0, reflect_deadband_n=0.0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    log = _run(ctrl, leader, follower, hf=np.zeros(7), n=3000, dt=1e-3)
    assert np.allclose(log["follower_pos"][-1], q_des, atol=0.05)


# --- telemetry / delay capture ----------------------------------------------- #
def test_commanded_target_is_the_delayed_leader_position():
    leader, follower = _pair(dof=1, wall=False)
    cfg = BilateralConfig(scheme="position", force=False, delay_steps=5)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    commanded = []
    for i in range(20):
        leader._pos = np.array([float(i)])  # march the leader
        tel = ctrl.step(dt=1e-3, human_force=None)
        commanded.append(float(tel.commanded_follower_target[0]))
    # delay_steps=5: target at tick 10 is the leader position from tick 5
    assert commanded[10] == 5.0
    # before the channel primes it holds the home fill (leader started at 0)
    assert commanded[0] == 0.0


def test_reanchor_recaptures_homes_at_current_pose():
    leader, follower = _pair(dof=1, wall=False)
    cfg = BilateralConfig(scheme="pf", force=True, delay_steps=3)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    leader._pos = np.array([0.3])
    follower._pos = np.array([0.5])
    ctrl.reanchor()
    assert np.allclose(ctrl.leader_home, [0.3])
    assert np.allclose(ctrl.follower_home, [0.5])
    assert np.allclose(ctrl._follower_target, [0.5])


def test_telemetry_fields_populated():
    leader, follower = _pair(wall=True)
    cfg = BilateralConfig(scheme="pf", force=True, delay_steps=0)
    ctrl = BilateralController(leader, follower, cfg, dt=1e-3)
    tel = ctrl.step(dt=1e-3, human_force=np.array([5.0]))
    assert isinstance(tel, TeleopTelemetry)
    for name in ("live_leader_pos", "live_follower_pos", "commanded_follower_target",
                 "leader_feedforward", "reflected_wrench", "spring_force",
                 "forward_force", "follower_wrench", "leader_wrench"):
        assert getattr(tel, name) is not None, name
    assert tel.delay_steps == 0
    assert tel.control_dt == 1e-3
