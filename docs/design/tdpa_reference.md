# TDPA reference (from franka_server_standalone) — for step 2 (crisp_gym-1.3)

Source: `franka_server/teleop/bilateral/{tdpa,master_controller,slave_controller,increment,config}.py`.

## MasterOnlyPOPC (deployed P-F mode) — the one to reimplement first

6-DOF, per-axis, all signals in master BASE frame. State (shape (6,), float64, init 0):
`e_m_in>=0, e_m_out<=0, e_s_in>=0, e_s_out<=0`.

Per-axis energy helpers:
```
input_energy_update(E,F,V,dt):  P=F*V; return E+P*dt if P>0 else E
output_energy_update(E,F,V,dt): P=F*V; return E+P*dt if P<0 else E
```

`update_master(Fh, Vm, Fe, dt)` — FIRST each tick. For each axis only when `abs(Fe[i])>contact_threshold_n` (3.0 N):
```
e_m_in[i]  = input_energy_update(e_m_in[i],  Fh[i], Vm[i], dt)
e_m_out[i] = output_energy_update(e_m_out[i], Fh[i], Vm[i], dt)
```
Velocity unchanged (returns Vm.copy()).

`update_slave(F=-Fe, V=Vm, dt)` — SECOND each tick. Tentative accumulate, then per axis correct:
```
e_s_in[i]  = input_energy_update(e_s_in[i],  F[i], V[i], dt)
e_s_out[i] = output_energy_update(e_s_out[i], F[i], V[i], dt)
if (e_s_out[i] + e_m_in[i] < 0) and (V[i]**2 > 0):
    alpha = (e_s_out[i] + e_m_in[i]) / (V[i]**2 * dt)   # alpha <= 0
    e_s_out[i] -= F[i]*V[i]*dt                           # revert tentative
    F_res[i]   -= alpha*V[i]                              # inject damping
    e_s_out[i] += F_res[i]*V[i]*dt                        # re-accumulate corrected
return F_res
```
**Order matters**: revert-then-reaccumulate preserves same-tick passivity. Master controller calls
`update_master` then `f_res=update_slave(-Fe,Vm,dt)`; reflects `-f_res` (net = Fe_modified), rotates base->TCP
(crisp uses local jacobian), clamps via shape_wrench.

## Distributed PCs (for PF-PF later): ImpedancePC (master, in=V out=F) + AdmittancePC (slave, in=F out=V).
Same revert/reaccumulate, vectorized 6-axis, gate `e_out + e_in_remote < 0`, clip alpha/beta to [-max,0],
death-zone sets `e_out=-e_in_remote` when gate<0 but |port|<eps, Ryu/Hannaford idle reset (|F|<0.2N for 0.01s -> zero e_in,e_out). `e_in_remote` exchanged over channel.

## Channel / packet (TeleopPacket, orjson over WS, keep-last both dirs)
incr(6 world pose incr), twist(6), wrench(6, sender base frame), energy(dict), seq(int), t(monotonic), pose(7 [x,y,z,qw,qx,qy,qz]). Slave integrates each `seq` once. **No artificial delay exists — add a timed buffer for dry-run.**

## Increment math (increment.py), pose = [x,y,z,qw,qx,qy,qz]
`increment_world(before,after)`: trans=after[:3]-before[:3]; rel=Rb.inv()*Ra; rotvec_world=Ra.apply(rel.as_rotvec()); return [trans, rotvec_world].
`integrate_pose(pose,incr,scale,R)`: new_t=pose[:3]+scale*(R@incr[:3]); new_R=Rotation.from_rotvec(scale*R@incr[3:]) * Rotation(pose[3:7]) (left-mul world delta).

## Tunables (defaults): rate_hz=200, horizon_hz=100, channel_timeout_s=0.2, pose_scale=1.0,
contact_threshold_n=3.0, ff_gain=1.0, ff_sign=1.0, ff_max_force_n=6.0, ff_max_torque_nm=1.0,
k_couple_pos=300, k_couple_rot=10, alpha/beta_max=1e6, eps_v/f=1e-9, po_reset_force_n=0.2, po_reset_time_s=0.01,
slave admittance uses NOMINAL dt=1/rate_hz (not measured) for v_cmd->energy.

No MSD plant / no delay exist there — both are built fresh in step 0.
