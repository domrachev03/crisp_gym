# Bilateral teleop scheme — hardware experiment ladder

Date: 2026-06-11
Branch: `feat/bilateral-teleop-recording`
Rig: KAIST panda-panda. Leader namespace `left`, follower namespace `right`.

Goal: feel each bilateral scheme, then record one episode of it, climbing from the
safest (position-only) to full 4-channel + TDPA. Every scheme runs the **same**
`BilateralController`; only the named config changes. Keep the **e-stop in hand**.

Two entry points per scheme:
- **Feel** (no recording): `bilateral_teleop.py` — bare control loop.
- **Record** (LeRobot dataset): `crisp-record-structured --teleop-scheme <s>`.

```
# Feel a scheme:
pixi run -e jazzy python crisp_gym/scripts/bilateral_teleop.py --teleop-scheme <s>

# Record a scheme (1 episode, ros stop key):
pixi run -e jazzy-lerobot crisp-record-structured \
    --repo-id test/bilateral-<s> --teleop-scheme <s> \
    --recording-manager-type ros --fps 30 --num-episodes 1
```

The recorder writes `observation.{leader,follower}_state`, the high-rate windows,
both cameras, the controller telemetry (`observation.follower_target`,
`reflected_wrench`, `forward_force`, `spring_force`, `leader_feedforward`), and an
`action` = the commanded (delayed) follower target. The delay is recoverable from
`leader_state` vs `follower_target`, the pose hrate windows, and the config metadata.

---

## Pre-flight (once)

- [ ] Both arms up, controllers loaded; `left` (leader) and `right` (follower) namespaces live.
- [ ] Leader F/T publishes:  `ros2 topic hz /left/netft_data_unbiased_tcp`  (needed by `pfpf*`, `*_tdpa`).
- [ ] Follower F/T publishes: `ros2 topic hz /right/netft_data_unbiased_tcp`  (needed by `pf*`, `pfpf*`).
- [ ] Leader is backdrivable (gravity-comp cartesian_impedance) after `prepare_for_teleop`.
- [ ] E-stop within reach. Start each run with hands clear; the follower homes first.

---

## E1 — `position` (baseline)

No force. Confirms the controller+recorder path end-to-end and the 1:1 world-axis mapping.

- Feel: `--teleop-scheme position`
- Record: `--teleop-scheme position`
- **Feel for:** follower mirrors leader translation/rotation 1:1; no jump at engage; no force on the leader.
- **Pass:** follower tracks; dataset has motion, both `_state` populated, 2 camera mp4s, clean exit.
- **Fail signs:** startup jump (home anchoring), axis mismatch (mapping), follower frozen.

## E2 — `pf` (position-force)

Reflected follower contact force on the leader (HW-validated scheme).

- Feel: `--teleop-scheme pf`   (try `--feedback-gain 0.5`, `1.0`, `1.5`)
- Record: `--teleop-scheme pf`
- **Feel for:** pushing the follower into the table is felt on the leader; free space is light (deadband nulls the noise floor).
- **Pass:** crisp wall contact, no buzzing in free space, leader doesn't drift away.
- **Tune:** `feedback_gain` (firmness), `reflect_deadband_n` (free-space float). Higher gain → more force, less stability margin.

## E3 — `pf_tdpa` (P-F + TDPA, the `Fh` fix)

Same as E2 but the reflected force is passivated; the TDPA master budget is driven
by the **leader** wrench (`Fh`) — the fix for the old over-damping. Prove it under delay.

- Feel A (no delay): `--teleop-scheme pf_tdpa`
- Feel B (delay):    `--teleop-scheme pf_tdpa --delay-steps 30`
- Compare:           `--teleop-scheme pf --delay-steps 30`  ← should buzz / go unstable on contact
- Record: `--teleop-scheme pf_tdpa --delay-steps 30`
- **Feel for:** with delay, `pf` contact buzzes/diverges; `pf_tdpa` stays stable (damped but not dead).
- **Pass:** TDPA contact is stable under `--delay-steps 30` where `pf` is not; free motion still feels live (not over-damped — that was the bug).
- **Tune:** `contact_threshold_n` (when passivation engages).

## E4 — `pfpf` (4-channel)

Position + force both ways: leader force fed forward to the follower, follower
position springs the leader back toward it.

- Feel: `--teleop-scheme pfpf`   (tune spring via config `position_spring_k`)
- Record: `--teleop-scheme pfpf`
- **Feel for:** stiffer coupling — the leader is gently pulled toward the follower's actual pose; pushing the leader drives the follower harder.
- **Pass:** stable two-way coupling; spring pulls leader to follower without oscillation.
- **Tune:** `position_spring_k` (start 150 N/m). Too high → oscillation; too low → no return-position feel.

## E5 — `pfpf_tdpa` (4-channel + TDPA)

E4 with the reflected force passivated. Repeat E4 + a delay.

- Feel: `--teleop-scheme pfpf_tdpa --delay-steps 30`
- Record: `--teleop-scheme pfpf_tdpa --delay-steps 30`
- **Pass:** 4-channel feel of E4 but stable under delay on contact.

## E6 — `joint_pf` (joint 1:1) — EXPERIMENTAL

Direct 1:1 joint coupling (no cartesian mapping). Force is **recorded only** (crisp
has no joint-torque streaming controller — the leader does not render reflected joint effort).

- Pre-check: `ros2 topic info /right/target_joint -v`  → confirm the controller is the **sole publisher** (extra publishers are dropped → follower freezes).
- Feel: `--teleop-scheme joint_pf`
- Record: `--teleop-scheme joint_pf`
- **Feel for:** each follower joint follows the matching leader joint.
- **Pass:** follower is NOT frozen; joints track 1:1.
- **Fail signs:** follower frozen at home → multiple `target_joint` publishers, or `joint_impedance_controller` not active.

---

## After the ladder

- Inspect any dataset: `pixi run -e jazzy-lerobot crisp-visualize-structured --repo-id test/bilateral-<s>`
- Delay sanity: cross-correlate `observation.leader_pose_hrate` vs `observation.follower_pose_hrate`,
  or compare `observation.leader_state` vs `observation.follower_target`.
- Report per scheme: tracking quality, force feel, stability (esp. under `--delay-steps`),
  and good gains. Those tuned values go back into `config/teleop/bilateral/<scheme>.yaml`.
