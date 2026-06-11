# Bilateral teleop schemes, unified controller, and recorder integration

Date: 2026-06-11
Status: Approved (design)
Branch: `feat/bilateral-teleop-recording`
Author: Ivan Domrachev

## Goal

Make every bilateral teleoperation scheme we care about (a) share **one** live
control law, (b) be selectable by **one** named config, and (c) be **recorded**
faithfully by the structured LeRobot recorder — including any channel delay.

Schemes in scope this round:

| scheme        | fwd pos | back force | fwd force | back pos spring | TDPA | mode      |
|---------------|:------:|:----------:|:---------:|:---------------:|:----:|-----------|
| `position`    | ✓      |            |           |                 |      | cartesian |
| `pf`          | ✓      | ✓          |           |                 |      | cartesian |
| `pf_tdpa`     | ✓      | ✓          |           |                 | ✓    | cartesian |
| `pfpf`        | ✓      | ✓          | ✓         | ✓               |      | cartesian |
| `pfpf_tdpa`   | ✓      | ✓          | ✓         | ✓               | ✓    | cartesian |
| `joint_pf`    | ✓      | ✓          |           |                 | —    | joint     |

`joint_pf` has **no** TDPA (decision: joint mode ships P-F only this round).

## Current state (what exists)

- `crisp_gym/scripts/bilateral_teleop.py` — live P-F / TDPA / coupling law **inline
  in `main()`**, single process, both robots on one ROS graph, optional artificial
  `--delay-steps` via `DelayedChannel`. HW-validated: P (position) and P-F. TDPA ran
  but is flawed. Joint mode: follower frozen (broken).
- `crisp_gym/bilateral/`: `interface.py` (`RobotInterface` Protocol),
  `loop.py` (`run_bilateral` dry-run), `tdpa.py` (`MasterOnlyPOPC`),
  `pose_math.py`, `filters.py`, `telemetry.py`.
- `crisp_gym/sim/`: `MSDRobot` (per-axis mass-spring-damper; supports
  `set_target_position` + `set_feedforward_force` on **both** arms → 4-channel is
  dry-run-testable), `DelayedChannel`.
- `crisp_gym/record/structured_record.py`: `make_structured_teleop_fn`
  (position-only, drives `env.step`) + `make_structured_sample_fn` (read-only).
- 4-channel (PF-PF) lives only in `franka_server_standalone` (separate repo,
  2-process WebSocket + distributed TDPA). We port the **scheme**, not the code.

Two known bugs folded into this work:

1. **TDPA `human_force = 0`** → master energy budget always 0 → per-axis
   over-damping / 180° force thrash. Fix: feed the **leader** netft wrench as `Fh`
   into `update_master` (real master port-power, = franka_server's approach).
2. **Joint `target_joint` dropped when >1 publisher** → follower frozen. Fix:
   controller is the sole publisher; switch `joint_impedance_controller`,
   `set_target_joint`. Ships **experimental** (HW-verify `ros2 topic info -v`).

## Architecture

### Layering (keeps the law dry-run-testable)

```
BilateralController            generic, axis-vector law: channels, reflection,
  (crisp_gym/bilateral/          force-fwd, position spring, TDPA, telemetry.
   controller.py)                Operates ONLY on RobotInterface. MSD-tested.
        |
        +-- leader:  RobotInterface
        +-- follower:RobotInterface
              |
        realized by either:
   MSDRobot (sim, tests)   OR   CrispCartesianAdapter / CrispJointAdapter (HW)
                                  thin wrappers over a crisp_py robot doing
                                  pose<->vec, aligned_pose mapping, wrench-frame
                                  rotation (cartesian) or joint passthrough.
```

The **core law is axis-generic** (operates on generalized position vectors), so it
is tested in full against `MSDRobot` for every scheme. Cartesian-specific framing
(quaternion handling, `aligned_pose`, `wrench_frame_rotation`/`rotate_wrench`) lives
in `CrispCartesianAdapter` and is already covered by `tests/bilateral/test_pose_math.py`.

### `BilateralController`

`crisp_gym/bilateral/controller.py`

- `__init__(leader, follower, config, channels?, tdpa?, ...)`: capture home anchors,
  build forward/return `DelayedChannel`s sized by `config.delay_steps` and scheme
  (4-ch adds a return-position channel + forward-force channel), build TDPA
  (`MasterOnlyPOPC`) if enabled, build reflected-wrench high-pass.
- `step(dt) -> TeleopTelemetry`: exactly one control tick.
  - Forward position: leader pos → (delay) → follower target (absolute via
    `aligned_pose`/`offset_joint`, or relative increment integration).
  - Return force (`force`): follower wrench → (delay) → leader feed-forward, after
    bias-subtract, gain/sign, high-pass, soft deadband, optional TDPA, mag clamp.
  - Forward force (`force_fwd`, 4-ch): leader wrench → (delay) → follower feed-forward.
  - Return position spring (`pos_spring`, 4-ch): leader feed-forward +=
    `k_spring * (follower_pos_mapped - leader_pos)`.
  - TDPA `Fh` = leader wrench (when available), else 0.
  - Returns `TeleopTelemetry`.
- `home()` / context to set anchors and switch controllers.

`bilateral_teleop.py::main()` collapses to: build adapters + `BilateralController`,
loop `controller.step(dt)`, write telemetry. **No control logic duplicated.**

### `TeleopTelemetry` (delay capture)

Returned by every `step()`:

```
live_leader_pos, live_follower_pos          # live, this tick
commanded_follower_target                   # what follower was told (DELAYED)
reflected_wrench                             # force sent to leader (post-TDPA)
follower_wrench, leader_wrench               # raw sensed
delay_steps, control_dt                      # the configured/realized delay
```

### Delay is recoverable three ways

1. **Config metadata** — `BilateralConfig` (incl. `delay_steps`, `control_frequency`,
   scheme name) written into the dataset per episode → intended delay is explicit.
2. **Implicit hrate** — `leader_pose_hrate` vs `follower_pose_hrate` are absolute-
   timestamped at native rate; cross-correlate → realized leader→follower lag.
3. **Explicit commanded-vs-live** — recorder logs `observation.leader_state` (live)
   and `observation.follower_target` / `action` (the delayed command from telemetry);
   their offset = the delay, per frame.

## Config

`crisp_gym/bilateral/bilateral_config.py`:

```python
@dataclass
class BilateralConfig:
    scheme: str                  # name, for metadata
    mode: str = "cartesian"      # cartesian | joint
    coupling: str = "absolute"   # absolute | relative
    force: bool = False          # return force reflection
    force_fwd: bool = False      # forward force feedforward (4-ch)
    pos_spring: bool = False     # return position spring (4-ch)
    tdpa: bool = False
    control_frequency: float = 100.0
    delay_steps: int = 0
    feedback_gain: float = 1.0
    feedback_sign: float = 1.0
    feedback_max_force: float = 20.0
    contact_threshold_n: float = 1.0
    reflect_highpass_hz: float = 0.0
    reflect_deadband_n: float = 0.0
    position_spring_k: float = 0.0
    follower_wrench_topic: str = "/right/netft_data_unbiased_tcp"
    leader_wrench_topic: str = "/left/netft_data_unbiased_tcp"
    leader_config: str = "left_leader_nogripper"
    leader_namespace: str = "left"
    follower_namespace: str = "right"
    follower_env_config: str = "panda_no_cam"

def make_bilateral_config(name: str) -> BilateralConfig: ...   # loads YAML
```

YAML one-per-scheme in `config/teleop/bilateral/{position,pf,pf_tdpa,pfpf,
pfpf_tdpa,joint_pf}.yaml`. Both the runner and the recorder take a single
`--teleop-scheme <name>`.

## Recorder integration

`crisp_gym/record/structured_record.py`:

- New `make_structured_bilateral_fn(env, leader, controller, hrate, cams, ...)`:
  per frame `tel = controller.step(dt)`; read follower obs **read-only**
  (`env._get_obs()`, controller already commanded the robot); assemble structured
  obs; fold telemetry → `observation.follower_target`, `observation.reflected_wrench`
  (+ existing `_state`/`_ft`/`_hrate`/images); `action = tel.commanded_follower_target`.
- `build_structured_features` extended to declare the new telemetry fields and
  (when present) the `joint`-mode action shape.
- The controller and the env share the **same** crisp_py follower robot handle
  (controller's follower adapter wraps `env.robot`), so there is one commander.
- `record_structured_leader_follower.py`: replace `--leader-config/--follower-config`
  knobs with `--teleop-scheme <name>` (still allow namespace overrides); build the
  controller from `make_bilateral_config`; write the config into dataset metadata.

## Testing (TDD, robots enabled → also HW smoke)

Dry-run (MSD), one or more failing-first tests each:

- `pf`: reflected force sign/magnitude correct, leader settles bounded with no delay.
- `pf` under delay: naive diverges (pins the bug; mirrors `test_loop`).
- `pf_tdpa`: with delay, leader energy **bounded** (vs naive diverging); `Fh` = leader
  wrench gives a non-degenerate master budget (no all-axis over-damp).
- `pfpf`: follower receives forward force feed-forward; leader spring pulls leader
  toward follower position; both bounded.
- `pfpf_tdpa`: passivity holds under delay.
- `joint_pf`: dof=7 1:1 coupling, follower tracks leader joints; reflected joint
  effort applied; no freeze.
- `TeleopTelemetry`: all fields populated; `commanded_follower_target` equals the
  **delayed** leader pose (delay recoverable).
- `make_bilateral_config`: each YAML loads to the right flag set; unknown name errors.
- Recorder frame-fn: with a **mock controller**, `make_structured_bilateral_fn`
  emits the expected obs keys incl. delay-capture fields; `action` = commanded target.

Then HW: headless import smoke, verify `/left/netft...` publishes, short recordings.

## HW experiment ladder (run on return)

Each = a named scheme + a `crisp-record-structured --teleop-scheme <s>` command +
what to feel + pass/fail. Staged:

- **E1 `position`** — baseline; confirm controller+recorder path, clean dataset.
- **E2 `pf`** — feedback-gain sweep (e.g. 0.5/1.0/1.5); feel wall contact; watch drift.
- **E3 `pf_tdpa`** — repeat E2 contact **+ `--delay-steps`** (e.g. 20/40); TDPA should
  keep it stable where naive (`pf`) buzzes/diverges. Proves the `Fh` fix.
- **E4 `pfpf`** — 4-channel; feel force both directions; leader spring should pull
  to follower. Tune `position_spring_k`.
- **E5 `pfpf_tdpa`** — E4 + delay; passivity.
- **E6 `joint_pf`** — joint 1:1; verify follower not frozen (`ros2 topic info -v`).

Each experiment emits a real LeRobot dataset so we can later inspect delay capture.

## Out of scope (this round)

- Distributed TDPA (`ImpedancePC`/`AdmittancePC`) — single-process port only.
- TDPA in joint mode.
- 2-process WebSocket topology (franka_server's domain).
```
