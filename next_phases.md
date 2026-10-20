# Proposal: Zombie Crawl — Phase 2 (heading-rate command) and Phase 3 (maneuvers)

**Project:** Zombie Crawl (SO-ARM100 / SO-101, Isaac Sim + Isaac Lab 3.0, `rsl_rl` 5.5.1 PPO)
**Status:** Phase 1 is complete: a policy exists that drags the free-floating base toward a **randomly commanded heading** (uniform ±180°, frozen per episode) at a curriculum target of 0.06 → 0.10 m/s. Shipped checkpoint: `checkpoints/model_1800.pt`. The policy is *not* a "+X crawler" and its eval numbers are modest (see §0.1) — baselines and regression gates must use these real numbers.
**Audience:** the project owner, and the coding agent that will implement these phases.
**Purpose of this document:** specify two consecutive phases.
- **Phase 2** replaces the one input that cannot exist on the real robot — the world-frame heading error — with a rate-based command and signal that a gyro and servo encoders can supply. Nothing else about the behavior changes.
- **Phase 3** adds the additional maneuvers on top of the Phase 2 policy.

---

## 0. Common context (applies to both phases)

### 0.1 What Phase 1 actually is (baseline facts)

These repo facts constrain the sections below; they are cited so implementers do not re-discover them.

- **Command semantics today:** the crawl direction is a per-episode uniform ±180° draw (`env.zombie_ref_yaw`, set by `mdp.reset_start_pose`), or a mid-episode-resampled heading when `use_dynamic_heading=true` (`mdp.UniformCrawlHeadingCommand`, `mdp/commands.py`). Directional rewards/terminations project onto it via `mdp.get_ref_dir`; the *locomotion* frame is the claw arrow via `mdp.get_claw_dir` (base→wrist, with a base-forward fallback `yaw − π/2` when the arm is curled).
- **Actor obs (26-D):** projected gravity (3) + `heading_error_wrist` (2, `(cos, sin)` of claw-minus-command — the only command signal the current actor sees) + base ang. vel (3) + joint pos rel. (6) + joint vel (6) + last action (6). Optional 2-D gait clock exists but is **off by default** (`use_gait_clock=False`).
- **Critic:** privileged group (base lin vel 3, base height 1, ground friction 1) concatenated for the critic only — asymmetric actor-critic is already wired (`obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}`, `agents/rsl_rl_ppo_cfg.py`).
- **Timing:** `dt=0.005`, `decimation=4` → 50 Hz control; episodes 20 s; `num_steps_per_env=40`; PPO nets 256/128/64 ELU, empirical obs norm, `init_std=0.6`, entropy 0.01, adaptive LR (`desired_kl=0.01`).
- **Terminations (`mdp/terminations.py`, `crawl_env_cfg.py`):** `time_out`, `base_nan`, `base_too_high` (0.45 m), **`out_of_bounds`** (|dx|>2.5 or |dy|>1.5 m; terminal "success"), **`stalled`** (failure if <0.05 m along the command axis within 12 s; re-anchors on command resample).
- **Events / DR:** spawn poses folded/on_side/face_down with weights (0.5, 0.25, 0.25); `upright_bonus` recovery reward (w=1.0); per-env ground friction 0.5–1.2 effective (startup event, multiplier 0.625–1.5 on μ=0.8); all-body disturbance `disturb_all_bodies` ±0.4 N / ±0.03 N·m every 0.2–0.5 s, scaled by `env.extras["push_scale"]` (difficulty curriculum); obs corruption via per-term `UniformNoiseCfg`.
- **Actuators:** presets `servo` (default: STS3215-class, 2.9 N·m, 6 rad/s, PD k=1.0 d=0.1, armature 0.01) and `ideal` (10 N·m).
- **Baseline quality (use these for regression gates, not idealized numbers):** curriculum saturated ~iter 400 (4096 envs); eval of the shipped `model_1800.pt` shows ~19% of steps in the ±50% speed band and mean |yaw drift| ≈ 81° on the deterministic eval path. `scripts/evaluate.py` has a **known bug**: `out_of_bounds` is bucketed as early termination, so escaping episodes are dropped (NaN displacement on some evals). Fixing this is a prerequisite (§2.11.0).
- **Runtime (verified on the owner's machine):** all software lives in the Python venv at **`../.venv`** (i.e. `../.venv/bin/python`, Python 3.12.15, alongside the repo root). Installed and confirmed: `isaacsim 6.1.0.0`, `isaaclab` (3.x line; `isaaclab-rl 1.3.3`), **`rsl-rl-lib 5.5.1`**, `torch 2.11.0+cu130`; `zombie_crawl` is installed editable from `source/zombie_crawl/`, and `isaaclab_tasks` resolves to the local `../IsaacLab3` checkout. GPU: RTX 5090 Laptop, 24 GB — the 4096-env runs in `report.md` were done on this class of GPU (~1.6 s/iter). Every sim command must use this interpreter (`../.venv/bin/python …`); there is no system-wide Isaac install.
- **Doc/code mismatches to reconcile during Phase 2** (decide once, then update docs): `soft_joint_pos_limit_factor` is 1.0 in `assets/so_arm100.py` but 0.95 in README; the `PolicyCfg` docstring says "27-D" but the term dims sum to 26.

### 0.2 Standing decisions (both phases)

1. **No sensor/IMU modeling in simulation (owner: "start simple").** Inputs derived from the IMU and servo encoders are computed from ground-truth sim states through a shared pure-function estimator module. Gyro bias/noise, tilt error, encoder quantization, and the fusion filter are deferred to the sensor-noise pass (§4). All IMU processing stays on the ESP32 (`imu/`, already built, with boot gyro-bias calibration and measured noise).
2. **No pre-training scripted feasibility gates (owner: "if I could script it, I wouldn't need reinforcement learning").** What the policy can achieve is discovered by RL and measured in post-training evaluation. Code-level unit tests of the estimator math are fine; open-loop physics PASS/FAIL harnesses are not used.
3. **IMU is the QST QMI8658** (6-axis, no magnetometer), so absolute yaw is unobservable on the real robot. This is the reason Phase 2 exists.
4. **Train from scratch at every phase (owner).** No checkpoint surgery, warm-starting, fine-tuning, or weight transfer between phases or runs. The purpose of each phase is to discover which reward functions and curriculum work; what carries from one phase to the next is code, config, and findings, never weights. The final state is **one training recipe (rewards + automatic curriculum) that produces the complete behavior from a fresh random initialization with a single command.** The Phase 1 checkpoint is used only as a measurement subject (gait period, §2.6.7) and as a quality baseline for comparison (§0.1).

---

# PHASE 2 — Replace the heading error with a heading-rate command

## 2.1 Goal and scope

**Goal.** The Phase 1 actor is steered by `heading_error_wrist`, which needs the wrist's world-frame position (external localization on real hardware; the repo's own docstring flags it as "NOT proprioception-only"). Replace it with:

1. a **yaw-rate command** `yaw_rate_cmd` (rad/s) that steers the crawl, and
2. a **filtered heading-rate signal** — the measured rate of change of the base-to-wrist (claw) direction — computed only from the gyro, gravity direction, and joint angles.

The policy keeps crawling forward at the Phase 1 curriculum speed `v_target` (0.06 → 0.10 m/s); it is steered by the commanded turn rate instead of a commanded heading.

**In scope**
- New command term, swapped observation inputs, retargeted tracking rewards, shared estimator module, gait-period measurement, from-scratch training with a curriculum, evaluation, export.

**Out of scope for Phase 2**
- Any new behavior beyond steered crawling (Phase 3).
- Sensor-noise modeling (§4).

## 2.2 Why a yaw-rate command instead of a heading error

- **Heading error needs absolute heading.** On hardware you would integrate a gyro (drifts), use a magnetometer (unreliable near servos and indoors — and the chosen QMI8658 has none), or add external tracking. None is available.
- **Yaw rate is directly measurable**; noise must be tolerated, not integrated away.
- Operationally it behaves like a joystick turn input. "Face that way" can be layered on top later (§4).

## 2.3 The complication: the gait sways the base

The crawl gait moves the base left/right, so instantaneous yaw rate oscillates even while traveling straight. Track the **net rotation rate over a window about as long as the gait repeat period**. Nothing detects the cycle online; the period is measured once offline from the Phase 1 policy and stored as a constant (§2.6.7).

## 2.4 Command interface

| Command | Range (initial) | Meaning |
|---|---|---|
| `yaw_rate_cmd` | −0.3 to +0.3 rad/s (**provisional training range**) | Net turn rate of the claw direction (positive = CCW from above). `0` means crawl straight along the claw direction. The achievable range is unknown and intentionally NOT pre-gated by scripted tests (§0.2): RL discovers it, and §2.11 evaluation measures and records it in `policy_io.md`. |

- **Forward speed is not a command in Phase 2.** It stays the Phase 1 curriculum `v_target`, applied along the **claw frame** (the rotating heading frame).
- The command (1 value, optionally scaled) enters **both** actor and critic observations.
- Sampling/resampling: §2.8.
- **Implementation anchor:** the stub `crawl_env_cfg.CrawlCommandsCfg.base_velocity` (`UniformVelocityCommandCfg`, `ang_vel_z=(-0.3, 0.3)`) is already wired into `forward_velocity_exp`/`overspeed_penalty` through their `command_name` parameter when `use_velocity_commands=true`. Phase 2 replaces the stub flag with a real yaw-rate command term (a subclass of `CommandTerm`, reusing the `just_resampled` re-anchoring pattern of `UniformCrawlHeadingCommand`). Only the yaw-rate component is active in Phase 2.

## 2.5 Observation design

### 2.5.1 Actor (must be reproducible on real hardware)

| Group | Dim | Source on real robot |
|---|---|---|
| Projected gravity in base frame | 3 | IMU tilt estimate (roll/pitch only matter) |
| Base angular velocity in base frame | 3 | IMU gyro (QMI8658) |
| Joint positions relative to default | 6 | Servo encoders (STS3215) |
| Joint velocities | 6 | **Finite difference of encoder position** (servos report position only), filtered, as computed on board (recipe documented in `policy_io.md`; sim uses the sim's joint velocities until the sensor-noise pass) |
| Previous action | 6 | Controller memory |
| **Command `yaw_rate_cmd`** | 1 | Operator / supervisor |
| **Filtered heading rate (estimated, §2.6)** | 1 | Computed on board from the above |

New actor total: **26-D** (unchanged size: 2 inputs removed, 2 added). **Removed vs Phase 1:** the 2-D `heading_error_wrist` term (it requires world-frame wrist localization; its rate-based replacement is the heading-rate term). This term is currently the *only* command pathway the actor has — removing it and adding the command is the replacement; do not ship an intermediate state with neither.

**Feeding (owner decision — start simple):** the sim computes every row above from ground-truth states (projected gravity from the true root orientation, root angular velocity as "gyro", true joint angles through FK/estimator) with **no sensor-noise model** (§2.9). Phase 1's existing per-term `UniformNoiseCfg` corruption stays untouched — that is the full extent of "noise" in Phase 2. The right-hand column describes the *hardware* source that the sensor-noise pass and the deployment client must reproduce.

Speed note: the real robot has no forward-velocity sensor; the policy never observes its own speed and produces `v_target` "blind" as a gait property (already true in Phase 1; intentional). Forward-speed tracking is judged only from sim ground truth.

### 2.5.2 Critic (privileged, simulation only)
Everything the actor sees, plus: base linear velocity (base frame), base height, ground friction, ground-truth filtered heading rate, and (optionally) the ground-truth world heading error.
**No fallback clause is needed:** the installed `rsl_rl` 5.5.1 already runs asymmetric actor-critic via `obs_groups` (policy group for the actor; policy+critic groups for the critic). Privileged terms must NOT be added to the `policy` group. The exact ordering/scaling is documented once in `policy_io.md` (§2.12 — created there, it does not exist yet).

### 2.5.3 Optional: gait-phase clock
The existing `mdp.gait_phase_clock` (2-D sin/cos, 0.5 Hz) is **off by default; keep it off** for Phase 2 so the reward and curriculum findings isolate the command change.

## 2.6 Heading-rate signal (must work on the real robot)

### 2.6.1 Definition
"Heading" is the horizontal-plane direction of the vector from the base origin to the wrist origin (same arrow as `get_claw_dir` / `wrist_pointing`). We need its **rate of change**, never its absolute world angle.

### 2.6.2 Step-by-step computation (exact — no level-base approximation)
Inputs: `ω_b` gyro (base frame), `q_imu` the tilt-filter quaternion (gravity-aligned; its yaw origin is arbitrary and slowly drifting, which is irrelevant to a rate), `q` joint angles, `dt`, filter state.

An earlier draft decomposed the rate as `r_base = ω_b·u` plus `dψ/dt` with `ψ` measured against the *projected base x-axis*. That is exact only for a level base or pure rotation about vertical; with simultaneous tilt and pitch/roll rate (which the crawl gait has) it carries a cross-term error `−(ω·x̂)(x̂·u)/(1−(x̂·u)²)` on the order of the 0.01–0.05 rad/s band this signal must resolve, and the §2.10 "tilted base" unit test would fail against that formula. Use one of the two exact formulations:

**Primary (recommended): world-azimuth differencing.**
1. `p_b = FK(q)` — wrist origin relative to base origin, base frame (URDF kinematics).
2. `p_w = R(q_imu) · p_b` (world frame; arbitrary constant yaw offset is harmless).
3. `ψ_w = atan2(p_w.y − base_w.y, p_w.x − base_w.x)` of the horizontal projection.
4. `r_inst = wrapped finite difference of ψ_w / dt`, low-pass filtered.
5. `r_filt = EMA(r_inst)` (§2.6.7 constants).
On hardware, the measured Madgwick yaw drift (~0.1°/s ≈ 0.0017 rad/s, `imu/README`) will enter this signal as a small constant rate bias; Phase 2 does **not** model it in training (§2.9) — it is handled on the robot by the existing firmware startup bias calibration, and gets trained against in the sensor-noise pass (§4).

**Alternative A (exact body-frame form, avoids explicit rotation):** with `u = −g_b/‖g_b‖`, `x̂` the base x-axis, `p_h = p_b − (p_b·u)u`, and `ψ = atan2(p_h·e2, p_h·e1)` where `e1 = normalize(x̂ − (x̂·u)u)`, `e2 = u × e1`, then
`r_inst = [(ω·u) − (ω·x̂)(x̂·u)] / (1 − (x̂·u)²) + dψ/dt` (wrapped, filtered).
**Alternative B:** analytic `dψ/dt` from joint velocities and the kinematic Jacobian (avoids differentiating encoder noise). Optional refinement.

### 2.6.3 Degenerate cases and fallbacks
- **Short vector** (arm curled close to the body): when `‖p_h‖ < ε` (a few cm), ψ is unstable. Blend smoothly to the plain gyro yaw rate: `r = α·r_inst + (1−α)·r_base`, `α: 0→1` as `‖p_h‖` goes `ε→2ε`. (Mirror the 0.01 m validity threshold and base-forward fallback already used in `mdp.get_claw_dir` so sim reward and estimator agree on when the arrow "exists".)
- **Base on its side / upright** (`x̂` nearly parallel to `u`): fall back to `r_base` only and log the event.
- Keep the config switch `heading_rate_mode: "arm_vector" | "base_gyro"` (§2.6.6) so both can be compared.

### 2.6.4 One implementation for sim and real
Write the estimator once, as a pure-function module at
`source/zombie_crawl/zombie_crawl/zombie_crawl/estimators/heading_rate_estimator.py`
(inside the package so both the sim task and a future hardware client import the same code), accepting PyTorch tensors (batched, in-sim) and NumPy arrays (on hardware), with the signature `(gyro, gravity_estimate_or_quat, joint_positions, dt, filter_state) -> (rate, filter_state)`. **In Phase 2 the sim feeds it ground-truth quantities** (projected gravity, root angular velocity, true joint angles) — no sensor model (§2.9, owner decision). On hardware the same function is fed by the existing `imu/` stream (the ESP32 does all IMU processing; the host client adds wrist FK from the servo-bus joint positions — the IMU knows nothing about joints, which is why this computation lives in a host-side shared module, not on the board). The one hard parity rule: **the heading-rate filter/EMA constants (`τ_filter`, §2.6.7) must be identical in sim and on hardware** — filter *lag* changes the closed loop itself even when noise is not modeled; the hardware client reads the same config file.

### 2.6.5 What the reward uses
- **Actor observation:** the filtered rate from the shared estimator, fed by ground-truth sim states in Phase 2 (§2.6.4, §2.9).
- **Reward:** the *ground-truth* net heading rate over a sliding window `T_w` (unwrapped true heading now minus `T_w` ago, ÷`T_w`; §2.6.7), computed from true sim poses. In Phase 2 the actor's input differs from the reward's target only by the estimator's filtering (no noise), so the training objective and the observation stay in agreement; the sensor-noise pass is what adds the input noise the policy must then cope with.
- **Ring-buffer re-anchoring (required):** the repo already learned that stateful terms must re-anchor when the command changes (see `just_resampled` in `mdp/commands.py`, consumed by `forward_progress_lpf` and `stalled`). The heading-rate ring buffer must do the same: on command resample and on episode reset, refill the buffer with the current heading (zero net rate) — do not mix old/new command phases in one window.

### 2.6.6 Simplified fallback
If the estimator is fragile, use only `r_base` (plain gyro yaw rate) filtered over a gait cycle — config switch above. Simpler, and behaves similarly over a full cycle for gentle steering.

### 2.6.7 Gait period and filter constants (measured offline, not detected online)

**Principle.** The policy has no explicit gait generator. The crawl is roughly periodic (repeating reach/pull); the filter just needs a time constant on the order of that period. Measure it once from the trained Phase 1 policy (`checkpoints/model_1800.pt`) and write it into config. Same constants in sim reward, actor obs, and hardware. No online period detection.

**Measurement procedure** (`scripts/measure_gait_period.py`):
1. Load the Phase 1 checkpoint in a play/eval-style env (reuse `scripts/evaluate.py`'s env setup — `Zombie-Crawl-SO100-Play-v0` + `play_mode`, deterministic path), 4096 envs (fall back to a few hundred if VRAM-bound), 60 s each, discard the first 5 s. Note Phase 1 conditions on the commanded heading via `heading_error_wrist`; run at the closest achievable forward speed (v_target = 0.05, 0.10, 0.15 via `--v-target`) on flat ground at default friction.
2. Log joint positions (6), base yaw rate, and base forward velocity at the 50 Hz control rate.
3. Per env and per joint: detrend, FFT (Hann) or autocorrelation; dominant peak above a 0.1 Hz floor → period `T = 1/f`.
4. Same on base yaw rate (sway is usually at the joint frequency or 2×).
5. Combine across envs with the median per speed; record spread and fraction of envs with a clear peak.
6. `T_gait` = **longest** median period across speeds (cover the slowest rhythm).
7. Write `configs/gait_period.yaml` and `docs/gait_period_report.md` (both created; `configs/` does not exist yet; spectra plots + period table).

**Constants derived from `T_gait`:**

| Constant | Value | Used for |
|---|---|---|
| `T_w` (reward window) | `2 × T_gait` | Sliding-window net heading rate (ground truth) for the reward. Ring buffer, re-anchored per §2.6.5. |
| `τ_filter` (observation EMA) | `1.0–1.5 × T_gait` | Filter on the estimated heading rate the actor sees and that runs on hardware. |

Both overridable config values.

**If there is no clear period:** use fixed defaults (`T_w = 3 s`, `τ_filter = 1.5 s`) and log it. A fixed window still averages the sway; the failure mode is a gait slower than the window (oscillation leaks) → lengthen.

**Trade-offs and the settle-time coupling.** Longer window = less sway, more lag. §2.11 demands ≤3 s settling after a yaw-rate command change, but the defaults above can put the observation lag at 1.5–4.5 s — the acceptance criterion may be unreachable by construction. Resolve explicitly: (a) tune `τ_filter` first (the EMA is the dominant lag), (b) if settling still exceeds 3 s, prefer relaxing the settle target to `max(3 s, 2·τ_filter)` and record the deviation, rather than shortening the window below `1×T_gait` (sway leak is worse than slow settling — the sway corrupts the reward everywhere, the lag only at command changes). Re-check at final-policy evaluation either way.

## 2.7 Reward design, terminations, resets

### 2.7.1 Rewards
Existing Phase 1 terms are kept with the changes below. All terms are logged separately. **Naming note:** the repo has no term called "heading alignment" or "world-frame yaw penalty". The directional set is: `wrist_pointing` (−1.5; the anti-backward-crawl exploit guard, documented as dominating the forward term), `lateral_velocity` (−1.0), `overspeed` (−2.0), all on the **claw axis**; plus `forward_velocity_exp` (+2.5) and `forward_progress_lpf` (+1.0), currently scored on the frozen world-frame **command axis**.

| Term | Form | Sign |
|---|---|---|
| `forward_velocity_exp` (retargeted) | `exp(−(vx − v_target)² / σ²)` with `vx` in the claw/heading frame | + |
| Yaw-rate tracking (new) | `exp(−(r_filt_true − yaw_rate_cmd)² / σ_r²)` | + |
| `forward_progress_lpf` (retargeted) | displacement along the **claw axis** ÷ target | + |
| `overspeed` | unchanged form, plus a cap on tracked yaw rate | − |
| `wrist_pointing`, `lateral_velocity` | unchanged (claw axis); see open item in §2.15 | − |
| `vertical_velocity`, `action_rate`, `joint_acc`, `joint_torques`, `orientation`, `recovery` | unchanged from Phase 1 | per Phase 1 |

Rules:
- **Keep `wrist_pointing`, `lateral_velocity`, `overspeed`** (claw-axis). They are Phase 1's anti-exploit levers; removing the claw-leading guard invites the backward/drag exploit it was built to kill. What Phase 2 *does* remove is scoring progress against a **frozen world-frame command axis** in the tracking terms — those become claw-frame `v_target` and `yaw_rate_cmd` targets.
- **Steered crawl:** both tracking terms are active at once. Forward velocity is measured in the claw/heading frame, so a curved path still counts as forward progress. No world-frame lateral penalty (a curve produces lateral world displacement by definition); keep the heading-frame `lateral_velocity` penalty.
- **Stateful re-anchoring:** `forward_progress_lpf`, the heading-rate ring buffer, and any command-windowed EMA must re-anchor on the new command term's `just_resampled` flag (same pattern as `mdp/commands.py`).
- **Curriculum heading gate:** the Phase 1 difficulty curriculum includes a heading gate (0.75) defined on the commanded-heading metric, which no longer exists. Re-base it on the yaw-rate tracking score (or disable it) and document the choice (§2.15).

### 2.7.2 Terminations and resets

| Term | Phase 2 handling |
|---|---|
| `time_out` (20 s) | truncation, unchanged |
| `base_nan`, `base_too_high` (0.45 m) | unchanged |
| `out_of_bounds` (2.5/1.5 m) | unchanged (terminal success) |
| `stalled` (0.05 m / 12 s) | unchanged, except the "command axis" it measures along is now the claw axis; it keeps re-anchoring on command resample (extend the existing hook to the new command term) |

- **Resets:** keep Phase 1 start poses **and weights** (folded 0.5 / on_side 0.25 / face_down 0.25) so recovery keeps exercising. Spawn yaw stays ±180° (the task is rotationally symmetric; there is no world reference direction anymore — the command supplies the turn rate).

## 2.8 Command sampling, curriculum, and from-scratch training

### 2.8.1 Sampling inside an episode
- Episodes stay at **20 s**; `yaw_rate_cmd` is **resampled every 5–8 s** (random, per env, staggered — reuse the CommandManager `time_left` pattern from `UniformCrawlHeadingCommand`).
- Sample magnitudes uniformly in the provisional §2.4 range; reserve roughly 20 % of samples for near-zero values so straight crawling is well represented.
- On every resample: flip `just_resampled`; all stateful readers re-anchor (§2.6.5, §2.7.1).

### 2.8.2 Curriculum stages
A single run from random initialization advances through the stages **automatically** on mean episode reward + the stage's acceptance metric (or after a fixed iteration budget); no manual hand-offs between stages. These Phase 2 stages are **separate from** (and sit on top of) the existing Phase 1 difficulty curriculum (v_target, friction_scale, difficulty/fallen/pushes) — keep that machinery, add the command stages around it.

| Stage | Command | Goal |
|---|---|---|
| 0 | `yaw_rate_cmd ≡ 0` (observation layout final, incl. command + heading-rate inputs) | Learn the plain crawl from scratch in the new layout; reaching Phase 1 crawl quality here (regression gate vs the §0.1 baseline) shows the new inputs and rewards are not a handicap |
| 1 | `yaw_rate_cmd` in a narrow range (about ±0.1 rad/s) | First steering |
| 2 | Full provisional range (±0.3 rad/s) | Full steering range |
| 3 | Full range with physics domain randomization (friction/pushes/fallen; sensor noise deferred, §2.9) | Robustness |

### 2.8.3 Training from scratch (no checkpoint reuse)
- Every Phase 2 run starts from random initialization. There is no weight transfer from the Phase 1 policy or from any earlier run.
- **Experiment log:** since the point of the phase is to find what works, record every run in `docs/phase2_report.md`: reward weights, curriculum thresholds, outcome, and why a variant failed (extend the style of `docs/training_failure_analysis.md`).
- **Final recipe:** the committed config + automatic curriculum must reproduce the Phase 2 policy from a fresh initialization with one command. Confirm with at least two seeds.
- **Quality gate:** the from-scratch Phase 2 policy's crawl metrics must be no more than 10–15 % worse than the Phase 1 policy under the same eval (§2.11, §0.1 numbers).

## 2.9 Sensor and noise model — deferred (owner decision: start simple)

**Phase 2 simulates no IMU.** The actor's heading-rate input is computed from ground-truth sim states through the shared estimator (§2.6.4); nothing degrades the gyro, gravity, or joint-angle inputs beyond Phase 1's existing per-term `UniformNoiseCfg`. No simulated accelerometer, no fusion filter in sim, no per-episode bias, no encoder quantization, no datasheet work. All IMU processing (Madgwick, boot gyro-bias calibration) stays where it already lives and works — the ESP32 firmware in `imu/`.

**Why this is defensible now:** the Phase 2 deliverables are the command interface and the heading-rate signal *definition and filter* — neither requires noise to be discovered. The hardware pipeline already exists and is measured (roll/pitch σ ≈ 0.02°, yaw drift ~0.1°/s at rest, `imu/README`), so Phase 2 can hand the robot a signal computed by the same code path.

**The hooks that keep the sensor-noise pass cheap (build these in Phase 2):**
1. The estimator is a pure shared function (§2.6.4) — the noise pass changes only what *feeds* it, not the module.
2. `τ_filter`/`T_w` come from one config read by both sim and the hardware client (`configs/gait_period.yaml`) — lag parity is maintained from day one.
3. The reward already uses the ground-truth window (§2.6.5), so adding input noise later does not move the objective.

**Explicitly accepted risk of the simple start:** the policy never trains against a biased gyro (which would show up as a constant offset in the measured heading rate), and never tests whether steering survives quantized encoders. Expect the first real-robot bring-up to expose exactly these; that is the trigger for the sensor-noise pass (§4).

Existing Phase 1 domain randomization (friction, pushes, fallen starts) stays — it is physics, not sensing.

## 2.10 Estimator unit tests (`pytest`, pure NumPy — no sim needed)

These verify the §2.6.2 math, not what the robot can do:
- **Pure base rotation** about vertical at a known rate, joints fixed → output matches truth within tolerance — **repeated with the base rolled/pitched** (the test the earlier formula would fail; §2.6.2).
- **Pure arm sweep:** base fixed, shoulder pan at known rate → matches the geometrically expected rate.
- **Tilted base** for both cases; results must not depend on tilt.
- **Short-vector/degenerate:** curl the arm → smooth blend to gyro rate, no spikes; on-side base → falls back with log flag.
- **Gait-window filter:** synthetic oscillation + small mean → filtered output recovers the mean within tolerance; ring-buffer re-anchor on command change emits zero rate, not a mix.

## 2.11 Evaluation and acceptance criteria

### 2.11.0 Prerequisite
Fix `scripts/evaluate.py`'s known bucketing bug (`out_of_bounds` counted as early termination → dropped episodes/NaN displacement; noted in README §5). Rework eval axes: Phase 1 projects displacement on the frozen `zombie_ref_yaw` reference; Phase 2 metrics project on the **claw axis** (forward) and on the heading-rate definition (§2.6) — the same axes the rewards use.

`scripts/evaluate.py` is extended with per-scenario runs, 100 envs each, with **no sensor-noise model** (§2.9); friction randomized (report 0.5 / 0.8 / 1.2 separately — matches the training range 0.5–1.2).

| Scenario | Acceptance (initial thresholds, adjust with owner) |
|---|---|
| **Straight crawl** (`yaw_rate_cmd=0`, `v_target=0.10`) | Mean forward speed within ±30 % of target; mean filtered heading rate near zero (initial threshold |r| < 0.05 rad/s); no early terminations — **and, because the Phase 1 baseline is modest (§0.1), state the Phase 1 number this was compared against in the report** |
| **Steered crawl** (`v_target=0.10`, `yaw_rate_cmd=±0.1`, ±0.2, plus other points in the provisional range) | Forward speed and filtered heading rate each within ±35 % of command; no early terminations. Record the maximum tracked rate; if clearly below the provisional range, adjust range/thresholds with the owner |
| **Command changes** (e.g. +0.1 → −0.1 rad/s mid-episode) | Heading rate settles within `max(3 s, 2·τ_filter)` (§2.6.7 coupling); no flips beyond Phase 1 norms |
| **Regression** | Crawl metrics no more than 10–15 % worse than the Phase 1 policy measured under the same eval (§0.1) |

Also report: settle time per command change; success rate per friction.

## 2.12 Deliverables

1. **Code** (existing extension project, `source/zombie_crawl/zombie_crawl/…`):
   - Task config: new yaw-rate command term (`just_resampled` re-anchoring), retargeted tracking rewards, new yaw-rate tracking reward, Phase 2 curriculum stages, swapped observations (heading-error out; command + heading-rate in).
   - `estimators/heading_rate_estimator.py` (shared, tensor + NumPy) with unit tests.
2. **Scripts:** `measure_gait_period.py` (new) → `configs/gait_period.yaml` + `docs/gait_period_report.md` (all created; `configs/` does not exist yet); `train.py` (curriculum-stage flags added to the existing wrapper, for debugging a single stage; the default is the full automatic curriculum from scratch); `play.py` (add yaw-rate command input from keyboard/gamepad/CLI to the existing wrapper); `evaluate.py` (bug fix + per-scenario runs).
3. **Trained artifacts:** best checkpoint; exported **ONNX and TorchScript** actor — note `checkpoints/exported/{policy.pt,policy.onnx}` already exists for Phase 1 and the export pipeline is in place, so this is "re-export", not build; TensorBoard logs (`logs/rsl_rl/`); the final training config that reproduces the policy from a fresh initialization.
4. **Videos:** straight crawl, steered crawl (left and right), a command change mid-episode (reuse the existing `train.py --video` / `play.py --video` recording pipeline).
5. **Docs:** **create** `policy_io.md`: exact observation ordering/scaling, command semantics, estimator pseudocode, filter constants (the sim↔hardware parity contract, §2.6.4), and the reachable `yaw_rate_cmd` range measured at final evaluation (§2.11); `docs/gait_period_report.md`; and a Phase 2 section — appended to **`report.md`** or a new `docs/phase2_report.md` (the real Phase 1 training docs are `README.md` §5, `report.md`, `docs/training_failure_analysis.md`).
6. **Tests (`pytest`):** estimator tests (§2.10), env smoke test with the new 26-D actor obs (extend `tests/test_smoke.py`), command-sampling statistics, short training run writing a checkpoint, play-script smoke test.

## 2.13 Implementation order

1. Measure Phase 1 gait period → filter constants (§2.6.7); build the estimator module + unit tests, ground-truth-fed, shared EMA constants (§2.6, §2.10). **No sensor-noise layer** (deferred, §2.9).
2. Environment changes: yaw-rate command term, 26-D observation swap, retargeted tracking rewards, new yaw-rate reward, curriculum heading-gate re-base.
3. Short from-scratch smoke run at Stage 0 (straight crawl) to confirm the new observation layout and reward set learn at all; compare against the Phase 1 baseline.
4. Iterate on rewards and curriculum thresholds with full from-scratch runs (stages 0–2 advancing automatically), logging every experiment (§2.8.3).
5. Stage 3 robustness (physics DR only); final evaluation vs §2.11 (with the `evaluate.py` prerequisite fixed); record the measured reachable `yaw_rate_cmd` range in `policy_io.md`. Re-run the final recipe from a fresh initialization with a second seed.
6. Export (ONNX/TorchScript re-export), videos, docs (`policy_io.md` et al.), tests.

Report after each step. Stop and ask the owner if: from-scratch training cannot reach Phase 1 crawl quality in the new layout, or the policy cannot track `yaw_rate_cmd` at all after training.

## 2.14 Risks and mitigations

| Risk | Mitigation |
|---|---|
| From-scratch training does not reach Phase 1 crawl quality with the new layout or rewards | Stage 0 regression gate vs the *real* Phase 1 baseline (§0.1); logged reward/curriculum experiments; add changed terms one at a time |
| Weak servos (2.9 N·m preset) cannot turn the crawl at the provisional rates | Discovered by RL itself (no scripted pretest): §2.11 steered-crawl row measures the achievable rate; if unreachable, lower the command range/thresholds with the owner |
| Removing the claw-leading guard re-opens the backward-crawl exploit | `wrist_pointing`, `lateral_velocity`, `overspeed` explicitly kept (§2.7.1); video review, not just curves |
| Tilt-dependent heading-rate error | Exact formulas only (§2.6.2); tilted-base unit tests |
| Estimator fragile when the arm is curled | Short-vector blending, `base_gyro` fallback mode, mirrored threshold of `get_claw_dir` |
| No clear gait period / period shifts after training | Fixed defaults (3 s / 1.5 s); re-run `measure_gait_period.py` on the final policy, update constants and re-train from scratch if `T_gait` moved >~25 % |
| Filter lag vs 3 s settling | Joint tuning rule in §2.6.7; relaxed criterion stated, not silently missed |
| Gyro bias on the **hardware** shows up as a constant heading-rate offset | Untrained in Phase 2 by design (§2.9); mitigated on hardware by the existing firmware boot bias calibration (`imu/`); the sensor-noise pass (§4) trains it |
| Policy leans on clean sim signals (noise/quantization untrained) | Deliberate simple start (§2.9); hooks keep the noise pass cheap; run the noise pass **before** trusting hardware behavior |
| Reward discontinuities at command changes | Re-anchored stateful terms (§2.6.5) |
| Physics exploits (hopping, jitter) | Keep vertical/action-rate/overspeed penalties; video as well as curves |
| Doc/code drift (soft-limit factor 0.95 vs 1.0, "27-D" docstring) | Reconcile during Phase 2 (§0.1) |

## 2.15 Owner decisions and remaining open items (Phase 2)

### Resolved
1. **Heading error is replaced by a yaw-rate command + filtered heading-rate signal** (owner); "heading" = the base-to-wrist (claw) direction, as in Phase 1.
2. **Max turn rate unknown** — the 0.3 rad/s provisional range IS the training range; no scripted pretest (owner). The reachable range is discovered by training, measured in §2.11, and recorded in `policy_io.md`.
3. **No sensor/IMU modeling in sim** (owner); all IMU processing stays on the ESP32 (§0.2, §2.9).
4. **`wrist_pointing`/`lateral_velocity`/`overspeed` stay** (Phase 1 exploit guards).
5. **Train from scratch at every phase (owner):** no checkpoint extension or fine-tuning; the deliverable is a reproducible from-scratch recipe (§0.2, §2.8.3).

### Remaining open (agent tasks unless marked owner)
- **(Owner/agent)** `wrist_pointing` is defined in Phase 1 as the claw arrow versus the *commanded heading*, which no longer exists in Phase 2. Decide what it compares against (for example the base's travel direction, to keep the anti-backward-crawl guard) and document it.
- **(Agent)** Re-base or disable the Phase 1 curriculum heading gate (§2.7.1).
- IMU mounting on the arm (assumed: base link) — record the mount pose at deployment time.
- Operator command input for the real robot later (gamepad, keyboard, supervisor program).
- Reconcile README/code mismatches (soft limit 0.95 vs 1.0; actor dim docstring) (§0.1).

---

# PHASE 3 — Additional maneuvers: sit still, rotate in place, variable-speed forward + turn

**Starting point:** the Phase 2 code and findings (26-D layout, yaw-rate command, heading-rate signal, shared estimator, `configs/gait_period.yaml`, and the reward set and curriculum that worked). Phase 3 **trains from scratch** with the extended command space; it does not warm-start from Phase 2 weights, and it does not redo Phase 2 work.

## 3.1 Goal and scope

Let one policy handle four command modes, using only signals measurable on the real robot:

- `vx_cmd > 0`, `yaw_rate_cmd ≈ 0`: crawl forward.
- `vx_cmd = 0`, `yaw_rate_cmd = 0`: **sit still** in a defined rest pose.
- `vx_cmd = 0`, `yaw_rate_cmd ≠ 0`: **rotate in place** without translating.
- `vx_cmd > 0`, `yaw_rate_cmd ≠ 0`: crawl forward while turning (variable speed; Phase 2 only steered at the fixed `v_target`).

### 3.1.1 Why one policy
- Command-conditioned policies are the standard approach in legged locomotion: "stand", "walk", "turn" are points in one command space.
- The hard part is the *transitions* (crawl↔sit↔rotate). A single policy sees them in training. Separate models need a hand-built switcher, and a bad switch drops the robot into a pose neither model has seen.
- Cost: training must include zero and rotate-only commands often enough, and rewards must change meaning with the command. Both are specified below.

## 3.2 Command interface

| Command | Range (initial) | Meaning |
|---|---|---|
| `vx_cmd` | 0 to 0.15 m/s | Forward speed target in the **claw frame**. Replaces Phase 2's fixed `v_target`; Phase 1's 0.10 m/s target stays inside this range. |
| `yaw_rate_cmd` | −0.3 to +0.3 rad/s (provisional; unchanged from Phase 2) | Net turn rate (positive = CCW from above). |

- **Deadband:** treat `|vx_cmd| < 0.02` m/s and `|yaw_rate_cmd| < 0.05` rad/s as zero. Derive `is_zero_cmd` and `is_rotate_only` indicators from these thresholds; use them to gate rewards and terminations (§3.5, §3.6).
- Both commands enter **both** actor and critic observations.
- **Implementation anchor:** extend the Phase 2 command term to include the `lin_vel_x` component (the existing stub already carries `lin_vel_x=(0.05,0.15)`; extend its lower bound to zero). Isaac Lab's stock `rel_standing_envs` option is the natural home for the sit fraction; use it or replicate its behavior deliberately. The Phase 1 `v_target` ramp (0.06 → 0.10) is superseded by sampled `vx_cmd`; decide whether to keep it as a cap in early stages and document the choice.

## 3.3 Observation changes

- **Actor:** add `vx_cmd` → **27-D** (Phase 2's 26 inputs + 1). All other inputs unchanged.
- **Critic:** add the command-mode indicators (`is_zero_cmd`, `is_rotate_only`) to the privileged group. Privileged terms must not enter the `policy` group.
- `vx` note: the real robot still has no forward-velocity sensor, so `vx_cmd` is tracked "blind"; tracking quality is judged from sim ground truth only.
- **Gait clock:** stays off by default. If it is later enabled, freeze its phase when `is_zero_cmd` so the policy receives no rhythmic input while sitting (term-level state keyed by the command indicator; `gait_phase_clock` currently advances unconditionally).

## 3.4 Rest pose ("sitting")

- Define a **rest pose**: base link on the ground, arm folded close to the body, low CoM, stable at rest with the servos holding position. **Folded (URDF zero) is the default**: it is the action-zero pose of `OffsetEMAJointPositionToLimitsAction` (the policy reaches it with a zero action) and holds still under the servo preset (measured: base speed ≈3e-4 m/s, zero drift, base rests at z≈0.033 m). A lower candidate (`shoulder_lift 0.3, wrist_flex 0.6, gripper 0.5`) also holds dead-stable at z≈0; naive mid-joint tucks collapse instead of sitting. Record the chosen winner in `docs/rest_pose.md` (joint angles, resting contacts).
- The candidates were verified once under the **`servo` preset** (not `ideal`); no settle-test script is kept in the repo (the hold is checked through training behavior and §3.9 sit acceptance instead).

## 3.5 Reward design by command mode

Phase 2 reward terms stay. Phase 3 changes and additions:

| Term | Active when | Form | Sign |
|---|---|---|---|
| `forward_velocity_exp` | always | target is now `vx_cmd` (was `v_target`): `exp(−(vx − vx_cmd)² / σ²)` in the claw frame | + |
| Yaw-rate tracking | always | unchanged from Phase 2 | + |
| `forward_progress_lpf` | `vx_cmd` above deadband only | claw-axis displacement ÷ target | + |
| Stillness: base speed | `is_zero_cmd` | `‖v_base‖²` penalty (after settle grace) | − |
| Stillness: joint velocity | `is_zero_cmd` | `‖q̇‖²` penalty | − |
| Rest-pose attraction | `is_zero_cmd` **and base upright** | `exp(−‖q − q_rest‖² / σ_q²)` | + |
| In-place rotation: no translation | `is_rotate_only` | penalty on `‖v_base,xy‖` | − |
| Overspeed | always | penalty above `2 × max(vx_cmd, small)` and above a yaw-rate cap | − |
| `wrist_pointing`, `lateral_velocity`, `vertical_velocity`, `action_rate`, `joint_acc`, `joint_torques`, `orientation`, `recovery` | always | unchanged | per Phase 1 |

Rules:
- **Keep the claw-axis guards.** `wrist_pointing`, `lateral_velocity`, `overspeed` stay: sit/rest rewards now make stationary poses attractive, so removing the claw-leading guard exactly then invites the backward/drag exploit it was built to kill.
- **Recovery precedence:** when `is_zero_cmd` and the base is tipped, `upright_bonus` stays active and rest-pose attraction is gated off (via the upright indicator) until the base is level. A fallen robot under a zero command must get up first, then sit.
- **Settling grace period:** scale stillness penalties down for ~2 s after the command changes to zero (unavoidable deceleration).
- **Gate smoothly** (short ramp ~0.5 s) to avoid reward discontinuities.
- **Push interaction (sit):** the training disturbance (`disturb_all_bodies`, ±0.4 N per body, curriculum-gated by `push_scale`) physically cannot be resisted while sitting. **Decision (owner):** pushes stay on during sit training; sit is graded twice — pushless (strict, <0.01 m/s) and with pushes (loose, "stays roughly put") (§3.9).
- **Forward + turn:** both tracking terms active at once; forward is measured in the claw/heading frame so a curved path still counts. No world-frame lateral penalty in this mode; keep the heading-frame `lateral_velocity` penalty.
- **Stateful re-anchoring:** every stateful reader (`forward_progress_lpf`, the heading-rate ring buffer, any command-windowed EMA, `stalled`) re-anchors on the command term's `just_resampled` flag, as in Phase 2.

## 3.6 Terminations and resets

| Term | Phase 3 handling |
|---|---|
| `time_out` (30 s, §3.7) | truncation; episodes lengthen from 20 s to 30 s |
| `base_nan`, `base_too_high` (0.45 m) | unchanged |
| `out_of_bounds` (2.5/1.5 m) | keep as terminal-success, but note it can now only fire in forward modes (sit/rotate stay <0.15 m). The curriculum escape gates built on it stay valid only while ≥30% of episodes are forward modes (§3.7 mix). |
| **`stalled`** (0.05 m / 12 s) | **must be command-gated:** active only when `vx_cmd` is above the deadband; re-anchor its window on command change. Without this, every sitting episode is failure-terminated at 12 s, i.e. the phase would punish its own target behavior. Add a regression test: a sit-policy rollout with zero commands must produce zero `stalled` terms. |

- **Resets:** keep the Phase 1/2 start poses **and weights** (folded 0.5 / on_side 0.25 / face_down 0.25) so recovery keeps exercising, **plus the rest pose** as a fourth entry with its own weight (start ~10–15% of episodes in the rest pose). Some fraction of episodes should start rest-pose + zero command so "sit and stay" is directly represented; command resampling inside the episode covers the rest of the transition space.

## 3.7 Command sampling and curriculum

### 3.7.1 Sampling inside an episode
- Episodes are **30 s**; commands **resampled every 5–8 s** (random, per env, staggered).
- Mode mix at full difficulty (per resample): **stop 20 %, forward 30 %, rotate-only 25 %, forward + turn 25 %**. Commands are sampled inside the provisional §3.2 ranges. Per-env mode is drawn per resample, not fixed per episode. `rel_standing_envs` is the reference implementation for the zero fraction.
- Within a mode, sample magnitudes uniformly in range, plus small command noise near zero to exercise the deadband (the `vx_cmd` deadband of 0.02 m/s is meaningful only if commands actually land below it).
- On every resample: flip `just_resampled`; all stateful readers re-anchor.

### 3.7.2 Curriculum stages
A single run from random initialization advances through the stages **automatically** on mean episode reward + the stage's acceptance metric (or after a fixed iteration budget). These stages sit on top of the existing Phase 1 difficulty curriculum (v_target ramp aside, §3.2).

| Stage | Command mix | Goal |
|---|---|---|
| 0 | Phase 2 behavior (`vx_cmd` constant 0.10, `yaw_rate_cmd` following the Phase 2 stages; obs layout final, incl. `vx_cmd`) | Re-learn crawl and steering from scratch in the 27-D layout (regression gate vs the Phase 2 policy) |
| 1 | Forward + stop | Sit and resume |
| 2 | Add rotate-only | In-place rotation |
| 3 | Full mix (forward + turn at sampled speeds, all transitions) | Transitions between all modes |
| 4 | Full mix with physics domain randomization (friction/pushes/fallen; sensor noise deferred, §3.8) | Robustness |

### 3.7.3 Training from scratch (no checkpoint reuse)
- Every Phase 3 run starts from random initialization. There is no weight transfer from Phase 2 or any earlier run.
- **Experiment log:** record every run in `docs/phase3_report.md` — which reward terms and curriculum thresholds made sit, rotate-in-place, and the transitions emerge, and which variants failed and why.
- **Final recipe:** the committed config + automatic curriculum must reproduce the complete behavior set (crawl, steering, sit, rotate in place, forward + turn) from a fresh initialization with one command. Confirm with at least two seeds.
- **Quality gate:** keep forward and steered commands in every stage; the final policy's crawl and steered-crawl metrics must be no more than 10–15 % worse than the Phase 2 policy under the same eval (§3.9).

## 3.8 Sensor and noise model — still deferred

Phase 3 inherits the Phase 2 decision (§2.9, §0.2): no simulated IMU. Two risks that are specific to the new maneuvers are accepted for now:
- The policy never trains against "a sitting robot whose biased gyro *believes* it is rotating", and never tests whether rotate/sit survive quantized encoders.
- The pushless sit acceptance (§3.9) is therefore optimistic for hardware; expect the first real-robot bring-up to expose exactly these, which is the trigger for the sensor-noise pass (§4).

## 3.9 Evaluation and acceptance criteria

`scripts/evaluate.py` gains per-mode scenarios (100 envs each), with no sensor-noise model; friction randomized (report 0.5 / 0.8 / 1.2 separately).

| Scenario | Acceptance (initial thresholds, adjust with owner) |
|---|---|
| **Crawl and steered crawl** (as in Phase 2, now with `vx_cmd`; also other speeds in range) | Same thresholds as Phase 2 §2.11 |
| **Sit** (zero command 20 s, from crawl and from rest starts; run with pushes ON and OFF) | After 3 s settle: mean base speed <0.01 m/s (pushes off; report pushes-on separately), joint velocities near zero, joints within 0.15 rad of rest pose, no creeping; **zero `stalled` terminations** (§3.6) |
| **Rotate in place** (`vx_cmd=0`, `yaw_rate_cmd=±0.2` provisional) | Filtered heading rate within ±30 % of command; base translation <0.15 m per 10 s. Record the maximum tracked rate; if clearly below the provisional range, adjust range/thresholds with the owner |
| **Forward + turn** (`vx_cmd=0.10`, `yaw_rate_cmd=±0.1`, plus other points in the provisional range) | Each of forward speed / filtered heading rate within ±35 % (looser than single-mode — the two interact); no early terminations |
| **Transitions** (crawl → sit → rotate → crawl+turn → sit → crawl) | Each transition settles within `max(3 s, 2·τ_filter)` (§2.6.7 coupling); no flips beyond Phase 1 norms; no "stuck" states |
| **Regression** | Crawl and steered-crawl metrics no more than 10–15 % worse than the Phase 2 policy measured under the same eval |

**Deferred with the sensor model:** the "noise robustness" row (sit/rotate thresholds under full sensor noise + per-episode gyro bias) — meaningless without a sensor model; re-add when the sensor-noise pass runs (§4).

Also report: settle time per command change; unintended-motion (creep) distance per 20 s while sitting; success rate per friction.

## 3.10 Deliverables

1. **Code:** command term extended with `vx_cmd` (and the sit fraction); mode-gated rewards and indicators; rest-pose reset entry; `stalled` command-gating; 30 s episodes; Phase 3 curriculum stages; 27-D observations.
2. **Scripts:** `train.py` (stage flags for debugging a single stage; default is the full automatic curriculum from scratch); `play.py` (add `vx_cmd` + `yaw_rate_cmd` input from keyboard/gamepad/CLI); `evaluate.py` (per-mode scenarios).
3. **Trained artifacts:** best checkpoint; ONNX/TorchScript re-export; TensorBoard logs; the final training config that reproduces the policy from a fresh initialization.
4. **Videos:** crawl, sit, rotate-in-place, steered crawl, combined transitions.
5. **Docs:** `docs/rest_pose.md`; `policy_io.md` updated (27-D layout, both commands, deadband semantics, the reachable `(vx, yaw_rate)` range measured at final evaluation); a Phase 3 section in `report.md` or `docs/phase3_report.md`.
6. **Tests (`pytest`):** env smoke test with the 27-D actor obs, command-sampling statistics (mode mix and deadband landing), the sit/`stalled` regression test (§3.6), short training run writing a checkpoint, play-script smoke test.

## 3.11 Implementation order

1. Environment changes: `vx_cmd` + sit fraction, 27-D observations, mode-gated rewards, `stalled` command-gating, rest-pose reset entry (default = folded, §3.4), 30 s episodes.
2. Short from-scratch smoke run at Stage 0 to confirm the 27-D layout learns the Phase 2 behaviors.
3. Iterate on rewards and curriculum thresholds with full from-scratch runs (stages advancing automatically), logging every experiment (§3.7.3).
4. Stage 4 robustness (physics DR only); final evaluation vs §3.9; record the measured reachable command range in `policy_io.md`. Re-run the final recipe from a fresh initialization with a second seed.
5. Export, videos, docs, tests.

Report after each step. Stop and ask the owner if: from-scratch training cannot reach the Phase 2 crawl or steering quality in the 27-D layout, the sit behavior cannot emerge at all through training, or the pushless sit cannot hold the §3.9 threshold (creep/rotations it physically could not prevent even before noise enters the picture).

## 3.12 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Arm cannot rotate in place with the weak servos (2.9 N·m preset) | Discovered by RL itself (no scripted pretest): §3.9 rotate/turn rows measure the achievable rate; if unreachable, lower the command range/thresholds with the owner |
| The new modes starve crawl/steering learning in the mixed curriculum | Forward and steered commands in every stage; regression metric vs the Phase 2 policy; adjust mode weights and stage thresholds |
| Policy creeps while sitting | Stillness + rest-pose terms, deadband (bias robustness deliberately deferred, §3.8) |
| Gyro bias makes the **hardware** robot think it is turning | Untrained by design (§3.8); mitigated on hardware only by the existing firmware boot bias calibration (`imu/`); the sensor-noise pass trains it |
| **Removing `wrist_pointing` would re-open the backward-crawl exploit** | Terms explicitly kept (§3.5); video review, not just curves |
| **`stalled` kills every sitting episode** | Command-gated termination + regression test (§3.6) |
| **Fallen starts vs sit ambiguity** | Recovery precedence rule (§3.5); rest-pose attraction gated upright |
| Sit vs push disturbance conflict | **Resolved (owner):** pushes stay on; sit graded with pushes (loose) and pushless (strict) (§3.9) |
| Filter lag vs 3 s transition settling | Joint tuning rule in §2.6.7; relaxed criterion stated, not silently missed |
| Reward discontinuities at mode changes | Smooth ramps, settle grace, re-anchored stateful terms |
| Policy leans on clean sim signals (noise/quantization untrained) | Deliberate simple start (§3.8); sensor-noise pass required **before** trusting hardware sit/rotate |

## 3.13 Owner decisions and remaining open items (Phase 3)

### Resolved
1. **Forward + turn is required** — full mode (25 %), part of the full-mix stage, acceptance row (§3.9).
2. **Max turn rate unknown** — provisional range is the training range; discovered by training and measured at §3.9.
3. **Sitting comes from a zero command only** (no sit flag); the supervisor sends in-deadband commands. Robustness of "sit" to gyro bias/noise is a sensor-noise-pass deliverable (§4), not a Phase 3 acceptance gate.
4. **Sit-vs-push resolved (owner):** pushes stay on during sit training; sit is graded twice — pushless (strict) and with pushes (loose) (§3.5, §3.9).
5. **`stalled` gets command-gated** (§3.6).
6. **Train from scratch (owner):** no warm-start from Phase 2; the deliverable is a reproducible from-scratch recipe (§0.2, §3.7.3).

### Remaining open (agent tasks unless marked owner)
- Whether to keep the Phase 1 `v_target` ramp as an early-stage cap on sampled `vx_cmd` (§3.2).
- Operator command input for the real robot later (gamepad, keyboard, supervisor program).

---

# 4. Future extensions (documented, not implemented)

- **Sensor-noise training pass (first future item):** implement the deferred checklist, retrain from scratch with the noise terms added to the recipe, and re-run the deferred evaluation rows. **Trigger:** real-robot bring-up or `imu/` recordings show drift/noise affecting the robot's behavior (for example constant heading-rate offsets, or creeping/"believing it is rotating" while sitting). Checklist:
  - Per-episode random **gyro bias** (±) — implemented as a `mode="startup"`/reset EventTerm (per-step `UniformNoiseCfg` cannot express persistence; see the `randomize_ground_friction` pattern), not term noise.
  - Gyro white noise (scaled up while the arm moves, mimicking servo vibration).
  - Per-episode **tilt offset** (start ±2–3°) on the gravity vector.
  - **Encoder quantization** (STS3215-class resolution, verify gearing vs datasheet) + noise **before** the FK that produces the heading vector, and joint velocities re-derived from the quantized positions (the on-board recipe).
  - Optional one-control-step latency flag.
  - Then re-run the deferred noise-robustness rows (sit under bias, steering under noise).
- **"Face this direction" supervisor:** external heading source (overhead camera, AprilTag, mocap) → yaw-rate commands to this policy; the heading problem lives outside the policy. Cheaper relative-turn variant: integrate the filtered heading rate for a few seconds, accepting drift (~0.1°/s measured).
- Full domain randomization and latency modelling for the real arm.
- Deployment on the real SO-ARM100/SO-101: Feetech servo interface + the existing `imu/` QMI8658 board (raw stream at 100 Hz already supports 2× the 50 Hz control rate).
- Zombie-style reward shaping (asymmetric drag/recover, lower posture).