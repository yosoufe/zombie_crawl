# Proposal (Rev. 4): Phase 2 — Command-Conditioned Crawl, Sit, and Rotate

**Project:** Zombie Crawl (SO-ARM100 / SO-101, Isaac Sim + Isaac Lab 3.0, `rsl_rl` 5.5.1 PPO)
**Status:** Phase 1 is complete: a policy exists that drags the free-floating base toward a **randomly commanded heading** (uniform ±180°, frozen per episode) at a curriculum target of 0.06 → 0.10 m/s. Shipped checkpoint: `checkpoints/model_1800.pt`. The policy is *not* a "+X crawler" and its eval numbers are modest (see §1.0) — Phase 2 baselines and regression gates must use these real numbers.
**Audience:** the project owner, and the coding agent that will implement this phase.
**Purpose of this document:** propose, justify, and specify the changes needed to let one policy (a) crawl forward, (b) sit still when there is no command, and (c) rotate in place on a yaw-rate command, using only signals measurable on the real robot.

> **Revision note.** Rev. 1 was written without inspecting this repo and conflicted with it in several places. Rev. 2 reconciles the spec with the actual code (`source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/…`, `scripts/`, `imu/`), reuses what already exists, and fixes two technical errors (heading-rate math, termination handling). **Rev. 3 (this one) reduces scope at the owner's direction: Phase 2a starts simple — no IMU or sensor-noise modeling in simulation.** The yaw-rate command signal and sitting are trained from ground-truth sim states through the shared (pure-function) estimator; gyro bias, tilt error, encoder imperfections, and the fusion filter are deferred to a Phase-2b noise pass, added only when the hardware's drifts and noise actually bite (§8, §14). All IMU processing stays on the ESP32 (`imu/`, already built). **Rev. 4:** pre-training scripted feasibility gates removed per the owner ("if I could script it, I wouldn't need RL") — the reachable command range is discovered by the RL training itself and measured in post-training evaluation (§9, §10).

---

## 1. Summary of the proposal

1. **Use one policy, not several.** A single network conditioned on a command input handles crawl, sit, and rotate. No second model, no hand-written switcher; transitions are learned.
2. **Command interface:** a 2-D command `(vx_cmd, yaw_rate_cmd)`.
   - `vx_cmd > 0`: crawl forward.
   - `vx_cmd = 0`, `yaw_rate_cmd = 0`: sit still in a defined rest pose.
   - `vx_cmd = 0`, `yaw_rate_cmd ≠ 0`: rotate in place without translating.
   - `vx_cmd > 0`, `yaw_rate_cmd ≠ 0`: crawl forward while turning (curved path).
3. **Replace the heading-error input with a yaw-rate command.** The real robot has no absolute heading source, but it has a gyro and servo encoders. A rate-based command needs nothing else.
4. **Define "heading" as the direction of the base-to-wrist (claw) vector** — the same arrow Phase 1 already uses (`mdp.get_claw_dir`, `heading_error_wrist`). Measure only its **rate of change**, computed exactly (no level-base approximation, §5.2) from the gyro, the gravity direction, and forward kinematics, and average it over ~a gait period because the gait sways the base. The period is measured once offline from the Phase 1 policy (§5.7).
5. **Fine-tune from the Phase 1 checkpoint** with a staged curriculum; compare against a from-scratch baseline.
6. **Phase 2a starts simple (owner decision): no sensor model in simulation.** The heading-rate input is computed from ground-truth sim states (gravity vector, root angular velocity, true joint angles) through the *same pure-function estimator module* the hardware client will later use, with identical filter constants on both sides. Gyro noise/bias, tilt error, and encoder quantization are deliberately **not** simulated yet — they become a Phase-2b fine-tune pass once real-hardware drift/noise proves to matter (§8, §14). All IMU processing stays on the ESP32 board, which is already built and measured (`imu/`).

### 1.0 What Phase 1 actually is (baseline facts for this spec)

These repo facts constrain several sections below; they are cited so implementers do not re-discover them.

- **Command semantics today:** the crawl direction is a per-episode uniform ±180° draw (`env.zombie_ref_yaw`, set by `mdp.reset_start_pose`), or a mid-episode-resampled heading when `use_dynamic_heading=true` (`mdp.UniformCrawlHeadingCommand`, `mdp/commands.py`). Directional rewards/terminations project onto it via `mdp.get_ref_dir`; the *locomotion* frame is the claw arrow via `mdp.get_claw_dir` (base→wrist, with a base-forward fallback `yaw − π/2` when the arm is curled).
- **Actor obs (26-D):** projected gravity (3) + `heading_error_wrist` (2, `(cos, sin)` of claw-minus-command — the only command signal the current actor sees) + base ang. vel (3) + joint pos rel. (6) + joint vel (6) + last action (6). Optional 2-D gait clock exists but is **off by default** (`use_gait_clock=False`).
- **Critic:** privileged group (base lin vel 3, base height 1, ground friction 1) concatenated for the critic only — asymmetric actor-critic is already wired (`obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}`, `agents/rsl_rl_ppo_cfg.py`).
- **Timing:** `dt=0.005`, `decimation=4` → 50 Hz control; episodes 20 s; `num_steps_per_env=40`; PPO nets 256/128/64 ELU, empirical obs norm, `init_std=0.6`, entropy 0.01, adaptive LR (`desired_kl=0.01`).
- **Terminations (`mdp/terminations.py`, `crawl_env_cfg.py`):** `time_out`, `base_nan`, `base_too_high` (0.45 m), **`out_of_bounds`** (|dx|>2.5 or |dy|>1.5 m; terminal "success"), **`stalled`** (failure if <0.05 m along the command axis within 12 s; re-anchors on command resample). The two bold terms directly interact with Phase 2 and are handled explicitly in §6.3.
- **Events / DR:** spawn poses folded/on_side/face_down with weights (0.5, 0.25, 0.25); `upright_bonus` recovery reward (w=1.0); per-env ground friction 0.5–1.2 effective (startup event, multiplier 0.625–1.5 on μ=0.8); all-body disturbance `disturb_all_bodies` ±0.4 N / ±0.03 N·m every 0.2–0.5 s, scaled by `env.extras["push_scale"]` (difficulty curriculum); obs corruption via per-term `UniformNoiseCfg`.
- **Actuators:** presets `servo` (default: STS3215-class, 2.9 N·m, 6 rad/s, PD k=1.0 d=0.1, armature 0.01) and `ideal` (10 N·m).
- **Baseline quality (use these for regression gates, not idealized numbers):** curriculum saturated ~iter 400 (4096 envs); eval of the shipped `model_1800.pt` shows ~19% of steps in the ±50% speed band and mean |yaw drift| ≈ 81° on the deterministic eval path. `scripts/evaluate.py` has a **known bug**: `out_of_bounds` is bucketed as early termination, so escaping episodes are dropped (NaN displacement on some evals). Fixing this is a prerequisite (§10.0).
- **Runtime (verified on the owner's machine):** all software lives in the Python venv at **`../.venv`** (i.e. `../.venv/bin/python`, Python 3.12.15, alongside the repo root). Installed and confirmed: `isaacsim 6.1.0.0`, `isaaclab` (3.x line; `isaaclab-rl 1.3.3`), **`rsl-rl-lib 5.5.1`** (matches §7.3's checkpoint-layout assumptions), `torch 2.11.0+cu130`; `zombie_crawl` is installed editable from `source/zombie_crawl/`, and `isaaclab_tasks` resolves to the local `../IsaacLab3` checkout. GPU: RTX 5090 Laptop, 24 GB — the 4096-env runs in `report.md` were done on this class of GPU (~1.6 s/iter). Every sim command must use this interpreter (`../.venv/bin/python …`); there is no system-wide Isaac install.
- **Doc/code mismatches to reconcile during Phase 2** (decide once, then update docs): `soft_joint_pos_limit_factor` is 1.0 in `assets/so_arm100.py` but 0.95 in README; the `PolicyCfg` docstring says "27-D" but the term dims sum to 26.

---

## 2. Background and motivation

### 2.1 Why one policy
- Command-conditioned policies are the standard approach in legged locomotion: "stand", "walk", "turn" are points in one command space.
- The hard part is the *transitions* (crawl↔sit↔rotate). A single policy sees them in training. Separate models need a hand-built switcher, and a bad switch drops the robot into a pose neither model has seen.
- Cost: training must include zero and rotate-only commands often enough, and rewards must change meaning with the command. Both are specified below.

### 2.2 Why a yaw-rate command instead of a heading error
- **Heading error needs absolute heading.** The current `heading_error_wrist` term uses the world-frame wrist position (external localization in reality; the repo docstring itself flags it as "NOT proprioception-only"). On hardware you would integrate a gyro (drifts), use a magnetometer (unreliable near servos, indoors — and the chosen QMI8658 is 6-axis, no magnetometer), or add external tracking. None is available.
- **Yaw rate is directly measurable**; noise must be tolerated, not integrated away.
- Operationally it behaves like a joystick turn input. "Face that way" can be layered on top later (§14).

### 2.3 The complication: the gait sways the base
The crawl gait moves the base left/right, so instantaneous yaw rate oscillates even while traveling straight. Track the **net rotation rate over a window about as long as the gait repeat period**. Nothing detects the cycle online; the period is measured once offline from the Phase 1 policy and stored as a constant (§5.7).

---

## 3. Command interface

| Command | Range (initial) | Meaning |
|---|---|---|
| `vx_cmd` | 0 to 0.15 m/s | Forward speed target in the **claw frame** (the rotating heading frame). Phase 1's 0.10 m/s target stays inside this range. |
| `yaw_rate_cmd` | −0.3 to +0.3 rad/s (**provisional training range**) | Net turn rate (positive = CCW from above). The achievable range is unknown and intentionally NOT pre-gated by scripted open-loop tests (Rev. 4): RL discovers it, and §10 evaluation measures and records it in `policy_io.md`. |

- **Deadband:** treat `|vx_cmd| < 0.02` m/s and `|yaw_rate_cmd| < 0.05` rad/s as zero. Derive `is_zero_cmd` and `is_rotate_only` indicators from these thresholds; use them to gate rewards and terminations (§6).
- The command (2 values, optionally scaled) enters **both** actor and critic observations.
- Sampling/resampling: §7.
- **Implementation anchor:** extend the existing Phase-2 stub. `crawl_env_cfg.CrawlCommandsCfg.base_velocity` (`UniformVelocityCommandCfg`) already carries `lin_vel_x=(0.05,0.15)`, `ang_vel_z=(-0.3,0.3)` and is already wired into `forward_velocity_exp`/`overspeed_penalty` via their `command_name` parameter when `use_velocity_commands=true`. Phase 2 replaces the stub flag with a real velocity+yaw-rate command term (a subclass of `CommandTerm`, reusing the `just_resampled` re-anchoring pattern of `UniformCrawlHeadingCommand`). Isaac Lab's stock `rel_standing_envs` option is the natural home for the sit fraction; use it or replicate its behavior deliberately.

---

## 4. Observation design

### 4.1 Actor (must be reproducible on real hardware)

| Group | Dim | Source on real robot |
|---|---|---|
| Projected gravity in base frame | 3 | IMU tilt estimate (roll/pitch only matter) |
| Base angular velocity in base frame | 3 | IMU gyro (QMI8658) |
| Joint positions relative to default | 6 | Servo encoders (STS3215) |
| Joint velocities | 6 | **Finite difference of encoder position** (servos report position only), filtered, as computed on board (recipe documented in `policy_io.md`; sim uses the sim's joint velocities until Phase 2b) |
| Previous action | 6 | Controller memory |
| **Command `(vx_cmd, yaw_rate_cmd)`** | 2 | Operator / supervisor |
| **Filtered heading rate (estimated, §5)** | 1 | Computed on board from the above |

New actor total: **27-D**. **Removed vs Phase 1:** the 2-D `heading_error_wrist` term (it requires world-frame wrist localization, and its rate-based replacement is the heading-rate term). Note this term is currently the *only* command pathway the actor has — removing it and adding the 2-D command is the replacement; do not ship an intermediate state with neither (§7.3 stage-0 check covers this).

**Phase 2a feeding (owner decision — start simple):** the sim computes every row above from ground-truth states (projected gravity from the true root orientation, root angular velocity as "gyro", true joint angles through FK/estimator) with **no sensor-noise model** (§8). Phase 1's existing per-term `UniformNoiseCfg` corruption stays untouched — that is the full extent of "noise" in Phase 2a. The right-hand column describes the *hardware* source that the Phase-2b pass and the deployment client must reproduce.

`vx` note: the real robot has no forward-velocity sensor; the policy never observes its own speed and produces commanded speed "blind" as a gait property (already true in Phase 1; intentional). Forward-speed tracking is judged only from sim ground truth.

### 4.2 Critic (privileged, simulation only)
Everything the actor sees, plus: base linear velocity (base frame), base height, ground friction, ground-truth filtered heading rate, ground-truth world heading error (optional), and the command-mode indicators.
**No fallback clause is needed:** the installed `rsl_rl` 5.5.1 already runs asymmetric actor-critic via `obs_groups` (policy group for the actor; policy+critic groups for the critic). Privileged terms must NOT be added to the `policy` group. The exact ordering/scaling is documented once in `policy_io.md` (§11 — created there, it does not exist yet).

### 4.3 Optional: gait-phase clock
The existing `mdp.gait_phase_clock` (2-D sin/cos, 0.5 Hz) is **off by default; keep it off** for the Phase 2 baseline so the from-scratch comparison isolates the command change. If it is later enabled and `is_zero_cmd`, freeze its phase so the policy receives no rhythmic input while sitting (term-level state keyed by the command indicator; `gait_phase_clock` currently advances unconditionally).

---

## 5. Heading-rate signal (must work on the real robot)

### 5.1 Definition
"Heading" is the horizontal-plane direction of the vector from the base origin to the wrist origin (same arrow as `get_claw_dir` / `wrist_pointing`). We need its **rate of change**, never its absolute world angle.

### 5.2 Step-by-step computation (exact — no level-base approximation)
Inputs: `ω_b` gyro (base frame), `q_imu` the tilt-filter quaternion (gravity-aligned; its yaw origin is arbitrary and slowly drifting, which is irrelevant to a rate), `q` joint angles, `dt`, filter state.

Rev. 1 decomposed the rate as `r_base = ω_b·u` plus `dψ/dt` with `ψ` measured against the *projected base x-axis*. That is exact only for a level base or pure rotation about vertical; with simultaneous tilt and pitch/roll rate (which the crawl gait has) it carries a cross-term error `−(ω·x̂)(x̂·u)/(1−(x̂·u)²)` on the order of the 0.01–0.05 rad/s band this signal must resolve, and the §9 "tilted base" unit test would fail against the spec's own formula. Use one of the two exact formulations:

**Primary (recommended): world-azimuth differencing.**
1. `p_b = FK(q)` — wrist origin relative to base origin, base frame (URDF kinematics).
2. `p_w = R(q_imu) · p_b` (world frame; arbitrary constant yaw offset is harmless).
3. `ψ_w = atan2(p_w.y − base_w.y, p_w.x − base_w.x)` of the horizontal projection.
4. `r_inst = wrapped finite difference of ψ_w / dt`, low-pass filtered.
5. `r_filt = EMA(r_inst)` (§5.7 constants).
On hardware, the measured Madgwick yaw drift (~0.1°/s ≈ 0.0017 rad/s, `imu/README`) will enter this signal as a small constant rate bias; Phase 2a does **not** model it in training (§8) — it is handled on the robot by the existing firmware startup bias calibration, and gets trained against in the Phase-2b noise pass (§14).

**Alternative A (exact body-frame form, avoids explicit rotation):** with `u = −g_b/‖g_b‖`, `x̂` the base x-axis, `p_h = p_b − (p_b·u)u`, and `ψ = atan2(p_h·e2, p_h·e1)` where `e1 = normalize(x̂ − (x̂·u)u)`, `e2 = u × e1`, then
`r_inst = [(ω·u) − (ω·x̂)(x̂·u)] / (1 − (x̂·u)²) + dψ/dt` (wrapped, filtered).
**Alternative B:** analytic `dψ/dt` from joint velocities and the kinematic Jacobian (avoids differentiating encoder noise). Optional refinement.

### 5.3 Degenerate cases and fallbacks
- **Short vector** (folded/sitting arm): when `‖p_h‖ < ε` (a few cm), ψ is unstable. Blend smoothly to the plain gyro yaw rate: `r = α·r_inst + (1−α)·r_base`, `α: 0→1` as `‖p_h‖` goes `ε→2ε`. (Mirror the 0.01 m validity threshold and base-forward fallback already used in `mdp.get_claw_dir` so sim reward and estimator agree on when the arrow "exists".)
- **Base on its side / upright** (`x̂` nearly parallel to `u`): fall back to `r_base` only and log the event.
- Keep the config switch `heading_rate_mode: "arm_vector" | "base_gyro"` (§5.6) so both can be compared.

### 5.4 One implementation for sim and real
Write the estimator once, as a pure-function module at
`source/zombie_crawl/zombie_crawl/zombie_crawl/estimators/heading_rate_estimator.py`
(inside the package so both the sim task and a future hardware client import the same code), accepting PyTorch tensors (batched, in-sim) and NumPy arrays (on hardware), with the signature `(gyro, gravity_estimate_or_quat, joint_positions, dt, filter_state) -> (rate, filter_state)`. **In Phase 2a the sim feeds it ground-truth quantities** (projected gravity, root angular velocity, true joint angles) — no sensor model (§8, owner decision). On hardware the same function is fed by the existing `imu/` stream (the ESP32 does all IMU processing; the host client adds wrist FK from the servo-bus joint positions — the IMU knows nothing about joints, which is why this computation lives in a host-side shared module, not on the board). The one hard parity rule: **the heading-rate filter/EMA constants (`τ_filter`, §5.7) must be identical in sim and on hardware** — filter *lag* changes the closed loop itself even when noise is not modeled; the hardware client reads the same config file.

### 5.5 What the reward uses
- **Actor observation:** the filtered rate from the shared estimator, fed by ground-truth sim states in Phase 2a (§5.4, §8).
- **Reward:** the *ground-truth* net heading rate over a sliding window `T_w` (unwrapped true heading now minus `T_w` ago, ÷`T_w`; §5.7), computed from true sim poses. In Phase 2a the actor's input differs from the reward's target only by the estimator's filtering (no noise), so the training objective and the observation stay in agreement; the Phase-2b pass is what adds the input noise the policy must then cope with.
- **Ring-buffer re-anchoring (required):** the repo already learned that stateful terms must re-anchor when the command changes (see `just_resampled` in `mdp/commands.py`, consumed by `forward_progress_lpf` and `stalled`). The heading-rate ring buffer must do the same: on command resample and on episode reset, refill the buffer with the current heading (zero net rate) — do not mix old/new command phases in one window. Combined with the 2 s settle grace (§6.2) this removes the Rev. 1 ambiguity of "leave the window running".

### 5.6 Simplified fallback
If the estimator is fragile, use only `r_base` (plain gyro yaw rate) filtered over a gait cycle — config switch above. Simpler, and behaves similarly over a full cycle for rotate-in-place and gentle steering.

### 5.7 Gait period and filter constants (measured offline, not detected online)

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
| `T_w` (reward window) | `2 × T_gait` | Sliding-window net heading rate (ground truth) for the reward. Ring buffer, re-anchored per §5.5. |
| `τ_filter` (observation EMA) | `1.0–1.5 × T_gait` | Filter on the estimated heading rate the actor sees and that runs on hardware. |

Both overridable config values.

**If there is no clear period:** use fixed defaults (`T_w = 3 s`, `τ_filter = 1.5 s`) and log it. A fixed window still averages the sway; the failure mode is a gait slower than the window (oscillation leaks) → lengthen.

**Trade-offs and the settle-time coupling.** Longer window = less sway, more lag. §10 demands ≤3 s transition settling, but the defaults above can put the observation lag at 1.5–4.5 s — the acceptance criterion may be unreachable by construction. Resolve explicitly: (a) tune `τ_filter` first (the EMA is the dominant lag), (b) if settling still exceeds 3 s with the rotate command, prefer relaxing the settle target to `max(3 s, 2·τ_filter)` and record the deviation, rather than shortening the window below `1×T_gait` (sway leak is worse than slow settling — the sway corrupts the reward everywhere, the lag only at transitions). Re-check at final-policy evaluation either way.

---

## 6. Behavior definitions and reward design

### 6.1 Rest pose ("sitting")
- Define a **rest pose**: base link on the ground, arm folded close to the body, low CoM, stable at rest with the servos holding position. **Folded (URDF zero) is the default**: it is the action-zero pose of `OffsetEMAJointPositionToLimitsAction` (the policy reaches it with a zero action) and holds still under the servo preset (measured: base speed ≈3e-4 m/s, zero drift, base rests at z≈0.033 m). A lower candidate (`shoulder_lift 0.3, wrist_flex 0.6, gripper 0.5`) also holds dead-stable at z≈0; naive mid-joint tucks collapse instead of sitting. Record the chosen winner in `docs/rest_pose.md` (joint angles, resting contacts).
- The candidates were verified once under the **`servo` preset** (not `ideal`); no settle-test script is kept in the repo (Rev. 4 — the hold is checked through training behavior and §10 sit acceptance instead).

### 6.2 Reward terms by command mode
Existing Phase 1 terms are kept with the changes below. All terms are logged separately. **Naming note:** the repo has no term called "heading alignment" or "world-frame yaw penalty". The directional set is: `wrist_pointing` (−1.5; claw-arrow vs commanded heading — the anti-crawl-backward exploit guard, documented as dominating the forward term), `lateral_velocity` (−1.0), `overspeed` (−2.0), all on the **claw axis**; plus `forward_velocity_exp` (+2.5) and `forward_progress_lpf` (+1.0) on the **command axis**.

| Term | Active when | Form | Sign |
|---|---|---|---|
| `forward_velocity_exp` (retargeted) | always (target = `vx_cmd`) | `exp(−(vx − vx_cmd)² / σ²)` in the claw/heading frame | + |
| Yaw-rate tracking (new) | always (target = `yaw_rate_cmd`) | `exp(−(r_filt_true − yaw_rate_cmd)² / σ_r²)` | + |
| `forward_progress_lpf` (retargeted) | `vx_cmd` above deadband only | command-axis displacement ÷ target | + |
| Stillness: base speed | `is_zero_cmd` | `‖v_base‖²` penalty (after settle grace) | − |
| Stillness: joint velocity | `is_zero_cmd` | `‖q̇‖²` penalty | − |
| Rest-pose attraction | `is_zero_cmd` **and base upright** | `exp(−‖q − q_rest‖² / σ_q²)` | + |
| In-place rotation: no translation | `is_rotate_only` | penalty on `‖v_base,xy‖` | − |
| Overspeed | always | penalty above `2 × max(vx_cmd, small)` and above a yaw-rate cap | − |
| `vertical_velocity`, `action_rate`, `joint_acc`, `joint_torques`, `orientation`, `recovery` | always | unchanged from Phase 1 | per Phase 1 |

Rules:
- **Keep `wrist_pointing`, `lateral_velocity`, `overspeed`** (claw-axis). They are Phase 1's anti-exploit levers; sit/rest rewards now make stationary poses attractive, so removing the claw-leading guard exactly then invites the backward/drag exploit it was built to kill. What Phase 2 *does* remove is scoring progress against a **frozen world-frame command axis** in the tracking terms — those become `vx_cmd`/`yaw_rate_cmd` targets. (Rev. 1's blanket "remove heading-alignment" sentence, which named non-existent terms and threatened `wrist_pointing`, is deleted.)
- **Recovery precedence:** when `is_zero_cmd` and the base is tipped, `upright_bonus` stays active and rest-pose attraction is gated off (via the upright indicator) until the base is level. A fallen robot under a zero command must get up first, then sit.
- **Settling grace period:** scale stillness penalties down for ~2 s after the command changes to zero (unavoidable deceleration).
- **Gate smoothly** (short ramp ~0.5 s) to avoid reward discontinuities.
- **Push interaction (sit):** the training disturbance (`disturb_all_bodies`, ±0.4 N per body, curriculum-gated by `push_scale`) physically cannot be resisted while sitting. Decision (owner, §15): (a) recommended — keep pushes on for training robustness and accept the noise, measuring sit metrics both with pushes on and off (§10); or (b) gate `push_robot` off during `is_zero_cmd` windows. Either way, document the choice; do not leave it implicit.
- **Forward + turn:** both tracking terms active at once; forward is measured in the claw/heading frame so a curved path still counts. No world-frame lateral penalty in this mode (a curve produces lateral world displacement by definition); keep the heading-frame `lateral_velocity` penalty.
- **Stateful re-anchoring:** `forward_progress_lpf`, the heading-rate ring buffer, and any command-windowed EMA must re-anchor on the new command term's `just_resampled` flag (same pattern as `mdp/commands.py`).

### 6.3 Terminations and resets
**Explicit handling of every existing term** (Rev. 1 listed only three and omitted the two that conflict with sitting):

| Term | Phase 2 handling |
|---|---|
| `time_out` (30 s, §7.1) | truncation, unchanged |
| `base_nan`, `base_too_high` (0.45 m) | unchanged |
| `out_of_bounds` (2.5/1.5 m) | keep as terminal-success, but note it can now only fire in forward modes (sit/rotate stay <0.15 m). The curriculum escape gates built on it stay valid only while ≥30% of episodes are forward modes (§7.1 mix). |
| **`stalled`** (0.05 m / 12 s) | **must be command-gated:** active only when `vx_cmd` is above the deadband; re-anchor its window on command change (it already re-anchors on heading resample — extend that hook to the new command term). Without this, every sitting episode is failure-terminated at 12 s, i.e. Phase 2 punishes its own target behavior. Add a regression test: a sit-policy rollout with zero commands must produce zero `stalled` terms. |

- **Resets:** keep Phase 1 start poses **and weights** (folded 0.5 / on_side 0.25 / face_down 0.25) so recovery keeps exercising, **plus the rest pose** as a fifth entry with its own weight (start ~10–15% of episodes in the rest pose). Spawn yaw stays ±180° (task is rotationally symmetric; there is no world +X reference anymore — the command supplies direction). Some fraction of episodes should start rest-pose + zero command so "sit and stay" is directly represented; command resampling inside the episode covers the rest of the transition space.

---

## 7. Command sampling and curriculum

### 7.1 Sampling inside an episode
- Episodes lengthen from 20 s to **30 s**; command **resampled every 5–8 s** (random, per env, staggered — reuse the CommandManager `time_left` pattern from `UniformCrawlHeadingCommand`).
- Mode mix at full difficulty (per resample): **stop 20 %, forward 30 %, rotate-only 25 %, forward + turn 25 %**. Forward + turn is a required behavior (owner-confirmed). Commands are sampled inside the provisional §3 ranges (no pre-training envelope, Rev. 4). Isaac Lab's `rel_standing_envs` is the reference implementation for the zero fraction; per-env mode is drawn per resample, not fixed per episode.
- Within a mode, sample magnitudes uniformly in range, plus small command noise near zero to exercise the deadband (the vx_cmd deadband of 0.02 m/s is meaningful only if commands actually land below it).
- On every resample: flip `just_resampled`; all stateful readers re-anchor (§5.5, §6.2).

### 7.2 Curriculum stages
Progress on mean episode reward + the stage's acceptance metric, or after a fixed iteration budget. These Phase-2 stages are **separate from** (and sit on top of) the existing Phase-1 difficulty curriculum (v_target, friction_scale, difficulty/fallen/pushes, heading gate at 0.75) — keep that machinery, add the command stages around it.

| Stage | Command mix | Goal |
|---|---|---|
| 0 | Forward only (constant `vx_cmd=0.10`, obs layout final incl. command + heading-rate inputs) | Checkpoint-surgery sanity: the converted policy still crawls (regression gate vs Phase-1 baseline) |
| 1 | Forward + stop | Sit and resume |
| 2 | Add rotate-only | In-place rotation |
| 3 | Add forward + turn | Curved crawling |
| 4 | Full mix | Transitions between all modes |
| 5 | Full mix with domain randomization (physics only: friction/pushes/fallen; sensor noise deferred to Phase 2b, §8) | Robustness |

### 7.3 Fine-tuning from the Phase 1 checkpoint
- **Weight surgery** (`scripts/extend_checkpoint.py`, new) for the rsl_rl 5.5.1 checkpoint layout (`model_state_dict` + `obs_norm_state_dict`):
  - Remove the 2 columns corresponding to `heading_error_wrist` and append `2+1` columns (command ×2, heading rate ×1) to **both** the actor's and the critic's first `nn.Linear` (the removed term is in the shared `policy` group, which feeds both nets via `obs_groups`). New columns **zero-initialized**; all other weights kept.
  - Same column surgery on the EmpiricalNormalization mean/var/count for actor and critic separately (new dims: mean 0, var 1).
  - Reset the Gaussian distribution's `std` parameter to `init_std` = 0.6 (the converged tiny std would kill exploration for sit/rotate). Reset optimizer state for the touched layers (or all of it); do not resume the iteration counter.
  - Unit test: run the converted checkpoint for Stage 0 with `vx_cmd` constant and assert crawl metrics match the Phase-1 baseline (no first-layer behavior change when new columns are zero and removed columns carry the heading-err input zeroed — verify numerically against the original net).
- **Guard against forgetting:** forward commands in every stage; track crawl metrics each evaluation; abort and report if forward accuracy degrades >30 % from the Phase-1 baseline (§1.0 numbers).
- **Baseline:** one from-scratch run with the full curriculum; report which produced the better final policy.

---

## 8. Sensor and noise model — **deferred to Phase 2b** (owner decision: start simple)

**Phase 2a simulates no IMU.** The actor's heading-rate input is computed from ground-truth sim states through the shared estimator (§5.4); nothing degrades the gyro, gravity, or joint-angle inputs beyond Phase 1's existing per-term `UniformNoiseCfg`. No simulated accelerometer, no fusion filter in sim, no per-episode bias, no encoder quantization, no datasheet work. All IMU processing (Madgwick, boot gyro-bias calibration, QST COD) stays where it already lives and works — the ESP32 firmware in `imu/`.

**Why this is defensible now:** the Phase-2a deliverables are the command interface, the yaw-rate signal *definition and filter*, and the sit/rotate behaviors — none of which require noise to be discovered. The hardware pipeline already exists and is measured (roll/pitch σ ≈ 0.02°, yaw drift ~0.1°/s at rest, `imu/README`), so Phase 2a can hand the robot a signal computed by the same code path.

**The hooks that keep Phase 2b cheap (build these in Phase 2a):**
1. The estimator is a pure shared function (§5.4) — Phase 2b changes only what *feeds* it, not the module.
2. `τ_filter`/`T_w` come from one config read by both sim and the hardware client (`configs/gait_period.yaml`) — lag parity is maintained from day one (§5.4).
3. Reward already uses the ground-truth window (§5.5), so adding input noise later does not move the objective.

**Deferred checklist — the Phase-2b training pass (triggered when hardware bring-up or `imu/` recordings show drift/noise mattering, §14):**
- Per-episode random **gyro bias** (±) — implemented as a `mode="startup"`/reset EventTerm (per-step `UniformNoiseCfg` cannot express persistence; see `randomize_ground_friction` pattern), not term noise.
- Gyro white noise (scaled up while the arm moves, mimicking servo vibration).
- Per-episode **tilt offset** (start ±2–3°) on the gravity vector.
- **Encoder quantization** (STS3215-class resolution, verify gearing vs datasheet) + noise **before** the FK that produces the heading vector, and joint velocities re-derived from the quantized positions (the on-board recipe).
- Optional one-control-step latency flag.
- Then: re-fine-tune briefly and re-run the deferred §10 rows (sit under bias, noise robustness).

**Explicitly accepted risks of the simple start:** Phase 2a never trains against "a sitting robot whose biased gyro *believes* it is rotating", and never tests whether rotate/sit survive quantized encoders. The pushless sit acceptance (§10) is therefore optimistic for hardware; expect the first real-robot bring-up to expose exactly these, which is the trigger for Phase 2b.

Existing Phase-1 domain randomization (friction, pushes, fallen starts) stays — it is physics, not sensing. Mass and PD-gain randomization remain optional, stage 5 if the owner wants a robustness pass.

---

## 9. Estimator unit tests (`pytest`, pure NumPy — no sim needed)

**There are no pre-training physics feasibility gates** (owner decision, Rev. 4: "if I could script it, I wouldn't need reinforcement learning"). The open-loop scripted tests and their PASS/FAIL harnesses were deleted from the repo (`scripts/scripted_crawl.py`, `scripts/rest_pose_test.py`). What RL can achieve is discovered by RL; the reachable command range is measured at §10 evaluation and recorded in `policy_io.md`. What remains here is ordinary code unit testing of the estimator module — these verify the §5.2 math, not what the robot can do:
- **Pure base rotation** about vertical at a known rate, joints fixed → output matches truth within noise — **repeated with the base rolled/pitched** (this is the test Rev. 1's formula would fail; §5.2).
- **Pure arm sweep:** base fixed, shoulder pan at known rate → matches the geometrically expected rate.
- **Tilted base** for both cases; results must not depend on tilt.
- **Short-vector/degenerate:** fold the arm → smooth blend to gyro rate, no spikes; on-side base → falls back with log flag.
- **Gait-window filter:** synthetic oscillation + small mean → filtered output recovers the mean within tolerance; ring-buffer re-anchor on command change emits zero rate, not a mix.

---

## 10. Evaluation and acceptance criteria

### 10.0 Prerequisite
Fix `scripts/evaluate.py`'s known bucketing bug (`out_of_bounds` counted as early termination → dropped episodes/NaN displacement; noted in README §5). Rework eval axes: Phase 1 projects displacement on the frozen `zombie_ref_yaw` reference; Phase 2 metrics project on the **command term** (forward) and on the heading-rate definition (§5) — same axis the reward uses.

`scripts/evaluate.py` extended with per-mode scenarios, 100 envs each. Phase 2a runs with **no sensor-noise model** (§8); friction randomized (report 0.5 / 0.8 / 1.2 separately — matches the training range 0.5–1.2).

| Scenario | Acceptance (initial thresholds, adjust with owner) |
|---|---|
| **Crawl** (`vx_cmd=0.10`, `yaw_rate_cmd=0`) | Mean forward speed within ±30 % of command; heading drift small; no early terminations — **and, because the Phase-1 baseline is modest (§1.0), state the Phase-1 number this was compared against in the report** |
| **Sit** (zero command 20 s, from crawl and from rest starts; run with pushes ON and OFF) | After 3 s settle: mean base speed <0.01 m/s (pushes off; report pushes-on separately), joint velocities near zero, joints within 0.15 rad of rest pose, no creeping; **zero `stalled` terminations** (§6.3) |
| **Rotate in place** (`vx_cmd=0`, `yaw_rate_cmd=±0.2` provisional) | Filtered heading rate within ±30 % of command; base translation <0.15 m per 10 s. Record the maximum tracked rate; if clearly below the provisional range, adjust range/thresholds with the owner |
| **Forward + turn** (`vx_cmd=0.10`, `yaw_rate_cmd=±0.1`, plus other points in the provisional range) | Each of forward speed / filtered heading rate within ±35 % (looser than single-mode — the two interact); no early terminations |
| **Transitions** (crawl → sit → rotate → crawl+turn → sit → crawl) | Each transition settles within `max(3 s, 2·τ_filter)` (§5.7 coupling); no flips beyond Phase-1 norms; no "stuck" states |
| **Regression** | Crawl metrics no more than 10–15 % worse than the Phase-1 policy measured under the same eval (§1.0) |

**Deferred to Phase 2b with §8:** the "noise robustness" row (sit/rotate thresholds under full sensor noise + per-episode gyro bias) — meaningless without a sensor model; re-add when Phase 2b runs.

Also report: settle time per command change; unintended-motion (creep) distance per 20 s while sitting; success rate per friction.

---

## 11. Deliverables

1. **Code** (existing extension project, `source/zombie_crawl/zombie_crawl/…`):
   - Task config: new command term (velocity + yaw-rate, `just_resampled` re-anchoring), mode-gated rewards, Phase-2 curriculum stages, command-conditioned observations; `stalled` command-gating (§6.3). (Sensor-noise/reset-event terms are Phase 2b, §8 — not delivered now.)
   - `estimators/heading_rate_estimator.py` (shared, tensor + NumPy) with unit tests.
2. **Scripts:** `measure_gait_period.py` (new) → `configs/gait_period.yaml` + `docs/gait_period_report.md` (all created; `configs/` does not exist yet); `extend_checkpoint.py` (new); `train.py` (curriculum-stage + checkpoint-init flags added to the existing wrapper); `play.py` (add command input from keyboard/gamepad/CLI to the existing wrapper); `evaluate.py` (bug fix + per-mode scenarios). (The open-loop feasibility scripts are removed — Rev. 4.)
3. **Trained artifacts:** best checkpoint; exported **ONNX and TorchScript** actor — note `checkpoints/exported/{policy.pt,policy.onnx}` already exists for Phase 1 and the export pipeline is in place, so this is "re-export", not build; from-scratch baseline run; TensorBoard logs (`logs/rsl_rl/`).
4. **Videos:** crawl, sit, rotate-in-place, combined transitions (reuse the existing `train.py --video` / `play.py --video` recording pipeline).
5. **Docs:** **create** `policy_io.md` (Rev. 1 referenced it as existing — it does not): exact observation ordering/scaling, command semantics, estimator pseudocode, filter constants (the sim↔hardware parity contract, §5.4), and the reachable `(vx, yaw_rate)` range measured at final evaluation (§10 — replaces the removed pre-training envelope); `docs/rest_pose.md`; `docs/gait_period_report.md`; and a Phase-2 section — appended to **`report.md`** or a new `docs/phase2_report.md` (Rev. 1's `docs/training_report.md` does not exist; the real Phase-1 training docs are `README.md` §5, `report.md`, `docs/training_failure_analysis.md`). (`docs/imu_model.md` is deferred with the sensor model to Phase 2b, §8.)
6. **Tests (`pytest`):** estimator tests (§9), env smoke test with the new 27-D actor obs (extend `tests/test_smoke.py`), command-sampling statistics, short training run writing a checkpoint, play-script smoke test, and the sit/`stalled` regression test (§6.3).

---

## 12. Implementation order

1. Measure Phase-1 gait period → filter constants (§5.7); build the estimator module + unit tests, ground-truth-fed, shared EMA constants (§5, §9). **No sensor-noise layer** (deferred, §8).
2. Environment changes: command term, 27-D observations, mode-gated rewards, `stalled` command-gating, rest-pose reset entry (default = folded, §6.1), 30 s episodes.
3. `extend_checkpoint.py`; verify Stage 0 numerically vs the original net, then confirm the converted policy still crawls.
4. Curriculum stages 1–4 with intermediate evaluation; from-scratch baseline in parallel if GPU time allows.
5. Stage 5 robustness (physics DR only, §7.2); final evaluation vs §10 (with the `evaluate.py` prerequisite fixed); record the measured reachable command range in `policy_io.md`.
6. Export (ONNX/TorchScript re-export), videos, docs (`policy_io.md` et al.), tests.

Report after each step. Stop and ask the owner if: fine-tuning loses the crawl, the sit behavior cannot emerge at all through training, or the pushless sit cannot hold the §10 threshold (creep/rotations it physically could not prevent even before noise enters the picture).

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Arm cannot rotate in place with the weak servos (2.9 N·m preset) | Discovered by RL itself (no scripted pretest, Rev. 4): §10 rotate/turn rows measure the achievable rate; if unreachable, lower the command range/thresholds with the owner |
| Policy forgets crawling | Forward commands in every stage, regression metric vs the *real* Phase-1 baseline (§1.0), from-scratch comparison |
| Policy creeps while sitting | Stillness + rest-pose terms, deadband (bias robustness deliberately deferred, §8) |
| Gyro bias makes the **hardware** robot think it is turning | **Untrained in Phase 2a by design (§8)**; mitigated on hardware only by the existing firmware boot bias calibration (`imu/`); Phase-2b noise pass trains it |
| Estimator fragile when arm folded | Short-vector blending, `base_gyro` fallback mode, mirrored threshold of `get_claw_dir` |
| **Removing `wrist_pointing` re-opens the backward-crawl exploit** | Terms explicitly kept (§6.2); video review, not just curves |
| **`stalled` kills every sitting episode** | Command-gated termination + regression test (§6.3) |
| **Fallen starts vs sit ambiguity** | Recovery precedence rule (§6.2); rest-pose attraction gated upright |
| Sit vs push disturbance conflict | **Resolved (owner, Rev. 3):** pushes stay on; sit graded with pushes (loose) and pushless (strict) (§10) |
| Tilt-dependent heading-rate error | Exact formulas only (§5.2); tilted-base unit tests |
| Policy leans on clean sim signals (noise/quantization untrained) | Deliberate simple start (§8); hooks keep Phase 2b cheap; Phase-2b pass required **before** trusting hardware sit/rotate |
| No clear gait period / period shifts after Phase 2 | Fixed defaults (3 s / 1.5 s); re-run `measure_gait_period.py` on the final policy, update constants + brief fine-tune if `T_gait` moved >~25 % |
| Filter lag vs 3 s transition settling | Joint tuning rule in §5.7; relaxed criterion stated, not silently missed |
| Reward discontinuities at mode changes | Smooth ramps, settle grace, re-anchored stateful terms |
| Physics exploits (hopping, jitter) | Keep vertical/action-rate/overspeed penalties; video as well as curves |
| Sim/real divergence | Single estimator module + identical filter constants (§5.4); the simple start (no noise trained, §8) is deliberate and is itself the entry criteria for Phase 2b |
| **Doc/code drift (soft-limit factor 0.95 vs 1.0, "27-D" docstring)** | Reconcile during Phase 2 (§1.0) |

---

## 14. Future extensions (documented, not implemented)

- **Phase 2b — sensor-noise training pass (first future item, §8):** implement the deferred checklist (per-episode gyro bias events, gyro noise, tilt offset, encoder quantization → FK, derived joint velocities, optional latency), briefly re-fine-tune the Phase-2a policy, and re-run the deferred §10 rows. **Trigger:** real-robot bring-up or `imu/` recordings show drift/noise affecting sit/rotate (e.g., the robot creeps or "believes" it is rotating while sitting).
- **"Face this direction" supervisor:** external heading source (overhead camera, AprilTag, mocap) → yaw-rate commands to this policy; heading problem lives outside the policy. Cheaper relative-turn variant: integrate the filtered heading rate for a few seconds, accepting drift (~0.1°/s measured).
- Full domain randomization and latency modelling for the real arm.
- Deployment on the real SO-ARM100/SO-101: Feetech servo interface + the existing `imu/` QMI8658 board (raw stream at 100 Hz already supports 2× the 50 Hz control rate).
- Zombie-style reward shaping (asymmetric drag/recover, lower posture).

---

## 15. Owner decisions and remaining open items

### Resolved
1. **Forward + turn is required** — full mode (25 %), own curriculum stage (3), acceptance row (§10).
2. **Max turn rate unknown** — the 0.3 rad/s provisional range IS the training range; no scripted pretest (owner, Rev. 4). The reachable range is discovered by training, measured at §10, and recorded in `policy_io.md`.
3. **IMU is the QST QMI8658** (6-axis); **all IMU processing stays on the ESP32** (`imu/`, already built with boot gyro-bias calibration and measured noise). The simulator models **no** IMU in Phase 2a (§8).
4. **Sitting comes from a zero command only** (no sit flag); supervisor sends in-deadband commands. Robustness of "sit" to gyro bias/noise is a **Phase-2b** deliverable (§8, §14), not a Phase-2a acceptance gate.
5. **`wrist_pointing`/`lateral_velocity`/`overspeed` stay; `stalled` gets command-gated** (Rev. 1 ambiguity resolved in favor of keeping Phase-1 exploit guards).
6. **Phase 2a starts simple (owner, Rev. 3):** no sensor/IMU modeling in sim; the yaw-rate signal and sit are trained from ground-truth states through the shared estimator; noise/drift work deferred to Phase 2b (§8, §14).
7. **Sit-vs-push resolved (owner, Rev. 3):** pushes stay on during sit training; sit is graded twice — pushless (strict <0.01 m/s) and with pushes (loose "stays roughly put") (§6.2, §10).

### Remaining open (agent tasks unless marked owner)
- IMU mounting on the arm (assumed: base link) — a hardware-side detail; record the mount pose when Phase 2b/deployment arrives (§8).
- Operator command input for the real robot later (gamepad, keyboard, supervisor program).
- Reconcile README/code mismatches (soft limit 0.95 vs 1.0; actor dim docstring) (§1.0).

---

## 16. Changelog

### Vs Rev. 1

1. **Phase-1 characterization corrected** (§1.0): commanded random heading (not "+X"), real baseline metrics used for regression gates, known `evaluate.py` OOB-bucketing bug made a prerequisite (§10.0).
2. **Terminations made explicit** (§6.3): `stalled` command-gated (Rev. 1 omitted it; it would have failure-terminated every sitting episode), `out_of_bounds` handling, rest-pose added to the *existing* pose-weight scheme, fallen-start/recovery precedence rule added.
3. **Reward removals disambiguated** (§6.2): Rev. 1's "remove heading-alignment and world-frame-yaw penalties" named non-existent terms and could be read as deleting `wrist_pointing`, Phase 1's anti-backward-crawl exploit guard. Rev. 2 names the real terms and keeps the claw-axis guard set.
4. **§5.2 heading-rate math fixed:** Rev. 1's `ω·u + dψ/dt` against the projected base x-axis is inexact under tilt + pitch/roll rate (error term ~0.01–0.05 rad/s); Rev. 2 gives the exact body-frame formula and the recommended world-azimuth formulation, and the §9 unit tests pin it.
5. **Ring-buffer / stateful-term re-anchoring required** on command change (repo's `just_resampled` pattern), replacing Rev. 1's open question.
6. **Existing assets adopted:** `use_velocity_commands` stub + `UniformVelocityCommandCfg` ranges; `UniformCrawlHeadingCommand` resampling pattern; `rel_standing_envs`; `scripted_crawl.py` extended instead of a new `scripted_rotate.py`; `checkpoints/exported/` re-export; `imu/` firmware (Madgwick, boot bias calibration, measured noise, 100 Hz stream, 1024 DPS) as the hardware baseline with a stated filter-placement decision.
7. **§4.2 asymmetric-AC fallback clause deleted** (rsl_rl 5.5.1 already configured with `obs_groups`); privileged terms must stay out of the actor group.
8. **§7.3 weight surgery made concrete** for rsl_rl 5.5.1: both actor and critic first layers (shared policy group), obs-norm statistics, `std` reset to `init_std`, numerical Stage-0 equivalence test.
9. **Per-episode sensor biases moved from "noise layer" to reset event terms** (the repo's `UniformNoiseCfg` is per-step).
10. **Gait clock:** off by default in the Phase-2 baseline; if enabled, 2-D and phase-frozen under zero command.
11. **Filter lag vs settle-time coupling made explicit** (§5.7/§10) with a stated priority rule.
12. **Deliverable paths fixed:** `policy_io.md`, `configs/`, `docs/*_report.md` created (Rev. 1 referenced `policy_io.md` and `docs/training_report.md` as existing); estimator placed inside the `zombie_crawl` package.
13. New risks added (sit-vs-push, fallen-vs-sit, `stalled`, doc/code drift).

### Vs Rev. 2 (Rev. 3 scope reduction — owner: "start simple")
1. **No sensor/IMU modeling in Phase 2a.** §8 shrank from a full sensor-noise spec to a deferred checklist. Deleted from Phase 2a: simulated accelerometer + fusion filter in sim, per-episode gyro-bias/tilt-offset event terms, encoder quantization, joint-velocity differencing, latency flag, QMI8658 datasheet task, and `docs/imu_model.md` (all → Phase 2b, §8/§14).
2. **Actor heading-rate + joints/gyro/gravity inputs are ground-truth-fed** (plus Phase 1's existing per-term `UniformNoiseCfg`); the §5.4 "sim must feed sensor-model outputs, never ground truth" rule is withdrawn.
3. **The "tilt-filter placement" owner question is moot and deleted:** all IMU processing stays on the ESP32 (`imu/` firmware, already built); the heading-rate computation is host-side on hardware only because it needs wrist FK from joint angles.
4. **One parity rule kept:** `τ_filter`/`T_w` identical in sim and hardware client via one config file (§5.4) — lag parity matters even without noise.
5. **Kept hooks for cheap Phase 2b:** pure shared estimator function, config-driven filter constants, ground-truth reward window unchanged by the later noise pass (§8).
6. **§10:** "Noise robustness" row and bias-related reporting deferred to Phase 2b; pushless-sit optimism named as an accepted risk.
7. **Curriculum stage 5** is physics-DR-only (§7.2); §12 steps 3/7 updated to drop sensor-noise work and the filter-placement decision.
8. **Sit-vs-push locked (owner):** pushes stay on; sit graded with and without pushes (§6.2, §10, §15).

### Vs Rev. 3 (Rev. 4 — owner directive: no scripted feasibility tests)
1. **Pre-training feasibility gates deleted (former §9.1/§9.2/§9.2a).** The open-loop scripted crawl/rotate/forward-turn harnesses and their PASS/FAIL thresholds were removed from the repo (`scripts/scripted_crawl.py`, `scripts/rest_pose_test.py`) — scripted open-loop tests answer the wrong question, and Phase 1's own history (a gate skipped, then run and mis-scaled to bang-bang) shows they add noise, not confidence.
2. **Command envelope → post-training measurement.** `configs/command_envelope.yaml` is no longer a deliverable; the reachable `(vx, yaw_rate)` range is measured at §10 evaluation and recorded in `policy_io.md`. Training samples inside the provisional §3 ranges.
3. **Rest-pose candidates carry their measured results** (§6.1: folded and the low tuck hold still under the servo preset; mid-joint tucks collapse) instead of a kept test script.
4. **Estimator unit tests stay** (§9) — they are pytest code tests of the §5.2 math, not physics gates.
5. §12 implementation order starts directly with gait-period measurement + estimator; stop-and-report clauses updated (no scripted-rotation gate).
