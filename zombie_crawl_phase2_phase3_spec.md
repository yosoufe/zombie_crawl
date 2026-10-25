# Specification: Zombie Crawl — Phase 2 (yaw-rate command) and Phase 3 (maneuvers)

**Audience:** an autonomous coding agent. Requirements use MUST / SHOULD / MUST NOT. Every requirement has an ID so progress and failures can be reported against it.
**Project:** free-floating SO-ARM100 / SO-101 learns to crawl in Isaac Sim + Isaac Lab 3.0 (`rsl_rl` 5.5.1 PPO).

---

## 0. Baseline and ground rules (read first, applies to both phases)

### 0.1 Starting point

- **Baseline code: commit `c450601e1955b1edde2e3b19acfb823ae9d2a8ff` of `github.com/yosoufe/zombie_crawl`. Implement on top of exactly this commit. MUST NOT use or copy anything from later commits** (they contain unfinished Phase 2 code).
- Phase 1 works: from scratch, its curriculum dials saturate by roughly iteration 400 (4096 envs) and the policy crawls toward a commanded heading and gets back up after falling.
- Runtime: Python env `../.venv` (Python 3.12), `isaacsim 6.1.0.0`, Isaac Lab 3.x, `rsl-rl-lib 5.5.1`, `torch 2.11.0+cu130`, RTX 5090 laptop. Every sim command uses that interpreter.

Files you will touch (all under `source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/` unless stated):
`crawl_env_cfg.py`, `mdp/commands.py`, `mdp/observations.py`, `mdp/rewards.py`, `mdp/terminations.py`, `mdp/curriculums.py`, `mdp/events.py`, `video_overlay.py`, plus `scripts/play.py`, `scripts/train.py`, the evaluation script, and `tests/test_smoke.py`. `agents/rsl_rl_ppo_cfg.py` MUST NOT change in Phase 2 (§A.10).

### 0.2 Why Phase 1 works (mechanisms that MUST be preserved)

1. **Staged difficulty.** Fallen starts and pushes are off at the start. `difficulty_curriculum` keeps `fallen_frac = 0` until the population crawls reliably, then raises it by 0.05 per check up to 0.5 (fills over difficulty 0 → 0.5); pushes fill over 0.5 → 1. Recovery is learned as a small addition to an existing gait.
2. **Posture gate + recovery reward.** `forward_velocity_exp` (despite its name) is `tanh(v/v_target)` and its forward half is multiplied by flatness `clamp(1 − (g_x² + g_y²), 0, 1)` (projected gravity), so a toppled robot gets no forward reward until level. `recovery` (`upright_bonus`, +1.0) pays for being level.
3. **Linear-from-rest forward reward.** `tanh(v/v_target)` has a gradient from a standstill. A Gaussian bell around the target does not, and MUST NOT replace it.
4. **Additive reward structure.** Terms are summed, never multiplied, so two negatives cannot multiply into a fake positive (this is what killed the tail-first exploit).
5. **Actor sees only proprioception-like signals plus the command**, critic gets privileged extras via `obs_groups`.

### 0.3 Standing decisions

1. **Train from scratch at every phase.** No checkpoint surgery, warm start, or weight transfer. Each phase's purpose is to discover which rewards and curriculum work; the deliverable is one training recipe (config + automatic curriculum) that reproduces the behavior from random initialization with a single command, confirmed on at least two seeds.
2. **No IMU / sensor-noise modeling in simulation.** Phase 1's existing per-term `UniformNoiseCfg` on observations stays; nothing else is added. A later sensor-noise pass is documented in §D.
3. **No scripted open-loop feasibility gates.** What the arm can achieve is discovered by RL and measured by evaluation.
4. **No gait-period measurement and no tuned filters on the actor input.** The actor receives no measured heading, heading error, or heading rate.
5. **No position box.** The `out_of_bounds` termination is removed (§A.7). Success is a logged metric that drives the curriculum.
6. **IMU hardware is a QST QMI8658 (no magnetometer):** absolute yaw is unobservable on the real robot, which is why the heading error cannot be an actor input.
7. All thresholds, windows and weights in this document are **initial values**. Tune them, but log every tuned value and the reason in the experiment log (§A.12).

### 0.4 Known pitfalls (each MUST be avoided)

| ID | Pitfall | Why it breaks learning |
|---|---|---|
| X1 | Fallen start poses on from iteration 0 | In `reset_start_pose`, the fallen fraction comes from the curriculum only if `env.extras["fallen_frac"]` exists; otherwise it silently falls back to the baked `pose_weights` (0.5 folded / 0.25 on_side / 0.25 face_down). If the curriculum that sets it is replaced or disabled, half of all episodes start toppled from the first iteration and neither crawling nor self-righting is discovered. |
| X2 | A success gate that counts fallen-start episodes | Fallen-start episodes cannot succeed early, so a gate that includes them can never open (deadlock). Gates MUST be computed on upright-start episodes only. |
| X3 | Projecting **world position** onto an axis that rotates | `forward_progress_lpf` computes `x = pos·axis` from `root_pos_w` (which includes the env-origin offset, up to about ±45 m at 4096 envs / 1.5 m spacing) and differences it. That is valid only for a fixed axis. If the axis rotates, `x` changes without any motion: huge fake progress. Same hazard for `stalled`. |
| X4 | Replacing `tanh` with a Gaussian forward reward | No gradient from rest, and the posture gate is lost. |
| X5 | Dropping the posture gate or the `recovery` term | Removes the incentive to get upright. |
| X6 | Leaving callers of `get_ref_dir`, `crawl_heading`, `zombie_ref_yaw`, `heading_error_wrist` behind | Runtime errors or silent use of a stale reference (video overlay, evaluation, curriculum). Grep and remove or update all of them. |
| X7 | Changing PPO hyperparameters while debugging the task | Confounds reward/curriculum findings. |
| X8 | Changing several things per experiment | Use the ladder in §A.11. |

---

# PHASE 2 — Replace the heading-error input with a yaw-rate command

## A.1 Goal and scope

**Main goal: change the actor's command input from the heading error angle (`heading_error_wrist`, a world-frame quantity that needs external localization) to a yaw-rate command (`yaw_rate_cmd`, rad/s).** Everything else in Phase 1 behavior and machinery MUST stay as close to Phase 1 as the change allows.

The crawl keeps the Phase 1 curriculum speed `v_target` (0.06 → 0.10 m/s). Steering is by commanded turn rate. Nothing else is added to the task.

## A.2 Definitions

- **Claw arrow:** horizontal direction of the vector from the base origin to the `wrist` body origin (`mdp.get_claw_dir`). It is valid when its horizontal length is > 0.01 m; otherwise `get_claw_dir` falls back to the base axis `yaw − π/2`. Keep this existing behavior.
- **Claw frame:** the frame whose forward axis is the claw arrow. It rotates with the robot.
- **Claw heading rate `r`:** `env.zombie_r_ema`, defined in §A.5.
- **Upright-start episode:** an episode whose spawn pose was `folded`. **Fallen-start episode:** spawn pose `on_side` or `face_down`.
- **Flatness (posture gate) `up`:** `clamp(1 − (g_x² + g_y²), 0, 1)` from `projected_gravity_b`.

## A.3 Command and observation

**P2-1 (observation swap).** The actor group MUST drop the 2-D `heading_err` term and gain one term, `yaw_rate_cmd` (1-D, rad/s, scaled by 1/0.3). Actor dimension: **26 → 25** (3 gravity + 3 ang. vel. + 6 joint pos + 6 joint vel + 6 last action + 1 command). No other actor input is added or changed. The 2-D gait clock stays off. Update the stale "27-D" docstring and the dimension test in `tests/test_smoke.py` to 25.

**P2-2 (command term).** Replace `UniformCrawlHeadingCommand` with `UniformYawRateCommand(CommandTerm)` in `mdp/commands.py`:
- `command`: shape `(num_envs, 1)`, yaw rate in rad/s.
- On every (re)sample: with probability 0.2 the command is exactly 0, otherwise `U(−r_max, +r_max)`, where `r_max = env.extras.get("yaw_rate_max", 0.0)`. While `yaw_rate_max == 0` every command is 0.
- Resample every 5–8 s (uniform, per env, staggered; use the CommandManager `resampling_time_range`) and at every episode reset.
- Keep a `just_resampled` boolean per env (set on resample and on reset), exactly as the old term did.
- Log metrics `yaw_rate_cmd/abs` and `yaw_rate_cmd/resamples_per_episode`.
- Remove the old term, its config fields (`use_dynamic_heading`, `heading_*`), and the heading gate. Add config fields for the resample range and `yaw_rate_max_final` (0.3 rad/s, provisional).

**P2-3 (critic).** The privileged critic group keeps `base_lin_vel`, `base_height`, `ground_friction`. It MUST also gain `r` (the claw heading rate, §A.5). Privileged terms MUST NOT enter the `policy` group.

## A.4 Rewards

Start from Phase 1's reward config. Weights below are Phase 1's unless marked **changed/new**.

| Term | Weight | Definition in Phase 2 |
|---|---|---|
| `forward_velocity` | +2.5 | **Changed axis only.** `tanh(v_c / v_target)` where `v_c` is the base horizontal velocity projected on the **current claw axis**. Keep the posture gate on the forward half (`v_c > 0`: multiply by `up`); the backward half is always the plain (negative) `tanh`. |
| `forward_progress` | +1.0 | **Changed implementation (P2-5).** Same LPF (`tau_s = 1.0`), same `/ v_target` normalization, but the input increments are axis-free. |
| `yaw_rate_tracking` | **+1.5 (new)** | `up · clamp(tanh(v_c / v_target), 0, 1) · mask · exp(−((r − yaw_rate_cmd) / 0.1)²)`. `mask` fades 0 → 1 linearly over `yaw_mask_s = 2.0 s` after every resample and reset (the filtered rate lags a new command). The factor `clamp(tanh(v_c/v_target), 0, 1)` makes this term pay only while crawling forward, so standing still can never earn it. Returns 0 while `yaw_rate_max == 0`. |
| `lateral_velocity` | −1.0 | Unchanged (already uses the claw axis). |
| `overspeed` | −2.0 | Unchanged (already uses the claw axis, factor 2.0). |
| `vertical_velocity` | −2.0 | Unchanged. |
| `base_height` | −0.5 | Unchanged (target 0.005). |
| `orientation` | −0.5 | Unchanged. |
| `recovery` | +1.0 | Unchanged (`upright_bonus`). |
| `action_rate` | −0.01 | Unchanged. |
| `joint_acc` | −1e-4 | Unchanged. |
| `joint_torques` | −1e-5 | Unchanged. |
| `alive` | +0.05 | Unchanged. |
| `joint_limits`, `joint_deviation`, `low_ground_contact`, `gait_asymmetry` | 0 | Unchanged (disabled). |
| `wrist_pointing` | — | **Removed.** Its reference (the commanded heading) no longer exists. Its job (claw leads, no tail-first crawl) is done by `forward_velocity` now being measured on the claw axis: moving against the arrow gives a negative reward. See watch item W2. |

**P2-4.** All reward terms MUST be additive. MUST NOT multiply two terms. (The factors inside `yaw_rate_tracking` are gates on one term, not products of separate rewards, and are the only exception.)

**P2-5 (axis-free bookkeeping — fixes pitfall X3).**
- `forward_progress`: per step, `dx = (p_t − p_{t−1}) · u_t` where `p` is the base horizontal position and `u_t` the current claw unit vector. Keep the existing LPF accumulation and the existing jump guard (zero out `|Δp| > 3·v_target·tau_s`). `reset_start_pose` MUST reseed the cached previous position for the resetting envs (replace the old `zombie_lp_x0`/`zombie_prev_x` seeding that used the axis projection).
- No term may project an absolute world position on a rotating axis.
- `_proj_vel_ema` in the curriculum projects **velocity** on the claw axis (instantaneous, so axis rotation is harmless) and needs no re-anchoring.

## A.5 Claw heading rate (reward/metric measurement, simulation only)

**P2-6.** Implement `claw_heading_rate(env)` in `mdp/rewards.py`:
- Each env step, `ψ_t` = claw heading (`atan2` of the claw unit vector). `Δψ = wrap(ψ_t − ψ_{t−1})`.
- Drop the increment (use `Δψ = 0`) if arrow validity flipped between the two steps or `|Δψ| > 0.5 rad`. This avoids spikes when the arm curls and the arrow falls back to the base axis.
- `r_inst = Δψ / dt_env`; `r ← r + (dt_env / τ_r)(r_inst − r)` with `τ_r = 1.5 s` (config `yaw_rate_ema_tau_s`). The EMA averages out the base sway of the gait; it is a fixed constant, not measured.
- Store `env.zombie_r_ema` and a per-env `time_since_resample`. Update exactly once per env step (inside the `yaw_rate_tracking` reward, which every consumer reads after). Reseed (`r = 0`, `ψ_prev = current ψ`, timer = 0) in `reset_start_pose`; reset the timer when the command term sets `just_resampled`.
- `r` is **never an actor input** and is never computed on the robot.

## A.6 Terminations

| Term | Phase 2 behavior |
|---|---|
| `time_out` (20 s) | Unchanged (truncation). |
| `base_nan` | Unchanged. |
| `base_too_high` (0.45 m) | Unchanged. |
| `out_of_bounds` | **Removed (P2-7).** Keep the function but remove it from the termination config; a config flag `use_box_termination` (default False) MAY re-enable it only for bisecting (§A.11). |
| `stalled` | **Reimplemented axis-free (P2-8).** |

**P2-8 (`stalled`).** Failure (non-bootstrapped terminal) if the base's **Euclidean** horizontal displacement from the window anchor is `< 0.05 m` when `episode_steps − anchor_steps > 12 s`. If the displacement is ≥ 0.05 m at that check, re-anchor (anchor position and step = now) and continue. Record the anchor **inside `reset_start_pose`** (provably runs on every reset) and delete the `episode_length_buf <= 1` heuristic. It does not re-anchor on command resamples. It applies to fallen starts too.

## A.7 Resets and start poses

**P2-9 (fixes pitfall X1).**
- The training config's `pose_weights` default MUST be `(1.0, 0.0, 0.0)` (folded only). Fallen starts MUST come only from `env.extras["fallen_frac"]` (set by the curriculum, §A.8).
- `reset_start_pose` MUST behave as follows when `fallen_frac` is absent: use folded only. The old implicit fallback to 0.5 / 0.25 / 0.25 is removed.
- Every place that relied on that fallback (the "training-spec" play clip, evaluation scenarios) MUST set `fallen_frac` explicitly.
- When `use_curriculum=True`, assert at env creation that the difficulty term exists.
- Record per env: `started_fallen` (bool) in `reset_start_pose`.
- Keep Phase 1's other reset behavior: `joint_noise 0.1`, `xy_offset 0.05`, spawn yaw uniform ±π. The old `zombie_ref_yaw` command draw MUST be removed (the command is now a rate; nothing is drawn).
- `env_spacing` and collision filtering: confirm `InteractiveSceneCfg.filter_collisions` is true so a robot that wanders far cannot touch a neighbor (episodes now always run to their end).

## A.8 Success metrics and curriculum

### P2-10 Metrics (logged to TensorBoard, computed per finished episode)

| Metric | Definition |
|---|---|
| `crawl_success` | Over **upright-start** episodes: `Σ_t v_c·dt / T_episode ≥ 0.6 · v_target` where `T_episode` is the nominal 20 s (so an episode that ends early by a failure term scores low). Population rate = mean over finished upright-start episodes (EMA over the last ~2000). |
| `recovery_success` | Over **fallen-start** episodes: flag set if `−projected_gravity_b.z ≥ 0.8` is reached within 10 s of the episode start. Also log `time_to_upright`. |
| `steer_success` | Over segments (spans between command resamples, or between a resample and the episode end) with `|yaw_rate_cmd| > 0.05` and at least `yaw_mask_s + 1 s` of unmasked time: `|mean(r over the unmasked part) − yaw_rate_cmd| ≤ max(0.35·|yaw_rate_cmd|, 0.03 rad/s)`. A failure termination inside the segment counts as a failure. |

The curriculum term runs at the start of the reset step, before events (verify in the installed Isaac Lab). Read per-env episode accumulators there, then re-initialize them in `reset_start_pose`.

### P2-11 Gates (replace the box-escape signal; everything else stays as Phase 1)

Keep Phase 1's three terms and parameters (`check_interval` counts curriculum calls, as before). Only the gate inputs change; "escape" becomes `crawl_success`.

| Dial | Phase 1 gate | Phase 2 gate |
|---|---|---|
| `v_target` ramp 0.06 → 0.10, step 0.005 (interval 100) | `mean_vx > 0.6·v_target` and escape ≥ 0.5 | `mean_v_c EMA > 0.6·v_target` and `crawl_success ≥ 0.5` |
| `friction_scale` 1.5 → 1.0, step 0.05 (interval 200) | same | same replacement |
| `difficulty` +0.05 per check (interval 100): `fallen_frac` fills over [0, 0.5], `push_scale` over [0.5, 1] | onset: escape ≥ 0.5; continue while `mean_vx > 0.6·v_now` | onset: `crawl_success ≥ 0.5` (upright starts only, pitfall X2); continue while `mean_v_c EMA > 0.6·v_now` |

### P2-12 Yaw-rate stages (new dial `yaw_rate_max`, stored in `env.extras`)

| Stage | `yaw_rate_max` | Unlock |
|---|---|---|
| 0 | 0 (all commands 0) | start |
| 1 | 0.10 rad/s | `difficulty ≥ 0.75` (same unlock point Phase 1 used for its dynamic heading) |
| 2 | +0.05 per check up to 0.30 rad/s | `steer_success ≥ 0.7`, checked every 100 curriculum calls |

Order of events in a normal run: crawl → fallen starts → pushes → steering. No dial moves backward. If a dial's gate does not open within 1500 iterations, stop the run and report the gate's input values (do not loosen thresholds silently). Log every dial (`v_target`, `friction_scale`, `difficulty`, `fallen_frac`, `push_scale`, `yaw_rate_max`) and every metric each iteration.

## A.9 Cleanup requirements

**P2-13.** Remove or update every user of the old command (pitfall X6): `heading_error_wrist`, `crawl_heading_command`, `crawl_heading_resample_flag`, `get_ref_dir`, `wrist_pointing_error` usage, `zombie_ref_yaw`, `use_dynamic_heading` and `heading_*` config fields, the `--dynamic_heading` CLI flag, and the cyan "commanded heading" arrow in `video_overlay.py` (show the claw arrow instead, or disable it). `scripts/play.py` MUST gain `--yaw_rate <rad/s>` that pins a constant command for all envs. Replace Hydra override docs accordingly. The evaluation script MUST project on the claw axis and use `r` (§A.13).

## A.10 Training configuration

**P2-14.** `agents/rsl_rl_ppo_cfg.py` MUST NOT change: 256/128/64 ELU, `init_std 0.6`, `entropy_coef 0.01`, adaptive LR (`desired_kl 0.01`), `gamma 0.995`, `lam 0.95`, `num_steps_per_env 40`, observation normalization on, 4096 envs. Train from random initialization.

## A.11 Mandatory development ladder (pitfall X8)

Run each rung from scratch, 4096 envs. A rung passes only if its criteria hold; do not start the next rung before then. Record each run in the experiment log (§A.12).

| Rung | Change | Pass criteria |
|---|---|---|
| R0 | Phase 1 baseline, unchanged, with the new metrics added (`crawl_success`, `recovery_success`, fallen_frac curve) | Record baseline numbers: iteration at which each dial saturates, `crawl_success`, `recovery_success`, final stalled fraction. These become the reference for later rungs. |
| R1 | Everything in §A.3–A.9 **except** steering: `yaw_rate_max` stays 0, `yaw_rate_tracking` returns 0. Actor has the `yaw_rate_cmd` input (always 0). | Every dial reaches its ceiling within 1.5× R0's iterations; `recovery_success ≥ 0.9×` R0; `crawl_success ≥ 0.9×` R0; sanity checks below all pass. If R1 fails and the box removal is suspected, re-enable `use_box_termination` for one diagnostic run. |
| R2 | Enable stages 1–2 and the `yaw_rate_tracking` reward | `steer_success ≥ 0.7` at `yaw_rate_max = 0.10`; then at the highest stage reached. R1 metrics stay within 10–15 %. |
| R3 | Final recipe from scratch, two seeds | Acceptance criteria §A.13. |

**Sanity checks to run in R1 (before trusting any curve):**
1. `fallen_frac` is exactly 0 at iteration 0 and only rises after `crawl_success ≥ 0.5` (pitfall X1).
2. For an untrained (random) policy, `Episode_Reward/forward_progress` averages near 0, not large positive or negative (pitfall X3).
3. Fraction of time the base moves against the claw arrow (`v_c < −0.02 m/s`) is logged (watch item W2).
4. No reference to the removed names remains (`grep`).
5. TensorBoard shows no surrogate-loss spike above 5× its recent mean without a note.

**If recovery still lags after R1 passes the checks above** (apply one at a time, log each): reward the *change* in flatness (potential-based) instead of only its level; introduce `on_side` starts before `face_down`; lengthen the stall window for fallen starts.

## A.12 Experiment log

`docs/phase2_report.md` MUST list every run: rung, git diff summary, tuned values, metrics from §A.8, outcome, and the reason a variant failed. The point of this phase is to learn which reward and curriculum choices work.

## A.13 Evaluation and acceptance

Evaluation uses the deterministic policy path (no observation corruption), 100 envs per scenario, friction reported at 0.5 / 0.8 / 1.2, **no sensor-noise model**. Episodes ended by a failure term count as failures and are never dropped from any statistic. Metrics project on the claw axis and use `r`.

| Scenario | Acceptance |
|---|---|
| Straight crawl, upright starts (`yaw_rate_cmd = 0`, `v_target = 0.10`) | Mean claw-axis speed within ±30 % of target; \|mean `r`\| < 0.05 rad/s; `crawl_success ≥ R0` baseline × 0.9 |
| Steered crawl, upright starts (`±0.1`, `±0.2`, and the highest stage reached) | Speed within ±35 % of target and mean `r` within `max(35 %, 0.03 rad/s)` of command in ≥ 70 % of segments. Report the highest tracked rate. |
| Fallen starts (`on_side`, `face_down`) | `recovery_success ≥ 0.9×` the R0 value; `time_to_upright` reported |
| Command change (e.g. +0.1 → −0.1 mid-episode) | `r` settles within 3 s of the unmasked period; no flips beyond Phase 1 norms |
| Regression | Straight-crawl metrics no more than 10–15 % worse than R0 under the same evaluation |

The training recipe MUST also reach these results from scratch with a second seed.

## A.14 Deliverables

1. Code changes per §A.3–A.9 on top of commit `c450601`.
2. `train.py` (stage flags for debugging one stage; default is the full automatic curriculum from scratch), `play.py` (`--yaw_rate`), the evaluation script, `video_overlay.py` fix.
3. Best checkpoint; exported ONNX and TorchScript actor; TensorBoard logs; the final training config.
4. Videos: straight crawl, steered crawl (left and right), a mid-episode command change, a fallen start recovering.
5. `policy_io.md` (new): actor observation ordering and scaling (25-D), command semantics (an operator-issued yaw rate; no heading measurement needed), joint-velocity recipe, the measured reachable `yaw_rate_cmd` range.
6. `docs/phase2_report.md` (§A.12).
7. Tests (`pytest`): 25-D observation test; command-sampling statistics (zero fraction ≈ 0.2, range obeys `yaw_rate_max`); `claw_heading_rate` unit tests (known ramp including the ±π wrap; oscillation + small mean recovers the mean; arrow-validity flip produces no spike; reset reseeds); `forward_progress` axis-free test (rotating the axis without moving the base yields ~0 progress); `stalled` Euclidean test; `fallen_frac` absent → folded-only test; short training smoke run writing a checkpoint.

## A.15 Watch list

| ID | Risk | Check / mitigation |
|---|---|---|
| W1 | The yaw-rate command is ignored | Log `steer_success`; if flat in R2, add the smoothed base-gyro yaw rate (~1 s moving average, hardware-measurable) as one extra actor input and re-run R2. |
| W2 | Tail-first crawl returns (no `wrist_pointing`) | Logged `v_c < −0.02` fraction; if it exceeds a few percent, add a reward penalty on velocity opposing the claw arrow before re-adding anything reference-based. |
| W3 | Turning is faked by swinging the shoulder-pan joint (the arrow turns, the base does not) | Log base yaw rate vs claw heading rate; if they decouple, add a small penalty on pan-joint velocity or a term requiring base yaw rate to agree in sign. |
| W4 | Filtered rate leaks gait sway or lags | `τ_r` and `yaw_mask_s` are tunable; log and adjust, do not introduce measurement scripts. |
| W5 | Gyro bias on hardware offsets the actor's gyro input | Untrained by design (§0.3.2); handled by the future sensor-noise pass (§D) and the board's startup calibration. |
| W6 | Physics exploits (hopping, jitter) | Keep vertical-velocity, action-rate, overspeed penalties; review videos, not only curves. |

## A.16 Decisions already made by the owner

1. Replace the heading error with a yaw-rate command; no heading, heading-error, or heading-rate input on the actor.
2. Turn-rate limit unknown: 0.3 rad/s is provisional; RL discovers the reachable range.
3. No sensor-noise model in sim; no scripted feasibility gates.
4. Train from scratch every time; the deliverable is a reproducible recipe.
5. Position box removed; success is a metric.

---

# PHASE 3 — Additional maneuvers: sit still, rotate in place, variable-speed forward + turn

**Starting point:** the Phase 2 code and the recipe that worked (rewards, curriculum, bookkeeping rules X1–X8 all still apply). Phase 3 trains **from scratch** with the extended command space; no warm start from Phase 2 weights. The ladder (§B.9) applies here too.

## B.1 Behaviors

| Command | Behavior |
|---|---|
| `vx_cmd > 0`, `yaw_rate_cmd ≈ 0` | Crawl forward |
| `vx_cmd = 0`, `yaw_rate_cmd = 0` | Sit still in the rest pose |
| `vx_cmd = 0`, `yaw_rate_cmd ≠ 0` | Rotate in place without translating |
| `vx_cmd > 0`, `yaw_rate_cmd ≠ 0` | Crawl while turning, at variable speed |

One policy handles all modes. Sit comes from a zero command only; there is no separate sit flag.

## B.2 Command and observation

**P3-1.** Add `vx_cmd` ∈ [0, 0.15] m/s, replacing the fixed `v_target` as the forward-speed target (Phase 2's `yaw_rate_cmd` range is unchanged). Actor dimension **25 → 26**. Critic gains the mode indicators `is_zero_cmd`, `is_rotate_only`.
- **Deadband:** `|vx_cmd| < 0.02` and `|yaw_rate_cmd| < 0.05` count as zero. Derive the indicators from these.
- The command term gains the `vx_cmd` component and a mode-mix sampler (§B.6). The `v_target` ramp (0.06 → 0.10) is superseded by sampled `vx_cmd`; keep it as an early-stage cap if it helps and document the choice.
- The gait clock stays off.

## B.3 Rest pose

**P3-2.** Rest pose = base on the ground, arm folded, low CoM. Default is **folded (all joints at URDF zero)**: it is the zero-action pose of `OffsetEMAJointPositionToLimitsAction`, holds still under the `servo` actuator preset, and rests at base z ≈ 0.033 m. A lower candidate (`shoulder_lift 0.3, wrist_flex 0.6, gripper 0.5`) also holds; naive mid-joint tucks collapse. Record the chosen pose and why in `docs/rest_pose.md`.

## B.4 Rewards by mode

Phase 2 rewards stay. Changes:

| Term | Active when | Form |
|---|---|---|
| `forward_velocity` | always | target is `vx_cmd` (posture gate and `tanh` unchanged) |
| `forward_progress` | `vx_cmd` above deadband | as Phase 2 |
| `yaw_rate_tracking` | `yaw_rate_max > 0` | as Phase 2, but the forward-speed factor is replaced by 1 when `is_rotate_only` (rotation is the goal) |
| Stillness: base speed (−) | `is_zero_cmd` | `‖v_base‖²`, after the settle grace |
| Stillness: joint velocity (−) | `is_zero_cmd` | `‖q̇‖²`, after the settle grace |
| Rest-pose attraction (+) | `is_zero_cmd` **and** `up` high | `exp(−‖q − q_rest‖² / σ_q²)`; gated off while tipped |
| No translation (−) | `is_rotate_only` | `‖v_base,xy‖²` |
| `overspeed` | always | above `2·max(vx_cmd, small)` |

Rules:
- **Recovery precedence:** while `is_zero_cmd` and the base is tipped, `recovery` stays active and the rest-pose attraction is gated off until level. A fallen robot gets up first, then sits.
- **Settle grace:** scale stillness penalties down for ~2 s after the command becomes zero.
- **Smooth gating:** ramp indicator gates over ~0.5 s; no hard reward discontinuities.
- **Pushes while sitting (owner decision):** pushes stay on during sit training. Sit is graded twice: strict (pushless) and loose (with pushes).
- **Rotate-only with a rotating claw arrow:** `forward_velocity` has no meaning at `vx_cmd = 0`; make sure it stays near zero there (target 0, `tanh` of a small value) and does not reward drifting.

## B.5 Terminations

**P3-3 (`stalled` MUST be command-gated).** Active only when `vx_cmd` is above the deadband; the anchor resets when the command changes. Without this, every sitting episode ends as a failure at 12 s. Test: a zero-command rollout produces zero `stalled` terminations. `out_of_bounds` stays removed; sit and rotate are scored by their own metrics.

## B.6 Command sampling

- Episodes become **30 s**; commands resample every 5–8 s.
- Full mix per resample: stop 20 %, forward 30 %, rotate-only 25 %, forward + turn 25 %. Add small command noise near zero so the deadband is exercised.
- Resets keep the Phase 2 rules: folded starts, fallen starts only through `fallen_frac` (X1), plus an optional rest-pose start at 10–15 % with zero command.

## B.7 Success metrics and gates

Metrics from §A.8 stay (same definitions, gated on upright starts, failures count). Per-segment definitions for the new modes (scored after the 2 s mask):

| Mode | Segment success (training-time) |
|---|---|
| Forward / forward + turn | claw-axis speed within ±30 % / ±35 % of `vx_cmd`; yaw as `steer_success` when turning |
| Sit | mean base speed < 0.03 m/s (pushes on), mean joint offset from rest < 0.15 rad, upright |
| Rotate in place | mean `r` within `max(30 %, 0.03 rad/s)` of command; base translation speed < 0.03 m/s |

Stage gates (monotonic, check every 100 curriculum calls, `advance ≥ 0.7`; earlier modes must stay above `hold = 0.6`; earlier-stage commands keep a 10–20 % floor of the samples; blend mixes over 50 iterations; a stage stuck 600 iterations stops the run and reports):

| Stage | Command mix | Advance when |
|---|---|---|
| 0 | Phase 2 behavior (`vx_cmd` constant 0.10) | forward and steered success ≥ 0.7 |
| 1 | forward + stop | `success/sit` ≥ 0.7; forward/steered ≥ hold |
| 2 | add rotate-only | `success/rotate` ≥ 0.7; others ≥ hold |
| 3 | full mix | every mode ≥ 0.7 |
| 4 | full mix + all physics dials at final values | none; train to the fixed iteration count |

Phase 2's dials (`v_target`/friction/difficulty/`yaw_rate_max`) keep running underneath unchanged.

## B.8 Evaluation and acceptance

Same setup as §A.13, with the extra scenarios below. Fallen-start recovery and straight/steered crawl keep their Phase 2 thresholds.

| Scenario | Acceptance |
|---|---|
| Sit (zero command 20 s, from crawl and from rest; pushes on and off) | After 3 s: base speed < 0.01 m/s pushless (loose bound with pushes, reported), joint velocities near zero, joints within 0.15 rad of rest, zero `stalled` |
| Rotate in place (`±0.2`, or the reachable maximum) | `r` within ±30 %; base translation < 0.15 m per 10 s |
| Forward + turn (`vx_cmd 0.10`, `±0.1`, plus other points) | Speed and `r` each within ±35 % |
| Transitions (crawl → sit → rotate → crawl+turn → sit → crawl) | Each settles within 3 s; no flips; no stuck states |
| Regression | Phase 2 scenarios no more than 10–15 % worse |

## B.9 Ladder and deliverables

Rungs (each from scratch, pass before continuing): **L1** `vx_cmd` input with commands always forward (must reproduce Phase 2); **L2** add stop/sit with `stalled` gating and rest pose; **L3** add rotate-only; **L4** full mix; **L5** robustness and two-seed reproduction. Log every run in `docs/phase3_report.md`.

Deliverables: code, scripts (`play.py` accepts `--vx` and `--yaw_rate`), best checkpoint and ONNX/TorchScript export, videos (crawl, sit, rotate, steered crawl, transitions), `docs/rest_pose.md`, updated `policy_io.md` (26-D), `docs/phase3_report.md`, tests (26-D observation, command-mix statistics, zero-command/`stalled` test, short training smoke run).

## B.10 Decisions already made

Forward + turn is required; sit from zero command only; pushes stay on during sit (graded loose/strict); `stalled` is command-gated; train from scratch; no position box.

---

# D. Future work (not part of Phases 2–3)

- **Sensor-noise pass:** per-episode gyro bias, gyro noise, tilt offset, encoder quantization (STS3215 resolution), optional one-step latency, then retrain from scratch. **Trigger:** real-robot bring-up shows drift or noise problems.
- **"Face this direction" supervisor:** an external heading source (camera, AprilTag, mocap) issues yaw-rate commands to this policy.
- Full domain randomization, deployment on the real SO-ARM100 / SO-101 with the Feetech servo interface and the QMI8658 IMU, zombie-style reward shaping.
