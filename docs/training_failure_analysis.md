# Zombie Crawl — why training is not converging (latest run only)

Scope: analysis of the latest training run **`logs/rsl_rl/zombie_crawl/2026-10-09_11-04-48`**
(3000 iterations, 4096 envs, `EMAJointPositionToLimitsAction`, gated velocity reward,
`stalled` termination, constant LR 1e-3, entropy coef 0.001 — see `params/agent.yaml`).
All numbers below were extracted from that run's tensorboard; all code references are to
the current source tree.

## What the run actually did (verified)

| iter | mean_reward | ep_len | action std | entropy | fwd_vel (per s) | forward_progress | stalled frac |
|---|---|---|---|---|---|---|---|
| 0 | −0.20 | 12.4 | 0.99 | +8.5 | 0.003 | 0.000 | **0.409** |
| 200 | +6.50 | 918 | 0.17 | −2.4 | 0.588 | 0.000 | 0.179 |
| 600 | +10.19 | 970 | 0.08 | −6.9 | 0.724 | 0.000 | 0.159 |
| ~700 | peak +11.1 | ~970 | 0.076 | −7.4 | 0.72 | 0.000 | 0.168 |
| 730–820 | **crash: 4.31 → −13.85** | 633–657 | 0.08 → **0.54** | −7.4 → **+4.5** | **0.013** | 0.000 | 0.46 → 0.75 |
| 1500 | −1.90 | 855 | 0.31 | +0.1 | 0.510 | 0.000 | 0.362 |
| 2300 | **−13.31** (2nd crash) | 740 | 0.38 | +1.6 | 0.252 | 0.000 | 0.460 |
| 2999 | −2.66 | 676 | 0.23 | −3.1 | 0.230 | 0.000 | **0.621** |

- Mean reward range: min −20.59, max +11.08; last-100 average **−7.05** (worse than start).
- `Episode_Reward/forward_progress` = **0.000 at every one of the 3000 iterations** — the
  policy never earned a single step of net +X displacement credit in the whole run.
- `Curriculum/v_target` took exactly one value (0.06) and `Curriculum/friction_scale`
  exactly one value (1.5) across all 3000 iterations — the curriculum never moved.
- No NaNs (`base_nan` max 0.0), no `out_of_bounds`, no `base_too_high` events.
- The run peaked at iter ~665–700 (entropy at its minimum, std 0.076 — a near-deterministic,
  overconfident policy), then **collapsed between iter 710 and 820** (surrogate loss spiked
  0.01 → 0.141–0.171, ~10–17× normal), partially recovered, and **crashed again at iter ~2300**.
  End state is a degraded, oscillating policy that is worse than iteration 0 in mean reward.

---

## Item 2 — Geometric contradiction: yaw-0 reward vs. the arm's reach axis

**Problem.** At yaw 0 the arm reaches along world **−Y** (FK of the USD asset, at
`pan=0` the gripper sits ~0.20 m along −Y from the base), i.e. perpendicular to the +X
travel direction. The reward structure enforces the opposite geometry:

- `yaw_error` penalizes `1 − cos(yaw)` vs **world +X** (`mdp/rewards.py:97-104`),
  weight −0.5 (`crawl_env_cfg.py:238`).
- The main `forward_velocity_exp` rewards world-frame vx (`mdp/rewards.py:61-71`),
  which the robot can only produce by pulling along its arm — i.e. with the base rotated
  ~±90° (or `shoulder_pan` driven ~±1.6 rad, near its soft limit).

The two terms are mutually exclusive: any crawl geometry costs up to the full yaw penalty.

**Evidence from the latest run.** `Episode_Reward/yaw_error` goes from −0.002/s at iter 0
to **−0.227/s at iter 2999** (last-100 avg −0.287/s) ⇒ mean `1 − cos(yaw)` ≈ 0.45–0.57 ⇒
the population is ending the run with its base rotated **~57–65° off yaw 0**, on average,
continuously paying ~45–57% of the maximum yaw penalty. Combined with `forward_progress`
= 0 for the whole run, the endgame equilibrium is: rotate the base to point the arm into
the travel direction, pay the yaw penalty, still move nowhere — the positive (velocity
bursts) and negative (yaw) terms nearly cancel, leaving no net selection pressure on
displacement. The `yaw_range` at reset is ±π/2 (`crawl_env_cfg.py:290`), so the heading
that makes crawling geometrically easiest (≈±90°) sits at the very edge of the spawn range:
only a sliver of resets is ever geometrically favorable.

**Potential solution.** Make the reward consistent with the asset's kinematics — pick one
travel convention and use it everywhere:

1. Simplest: reward **body-frame** forward velocity instead of world vx, and measure
   yaw error against the heading the policy actually adopts. Concretely, in
   `mdp/rewards.py`:
   - rotate `root_lin_vel_w` into the base frame and reward its component along the arm
     reach axis (body −Y at pan 0) — or equivalently keep world vx but drop the yaw
     penalty and instead penalize *lateral* world velocity, which already exists
     (`lateral_velocity`, weight −1.0, keeps the net trajectory on a line without forcing
     a specific heading).
   - If you keep a yaw term, make it relative to a chosen crawl heading (e.g. the heading
     at reset, stored per env in the `reset_start_pose` event) rather than world +X —
     this rewards "keep your heading steady" (which a crawler needs) instead of "face +X".
2. Alternatively (bigger change): change the spawn so the arm reaches along +X at reset —
   rotate each env's spawn by +90° (or set `shoulder_pan` ≈ +1.57 in the start poses in
   `crawl_env_cfg.py:66-103`), so yaw-0 and travel direction coincide. Then the existing
   world-frame rewards become consistent.
3. Either way: verify with a short 100-iter smoke run that `Episode_Reward/yaw_error`
   trends toward 0 *while* `Episode_Reward/forward_progress` becomes nonzero — the two
   should no longer trade off.

---

## Item 3 — Action space: raw 0 = soft-limit midpoint attractor + bang-bang exploration

**Problem.** The latest run uses `EMAJointPositionToLimitsAction` with
`scale=1.0, rescale_to_limits=True` (`crawl_env_cfg.py:183-189`). With
`rescale_to_limits`, raw action 0 maps to the **midpoint of each joint's soft range**
(`IsaacLab3/.../actions/joint_actions_to_limits.py:165-177`, `unscale_transform` centered
on (low+high)/2) — pose [0, 1.75, −1.571, −0.65, 0, 0.9] for this asset. That pose is not
any start pose and not `DEFAULT_JOINT_POS` [0, 0.8, −1.2, 0.5, 0, 0.3]. Consequences:

- A PPO actor initializes its mean at 0, so the first thing every episode is commanded to
  do is move to the midpoint pose; with weak PD gains (k=1.0, c=0.1,
  `so_arm100.py:77-78`) the arm gravity-sags into a fixed sprawl attractor within ~1 s.
  All episodes effectively start from the same unusable posture no matter which named
  start pose (`START_POSES`, `crawl_env_cfg.py:66-103`) was reset.
- `init_std = 1.0` on a ±1-clipped Gaussian (`rsl_rl_ppo_cfg.py:45`, confirmed in the
  run's `params/agent.yaml`) puts ~76% of the sampled mass outside the clip: early
  exploration is bang-bang limit-slamming with near-zero policy gradient through the clip.
- The observation side is inconsistent with the action side: `joint_pos` is
  `joint_pos_rel` w.r.t. `DEFAULT_JOINT_POS` (`crawl_env_cfg.py:202`,
  `isaaclab/envs/mdp/observations.py:214-221`) while action-zero references the midpoint
  pose — obs-zero and action-zero describe two different configurations.

**Evidence from the latest run.** This run's only structural change vs. the previous
configuration was this action-term swap — and it is the run that **crashed**: a clean
climb to +11.1 reward by iter ~700, then a collapse at iter ~730 (reward → −11.3, std
0.08 → 0.54, entropy flipping sign) with no recovery to the pre-crash level for the rest
of the 3000 iterations, and a second crash at ~2300. The crash pattern (surrogate spikes
10–17× normal exactly when the previously-deterministic policy, trained against one action
geometry, must keep tracking a moving sprawl attractor) is the signature of the action
reference being wrong for the task: the policy spent its early iterations learning to
hold the attractor, then a large update pushed it off, and the only stable attractor it
found is "tumble to the sprawl and shuffle".

**Potential solution.**

1. Center the action space on a task-relevant pose, not the soft-limit midpoint. Options,
   in order of preference:
   - Use plain `JointPositionAction` (or `EMAJointPositionAction`) with an explicit
     `offset` = a crawl-usable reference pose (e.g. the FK-identified planted pose
     pan≈1.2, lift≈0.8, elbow≈−0.2, wrist≈0.5, gripper≈0.3) and `scale` ≈ 0.2–0.5 rad per
     raw unit, so raw 0 = hold the useful posture and raw ±1 stays well inside joint
     limits. `mdp/actions.py` already contains an unused
     `SmoothedJointPositionAction` for exactly this — wire it up, or delete it and use
     the stock term with the offset.
   - Or keep the limits-relative mapping but apply a constant offset to the raw action
     space so that 0 maps to the reference pose.
2. `init_std = 1.0 → 0.4` (`rsl_rl_ppo_cfg.py:45`). With ±1 clipping, std 0.4 keeps the
   bulk of exploration inside the clip so the gradient is dense.
3. Make obs and action references agree: either point `joint_pos_rel` at the same
   reference pose used by the action offset, or add the reference pose as a constant
   offset in the obs. Update the stale "25-D" docstring (`crawl_env_cfg.py:198`; the
   policy group is 24-D) and `tests/test_smoke.py` if it asserts the old count.
4. Keep the start poses meaningful: initialize the action term's internal state (EMA /
   target) to the *reset pose* on every reset so the first second of each episode does not
   fight toward the zero-action attractor.

---

## Item 4 — The actor never sees base linear velocity, but the main reward is a function of it

**Problem.** The actor's policy group is 24-D: `projected_gravity`(3) + `base_ang_vel`(3)
+ `joint_pos_rel`(6) + `joint_vel`(6) + `last_action`(6) (`crawl_env_cfg.py:200-204`).
There is **no base linear velocity**. The dominant reward `forward_velocity_exp` is a
function of world-frame vx (`mdp/rewards.py:68-71`), and `forward_progress` is a function
of world +X displacement (`mdp/rewards.py:84-94`). The critic sees `base_lin_vel`
(`crawl_env_cfg.py:216`) — the asymmetry is fine in principle, but the *policy* must
internally integrate noisy joint velocities and gravity to estimate the very quantity it
is rewarded on. That is an extra inference layer on the single signal that matters, and
it is why standard Isaac Lab locomotion tasks put `base_lin_vel` in the actor group.

**Evidence from the latest run.** `Episode_Reward/forward_progress` is **0.000 at all
3000 iterations** and `forward_velocity` only fires on transient vx bursts (rate peaks at
0.72/s at iter ~665 then oscillates 0.01–0.72 — bursts, not sustained motion). The
policy never built a consistent internal model of its own forward velocity; it learned to
poke the gate open in bursts. After the iter ~730 crash it did not regain it.

**Potential solution.** Add `base_lin_vel` to the policy group:

```python
# crawl_env_cfg.py, CrawlObservationsCfg.PolicyCfg
base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
```

(policy obs becomes 27-D; the critic group already concatenates it, so no change needed
there — just make sure the runner's `obs_groups` still maps actor→["policy"],
`rsl_rl_ppo_cfg.py:37`). Use the same noise magnitude as the other velocity terms. This is
a one-line change with no reward interaction; retrain from scratch (checkpoint
incompatibility due to obs dim change). If you want to stay strictly hardware-plausible
later for sim-to-real, keep a variant without it — but for getting learning to work, give
the actor the sensor the reward depends on.

---

## Item 5 — Curriculum deadlock: both knobs frozen at their hardest values for all 3000 iterations

**Problem.** `Curriculum/v_target` and `Curriculum/friction_scale` advanced **zero** times
in the run (verified: exactly one distinct value each — 0.06 and 1.5 — across all 3000
tensorboard points). The advance conditions:

- `v_target`: `mean_vx > 0.6 · v_target` **and** `mean_episode_len > 0.5 · max`
  (`mdp/curriculums.py:55`, params at `crawl_env_cfg.py:335-345`).
- `friction_scale`: `mean_vx > 0.6 · v_target` (`mdp/curriculums.py:84`,
  `crawl_env_cfg.py:346-355`).

Both are keyed on **instantaneous** `root_lin_vel_w[:,0].mean()` — a scalar sampled at
check time that reflects oscillation, not net progress. In this run the population's mean
vx never sustained above the 0.036 m/s bar (and `forward_progress` shows net displacement
was ~0), so the curriculum was inert for the entire run. Side effects:

- Training ran the whole time with the friction multiplier at its maximum (1.5), i.e.
  pair static friction 0.8 × 1.5 × U(0.625, 1.5) ≈ 0.75–1.8, while play/eval freezes the
  multiplier at 1.0 (`crawl_env_cfg.py:615, 648`) — the policy is trained on a different
  floor than it is evaluated on.
- `v_target` stayed at 0.06 while the eval/success bar is 0.10 — the end target was never
  in scope.

**Evidence from the latest run.** The tensorboard `Curriculum/*` series are flat lines
for all 3000 iterations; `Episode_Reward/forward_velocity` oscillates without any trend
that would track an advancing target.

**Potential solution.**

1. Key curriculum progress on the quantity that matters — **net displacement** — not
   instantaneous vx. In `mdp/curriculums.py`, track per-env `x − start_x` (the same state
   `stalled` already maintains, see item 6) and advance when the population median net
   displacement over the last N iterations exceeds a threshold (e.g. > 0.5 m at the 20 s
   horizon, scaled).
2. Lower the bar for the first advance: `min_fraction` 0.6 → 0.3 for `v_target`
   (`crawl_env_cfg.py:341`), and decouple `friction_scale` from it entirely — decay the
   friction multiplier on a fixed iteration schedule (e.g. linearly from 1.5 → 1.0 over
   the first 1000–1500 iterations) so the policy is gradually exposed to the eval friction
   instead of training at max grip for 3000 iterations.
3. Optionally start easier: `v_start` 0.06 → 0.03–0.04 so the reward bell
   (`std=0.05` around the target, `crawl_env_cfg.py:234`) overlaps what the policy can
   actually do early, then ease up.
4. Add a sanity log line (or tensorboard scalar) for the advance condition's inputs
   (mean net displacement, mean ep len) so a frozen curriculum is visible at a glance next
   time.

---

## Item 6 — `stalled` termination: a real bug + a bar that converts "no learning" into "constant failure"

**Problem A (bug).** `stalled()` detects "just reset" with
`just_reset = env.episode_length_buf <= 1` (`mdp/terminations.py:81`). The run used
`init_at_random_ep_len: true` (`params/agent.yaml`), which randomizes
`episode_length_buf` to U[0, 1000) at training start **without resetting the robots**. On
the first step, every env whose randomized length is already > 600 (`stall_steps` =
12 s / 0.02 = 600) has `|x − start_x| ≈ 0 < 0.05` and is terminated as a **failure** on
step 1. Verified in the run: `Episode_Termination/stalled` = **0.409 at iter 0** — exactly
the ~40% of envs whose random length starts above 600. One PPO update is trained on ~40%
fake stall-failures. The `zombie_start_x` state is also lazily created on first call
(`mdp/terminations.py:77-79`) and re-recorded only via the buf heuristic, so any reset
path that doesn't pass through it leaves stale start positions.

**Problem B (bar).** The stall bar is 5 cm net +X within 12 s = **0.42 cm/s sustained**.
In this run `forward_progress` never exceeded 0.000 (no net displacement at any point),
so the stall termination is firing on essentially the entire population late in training:
`stalled` fraction ends at **0.621** at iter 2999 (last-100 avg 0.404, peaking 0.75 during
the crash). Failure terminations give no value bootstrap, so 40–75% of all transitions in
the late run are hard negatives for "everything the robot can do". The value at the reset
state is pinned near zero/negative and the only remaining contrast is "stall vs. not
stall", which — given item 4 — correlates with vx bursts, not crawling.

**Evidence from the latest run.** `stalled` = 0.409 @0 (bug), 0.159 @600 (best phase —
policy was timeouting most episodes), 0.746 @800 (crash), 0.621 @2999 (end). Mean episode
length 676 steps at the end ≈ 13.5 s, consistent with the population dying at the 12 s
stall line.

**Potential solution.**

1. Fix the reset detection: record `start_x` **inside** the `reset_start_pose` event
   (`mdp/events.py`, which provably runs on every reset) instead of the
   `episode_length_buf <= 1` heuristic, and remove the lazy-init branch in
   `mdp/terminations.py:77-83`. Alternatively set `init_at_random_ep_len: false` in
   `CrawlPPORunnerCfg` (rsl_rl runner cfg) — the bug disappears, at the cost of a slightly
   noisier first update.
2. Until the policy can actually crawl, relax the bar: `min_displacement` 0.05 → 0.02 and
   `stall_time_s` 12 → 18 (`crawl_env_cfg.py:268`), or temporarily set the term's weight
   to truncation semantics (bootstrap) so "slow" is not "failed".
3. Re-tighten the bar via the curriculum (item 5): only shorten `stall_time_s` / raise
   `min_displacement` once median net displacement is consistently above the current bar.

---

## Item 7 — Tiny per-step reward signal + 24-step rollouts + unmanaged confidence → repeated policy collapse

**Problem.** Three compounding optimization issues, all visible in the latest run:

1. **Reward magnitude.** Isaac Lab 3's reward manager multiplies every term by
   `dt = 0.02` (`IsaacLab3/.../managers/reward_manager.py:151-152`), so the configured
   weights are 50× smaller per step than they read: the max step reward from the dominant
   term is `1.5 · 0.02 = 0.03`, and at the velocities this run actually produces it is an
   order of magnitude less. Whole-episode returns were ~+11 at the peak and ~−2.7 at the
   end — small absolute signal for a 20 s skill.
2. **Short rollouts for a long-horizon skill.** `num_steps_per_env = 24`
   (`rsl_rl_ppo_cfg.py:27`) = 0.48 s of experience per update, with `gamma = 0.99`,
   `lam = 0.95` (effective credit-horizon a few steps) while the task (stall line at 12 s,
   episode at 20 s) rewards outcomes 25–40× farther out. The value function must bridge a
   long gap from short segments; with the `stalled` failure (item 6) injecting hard
   negatives, advantages are high-variance.
3. **Unmanaged confidence.** `entropy_coef = 0.001` is constant with no floor
   (`rsl_rl_ppo_cfg.py:61`) and the schedule is constant LR 1e-3 with
   `desired_kl = 0.01` inert (`rsl_rl_ppo_cfg.py:68, 71`). The run's `mean_std` collapsed
   to **0.076 at iter 665** (entropy −7.4 nats — a near-deterministic policy), and within
   ~60 iterations the policy **crashed** (surrogate 0.01 → 0.14–0.17, reward +8 → −11).
   Once the std is that small, a single large update (max_grad_norm 1.0 is not enough at
   this signal level) can move the policy far, and there is no entropy restoring
   exploration. The second crash at ~2300 has the same signature (surrogate 0.171, std
   jump, reward −13.3).

**Evidence from the latest run.** Surrogate loss: ~0.01 in healthy phases vs. 0.141 @700,
0.155 @800, 0.171 @2300. `mean_std`: 0.99 @0 → 0.076 @665 → 0.353 @800 → 0.544 @1000 →
0.232 @2999 (oscillating, never settling). Entropy flips sign at each crash (−7.4 → +2.1,
+0.1 → +1.6). The run ended at last-100 mean reward **−7.05**, i.e. below the iteration-0
value of −0.20.

**Potential solution.**

1. Scale the signal up: since the manager divides by dt anyway, work in per-second
   magnitudes — raise the dominant weights ~5–10× (e.g. `forward_velocity` 1.5 → 8–15,
   `forward_progress` 0.1 → 0.5–1.0, `crawl_env_cfg.py:233-236`) so per-step rewards are
   O(0.1) and episode returns O(100). PPO is not scale-invariant in practice; the current
   scale makes the policy loss ~100× smaller than a well-scaled one.
2. Extend the rollout: `num_steps_per_env` 24 → 48 (0.96 s) or 96 — cheaper, more stable
   advantages for a 12–20 s skill; compensate by dropping `num_learning_epochs` 5 → 3 if
   wall-clock matters. (Keep it ≥ the stall-time scale ratio so each segment contains
   meaningful progress contrast.)
3. Manage confidence:
   - set an **entropy floor** on the std (e.g. clamp std ≥ 0.1–0.2) or use a KL-adaptive
     entropy coefficient;
   - or revert `schedule` to `"adaptive"` with `desired_kl = 0.01` **and** a sensible
     entropy coef (0.01) so the LR backs off when updates get large — the earlier
     constant-LR change fixed LR freezing but removed the last guard against large
     policy moves;
   - consider `normalize_advantage_per_mini_batch: true` (`params/agent.yaml`) to reduce
     advantage variance from the stall-failure negatives.
4. Add a training-time guardrail: stop or log-loud when `Loss/surrogate` > 5× its
   last-100-iter mean, and auto-load the last checkpoint — the two crashes in this run
   each destroyed ~1000+ iterations of a policy that never got a chance to recover.

---

## Suggested implementation order (cheapest → highest impact)

1. One-liners, no behavior trade-offs: item 4 (add `base_lin_vel` to actor obs),
   item 6.1 (fix `stalled` reset detection / `init_at_random_ep_len`), item 7.3
   (entropy floor or adaptive schedule), item 7.4 (crash guardrail).
2. Config rebalance: item 7.1 (reward scale ×5–10), item 7.2 (rollouts 24 → 48),
   item 6.2 (relax stall bar), item 5.2–5.3 (schedule-based friction decay, lower
   `v_start`/`min_fraction`).
3. Structural: item 3.1 (re-center action space on a planted reference pose,
   `init_std` → 0.4), item 2 (resolve the yaw-vs-travel-axis conflict — body-frame
   forward reward or +90° spawn).
4. Re-run 3000 iters and watch, in order: `forward_progress` leaves 0 (item 2/4 worked),
   `stalled` drops < 0.2 (item 6), `Curriculum/v_target` moves (item 5), surrogate
   stays < 0.05 with std in [0.1, 0.5] (item 7).
