# Zombie Crawl — RL Specification

A free-floating **SO-ARM100** (6-DOF arm, no fixed base) that is trained with **PPO**
(RSL-RL) to drag its body forward along world **+X** on flat ground, crawling at
~6–10 cm/s. This file documents the three things that define the task: the **reward
function**, the **neural-network models**, and the **termination conditions**.

Source of truth (all under `source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/`):

| Concern       | File                                    |
|---------------|-----------------------------------------|
| Env + all configs | `crawl_env_cfg.py`                  |
| Reward terms  | `mdp/rewards.py`                        |
| Terminations  | `mdp/terminations.py`                   |
| PPO / networks| `agents/rsl_rl_ppo_cfg.py`              |
| Observations  | `crawl_env_cfg.py` + `mdp/observations.py` |
| Curriculum    | `mdp/curriculums.py`                    |

**Coordinates:** world frame, right-handed. **+X = forward crawl direction**, +Y = lateral,
+Z = up. The robot's long axis (its "front") points down +X at a yaw-0 spawn. The reward is
**world-frame**, not robot-body-frame — the robot's own heading is irrelevant; it must
physically face and move toward +X.

---

## 1. Reward function

Scored at every control step (50 Hz). The episode reward is the sum of the terms below,
each weighted. Positive terms maximize, negative terms penalize.

| # | Term | Weight | Shape | Plain meaning |
|---|------|-------:|-------|---------------|
| 1 | `forward_velocity`  | **+1.5** | `exp(-(vx - v_target)² / 0.05²)` | The main term. Bell-curve bonus peaking when forward (+X) speed equals `v_target`. |
| 2 | `forward_progress`  | +0.1 | per-step +X displacement, clipped ±0.05 m | Small credit for actually moving forward; capped so it can't be farmed. |
| 3 | `alive`             | +0.1 | 1.0 each step the episode is alive | Keeps the running score positive. |
| 4 | `yaw_error`         | −0.5 | `1 − cos(yaw)` vs +X | Penalizes pointing anywhere but straight ahead; full penalty facing backwards. |
| 5 | `lateral_velocity`  | −1.0 | `vy²` | Penalizes swaying left/right. |
| 6 | `vertical_velocity` | −2.0 | `vz²` | Anti-hop: penalizes moving up/down. |
| 7 | `base_height`       | −1.0 | `(z − 0.06)²` | Anti-launch: penalizes the base rising above its ~6 cm resting height. |
| 8 | `orientation`       | −0.5 | L2 of flat-orientation error | Penalizes tilting/rolling off its "belly" orientation. |
| 9 | `overspeed`         | −2.0 | `relu(vx − 2·v_target)²` | Anti-exploit: quadratic wall once faster than 2× target. |
| 10| `action_rate`       | −0.01 | `‖a_t − a_{t-1}‖²` | Smoothness: penalizes whipping joints. |
| 11| `joint_acc`         | −1e-4 | joint-acceleration L2 | Smoothness: penalizes jerk. |
| 12| `joint_limits`      | −0.5 | L2 of limit violation | Penalizes driving joints to the edge of their range. |
| 13| `joint_torques`     | −1e-5 | torque L2 | Cheap-motion: penalizes high torque. |
| 14| `joint_deviation`   | −1e-3 | `‖q − q0‖²` | Very weak pull back toward the default rest pose. |
| 15| `low_ground_contact`| 0.0 | *(disabled)* | Optional "stay low to the ground" bonus — off by default. |
| 16| `gait_asymmetry`    | 0.0 | *(disabled)* | Optional "alternate legs" bonus — off by default. |

**How to read the scale.** The forward-speed bell (weight 1.5) dominates. The
`v_target` it centers on is **curriculum-driven**: it starts at **0.06 m/s** and eases up
to **0.10 m/s** as the population learns (see §4). The bell is narrow (std = 0.05 m/s),
so "right speed" and "not too fast" are two faces of the same hill — the bell provides the
pull, the `overspeed` wall (at 2× target) stops sprinting.

**A known gradient problem.** At ~1 cm/s actual crawl speed, the main bell is centered at
0.06 m/s, so it contributes `exp(-(0.05)²/0.05²) ≈ 0.37` nearly constantly — almost
saturated and giving little gradient. The `forward_progress` term (weight 0.1) carries
almost no weight. This, plus rarely-failing episodes (§3), is a leading suspect for slow
learning (see §5).

---

## 2. Neural-network models

Framework: **RSL-RL 5.5.1 PPO**, **asymmetric actor-critic** (the critic sees privileged
state the actor never sees). Two MLPs, both with empirical observation normalization.

| Network | Architecture | Activation | Notes |
|---------|--------------|------------|-------|
| **Actor** | `256 → 128 → 64` | ELU | Outputs a **diagonal Gaussian** over the 6-DOF limits-relative joint-position action. `init_std = 0.4`. |
| **Critic** | `256 → 128 → 64` | ELU | Value estimate; sees the actor obs **+ 5 privileged dims**. |

**Observations**

- **Actor (policy) group — 25-D** (27-D if the optional gait clock is on). Hardware-plausible
  only, with observation noise: projected gravity, base angular velocity, joint positions
  (relative), joint velocities, last actions.
- **Critic (privileged) extra group — 5-D:** base linear velocity, base height, ground
  friction. Concatenated with the policy group by the runner — this is the "asymmetric" part.

**Action:** 6-D joint **positions** via the **limits-relative** mapping
(`EMAJointPositionToLimitsAction`) — a raw action in [−1, +1] maps onto each joint's own
**soft** position limits (−1 = low, +1 = high; with `soft_joint_pos_limit_factor = 0.95`
that is the middle 95% of each joint's hard range, shrunk symmetrically about the range
midpoint). The policy can therefore
command essentially every joint's full range of motion instead of the old fixed ±0.5 rad
window around the default pose (which covered only 16–45% of each joint's range).
EMA-smoothed (`alpha=0.3`, state re-initialized to the current joint positions on reset,
so episodes don't start with a jump); raw magnitude clipped to ±1.

**PPO hyperparameters** (`rsl_rl_ppo_cfg.py`):

| Param | Value | Param | Value |
|-------|-------|-------|-------|
| `num_steps_per_env` | 24 (0.48 s/rollout) | `gamma` | 0.99 |
| `max_iterations` | 3000 | `lam` | 0.95 |
| `clip_param` | 0.2 | `entropy_coef` | 0.005 |
| `learning_rate` | 1e-3 (adaptive) | `desired_kl` | 0.01 |
| `value_loss_coef` | 1.0 | `num_learning_epochs` | 5 |
| `use_clipped_value_loss` | True | `num_mini_batches` | 4 |
| `max_grad_norm` | 1.0 | `schedule` | adaptive |

**Known model problem (fixed: `init_std` set to 0.4).** `init_std = 1.0` on a **±1-clipped** action space puts most of the
Gaussian's mass outside the clip, so sampled actions saturate (observed raw actions
`|a| ≈ 18`, clipped to ±1) and the policy gradient through the clip is sparse. A standard
fix is `init_std ≈ 0.3–0.5`. With the limits-relative action space, a saturated raw action
means the policy is pressing a joint toward its **hard limit**, not toward ±0.5 rad —
persistent saturation on one joint is now a signal to check the `joint_limits` reward
weight or that joint's effective range.

---

## 3. Termination conditions

**Design intent: tipping over is *allowed*.** A crawler may flip itself; that is *not* a
failure. Episodes end only on numerical failure, being flung into the air, leaving the
bounding area, or the time limit.

| Term | Trigger | Type |
|------|---------|------|
| `time_out` | episode reaches **20 s** | truncation (bootstrap) |
| `base_nan` | any NaN/Inf in root or joint state | failure |
| `base_too_high` | base origin z > **0.45 m** (physics-exploit guard) | failure |
| `out_of_bounds` | \|dx\| > **2.5 m** or \|dy\| > **1.5 m** from the env origin | failure |

**Why this matters for learning.** Because tipping is not a failure and the bounds are
large, most episodes run the full 20 s and `alive` is earned almost every step. That yields
flat value estimates and little contrast between good and bad behavior — another suspected
contributor to slow learning.

---

## 4. Curriculum (drives `v_target` and friction)

Enabled by default (`use_curriculum=True`). Two knobs advance as the population improves:

- **`v_target`:** 0.06 → 0.10 m/s, +0.005 step. Advances only when mean forward speed >
  60% of the current target **and** mean episode length > 50% of max.
- **`friction_scale`:** 1.5 → 1.0, −0.05 step. Decays the extra ground-friction multiplier
  as the robot gets faster. (Per-env base friction is sampled at startup, effective
  ~0.5–1.2, and is also exposed to the critic as a privileged observation.)

---

## 5. "Not learning" — open items

Observed: the best checkpoint moves ~1 cm/s and raw actions saturate at the ±1 clip. The
likely culprits, in order:

1. **`init_std = 1.0` too large** → saturated actions, sparse gradient. **Fixed: set to 0.4.**
2. **Weak reward gradient:** main bell nearly saturated at slow speeds; `forward_progress`
   under-weighted. Rebalance (widen the bell / raise progress weight / lower `v_target_start`).
3. **Long, rarely-failing episodes:** flat values, low contrast. Consider a tip/instability
   termination or a shorter horizon.
4. **Action-space coverage (addressed):** the old ±0.5 rad window around the default pose
   covered only 16–45% of each joint's measured range of motion (gripper bottom edge was
   exactly its hard limit; wrist_roll 16%), so a gait needing larger excursions was
   unrepresentable. The action term is now **limits-relative** (§2): raw −1…+1 spans each
   joint's soft range (95% of the hard range). Note raw 0 now means "soft-range midpoint",
   not "default pose" — the EMA state is re-initialized to the reset pose, so this only
   matters as a steady-state bias, not as an episode-start jump.

---

## Run

```bash
source /home/yousof/robotics/experimental/zombie_crawl/scripts/env.sh
cd /home/yousof/robotics/experimental/zombie_crawl
```

### 1. Smoke test, headless, with video

Short run (64 envs, 30 iterations ≈ 2 min of sim) that verifies the full pipeline and
writes at least one complete training clip. A clip is 500 env steps long, so use
`--max_iterations 30` (not 2–3) or no clip will finish.

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 64 --max_iterations 30 --video --headless
# → logs/rsl_rl/zombie_crawl/<timestamp>/videos/train/train_overview_0000.mp4 (4K)
```

### 2. Full training, headless, with video

The real run. One ~10 s 3840×2160 overview clip every 100 PPO iterations (all clips kept on disk)
(`keep_last_n_clips=10` caps disk use).

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 4096 --max_iterations 3000 --video --headless
```

Notes on `--video`:

- `--video` is what turns recording on; without it no `videos/` directory is created.
- The source is a SINGLE world-space viewer camera framing the env-grid centre (several
  robots visible at once), never a per-env camera (that OOMs at 4096). Cost is one 4K
  viewport render per step and is independent of `--num_envs`.
- Tune with `--video_length` / `--video_interval` (env steps); resolution/fps come from
  `CrawlEnvCfg`. Headless frames are fresh (not the old frozen-viewport bug) — see
  [docs/video_fix.md](docs/video_fix.md).

### 3. GUI mode (watch it live)

In 3.0 the launcher is headless by default; `--viz kit` opens the Kit window (and
`--headless` forces headless, e.g. to override a visualizer that would otherwise open a
window — relevant with `--video`). Use a small `--num_envs` so the viewport stays fast;
only env_0 is rendered regardless.

```bash
# Watch training in the Kit viewport:
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 64 --max_iterations 100 --viz kit

# Same, but also recording headless overview clips to disk:
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 64 --max_iterations 30 --video --viz kit
```

### Diagnostics & playback

```bash
# View the robot idle:
python scripts/view_robot.py --viz kit --pose folded --preset servo

# Print the joint limits the action term (raw -1..+1) maps onto:
python scripts/print_action_limits.py

# Physically drive each joint soft-max -> soft-min through the action path:
python scripts/drive_joint_limits.py --viz kit [--preset servo]
python scripts/drive_joint_limits.py --no-idle        # headless: exit after the summary

# Record a 20 s video of a checkpoint (two angles: side + top-down):
python scripts/play.py --task Zombie-Crawl-SO100-Play-v0 \
    --checkpoint logs/rsl_rl/zombie_crawl/<run>/model_3499.pt --video
# → <run>/videos/play/side_model_3499_0000.mp4 and top_model_3499_0000.mp4
```

- `--preset servo` (2.9 N·m, real servo) vs `--preset ideal` (10 N·m, separates task-design
  from torque limits).
- `--pose folded|extended|on_side`.
- `--headless` runs without a GUI window (on `train.py`/`play.py` and the test helpers).
  Isaac Lab 3.0 dropped `--headless` from the launcher CLI — headless is the default when
  no visualizer is requested — so this repo re-adds the flag as a compatibility shim that
  maps it to the launcher's `HEADLESS=1` env var (and keeps `--viz kit` working for the GUI).
- Headless `--video` records from scene camera sensors, not the GUI viewport —
  see [docs/video_fix.md](docs/video_fix.md) for how it works and why the old
  viewport-based clips came out frozen.
