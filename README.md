# Zombie Crawl — RL Specification

A free-floating **SO-ARM100** (6-DOF arm, no fixed base) trained with **PPO** (RSL-RL) to
drag its body forward on flat ground, *crawling claw-first toward a randomly commanded
direction* at ~6–10 cm/s. This file documents the three things that define the task: the
**reward function**, the **neural-network models**, and the **termination conditions** — plus
the **curriculum**.

![Zombie crawl training overview](docs/assets/zombie_crawl_training.gif)

*Training overview clip: a grid of parallel SO-ARM100 policies crawling toward their own
randomly commanded headings (single world-space viewer camera, one clip every 200 PPO
iterations). 4K source: `docs/assets/zombie_crawl_discord.mp4`.*

> [!NOTE]
> The whole repo was **vibe-coded** with [opencode](https://opencode.ai) using the
> **Qwen3.8-flash-next** model — and it works.

Source of truth (all under `source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/`):

| Concern           | File                                        |
|-------------------|---------------------------------------------|
| Env + all configs | `crawl_env_cfg.py`                          |
| Reward terms      | `mdp/rewards.py`                            |
| Terminations      | `mdp/terminations.py`                       |
| PPO / networks    | `agents/rsl_rl_ppo_cfg.py`                  |
| Observations      | `crawl_env_cfg.py` + `mdp/observations.py`  |
| Curriculum        | `mdp/curriculums.py`                        |
| Reset / randomization | `mdp/events.py`                         |

**Coordinates & the command.** World frame, right-handed, +Z up. Each episode draws a
**commanded direction: uniform ±180°, independent of the spawn facing**, frozen for the
episode in `env.zombie_ref_yaw` by `mdp.reset_start_pose`. Every directional
reward/curriculum/termination scores motion along that commanded axis, and the
`heading_error_wrist` observation shows the actor the claw-vs-command angle — so the skill
is *turn to the command, then crawl it claw-first*. The command never follows the robot, so
turning cannot farm it. Spawn facings stay uniform ±180°. Evals force spawn yaw 0;
`evaluate.py` scores displacement projected on the command axis.

**Two reference axes** are used deliberately (see `mdp/rewards.py`):

- **Command axis** — `get_ref_dir(env)` = `(cos, sin)` of `zombie_ref_yaw`. A FIXED per-episode
  goal direction. `forward_velocity` and `forward_progress` (and the stall/curriculum gates)
  project onto it: they answer *"am I advancing toward the goal?"*
- **Claw axis** — `get_claw_dir(env)` = the horizontal **base→wrist** arrow (falls back to the
  base −Y yaw when the arm is curled). `wrist_pointing` measures how well this arrow points at
  the command; `lateral_velocity` and `overspeed` measure velocity relative to it. This is the
  locomotion frame that makes the **claw lead**.

---

## 1. Reward function

Scored every control step (50 Hz). The episode reward is the weighted sum of the terms below.
Positive terms maximize, negative terms penalize. Weights are the values in
`CrawlRewardsCfg` (`crawl_env_cfg.py`).

| # | Term | Weight | Shape (axis) | Plain meaning |
|---|------|-------:|--------------|---------------|
| 1 | `forward_velocity`  | **+2.5** | `tanh(v_goal / v_target)`, posture-gated, negative down (**command axis**) | Main term: advance the base toward the commanded heading; flattens past target; moving away scores negative. |
| 2 | `forward_progress`  | **+1.0** | 1 Hz low-passed command-axis displacement, **normalized by `v_target`** (τ = 1 s) | Immune to back-and-forth oscillation; shapes end-of-episode displacement directly. The anti-park lever. |
| 3 | `wrist_pointing`    | **−1.5** | `1 − cos(Δθ)` of **base→wrist arrow vs command** (0 aligned, 2 reversed) | **Anti-tail-first:** penalizes pointing the claw off the command, so the claw must LEAD. Strong enough that crab-walking/backward crawling never pays. |
| 4 | `lateral_velocity`  | −1.0 | `v⊥²` perpendicular to the **claw** axis | Penalizes swaying sideways. |
| 5 | `overspeed`         | −2.0 | `relu(v‖ − 2·v_target)²` along the **claw** axis | Anti-exploit: quadratic wall once faster than 2× target. |
| 6 | `vertical_velocity` | −2.0 | `vz²` | Anti-hop: penalizes moving up/down. |
| 7 | `base_height`       | −0.5 | `(z − 0.005)²` | Anti-launch: penalizes rising off the crouch resting height (~0.5 cm). Softened so it never out-weighs recovery. |
| 8 | `orientation`       | −0.5 | L2 of flat-orientation error | Penalizes tilting/rolling off the "belly" orientation. |
| 9 | `recovery`          | **+1.0** | `clamp(−projected_gravity_b.z, 0, 1)` (1 level, 0 on-side) | **Recovery:** dense pull to right the base when toppled. A level crawl crouch scores the same ~1, so it can't be farmed by rearing up. |
| 10| `action_rate`       | −0.01 | `‖a_t − a_{t-1}‖²` | Smoothness: penalizes whipping joints. |
| 11| `joint_acc`         | −1e-4 | joint-acceleration L2 | Smoothness: penalizes jerk. |
| 12| `joint_limits`      | 0.0 | L2 of limit violation | *(disabled)* Penalizes driving joints to the edge of range. |
| 13| `joint_torques`     | −1e-5 | torque L2 | Cheap-motion: penalizes high torque. |
| 14| `joint_deviation`   | 0.0 | `‖q − q0‖²` | *(disabled)* Weak pull toward the default pose. |
| 15| `alive`             | +0.05 | 1.0 per alive step | Keeps the running score positive (halved for value contrast). |
| 16| `low_ground_contact`| 0.0 | *(disabled)* | Optional "stay low" bonus — off by default. |
| 17| `gait_asymmetry`    | 0.0 | *(disabled)* | Optional "alternate limbs" bonus — off by default. |

**Two additive goal terms (no product).** `forward_velocity` (speed toward the goal) and
`wrist_pointing` (claw aimed at the goal) are **independent numbers added together**. Both the
old failure modes are closed:

- **Off-goal crawl** is punished by `forward_velocity`/`forward_progress` being on the *fixed
  command axis* (advancing away from the goal is negative, not reward-able by pointing the claw
  some other way).
- **Tail-first (backward) crawl** is dominated by `wrist_pointing` (−1.5): a robot that advances
  toward the goal while dragging the claw behind it has claw→command ≈ reversed, so the aim term
  pays ≈ −3.0 and overwhelms the +2.5 speed term.

A previous design multiplied claw-speed by `cos(claw→command)`; two negatives (claw pointed away
**and** base moving opposite the claw) multiplied into a positive, so the policy learned to crawl
tail-first. The additive split removed that sign flip.

**Scale.** `forward_velocity` is `tanh(v_goal/v_target)` at weight 2.5 — a linear gradient at low
speed, capped by the `overspeed` wall at 2× target. `forward_progress` is the 1 Hz LPF of
command-axis displacement **divided by `v_target`** so it sits on the same [0,1]-ish scale as the
speed term (raw m/s was numerically negligible). `v_target` is curriculum-driven (0.06 → 0.10 m/s, §4).
A *posture gate* multiplies only the forward half of `forward_velocity`, so a face-plant topple no
longer pays for the forward push.

---

## 2. Neural-network models

Framework: **RSL-RL 5.x PPO**, **asymmetric actor-critic** (the critic sees privileged state the
actor never sees). Two MLPs, both with empirical observation normalization.

| Network | Architecture | Activation | Notes |
|---------|--------------|------------|-------|
| **Actor** | `256 → 128 → 64` | ELU | Diagonal Gaussian over the 6-DOF limits-relative joint-position action. `init_std = 0.6`. |
| **Critic** | `256 → 128 → 64` | ELU | Value estimate; sees the actor obs **+ 5 privileged dims**. |

**Observations** (`CrawlObservationsCfg`):

- **Actor (policy) group — 26-D** (27-D with the optional gait clock; both use
  `enable_corruption` uniform noise). Hardware-plausible: projected gravity (3),
  `heading_error_wrist` (2), base angular velocity (3), joint positions rel. to folded default (6),
  joint velocities (6), last action (6).
- **Critic (privileged) extra group — 5-D:** base linear velocity (3), base height (1), ground
  friction (1). Concatenated with the policy group by the runner (26 → **31-D** critic input) —
  this is the "asymmetric" part.

`heading_error_wrist` is the (cos, sin) of the angle between the world-frame **base→wrist** arrow
and the commanded heading — the actor directly sees the pointing error `wrist_pointing` uses.
**Caveat:** it needs the wrist's *world* position (external localization), so it is the only
non-proprioceptive observation — drop it for a strict sim-to-real policy.

**Action:** 6-D joint **positions** via the **limits-relative** mapping
(`OffsetEMAJointPositionToLimitsAction`, `mdp/actions.py`) — a raw action in [−1, +1] maps
piecewise about a **neutral** angle (`a = 0` = the folded spawn, not the joint-range midpoint);
`a = ±1` reach each joint's full **soft** limits (`soft_joint_pos_limit_factor = 0.95`, i.e. the
middle 95% of each hard range). A custom subclass was needed because the stock map centered on the
range midpoint (a stand-up kick every reset) and an affine-offset variant clipped away ~half of
each one-sided joint's range (forcing a sideways crab-walk). EMA-smoothed (`alpha = 0.3`, state
re-initialized to the current joint positions on reset, so no episode-start jump); raw magnitude
clipped to ±1.

**PPO hyperparameters** (`agents/rsl_rl_ppo_cfg.py`):

| Param | Value | Param | Value |
|-------|-------|-------|-------|
| `num_steps_per_env` | **40** (0.8 s/rollout) | `gamma` | **0.995** |
| `max_iterations` | 3000 (config default) | `lam` | 0.95 |
| `save_interval` | 50 | `clip_param` | 0.2 |
| `clip_actions` | 1.0 | `entropy_coef` | **0.01** |
| `init_std` | **0.6** | `learning_rate` | 1e-3, **adaptive** (`desired_kl = 0.01`) |
| `value_loss_coef` | 1.0 | `num_learning_epochs` | 5 |
| `use_clipped_value_loss` | True | `num_mini_batches` | 4 |
| hidden dims | `[256, 128, 64]` | `max_grad_norm` | 1.0 |

`NUM_STEPS_PER_ENV = 40` is a **single source of truth**: the training-video timing
(`CrawlEnvCfg._setup_training_video`) converts between the recorder's env-step counter and the
TensorBoard iteration axis using this same constant (a stale 24 skewed every clip ~1.67×).

**Stability notes.** `init_std` went `1.0 → 0.6`: on the ±1-clipped action space, std=1.0 pushed
most Gaussian mass past the clip, saturating actions and starving the gradient (a run blew up
0.08 → 0.49 at iter ~710). `entropy_coef` is now **0.01** to keep the policy stochastic long enough
to discover *turning then crawling* (the old 0.001 let it sit at a zero-reward "point-and-sit"
plateau). Because rsl-rl 5.5.1 has **no KL early-stop**, the **adaptive LR schedule** is the only
feedback loop containing a std blow-up — kept on for that reason.

---

## 3. Termination conditions

**Design intent: tipping over is *allowed*.** A crawler may flip itself; that is *not* a failure
(recovery, §1 term 9, pays for getting back up). Episodes end only in success (crawled out of the
box), on a guard failure (numerical blow-up, flung into the air, stalled out), or at the time limit.

**`out_of_bounds` is the goal, not a failure.** Reaching the box edge means the robot crawled the
commanded distance — it is the success condition and the curriculum's escape gate (§4).

| Term | Trigger | Outcome / bootstrap |
|------|---------|---------------------|
| `time_out` | episode reaches **20 s** | **truncation** (bootstrap `V(s)`) |
| `base_nan` | any NaN/Inf in root or joint state | guard failure (terminal, value 0) |
| `base_too_high` | base origin z > **0.45 m** (physics-exploit guard) | guard failure (terminal, value 0) |
| `out_of_bounds` | \|dx\| > **2.5 m** or \|dy\| > **1.5 m** from the env origin | **success** (terminal, value 0) |
| `stalled` | < **0.05 m** progress along the command axis within **12 s** | guard failure (terminal, value 0) |

> [!NOTE]
> In Isaac Lab a termination is only ever a **terminal** (`time_out=False` → value target 0) or a
> **truncation** (`time_out=True` → bootstrap `V(s)`); there is **no first-class "success" flag** —
> good-vs-bad is just bookkeeping. So the only thing that actually matters is *bootstrap or not*:
> - **`out_of_bounds` is correctly a plain terminal (value 0)** — this is not a penalty. A robot that
>   has left the play field has no meaningful future state to bootstrap, and the termination predicate
>   adds **no negative reward** (only the §1 reward terms pay). Escaping is rewarded purely by
>   `forward_velocity`/`forward_progress` and then simply ends the episode. Calling it a "failure" would
>   be wrong — it is the success signal the curriculum gates on.
> - Only `time_out` is a truncation, because a 20 s cutoff is artificial and its value *should* be
>   bootstrapped. The guard terms (`base_nan`/`base_too_high`/`stalled`) are genuine failures and are
>   terminals on purpose, so "doing nothing"/exploiting reads as value 0.

Caveat: `evaluate.py` currently buckets **any** early `done` (including `out_of_bounds`) as an early
termination and drops those episodes from the displacement stats — so read the eval's termination
breakdown, not just its displacement, before judging a policy.

---

## 4. Curriculum

Enabled by default (`use_curriculum=True`). Three terms in `CrawlCurriculumCfg`, each returning
its own TensorBoard card. All gates read a **sustained** signal (a 2 s EMA of command-projected
speed, `_proj_vel_ema`) **plus** the **box-escape rate** (`_escape_rate` = the value the
TerminationManager logs for `out_of_bounds`), *not* mean episode length — gating on long episodes
is self-defeating for a task whose success is *leaving the box early*.

| Term | Knob | Range / step | Gate (every `check_interval` resets) |
|------|------|--------------|--------------------------------------|
| `v_target` | forward speed target | 0.06 → 0.10, +0.005 | `mean_vx > 0.6·v_target` **and** `escape ≥ 0.5` (interval 100) |
| `friction_scale` | extra ground-friction multiplier | 1.5 → 1.0, −0.05 | same gate (interval 200) |
| `difficulty` | recovery/robustness dial | 0 → 1, +0.05 | onset `escape ≥ 0.5`; continue while `mean_vx > 0.6·v_target` (interval 100) |

- **`friction_scale`:** per-env base friction is sampled at startup (effective ~0.5–1.2) and exposed
  to the critic as a privileged observation; the curriculum decays the extra multiplier as the robot
  gets faster (`apply_friction_scale` interval event).
- **`difficulty` (recovery + robustness):** a single dial drives two env knobs, **staggered** so the
  fragile gait is built first:
  - `fallen_frac` — fraction of resets that **start toppled** (`on_side`/`face_down`), filled over
    `difficulty ∈ [0, 0.5]` (max 0.5). The robot learns to **get up** (via `recovery`) before it is
    ever pushed.
  - `push_scale` — multiplier on the **all-body disturbance** (`disturb_all_bodies`: random ±0.4 N
    force + ±0.03 N·m torque per rigid body, re-rolled every 0.2–0.5 s), filled over
    `difficulty ∈ [0.5, 1]` (`push_start_frac = 0.5`). No poking until it can already crawl and
    self-right.

---

## 5. Current status

The claw-first crawl-toward-the-command **works** and the curriculum **saturates early** (all three
dials hit their ceiling by roughly **iteration ~400** of the 4096-env run — `v_target 0.10`,
`difficulty 1.0`, `fallen_frac 0.5`, `push_scale 1.0`). At ~iteration 600 the population metrics were
approximately:

- `out_of_bounds` (crawls out of the box) ≈ **0.56**, `forward_velocity` ≈ **0.65** of ~1.0,
  `wrist_pointing` ≈ **−0.22** (claw well-aimed — no tail-first regression).
- Deterministic `evaluate.py` on `model_600.pt` (16 envs, random ±180° commands) reported only ~19%
  of steps in the target-speed band and ~81° mean yaw drift — consistent with the ~56% escape rate.
  (See the §3 note: its displacement came out NaN because every episode ended early and the script
  discards early-terminated episodes.)

Open directions (pick one, not all):

1. **Just keep training** at the maxed curriculum and watch escape/speed polish.
2. **A real Phase-3 ceiling raise:** higher `v_final` (0.15–0.20), stronger `push_max`, more
   `fallen_max`, and tighter gates (`min_escape`/`min_fraction`) so difficulty only climbs once the
   crawl is genuinely solid.
3. **Command tracking:** flip the stubbed `use_velocity_commands` so it tracks commanded (vx, vy,
   yaw_rate) instead of a fixed speed.
4. **Sim-to-real hardening:** drop the privileged `heading_error_wrist` observation and stay
   proprioception-only.

Historical note: the early "not learning" symptoms (raw actions saturating the ±1 clip, ~1 cm/s)
were driven by `init_std = 1.0` (now 0.6) and an action window that covered only 16–45% of joint
range (now limits-relative, §2).

---

## Setup

### Requirements

| Need | Detail |
|------|--------|
| OS | Linux x86_64 (developed/tested on Ubuntu 26.04) |
| GPU | NVIDIA RTX GPU (tested: RTX 5090 / Blackwell) with a current driver (CUDA 13.x driver stack) |
| Disk | ~30 GB free (Python env ≈ 24 GB + Isaac Sim pip packages ≈ 6 GB) |
| Tools | git, conda (Miniconda is fine), `ffmpeg` on PATH (the video recorder streams raw frames into it) |

Isaac Sim pins **Python 3.12** — let conda provide it; don't rely on the distro python.

Below, `<repo>` is your clone of this repository.

### 1. Create the env and install Isaac Lab + Isaac Sim (all via pip/conda)

From the repo root — the env lives at `<repo>/../.venv` (a plain `python3.12 -m venv`
off a uv-managed interpreter is broken on this stack, hence conda):

```bash
cd <repo>
conda create --prefix ../.venv python=3.12
conda activate "$(realpath ../.venv)"
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
```

Official install docs (for other install methods / troubleshooting):
[Isaac Sim pip installation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html) ·
[Isaac Lab pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)

### 2. Install the zombie_crawl package

```bash
pip install -e source/zombie_crawl
```

### 3. Per-shell environment

```bash
source scripts/env.sh
```

It activates the project env (`$ZOMBIE_CRAWL_ENV` > `<repo>/.venv` > `<repo>/../.venv`,
overriding e.g. your auto-activated conda base), exports `OMNI_KIT_ACCEPT_EULA=YES`
(mandatory for every headless/background Kit run), and prepends libxml2 to
`LD_LIBRARY_PATH` if it is present (step 4; not needed for training/play/eval). If your
env lives elsewhere, export `ZOMBIE_CRAWL_ENV` pointing at it before sourcing.

### 4. Optional: clean libxml2 (URDF conversion only)

The Isaac Sim URDF importer links `libxml2.so.2`, which Ubuntu 26.04 no longer ships
(only `.so.16`). Needed only when re-running the asset pipeline
(`scripts/asset_pipeline/convert_asset.py`). Build a minimal libxml2 2.12.10 (source
from gnome.org) — **no ICU** (conda's libxml2 pulls ICU73 + its own libstdc++ and heap
crashes) and **no iconv**:

```bash
./configure --prefix="<repo>/.libxml2" --without-python --without-iconv
make -j"$(nproc)" && make install
```

`env.sh` picks it up at `<repo>/.libxml2/lib` (or point `ZOMBIE_CRAWL_LIBXML2` at the
prefix). Only runtime deps: system `libc`/`libm`/`libz`.

### 5. Optional: SO-ARM100 upstream clone (asset pipeline only)

The converted URDF/USD are committed under
`source/zombie_crawl/zombie_crawl/assets/`, so this is only needed to re-run
`scripts/asset_pipeline/` (URDF prep + analysis + USD conversion):

```bash
git clone https://github.com/TheRobotStudio/SO-ARM100.git <repo>/../third_party/SO-ARM100
# or point SO_ARM100_DIR at an existing clone
```

### First-run sanity check

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 8 --max_iterations 1 --headless
```

First boot unpacks Kit caches and takes a few minutes; a completed PPO iteration with no
traceback means the stack is good. (Headless `import isaaclab_tasks` failing with
`No module named 'pxr'` in a *bare* shell is expected — `pxr` only exists inside the
running Isaac Sim app.)

---

## Run

```bash
# from the repo root:
source scripts/env.sh
```

> The `zombie_crawl` package is installed **editable from the main repo**, so a curriculum
> worktree run must export `PYTHONPATH=<worktree>/source/zombie_crawl` or it will silently import
> the main repo's code.

### 1. Smoke test, headless, with video

Short run (64 envs) that verifies the full pipeline and writes at least one complete training clip.
A clip is `training_video_length = 2000` env steps (40 s at 50 Hz = two full episodes) and recording
does **not** start until `training_video_start_iters` (default **200**) — so for a quick clip pass a
low start:

```bash
python scripts/train.py \
  --task Zombie-Crawl-SO100-v0 \
  --num_envs 64 --max_iterations 30 \
  --video --video_start_iters 10 \ # don't use less than 10, the video won't be created properly
  --headless
# → logs/rsl_rl/zombie_crawl/<timestamp>/videos/train/train_overview_*.mp4 (4K)
```

### 2. Full training, headless, with video

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 4096 --max_iterations 5000 --video --video_start_iters 10 --headless
```

One ~40 s 3840×2160 overview clip every `training_video_interval_iters` (200) PPO iterations, all
clips kept on disk (`keep_last_n_clips = None`). Frames stream straight into ffmpeg at full rate
(`frame_stride = 1`, 50 fps) via the task's `StreamingVideoRecorder` (`streaming_recorder.py`), so
host RAM stays constant (the stock in-RAM buffer would need ~66 GB for a clip this size). Clip
filenames bake the TensorBoard iteration/step range, and a `--checkpoint model_N.pt` resume shifts
them by the checkpoint's iteration so they line up with the TensorBoard axis (`scripts/rename_videos.py`
retro-renames existing clips).

Notes on `--video`:

- `--video` is what turns recording on; without it no `videos/` directory is created.
- The source is a SINGLE world-space viewer camera framing the env-grid centre (several robots
  visible at once), never a per-env camera (that OOMs at 4096). Cost is one 4K viewport render per
  step and is independent of `--num_envs`.
- Headless frames are fresh (not the old frozen-viewport bug) — see [docs/video_fix.md](docs/video_fix.md).

### 3. Resuming interrupted training

Checkpoints are written every `save_interval` (50) iterations as `<run_dir>/model_<iter>.pt` under
`logs/rsl_rl/zombie_crawl/`. Resume the **latest checkpoint of the latest run** with:

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 \
    --num_envs 4096 --max_iterations 5000 --video --headless --checkpoint latest
```

Pitfalls (all verified):

- `--checkpoint latest` picks the *newest directory* in `logs/rsl_rl/zombie_crawl/`. An aborted
  launch leaves an empty new dir that outranks the real one — delete stray dirs (contents just
  `run.json`/`model_0.pt`) before resuming. **Never delete run dirs you did not name.**
- Resuming creates a NEW run dir that continues the iteration counter; the old dir stays untouched.
- Do NOT use `--load_run` (rejected: the 3.0 CLI is Hydra-based) and do NOT pass a run-dir path to
  `--checkpoint` (the path gets joined twice → `FileNotFoundError`). Pass a real `model_N.pt`.
- `--max_iterations` is the TOTAL budget; resuming at 600 with 5000 keeps 4400 to go.

### 4. GUI mode (watch it live)

The 3.0 launcher is headless by default; `--viz kit` opens the Kit window (`--headless` forces
headless). Use a small `--num_envs` so the viewport stays fast.

```bash
python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100 --viz kit
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/zombie_crawl
# → http://localhost:6006  (add --port 6007 if busy, or point at a single <run> dir)
```

### Evaluate & playback

```bash
# Deterministic metrics (report includes the termination breakdown — read it):
python scripts/evaluate.py --num_envs 16 --checkpoint logs/rsl_rl/zombie_crawl/<run>/model_600.pt
# → prints displacement/speed/termination stats; --json writes them out.

# Record a 20 s video of a checkpoint (two angles: side + top-down):
python scripts/play.py --task Zombie-Crawl-SO100-v0 \
    --checkpoint logs/rsl_rl/zombie_crawl/<run>/model_600.pt --video --headless --num_envs 8
# → <run>/videos/play/side_model_600_0000.mp4 and top_model_600_0000.mp4
```

- `--headless` runs without a GUI window. Isaac Lab 3.0 dropped `--headless` from the launcher CLI
  (headless is the default when no visualizer is requested); this repo re-adds it as a compatibility
  shim mapped to the launcher's `HEADLESS=1` env var, and keeps `--viz kit` working for the GUI.
- Headless `--video` records from scene camera sensors, not the GUI viewport — see
  [docs/video_fix.md](docs/video_fix.md) for how it works and why the old viewport clips came out frozen.

### Play the shipped checkpoint (`checkpoints/model_1800.pt`)

`checkpoints/model_1800.pt` is the best policy shipped with the repo: iteration 1800 of the
4096-env training run (`~2.5 m/s-scale escapes at maxed curriculum`, 2026-10-14). It is the
**only** file needed to replay it — rsl_rl 5.x checkpoints store just the actor/critic/optimizer
state dicts; the network architecture and PPO hyperparameters come from the task code
(`agents/rsl_rl_ppo_cfg.py`), so editing the hidden dims there would make this checkpoint
incompatible. (`params/agent.yaml` and `run.json` in training run dirs are logging artifacts,
not load inputs.)

```bash
source scripts/env.sh

# Watch it live (GUI):
python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint checkpoints/model_1800.pt

# Record the 4K training-spec clip (4096 robots, full disturbance + 50% fallen starts,
# ~40 s clip from the spawn moment):
python scripts/play.py --task Zombie-Crawl-SO100-v0 \
    --checkpoint checkpoints/model_1800.pt --video --headless
# → checkpoints/videos/play/play_overview_*.mp4   (log dir follows the checkpoint's dir)
# → checkpoints/exported/policy.pt + policy.onnx  (auto-exported JIT/ONNX of the loaded policy)

# Deterministic metrics (termination breakdown included — see the §3 note):
python scripts/evaluate.py --num_envs 16 --checkpoint checkpoints/model_1800.pt
```

Tip: to re-render old runs instead, `--checkpoint` also accepts any `model_N.pt` under
`logs/rsl_rl/zombie_crawl/`, and `--checkpoint latest` picks the newest checkpoint of the
newest run dir.
