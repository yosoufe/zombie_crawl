# Zombie Crawl — Progress & Installation Report

Working directory: `<workspace>`
Last updated: 2026-10-03 (session)

This file is a living report. It records (a) what has been done so far, (b) every
software/tool that was installed and where, and (c) the current state and next steps.
It is updated as work progresses.

Stack: **Isaac Sim 6.1.0.0 + Isaac Lab 3.0** (Python 3.12) — the latest stable
Blackwell-capable release pair, as required by `prompt.md` line 29. All project work
lives in the project-local env `.venv`.

---

## 1. Objective

Implement the "Zombie Crawl" project end-to-end per `prompt.md`: a free-floating
SO-ARM100 arm that RL-trains (PPO via `rsl_rl`) to crawl along +X on flat ground in
Isaac Lab **3.0** simulation on **Isaac Sim 6.1.0.0**, delivering train/play/evaluate
scripts, a video, a checkpoint, a README (with exact pinned versions per prompt.md
line 29), and smoke tests.

---

## 2. Milestone Status

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Repo/asset prep (IsaacLab + SO-ARM100 clones, URDF analysis) | Done |
| M1 | Install Isaac Lab 3.0 + Isaac Sim 6.1 + rsl_rl; headless smoke test | **Done** — per official pip-install manuals: `create_empty.py` ran clean headless; `train.py --rl_library rsl_rl --task Isaac-Ant --num_envs 64 --max_iterations 2` finished 2 PPO iterations in 17.9 s on RTX 5090 |
| M2 | Convert prepared URDF → USD (`so_arm100_float.usd`) | **Done (verified on 3.0)** — asset loads & simulates under Isaac Sim 6.1; collision hulls present on all 7 bodies (incl. gripper/jaw, added by the URDF→USD converter), confirmed by a floor-penetration test |
| M3 | Asset inspection / start-pose settling | **Done** — 3.0 API migration complete; `inspect_asset.py` + 0.3 m drop settle test PASS (settles z≈0.05 m, drift 0.02 m, no NaNs) |
| M4 | Manager-based env + reward/obs/termination wiring | **Done** — all 16 reward terms, obs, terminations, events, curriculum run end-to-end on 3.0 (verified via training smoke + 20 s env runs) |
| M5 | Train PPO policy | **Done** — 3500 iters total (500 + 3000 resumed) @ 4096 envs on RTX 5090, ~1.6 s/iter. Final eval (100 envs, 20 s, friction 0.8): **mean displacement 0.255 m (1.27 cm/s)**, std 6.3 cm, 0 early terminations. Reward curves: `zombie_crawl/reward_curves.png`. §7.2 scripted-crawl gate **skipped by owner decision** (2026-10-02) |
| M6 | Play/evaluate + video + README | **In progress** — `play.py` (GUI: `--viz kit`, confirmed working) + `evaluate.py` verified; **headless `--video` root-caused & fixed** (broken path = default headless Kit-visualizer viewport source; fix = per-env scene `CameraCfg` sensors + `VideoRecorderCfg(source="sensor:...")`, 2 views; §7.3) and **all 7 checkpoints re-rendered**: 14 unique clips (side+top), each 20.000 s / 1000 frames / 1280×720@50fps, per-clip frame-diff confirms continuous motion (0 duplicate md5s); README pending |

---

## 3. Software Installed & Locations

### 3.1 Tooling (host)
| Software | Version | Location |
|----------|---------|----------|
| `uv` (system) | 0.8.4 | `~/.local/bin/uv` — **incompatible with Isaac Lab 3.0** (can't parse `[tool.uv.extra-build-variables]`; drops `[tool.uv]` so `prerelease="allow"` is ignored). Kept only as fallback. |
| `uv` (local, fallback) | 0.12.22 | `<workspace>/.tools/uv` |
| Miniconda | 26.1.1 | `<miniconda3>` (base Python 3.13.5; used only to **create** the project env) |

### 3.2 Source checkouts (git)
| Repo | Ref | Location |
|------|-----|----------|
| NVIDIA Isaac Lab | branch `release/3.0.0`, commit `ad728bdac` = `v3.0.0-EA-208-gad728bdac` | `<workspace>/IsaacLab3` (plain clone) |
| SO-ARM100 (LeRobot) | commit `5f6d2b8` | `<workspace>/third_party/SO-ARM100` |

Note: for fresh setups do **not** follow this section — use the README "Setup" flow
(plain pip/conda, no Isaac Lab source checkout needed):
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130` then
`pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com`.
Official docs: <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html>
(Isaac Sim), <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html>
(Isaac Lab).

### 3.3 Python virtual environment (active)
| Item | Detail |
|------|--------|
| Env path | `<workspace>/.venv` (**conda prefix env**, ~24 GB) |
| Python | 3.12.15 (required by Isaac Sim 6.1) |
| Created | `conda create -y -p <workspace>/.venv python=3.12 pip` |
| Activate | No `bin/activate`; use: `export CONDA_PREFIX=<workspace>/.venv; export PATH="$CONDA_PREFIX/bin:$PATH"` |
| Why conda, not venv | The PATH `python3.12` is a uv-managed build with `sys.base_prefix=/install` → `python3.12 -m venv` produces a broken venv (`ModuleNotFoundError: No module named 'encodings'`). No `/usr/bin/python3.12`. |

### 3.4 Packages installed in `.venv` (active stack)
| Package | Version | Notes / install source |
|---------|---------|------------------------|
| `isaacsim[all,extscache]` | 6.1.0.0 | per official 6.1.0 pip manual, `--extra-index-url https://pypi.nvidia.com` (includes 5.9 GB `isaacsim-extscache-kit`) |
| `isaaclab` (core) | 27.0.0 | editable → `IsaacLab3/source/isaaclab` |
| `isaaclab_ppisp` | 0.3.1 | editable |
| `isaaclab_contrib` | 3.1.1 | editable |
| `isaaclab_newton` | 9.1.1 | editable |
| `isaaclab_physx` | 7.3.2 | editable (provides the 3.0 `KitLauncher` / `PhysxCfg`) |
| `isaaclab_assets` | 2.0.0 | editable |
| `isaaclab_experimental` | 0.2.5 | editable |
| `isaaclab_ov` | 5.1.2 | editable (pulls `ovstage 0.2.0.377349`) |
| `isaaclab_tasks` | 22.0.0 | editable |
| `isaaclab_tasks_experimental` | 1.0.0 | editable |
| `isaaclab_rl` | 1.3.3 | editable (unified RL entrypoints: `scripts/reinforcement_learning/train.py`) |
| `isaaclab-visualizers` | 2.0.0 | editable |
| `rsl-rl-lib` (rsl_rl) | 5.5.1 | RL framework for PPO (3.0 core dependency, installed via installer token `rl[rsl-rl]`) |
| `torch` | 2.11.0+cu130 | `https://download.pytorch.org/whl/cu130` — the exact pin of the Isaac Sim 6.1.0 manual and the `v3.0.0-EA` tag (see I-4) |
| `torchvision` | 0.26.0+cu130 | cu130 index |
| `torchaudio` | 2.11.0+cu130 | cu130 index (no cu130 build exists for 2.12.0 — I-4) |
| `newton` | 1.6.1rc1 | physics backend (3.0 pin `release-1.6`) |
| `warp` | 1.17.0 | |
| `mujoco` | 3.12.0 | |
| `usd-exchange` | 3.0.0 | |
| `urdf-usd-converter` | 0.3.2 | M2 URDF→USD |
| `opencv-python-headless-noffmpeg` | 4.14.0.94rc1 | pre-release (needs `prerelease="allow"`) |
| `numpy` | 2.3.1 | |
| `gymnasium` | 1.3.0 | env registration |
| `onnx` / `onnxscript` | 1.23.1 / 0.7.2 | policy export |
| `triton` | 3.6.0 | |
| `setuptools` | 78.1.0 | |

### 3.5 Non-Python state
| Item | Location |
|------|----------|
| Isaac Sim EULA acceptance (persistent) | `~/.local/share/ov/...`; **all non-interactive runs must still set `OMNI_KIT_ACCEPT_EULA=YES`** |
| uv / pip caches (redirected to keep the tree clean) | `<workspace>/.cache/uv`, `<workspace>/.cache/pip` |
| **All install/verify logs (user rule: local folder only)** | `<workspace>/.install/` — incl. `pip-isaaclab3-*.log` (Isaac Sim 6.1), `isaaclab3-install-*.log` (Lab 3.0 submodules + rsl-rl), `torch-downgrade-*.log`, `verify-create_empty-*.log`, `verify-ant-train-*.log`, `headless_check.py` + `headless-check.log` |
| Clean libxml2 2.12.10 (M2 workaround, still needed in 6.1) | `<workspace>/.libxml2/` (use via `LD_LIBRARY_PATH=.../.libxml2/lib`) |

### 3.6 GPU / driver (pre-existing, verified)
| Item | Detail |
|------|--------|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU (Blackwell), 24463 MiB |
| Driver | 595.91.07 |
| CUDA (driver) | 13.2 |
| `torch.cuda.is_available()` | True (device `cuda:0`) |

---

## 4. Work Done So Far

### zombie_crawl project (stack-independent, done)
- M0: Cloned Isaac Lab and SO-ARM100; produced URDF analysis at
  `zombie_crawl/docs/so_arm100_urdf_analysis.md`.
- API research (via subagent, completed): API reference covering the external-project
  template, `UrdfConverter`, manager-based task structure, PPO cfg,
  `ArticulationCfg`/actuator fields, friction randomization, video recording, and policy
  export.
- Wrote the full `zombie_crawl` project (assets cfg, env cfg + Play variant, all MDP
  modules, PPO cfg, task registration, all scripts, all tests). Every file passes
  `python3 -m py_compile`.
- Ran `prepare_urdf.py`: base link mass → 0.15 kg, gripper/jaw collision added, total
  robot 0.759 kg; produced `so_arm100_float.urdf` + copied 13 meshes.
- FK clearance analysis: identified start poses that put the jaw on the ground facing +X
  (pan≈1.2, lift≈0.8, elbow≈-0.2, wrist≈0.5). `DEFAULT_JOINT_POS` still needs updating.

### M1: environment setup & installation (2026-10-02/03)
- prompt.md line 29 requires the **latest stable Blackwell-capable release pair** =
  Isaac Sim 6.1.0.0 + Isaac Lab 3.0 (Python 3.12). User approved a fresh project-local
  env and the **official pip flow (no uv)** — verified against the online manuals:
  - Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html
  - Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- Created the env as a conda **prefix** env at `.venv` (venv route broken — I-1;
  conda creation log in `.install/`).
- Installed Isaac Sim 6.1.0.0 exactly per the manual (torch 2.11.0 first from the cu130
  index, then `isaacsim[all,extscache]==6.1.0.0` from pypi.nvidia.com). Verified torch
  CUDA on the RTX 5090, then verified a full headless Kit launch
  (`.install/headless_check.py` → launch OK, 3 updates, clean shutdown).
- Installed Isaac Lab 3.0 from the `release/3.0.0` clone via the CLI installer
  (`isaaclab.cli`, token `rl[rsl-rl]`): 12 core submodules editable + centralized deps +
  `rsl-rl-lib 5.5.1`. The installer's uv-detection trap was avoided by removing
  `~/.local/bin` from PATH so plain pip is used (I-2, I-3).
- Aligned the torch stack to 2.11.0/0.26.0/2.11.0 (+cu130) per the official manuals /
  `v3.0.0-EA` (the `release/3.0.0` branch's 2.12.0 pin conflicts with Isaac Sim 6.1.0.0
  and has no cu130 torchaudio — I-4).
- **Ran the manual's verification steps, both passed** (logs in `.install/`):
  1. `scripts/tutorials/00_sim/create_empty.py --kit_args --headless` → 150 s clean run.
  2. "Train a robot!": `scripts/reinforcement_learning/train.py --rl_library rsl_rl
     --task Isaac-Ant --num_envs 64 --max_iterations 2` → 2 PPO iterations, 17.9 s, no
     errors. (3.0 task name is `Isaac-Ant`, no `-v0`; headless via `HEADLESS=1` /
     `--kit_args --headless`.)
- Reconnoitered the 3.0 API surface for the `zombie_crawl` migration: migration guide
  `IsaacLab3/docs/source/migration/migrating_to_isaaclab_3-0.rst` (3461 lines), unified
  RL entrypoints (`isaaclab_rl.entrypoints`), `isaaclab_physx.app` KitLauncher,
  `PhysxCfg`/`PresetCfg` multi-backend simulation config.

### Design decisions (zombie_crawl)
- Floating base (fix_base=False), convex-hull collision, self-collision on.
- Actuator presets: `servo` (2.9 N·m, 6 rad/s, default) vs `ideal` (10 N·m); PD k=1.0/d=0.1.
- Friction: ground 0.8/0.6 with `friction_combine_mode="multiply"`; per-env robot friction
  multiplier (effective 0.5–1.2); sampled at startup, exposed to privileged obs.
- Curriculum: v_target 0.06→0.10 m/s + friction scale 1.5→1.0.
- EMA-smoothed joint-position actions (alpha=0.3); 6-DOF.
- Asymmetric actor-critic (privileged critic extras: base_lin_vel, base_height, ground_friction).
- Rewards: forward_velocity_exp (main) + progress/yaw/lateral/overspeed/vertical/height/
  orientation/action-rate/joint terms + alive; optional zombie-style terms off by default.
- PPO: 256/128/64 MLP (ELU), num_steps_per_env=24, empirical normalization,
  gamma 0.99, lam 0.95, entropy 0.005.
- Task IDs: `Zombie-Crawl-SO100-v0` (train), `Zombie-Crawl-SO100-Play-v0` (deterministic).

---

## 5. Issues Encountered & Resolutions

- **I-1 `python3.12 -m venv` produces a broken venv.** The PATH `python3.12` is a
  uv-managed build whose `sys.base_prefix` is the nonexistent `/install`; the created
  venv can't even import `encodings` (ensurepip and get-pip both die). No
  `/usr/bin/python3.12` exists. **Resolved:** create the project env as a conda
  **prefix** env (`conda create -p .venv python=3.12 pip`).
- **I-2 System uv 0.8.4 cannot parse the Isaac Lab 3.0 `pyproject.toml`.** It fails to
  parse `[tool.uv.extra-build-variables]` and drops the `[tool.uv]` table, so
  `prerelease="allow"` is silently ignored → resolution failure on the pre-release
  `opencv-python-headless-noffmpeg==4.14.0.94rc1`. **Resolved:** use the official pip
  flow; local uv 0.12.22 at `.tools/uv` kept only as fallback.
- **I-3 Isaac Lab installer silently prefers `uv pip`.** `get_pip_command()`
  (`cli/utils.py`) returns `["uv","pip"]` whenever `uv` is on PATH in a conda/venv
  env. **Resolved:** strip `~/.local/bin` from PATH for the installer call so plain pip
  is used.
- **I-4 torch pin conflict between `release/3.0.0` and Isaac Sim 6.1.0.0.** The branch
  pins torch 2.12.0/torchvision 0.27.0 (and its installer *uninstalled* torchaudio),
  while `isaacsim-core 6.1.0.0` hard-requires torch==2.11.0/torchvision==0.26.0/
  torchaudio==2.11.0 — and the cu130 (Blackwell) index has **no** 2.12.0 torchaudio.
  **Resolved:** followed the official manuals + `v3.0.0-EA` tag and aligned the env to
  torch 2.11.0+cu130 / torchvision 0.26.0+cu130 / torchaudio 2.11.0+cu130. Validated
  by the headless Ant PPO run (2 iterations, no errors).
- **I-5 3.0 API breakage in `zombie_crawl` (expected, to fix).** e.g.
  `crawl_env_cfg.py:373` `self.sim.physx.gpu_max_rigid_patch_count` — in 3.0
  `SimulationCfg` has no `.physx` (physics moves to `isaaclab_physx.physics.PhysxCfg`
  + `PresetCfg`). Full delta per the 3.0 migration guide. Plus the
  `mdp/events.py` `randomize_ground_friction` `len(None)` bug (startup events pass
  `env_ids=None`; fix: `if env_ids is None: env_ids = slice(None)`).
- **I-6 Isaac Sim EULA prompt in non-interactive runs.** Resolved by setting
  `OMNI_KIT_ACCEPT_EULA=YES` for every headless/background run (acceptance is also
  persisted on disk under `~/.local/share/ov/`).
- **I-7 `import isaaclab_tasks` fails with `No module named 'pxr'` in a bare shell.**
  **Expected** — `pxr` (OpenUSD) is only available inside the running Isaac Sim app,
  not in a plain Python interpreter.

---

## 5b. Ubuntu 26.04 Notes (Isaac Sim 6.1)

Isaac Sim 6.1 runs headless natively on this OS (verified: `create_empty.py` and a
full PPO training run both succeeded, see §4). Remaining 26.04-specific workarounds:

1. **libxml2** — `omni.kit.asset_converter` / URDF asset importer fail to import
   (`libxml2.so.2` missing; 26.04 ships only `.so.16`). Use
   `LD_LIBRARY_PATH=<workspace>/.libxml2/lib` for M2
   (URDF→USD).
2. **EULA** — every non-interactive run needs `OMNI_KIT_ACCEPT_EULA=YES`.
3. `zombie_crawl/scripts/env.sh` needs updating for the `.venv` activation (conda
   prefix: `CONDA_PREFIX` + PATH, no `bin/activate`).

**M6 video:** 6.1's `train.py`/`play.py` support `--video`, but by default it auto-injects
a headless **Kit-visualizer** and records its viewport render product — which does not
refresh per-step in pure headless and produced byte-identical (frozen) clips. Resolved:
the task config now adds per-env scene `CameraCfg` sensors and records from them
(`VideoRecorderCfg(source="sensor:...")`), which drives the headless RTX pipeline
(`rtx_sensors` → `physx_headless_video_pump`) and yields fresh frames. Full root cause +
fix in §7.3.

### Clean libxml2 build (this session)
- Source: libxml2 **2.12.10** (gnome.org), built with system gcc 15.2, `--without-iconv`,
  **no ICU** (conda's libxml2 pulled ICU73 + conda libstdc++ → heap crash; rejected).
- Installed to: `<workspace>/.libxml2/` (prefix).
- Runtime deps: only system `libz`/`libm`/`libc` (verified via `ldd`).
- Build tree (scratch): `/tmp/opencode/libxml2-2.12.10/`.

---

## 6. Run Commands

### 6.1 Watch learning live (GUI)

Opens the Omniverse Kit viewport (shows env_0) while PPO trains. In 3.0 the launcher
forces headless **unless** `--viz kit` is passed (`kit_launcher.py`:
"Running headless because '--viz kit' was not requested.").

```bash
cd <workspace>/IsaacLab3
export CONDA_PREFIX=<workspace>/.venv
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=0
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Ant --num_envs 64 --max_iterations 100 --viz kit
```

- `--num_envs 64`: fewer parallel envs → faster PPO iterations, so learning is visible quickly.
- `--max_iterations N`: stop training after N PPO iterations (omit to run until Ctrl+C).
- Only env_0 is rendered regardless of `--num_envs`.

### 6.2 Headless training (no window, for the real run)

```bash
cd <workspace>/IsaacLab3
export CONDA_PREFIX=<workspace>/.venv
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Ant --num_envs 4096 --max_iterations 1500
```

### 6.3 Zombie crawl — full training run with live video

The real run trains the crawler task and records a **single world-space overview
camera** of the whole scene (several robots visible at once) — **not** a per-env
camera, which OOMs at 4096 envs.

```bash
cd <workspace>/zombie_crawl
source scripts/env.sh                       # .venv + OMNI_KIT_ACCEPT_EULA + libxml2
HEADLESS=1 python scripts/train.py \
    --task Zombie-Crawl-SO100-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --video
```

**Video flags** (added by the unified 3.0 train entrypoint; `--video` is what turns
recording on):

- `--video` — enable live training video. Adds a headless world-space
  `KitVisualizerCfg(origin_type="world")` framing the env-grid centre at a ~3/4 angle,
  and a `VideoRecorderCfg(source="visualizer:kit")` that writes
  `<run_dir>/videos/train/train_overview_*.mp4`.
- `--video_length N` — clip length in **env steps** (default `500` = 10 s at 50 Hz).
  Omit to use the task-config default.
- `--video_interval N` — env steps between new clips (default `600` = every 25 PPO
  iterations, since `num_steps_per_env` = 24). Omit to use the task-config default.

**Output:** 3840×2160 (4K) @ 30 fps, one new ~10 s clip every 25 iterations, in
`logs/rsl_rl/zombie_crawl/<timestamp>/videos/train/train_overview_*.mp4`
(`keep_last_n_clips=10` caps disk use). Resolution/interval defaults live in
`CrawlEnvCfg` (`training_video_width/height`, `training_video_interval_iters`,
`training_video_length`); `--video_length`/`--video_interval` override the length and
interval only — `fps`, `step_offset`, `source` and the 4K size come from the config.

**Why the frames are fresh (not the old frozen-viewport bug):** a world-space
`KitVisualizerCfg` creates no `Camera` sensor, so the `rtx_sensors` /
`fabricUpdateTransformations` settings a sensor would normally set are never flipped,
`is_rendering` stays `False`, and the viewport renders once. The task's
`CrawlRLEnv(ManagerBasedRLEnv)` subclass forces both settings `True` when training video
is on, so the env renders every step with the current physics. See
`zombie_crawl/docs/video_fix.md` (the “Training video” section).

**Cost:** one 4K viewport render per step, only during each ~10 s clip window
(500 of every 600 env steps). The viewer is never cloned per env, so cost is
independent of `--num_envs`. For a high-res video of a *finished* policy, use
`scripts/play.py --video` on a checkpoint instead.

### 6.4 Print the joint limits the action term uses

`EMAJointPositionToLimitsAction` rescales raw actions in [-1, +1] onto each joint's
**soft** position limits, read live from `asset.data.soft_joint_pos_limits` every step.
This script boots the task (headless, ~2 min startup) and prints exactly the
raw -1 / raw +1 endpoints per joint, with the hard limits and default pose for
reference:

```bash
cd <workspace>/zombie_crawl
source scripts/env.sh                       # .venv + OMNI_KIT_ACCEPT_EULA + libxml2
python scripts/print_action_limits.py [--preset servo]
```

```
Action term: EMAJointPositionToLimitsAction
Raw action [-1, +1] maps onto the SOFT limits below, per joint:

joint               raw -1    raw +1  range rad  range deg |   hard lo   hard hi   default
------------------------------------------------------------------------------------------
shoulder_pan       -1.9000    1.9000     3.8000      217.7 |   -2.0000    2.0000    0.0000
shoulder_lift       0.0875    3.4125     3.3250      190.5 |    0.0000    3.5000    0.8000
elbow_flex        -3.0630   -0.0785     2.9845      171.0 |   -3.1416    0.0000   -1.2000
wrist_flex        -2.4075    1.1075     3.5150      201.4 |   -2.5000    1.2000    0.5000
wrist_roll        -2.9845    2.9845     5.9690      342.0 |   -3.1416    3.1416    0.0000
gripper           -0.1450    1.9450     2.0900      119.7 |   -0.2000    2.0000    0.3000

soft_joint_pos_limit_factor = 0.95
```

Soft limits = middle 95% of each joint's hard range (shrunk symmetrically about the
range midpoint). To change what the policy can command: edit the URDF/USD limits,
change `soft_joint_pos_limit_factor` in `so_arm100.py`, or write new limits at runtime
(`write_joint_position_limit_to_sim_index`) — the action term picks up any change on
the next step, and re-running this script shows the new endpoints.

A related diagnostic, `scripts/drive_joint_limits.py`, boots the same scene and
physically drives each joint from its soft max to its soft min through the real action
path, reporting the achieved range of motion per joint (useful to separate
torque/contact shortfalls from the commanded range):

```bash
python scripts/drive_joint_limits.py --viz kit [--preset servo] [--phase 3.0]
python scripts/drive_joint_limits.py --no-idle          # headless: exit after the summary
```

---

## 7. Current State & Next Steps

- **M1 DONE:** Isaac Sim 6.1.0.0 + Isaac Lab 3.0 (`release/3.0.0` @
  `ad728bdac`) + rsl-rl 5.5.1 installed in `.venv` exactly per the official pip
  manuals. Both documented verifications passed: `create_empty.py` headless, and
  `train.py --rl_library rsl_rl --task Isaac-Ant --num_envs 64 --max_iterations 2`
  (2 PPO iterations, 17.9 s, clean). Torch aligned to 2.11.0+cu130.
- **M2 DONE (verified on 3.0):** the existing USD asset loads and simulates under
  6.1. All 7 bodies carry collision geometry — the URDF→USD converter
  (`URDF USD Converter v0.3.2`) added `PhysicsMeshCollisionAPI`/`convexHull` twins
  for every link, **including `gripper`/`jaw`**, even though the source URDF omits
  them (confirmed three ways: USD payload inspection, live contact-sensor prim
  count = 7, and a floor-penetration test where a jaw teleported 3 cm below the
  floor was pushed back out by PhysX).
- **M3 DONE:** `zombie_crawl` fully migrated to the 3.0 API. `scripts/inspect_asset.py`
  + the 0.3 m zero-action drop **settle test PASS** (settles z≈0.05 m, xy drift
  0.02 m over 5 s, no NaNs).
- **M4 DONE:** manager-based env, all 16 reward terms, obs/termination managers,
  events and curriculum run end-to-end on 3.0 (verified by the training smoke run
  and full 20 s env executions).
- **§7.2 scripted-crawl feasibility gate: SKIPPED by owner decision (2026-10-02).**
  For the record, the open-loop sinusoid test runs cleanly but does not reach the
  0.2 m / 20 s bar: with the servo preset the arm's distal links never plant a
  sustained horizontal pull (measured sustained v_x ≈ 0.16 cm/s best-case vs the
  1 cm/s target; Fx ≈ 0 N throughout), and the gripper/jaw, though collidable, stay
  14–31 cm above the floor in all three start poses.
- **M5 IN PROGRESS:** full PPO run launched 2026-10-02 23:11 (headless, 4096 envs,
  3000 iters, ~1.9 s/iter on RTX 5090, checkpoint every 50 iters →
  `logs/rsl_rl/zombie_crawl/2026-10-02_23-11-09/`).
- **M6 IN PROGRESS:** `scripts/play.py` (unified 3.0 play entrypoint; runs until
  Ctrl+C unless `--video` bounds the clip) and `scripts/evaluate.py` (100-env
  displacement/speed/yaw report + friction sweep) both verified against early
  checkpoints. Headless `--video` **root-caused & fixed** (was byte-identical frozen
  frames): the default headless source records the injected Kit-visualizer viewport,
  whose render product does not refresh per-step headless. The task config now adds
  per-env scene `CameraCfg` sensors and records from them via
  `VideoRecorderCfg(source="sensor:...")` (2 views: side + top). Verified on a 30-frame
  smoke run (real per-step motion, both views) and a full 20 s / 1000-frame 720p@50fps
  render of `model_500` (side + top; frame-diff shows continuous motion across all 20 s).
  See §7.3. `render_progression.py` re-rendered all 7 checkpoints (500…3499) with the new
  pipeline: **14 clips, all unique (0 duplicate md5s), each 20.000 s / 1000 frames /
  1280×720@50fps**, per-clip 5 s-sample frame-diff shows continuous motion in every clip.
- **Next:** README with pinned versions (prompt.md line 29). (M5 policy quality —
  `model_3499` measured +0.098 m / 10 s ≈ 1 cm/s, actions saturating — is a separate,
  owner-tracked item.)

### 7.1 3.0 API migration notes (what actually broke, 2026-10-02/03)

1. `gym.make(...)` returns a gymnasium wrapper — scripts use `env.unwrapped`;
   `reset()` is mandatory before `step()`.
2. `data.*` properties are `ProxyArray` — resolve with `.torch` before tensor ops or
   warp-declared math functions (e.g. `euler_xyz_from_quat`).
3. `write_root_pose_to_sim_index(root_pose=..., env_ids=...)` etc. are **keyword-only
   with renamed params**; whole-env resets pass `slice(None)` as `env_ids`.
4. `math_utils.sample_uniform(lower, upper, size, device)` is the 4-arg 3.0 form.
5. Limits layout is `(num_envs, num_joints, 2)` = [low, high]; `env_origins` is a CPU
   tensor on the no-terrain path.
6. Event-term signatures must be exactly `(env, env_ids[, named params])`: 3.0 injects
   defaulted args into `term_cfg.params` (colliding with the positional `env_ids`) and
   its static check treats `**kwargs` as a required param.
7. **Critical:** importing the concrete `ManagerBasedEnv` class before `SimulationApp`
   drags in `pxr` (via `isaaclab.assets.visual_material`), which breaks Kit extension
   loading (`omni.UsdMdl` → `TfNotice` wrapper error, then `omni.usd.get_context`
   missing). All mdp modules now import it under `TYPE_CHECKING` only, so
   `import zombie_crawl.tasks` is pxr-free and the unified train/play entrypoints
   work.
 8. `evaluate.py` needs `handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())`
    before `create_rsl_rl_runner` (strips deprecated `stochastic`/`init_noise_std`).

### 7.3 Headless `--video` root cause & fix (2026-10-03)

**Symptom.** Every clip in `logs/.../2026-10-02_23-35-11/videos/play/` (7 clips, one per
checkpoint) was **byte-identical** (md5 `a54011ef…`, 417 500 B, 1280×720@50fps, 1000
frames) even though the checkpoints differ (13 actor tensors differ). Frame-diff showed
a single change at frame ~50 then only tiny jolts at 250/500/750 — i.e. a **frozen
viewport**, not per-step motion.

**Root cause.** With `--video` but no `--viz`, the unified 3.0 entrypoint
(`pre_launch_video_config` / `apply_video_recording`, `isaaclab_rl/entrypoints/common.py`)
auto-injects a headless `KitVisualizerCfg` and the `VideoRecorderCfg` records from the
**viewport render product** (`KitVisualizer.render_rgb_array`). In a pure headless run that
render product is not re-driven per env step, so the camera keeps returning the last
viewport frame. Play is fully deterministic (fixed seed, single "folded" start pose, zero
noise/offset/yaw, friction 1.0), so a stale viewport + same start pose → byte-identical
clips.

**Why the sensor path works.** A scene `CameraCfg` with `renderer_type="isaac_rtx"` sets
`/isaaclab/render/rtx_sensors=True` → `sim.is_rendering=True` → per-step `render()` → the
`physx_headless_video_pump` callback (`isaaclab_physx/physics/physx_manager.py`) forces an
RTX render update every step. This is the same pipeline SDG relies on, and it is known
good headless. The `VideoRecorderCfg(source="sensor:<name>")` reads
`sensor.data.output[0]` after each step's render.

**Fix (task-local, no IsaacLab edits).**
- New `video_flags.py` (process-local `sensor_video_enabled` bool). The unified entrypoint
  builds the env config internally, so `play.py`/`train.py` set the flag from `--video`
  before dispatch (a Hydra flag can't be threaded through).
- `CrawlEnvCfg` gains `record_video`/`video_width`/`video_height`; when the flag (or the
  field) is set, `_setup_video_recording()` adds two **per-env** `CameraCfg` sensors
  (`{ENV_REGEX_NS}/video_camera` side 3/4 view, `…/video_camera_top` near-plumb top-down,
  poses from `create_rotation_matrix_from_view`/`quat_from_matrix`, env_0 framed) and two
  `VideoRecorderCfg(source="sensor:...")` streams (`side_…`, `top_…`). Because recorders
  are declared, `pre_launch_video_config` **early-returns** and never injects the broken
  Kit-visualizer source.
- 3.0's `Camera` sensor requires **one camera prim per env**, so the cameras are cloned
  per env (each frames its own robot; recorders read env_0) and the play path renders a
  **single env** when video is active (`play_mode` + `CrawlPlayEnvCfg` set `num_envs=1`).
  This also makes rendering cheap.
- `render_progression.py` updated: drops `--num_envs 8` (it would override the single-env
  gate) and skips only when **both** `side_…` and `top_…` clips already exist (old
  `clip_model_*` glob no longer matches). The 7 stale byte-identical clips were moved to
  `videos/play/_stale_kit_visualizer/` (kept for reference, not deleted).

**Verification.** 30-frame headless smoke run → both streams wrote fresh frames
(frame-diff 4–9k changed px between samples). Full `model_500` 20 s render: 1000 frames,
20.000 s, 1280×720@50fps for **both** side and top; frame-diff between successive 5 s
samples shows 9.6k–23k changed px (continuous motion, no frozen segments). Side view has
the robot centered and moving across the frame; top view is a clean near-plumb overhead.
All 7 checkpoints then re-rendered with the new pipeline: 14 unique clips (side+top each),
each 20.000 s / 1000 frames / 1280×720@50fps, 5 s-sample frame-diff confirms continuous
motion in every clip, 0 duplicate md5s.

**Open / separate.** The policy itself is still weak (M5): `model_3499` moves +0.098 m in
10 s (~1 cm/s) and its raw actions saturate (max |a| ≈ 18 → clipped to ±1). That is a
training/curriculum question, independent of this recording fix.

**Standalone write-up:** `zombie_crawl/docs/video_fix.md`.
