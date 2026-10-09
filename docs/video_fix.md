# Headless Video Recording Fix (static/frozen mp4 → live per-step frames)

## TL;DR

Two recording paths, one shared root cause (the headless Kit viewport is static):

- **Play/eval video** (`play.py --video`, without `--viz`): recorded from the headless
  **Kit viewport** render product, which is not re-driven per env step in a pure headless
  run → every clip was a frozen, byte-identical image. The fix records from **scene
  camera sensors** (`VideoRecorderCfg(source="sensor:...")`), which drive the proven
  headless RTX render pipeline and produce a fresh frame every step.
- **Live training video** (`train.py --video`): a per-env overview camera sensor OOMs at
  4096 envs (RTX tiles all per-env camera renders into one giant texture — `Failed to
  allocate 40960x23040 LdrColor resource`). The fix records from a **single world-space
  `KitVisualizerCfg(origin_type="world", headless=True)`**
  (`VideoRecorderCfg(source="visualizer:kit")`): one 1280x720 viewport over the whole
  scene, never cloned per env, fresh frame per captured step.

No IsaacLab edits; everything is in the `zombie_crawl` task config + scripts.

## Symptom & evidence

- All 7 clips in `logs/rsl_rl/zombie_crawl/2026-10-02_23-35-11/videos/play/`
  (`clip_model_{500…3499}_0000.mp4`) were **byte-identical**: same md5
  (`a54011ef7a1b62c5d490248d593381fa`), 417,500 B, 1280×720 @ 50 fps, 1000 frames —
  despite the checkpoints genuinely differing (13 actor tensors differ between them).
- Frame diff: one large change at ~frame 50, then only tiny jolts at 250/500/750 — a
  stale viewport, not per-step motion. The yellow robot's centroid shifted only ~7.6 px
  over the whole 20 s.
- Play mode is fully deterministic (fixed seed, single "folded" start pose, zero
  noise/offset/yaw, friction 1.0) → same start pose + same stale frame = byte-identical
  clips for every checkpoint.

## Root cause

- With `--video` but no `--viz`, the unified Isaac Lab 3.0 entrypoint
  (`pre_launch_video_config` / `apply_video_recording` in
  `isaaclab_rl/entrypoints/common.py`) auto-injects a headless `KitVisualizerCfg`. The
  recorder's default source (`visualizer`) grabs the **viewport render product**
  (`KitVisualizer.render_rgb_array`).
- In a pure headless run that render product is not re-rendered per env step, so the
  recorder keeps capturing the last viewport frame. The clip therefore shows the robot
  in (nearly) its start pose for all 1000 frames.

## Why the scene-sensor path works

- A `CameraCfg` with the `isaac_rtx` renderer sets `/isaaclab/render/rtx_sensors=True`
  → `sim.is_rendering=True` → `sim.render()` is called every env step → the
  `physx_headless_video_pump` callback
  (`isaaclab_physx/physics/physx_manager.py`) forces a real RTX render update.
- This is the same pipeline synthetic data generation (SDG) relies on and is
  known-good headless.
- `VideoRecorderCfg(source="sensor:<name>")` reads `sensor.data.output[0]` (the freshly
  rendered RGB) after each step's render, so every recorded frame is current.
- This is the right tool for **play/eval** (one env, deterministic). It does not scale to
  a whole training run — per-env cameras are `num_envs x resolution` of RTX work and OOM
  at 4096 — which is why the training video uses a world-space viewer instead (below).

## The fix (task-local — no IsaacLab edits)

1. **Switch the frame source.** Two `CameraCfg` sensors in the scene — a side 3/4 view
   and a near-plumb top-down view (look-at poses computed via
   `create_rotation_matrix_from_view` / `quat_from_matrix`, verified exact) — and two
   `VideoRecorderCfg(source="sensor:...")` streams producing `side_…` / `top_…` clips
   (the two angles requested by the prompt).
2. **Bypass the broken path.** Because the env config *declares* its own
   `video_recorders`, `pre_launch_video_config` early-returns and never injects the
   headless Kit visualizer, so the frozen viewport source is simply never used.
3. **Per-env cameras + single env for play.** Isaac Lab 3.0's `Camera` sensor requires
   one camera prim per environment, so the cameras are cloned per env (each frames its
   own robot; the recorders read env_0). The play/eval path renders **one env** when
   video is active — deterministic and cheap (~80 s per 20 s clip).
4. **Flag gating.** The unified entrypoint builds the env config internally, so a
   Hydra flag can't thread `--video` through. Instead `play.py` / `train.py` set a
   process-local flag (`zombie_crawl/.../crawl/video_flags.py`) before dispatch, and
   `CrawlEnvCfg.__post_init__` reads it. Normal training/play pays no camera cost.

## Training video (`train.py --video`): world-space visualizer (per-env cameras OOM at 4096)

The live training video first reused the play path's pattern: a per-env top-down
`CameraCfg` overview sensor (`{ENV_REGEX_NS}/overview_cam`) recorded via
`VideoRecorderCfg(source="sensor:overview_cam")`. At scale that is fatal:

- RTX renders every per-env camera and tiles all of them into a single texture, so the
  cost is `num_envs x resolution`. At 4096 envs the tiled texture could not be
  allocated — `Failed to allocate 40960x23040 LdrColor resource` (out of GPU memory).
  The approach only works at small `num_envs`, which is exactly the setting production
  training (2048/4096 envs) does not use.

The current approach instead records from ONE world-space viewer
(`CrawlEnvCfg._setup_training_video()`):

1. **One camera for the whole scene.** A single
   `KitVisualizerCfg(eye=(0, -7.0, 4.5), lookat=(0, 0, 0.1), origin_type="world",
   headless=True, window_width=1280, window_height=720)` is added to `sim.visualizer_cfgs`,
   and the recorder uses
   `VideoRecorderCfg(source="visualizer:kit", video_length=500, video_interval=100 * 24,
   output_filename_prefix="train_overview")` →
   `<run_dir>/videos/train/train_overview_*.mp4`.
2. **Never cloned per env.** The cloner only clones prims whose path matches the
   `{ENV_REGEX_NS}` env template (`InteractiveScene._collect_asset_cfgs` checks
   `cloner.path.match(prim_path, env_fmt)` before adding a prim to the clone plan), and
   the visualizer's camera is a single world-space prim outside that template. So there
   is exactly ONE camera no matter how many envs exist — no per-env tiling, no OOM, cost
   independent of `num_envs`.
3. **Fresh frames.** The root cause above (headless viewport static) still applies to the
   *viewport*, but the recorder does not read the viewport: `source="visualizer:kit"`
   pulls each frame via `KitVisualizer.render_rgb_array()`, which refreshes the Fabric
   transforms, pumps `omni.kit.app.get_app().update()`, and (headless) resumes the
   Replicator render product before reading the RGB array — so every captured frame
   reflects the current physics state.
4. **Framing.** The view is anchored on the grid centre (world origin, where the env grid
   is centred) at a natural ~3/4 angle (`training_video_cam_height=4.5 m` above,
   `training_video_cam_distance=7.0 m` out → ~33 deg down from horizontal), so several
   robots are in frame at ~`env_spacing` regardless of the total env count.
5. **Cost.** Recording is interval-driven: a new ~10 s clip every
   `training_video_interval_iters` (100) PPO iterations (100 x 24 = 2400 env steps),
   each clip `training_video_length` (500) env steps. That is one 1280x720 viewport
   render per step, but only during each ~10 s clip window — 500 of every 2400 env steps
   — negligible at any `num_envs`. For high-res videos of a finished policy use
   `play.py --video` on a checkpoint.

## Files touched

| File | Change |
|------|--------|
| `source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/video_flags.py` | New: process-local `sensor_video_enabled` (play) + `training_video_enabled` (train) flags |
| `source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/crawl_env_cfg.py` | `record_video` / `video_width` / `video_height` fields; `_look_at_pose()` helper; `_setup_video_recording()` (2 per-env cameras + 2 sensor recorders, play path); `_setup_training_video()` (single world-space `KitVisualizerCfg` + interval-based `visualizer:kit` recorder, train path); single-env gate in `play_mode()` and `CrawlPlayEnvCfg` |
| `scripts/play.py` | Sets the flag when `--video` is in argv |
| `scripts/train.py` | Same; prints a cost note (one 720p viewport render per step, only during each ~10 s clip window) |
| `scripts/render_progression.py` | Dropped `--num_envs 8` (it would override the single-env gate); skip-glob now matches the `side_/top_` clip names and requires both views present |

## Verification

- 30-frame headless smoke run (`--video_length 30`): both streams wrote fresh frames
  (4–9k changed pixels between samples).
- Full re-render of all 7 checkpoints (500…3499): **14 unique clips** (0 duplicate
  md5s), each **20.000 s / 1000 frames / 1280×720 @ 50 fps**; per-clip 5 s-sample
  frame-diff shows continuous motion in every clip (9.6k–27k changed px per 5 s sample).
- Side view: robot centered, crawling across the frame (+X). Top view: clean
  near-plumb overhead with no roll.
- Clips: `logs/rsl_rl/zombie_crawl/2026-10-02_23-35-11/videos/play/`
  (`side_model_<iter>_0000.mp4`, `top_model_<iter>_0000.mp4`). The stale frozen clips
  are preserved in `videos/play/_stale_kit_visualizer/` for reference.

## Known limitation (separate item)

Policy quality (M5), independent of recording: `model_3499` crawls +0.098 m over 10 s
(~1 cm/s vs the 10 cm/s target) and its raw actions saturate (max |a| ≈ 18, clipped to
±1). That is a training/curriculum question, not a recording bug.
