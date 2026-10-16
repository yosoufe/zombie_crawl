# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Process-local flags for video recording.

The unified RL entrypoints (``run_play_cli`` / ``run_train_cli``) build the env config
internally, so the scripts cannot pass a ``--video``-driven config flag through Hydra.
Instead, the scripts set the flags below before invoking the entrypoint;
``CrawlEnvCfg.__post_init__`` reads them and adds the matching cameras (per-env sensor
cameras for play, a single world-space visualizer for training) and ``VideoRecorderCfg``
entries.
"""

sensor_video_enabled: bool = False
"""When set by ``scripts/play.py`` from ``--video``, ``CrawlEnvCfg.__post_init__`` adds a
per-env camera pair (side + top) and sensor-based ``VideoRecorderCfg`` entries that record
one env for the whole episode (see :meth:`CrawlEnvCfg._setup_video_recording`).
"""

training_video_enabled: bool = False
"""When set by ``scripts/train.py`` from ``--video``, ``CrawlEnvCfg.__post_init__`` adds a
single headless Kit-visualizer camera (ONE viewport camera for the whole stage, many
robots per frame) plus an interval-based ``VideoRecorderCfg`` so a fresh clip of live
training robots is written every N PPO iterations (see
:meth:`CrawlEnvCfg._setup_training_video`). This is Isaac Lab's official training-video
path (how-to: "Recording video clips during training").
"""

play_video_training_spec: bool = False
"""When set by ``scripts/play.py`` from ``--video``, playback records with the EXACT
training-video spec: the single world-space viewer camera + 4K recorder of
:meth:`CrawlEnvCfg._setup_training_video` (instead of the per-env sensor camera pair),
``play_video_num_envs`` (4096) robots in the scene, and the training disturbances and
fallen-start rate kept at their full training values (push_scale=1.0, fallen 50%).
The only deviation from a training clip: recording starts at the spawn moment
(clip step-offset = RTX warm-up only, no 200-iteration delay).
"""

gui_overlay_enabled: bool = False
"""When set by ``scripts/play.py`` / ``scripts/train.py`` (from ``--viz kit``), the env drives
:class:`~...video_overlay.VectorArrowOverlay` (cyan commanded-direction + red per-body
disturbance arrows, drawn as stage ``VisualizationMarkers``) on EVERY env step, so the arrows
are visible in the live Kit GUI window even without ``--video``. While this flag is set the
recorders delegate to the env's single overlay instance instead of creating their own, so a
recording GUI run never double-draws (see :meth:`StreamingVideoRecorder._update_overlay`).
"""

play_disturbance_enabled: bool = False
"""When set by ``scripts/play.py`` from ``--disturbance``, playback KEEPS the live external
disturbance that deterministic ``play_mode()`` otherwise switches off:
``events.push_robot`` (``mdp.disturb_all_bodies``) keeps applying random force/torque
wrenches to every rigid body at full training strength. The red per-body arrows show the
instantaneous force when the GUI/recorder overlay is active (``--viz kit`` or ``--video``).
Default off -> playback stays deterministic (no pushes), exactly as before.
"""

play_dynamic_heading: bool = False
"""When set by ``scripts/play.py`` from ``--dynamic_heading``, playback keeps the
mid-episode resampled command heading ALIVE: the deterministic ``play_mode()`` path does
NOT pin ``crawl_heading.resampling_time_range`` to one-episode-length, so every env gets
a fresh +-180 deg crawl direction mid-episode (staggered every
``heading_resample_range`` seconds) exactly like a unlocked training run. The script also
appends the Hydra override ``env.use_dynamic_heading=true`` so the command term exists.
Without this flag, playback stays deterministic (one heading per episode).
"""

def enable_gui_overlay_from_argv(argv: list[str]) -> bool:
    """Set :data:`gui_overlay_enabled` when *argv* requests the Kit GUI visualizer.

    Recognizes every launcher spelling of ``--viz kit`` (``--viz kit``, ``--viz=kit``,
    comma combos like ``--viz kit,rerun``). Returns the resulting flag.
    """
    global gui_overlay_enabled
    values: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--viz" and i + 1 < len(argv):
            values += argv[i + 1].split(",")
            i += 2
            continue
        if arg.startswith("--viz="):
            values += arg.split("=", 1)[1].split(",")
        i += 1
    gui_overlay_enabled = any(v.strip() == "kit" for v in values)
    return gui_overlay_enabled


training_video_start_iters: int | None = None
"""When set by ``scripts/train.py`` from ``--video_start_iters N``, overrides
``CrawlEnvCfg.training_video_start_iters`` (the PPO iteration at which the first clip is
captured). Use ``--video_start_iters 1`` to record a clip immediately (e.g. to validate the
vector overlay on an untrained policy) instead of the default post-warm-up iteration.
"""

training_video_start_offset: int = 0
"""Absolute PPO iteration this run's TensorBoard axis starts at, set by ``scripts/train.py``.
0 for a from-scratch run; for a ``--checkpoint model_N.pt`` resume it is ``N`` (the checkpoint's
stored ``iter``). The video recorder's own step counter always restarts at 0 (it is rebuilt
when the env is created, before the runner loads the checkpoint), but rsl_rl *continues* logging
TensorBoard scalars from ``N``. So a clip recorded at recorder env-step ``S`` sits at TensorBoard
iteration ``training_video_start_offset + S / num_steps_per_env`` -- the offset is what lets the
clip filename line up with the matching point on the TensorBoard curve even after a resume.
"""
