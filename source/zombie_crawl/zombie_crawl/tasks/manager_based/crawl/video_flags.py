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
