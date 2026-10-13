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

training_video_start_iters: int | None = None
"""When set by ``scripts/train.py`` from ``--video_start_iters N``, overrides
``CrawlEnvCfg.training_video_start_iters`` (the PPO iteration at which the first clip is
captured). Use ``--video_start_iters 1`` to record a clip immediately (e.g. to validate the
vector overlay on an untrained policy) instead of the default post-warm-up iteration.
"""
