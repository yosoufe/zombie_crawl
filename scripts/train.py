# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the zombie crawl task with RSL-RL.

Thin wrapper around the Isaac Lab 3.0 unified RL entrypoint
(``isaaclab_rl.entrypoints.run_train_cli``). The zombie_crawl task is registered by
importing the package before dispatch, so the standard CLI is used unchanged:

    python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 2048 --max_iterations 3000
    python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100 --viz kit

``--video`` adds live training video: a SINGLE world-space viewer camera (a headless
``KitVisualizerCfg`` with ``origin_type="world"`` that frames the env-grid centre at a
natural ~3/4 angle, so several robots are visible at once regardless of ``num_envs``) and
a new ~10 s 1280x720 clip is written every 100 PPO iterations into
``<run_dir>/videos/train/train_overview_*.mp4`` (see
``CrawlEnvCfg._setup_training_video``). Frames are fresh each step: the recorder pulls
each frame via ``render_rgb_array()``, which pumps the Kit app (docs/video_fix.md).

COST: the viewer is never cloned per env, so ``--video`` costs one 720p viewport render
per step and only during each ~10 s clip window (500 of every 2400 env steps) —
negligible at any ``num_envs`` (a per-env camera sensor OOMs at 4096). For high-res
videos of a finished policy use ``play.py --video`` on a checkpoint instead.
Tune with ``--video_length`` / ``--video_interval`` (env steps).
"""

import sys

import zombie_crawl.tasks  # noqa: F401  # register the task with the gym registry
from zombie_crawl.tasks.manager_based.crawl import video_flags

from isaaclab_rl.entrypoints import run_train_cli

if __name__ == "__main__":
    # see CrawlEnvCfg._setup_training_video: --video adds a single world-space viewer
    # camera + interval-based recorder via the process-local flag (fresh clip every 100 iters)
    if "--video" in sys.argv[1:]:
        video_flags.training_video_enabled = True
        print(
            "[INFO] --video: live training video enabled (single world-space viewer camera "
            "over the whole scene, ~3/4 view of the grid centre with several robots visible; "
            "one ~10 s 1280x720 clip every 100 iters -> <run_dir>/videos/train/"
            "train_overview_*.mp4). Frames are fresh each step. Tune with --video_length / "
            "--video_interval (env steps)."
        )
        print(
            "[INFO] --video cost: one 720p viewport render per step (a single world-space "
            "viewer, not per-env), and only during each ~10 s clip window (500 of every "
            "2400 env steps) - negligible at any num_envs. For high-res videos of a "
            "finished policy use play.py --video on a checkpoint."
        )
    sys.exit(run_train_cli(sys.argv[1:]))
