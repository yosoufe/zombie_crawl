# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the zombie crawl task with RSL-RL.

Thin wrapper around the Isaac Lab 3.0 unified RL entrypoint
(``isaaclab_rl.entrypoints.run_train_cli``). The zombie_crawl task is registered by
importing the package before dispatch, so the standard CLI is used unchanged:

    python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 2048 --max_iterations 3000
    python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100 --headless
    python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100 --viz kit

Phase 2 command: the actor's single command input is a per-env yaw rate (rad/s)
resampled every ``env.commands.yaw_rate_cmd.resampling_time_range`` seconds (default
[5,8], per env, staggered). The yaw-rate STAGE curriculum (P2-12) climbs the command
range automatically: 0 (straight only) -> 0.10 rad/s at difficulty >= 0.75 -> +0.05 per
check up to ``env.yaw_rate_max_final`` (0.30 provisional) once ``steer_success >= 0.7``.
Useful Hydra overrides:

    env.yaw_rate_max_final=0.3        # provisional command cap [rad/s]
    env.yaw_rate_stage_lock=0         # R1: keep steering locked off (commands always 0)
    env.commands.yaw_rate_cmd.resampling_time_range=[5,8]
    env.use_box_termination=true      # diagnostic: re-add the removed position box (§A.11)

Watch in TensorBoard: ``Curriculum/crawl_metrics/*`` (crawl/recovery/steer_success,
back_frac, base_yaw_rate_abs, r_abs), ``Curriculum/yaw_rate_stage/*`` (dial + stage),
``Curriculum/difficulty/*``, and ``Metrics/yaw_rate_cmd/*``.

``--headless`` is accepted for compatibility with Isaac Lab 2.x command lines: the
unified 3.0 launcher CLI no longer registers the flag (headless is the default when
no visualizer is requested), and forwarding it as-is would crash Hydra's override
parser. This wrapper therefore strips it and forces ``HEADLESS=1`` in the environment,
which reproduces the old semantics, including a window-less Kit visualizer when
combined with ``--viz kit``.

``--video`` adds live training video: a SINGLE world-space viewer camera (a headless
``KitVisualizerCfg`` with ``origin_type="world"`` that frames the env-grid centre at a
natural ~3/4 angle, so several robots are visible at once regardless of ``num_envs``) and
a new ~10 s 3840x2160 (4K) clip is written every 25 PPO iterations into
``<run_dir>/videos/train/train_overview_*.mp4`` (see
``CrawlEnvCfg._setup_training_video``). Frames are fresh each step because the train
task's ``CrawlRLEnv`` forces ``rtx_sensors``/``fabricUpdateTransformations`` on — a
world-space viewer has no ``Camera`` sensor to set them, so without that the viewport
freezes (docs/video_fix.md).

COST: the viewer is never cloned per env, so ``--video`` costs one 4K viewport render
per step and only during each ~10 s clip window (500 of every 600 env steps) —
independent of ``num_envs`` (a per-env camera sensor OOMs at 4096). For a high-res
video of a finished policy use ``play.py --video`` on a checkpoint instead.
Tune with ``--video_length`` / ``--video_interval`` (env steps).
"""

import os
import sys

import zombie_crawl.tasks  # noqa: F401  # register the task with the gym registry
from zombie_crawl.tasks.manager_based.crawl import video_flags

from isaaclab_rl.entrypoints import run_train_cli

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    # --video_start_iters N: override the PPO iteration at which the first training clip is
    # captured (default 500, see CrawlEnvCfg.training_video_start_iters). Accepted in both
    # "--video_start_iters N" and "--video_start_iters=N" forms; consumed here (the unified
    # launcher forwards any leftover arg to Hydra, which would reject an unknown flag).
    for i, a in enumerate(argv):
        if a == "--video_start_iters" and i + 1 < len(argv):
            video_flags.training_video_start_iters = int(argv[i + 1])
            del argv[i : i + 2]
            break
        if a.startswith("--video_start_iters="):
            video_flags.training_video_start_iters = int(a.split("=", 1)[1])
            argv.pop(i)
            break
    # --video_length N: clip length in env steps (see CrawlEnvCfg.training_video_length);
    # consumed here for the same reason as --video_start_iters (Hydra has no `video` group).
    for i, a in enumerate(argv):
        if a == "--video_length" and i + 1 < len(argv):
            video_flags.training_video_length = int(argv[i + 1])
            del argv[i : i + 2]
            break
        if a.startswith("--video_length="):
            video_flags.training_video_length = int(a.split("=", 1)[1])
            argv.pop(i)
            break
    # --checkpoint model_N.pt: capture the resume offset so training-clip filenames can line up
    # with the TensorBoard iteration axis. rsl_rl CONTINUES logging from the checkpoint's stored
    # iter (N), but the video recorder rebuilds its step counter at 0, so the filename must add N.
    # The token is intentionally NOT stripped here -- the unified launcher still needs it to resume.
    import re as _re

    for i, a in enumerate(argv):
        if a == "--checkpoint" and i + 1 < len(argv):
            _ckpt = argv[i + 1]
        elif a.startswith("--checkpoint="):
            _ckpt = a.split("=", 1)[1]
        else:
            continue
        _m = _re.search(r"model_(\d+)\.pt$", os.path.basename(_ckpt))
        if _m:
            video_flags.training_video_start_offset = int(_m.group(1))
            print(
                f"[INFO] --video resume offset: TensorBoard it-axis starts at "
                f"{int(_m.group(1))} (from {os.path.basename(_ckpt)}); training-clip filenames "
                f"will be shifted to match."
            )
        break
    # see CrawlEnvCfg._setup_training_video: --video adds a single world-space viewer
    # camera + interval-based recorder via the process-local flag (fresh 4K clip every 25 iters)
    if "--video" in argv:
        video_flags.training_video_enabled = True
        print(
            "[INFO] --video: live training video enabled (single world-space viewer camera "
            "over the whole scene, ~3/4 view of the grid centre with several robots visible; "
            "one ~10 s 3840x2160 (4K) clip every 25 iters -> <run_dir>/videos/train/"
            "train_overview_*.mp4). Frames are fresh each step. Tune with --video_length / "
            "--video_interval (env steps)."
        )
        print(
            "[INFO] --video cost: one 4K viewport render per step (a single world-space "
            "viewer, not per-env), and only during each ~10 s clip window (500 of every "
            "600 env steps) - independent of num_envs. For a high-res video of a finished "
            "policy use play.py --video on a checkpoint."
        )
    # --viz kit -> live GUI overlay: the env drives the yaw-arc/disturbance overlay every
    # step so the GUI window shows it continuously, not just inside --video clip windows
    if video_flags.enable_gui_overlay_from_argv(argv):
        print(
            "[INFO] --viz kit: live vector overlay ON -- green arc = commanded yaw rate, "
            "magenta arc = actual filtered claw rate (full circle = 0.8 rad/s (0.3 = 3/8 circle), swept from "
            "the claw axis), red = current disturbance force per rigid body "
            "(updated every env step in the GUI window)."
        )
    # --headless compatibility: the unified Isaac Lab 3.0 launcher CLI no longer
    # registers the flag, so translate it to HEADLESS=1 (which the launcher respects)
    # and strip it before dispatch, where it would otherwise crash Hydra's parser.
    if "--headless" in argv:
        os.environ["HEADLESS"] = "1"
        argv = [a for a in argv if a != "--headless"]
        print(
            "[INFO] --headless: running headless. Isaac Lab 3.0 removed the --headless "
            "flag from the launcher CLI (headless is the default unless a visualizer is "
            "requested); it was translated to HEADLESS=1."
        )
    sys.exit(run_train_cli(argv))
