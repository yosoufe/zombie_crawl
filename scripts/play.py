# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained zombie crawl checkpoint with RSL-RL.

Thin wrapper around the Isaac Lab 3.0 unified RL play entrypoint
(``isaaclab_rl.entrypoints.run_play_cli``). Playback applies the env's
``play_mode()`` overrides (deterministic reset, no curriculum) automatically:

    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --headless
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --video

``--headless`` is accepted for compatibility with Isaac Lab 2.x command lines: the
unified 3.0 launcher CLI no longer registers the flag (headless is the default when
no visualizer is requested), and forwarding it as-is would crash Hydra's override
parser. This wrapper therefore strips it and forces ``HEADLESS=1`` in the environment,
which reproduces the old semantics, including a window-less Kit visualizer when
combined with ``--viz kit``.

``--video`` records with the EXACT training-video spec (see
``CrawlEnvCfg._setup_training_video`` / ``_apply_training_video_play_spec``):
the SAME single world-space ~3/4 viewer camera over the grid centre, the SAME
4096-robot population, and the SAME disturbances at full training strength with the
50% fallen-start rate (push_scale 1.0, pose_weights 0.5/0.25/0.25). The only
difference to a training clip: recording starts at the SPAWN moment (one ~40 s
3840x2160 clip, no 200-iteration warm-up delay) ->
``<log_dir>/videos/play/play_overview_*.mp4``.
"""

import os
import sys

import zombie_crawl.tasks  # noqa: F401  # register the task with the gym registry
from zombie_crawl.tasks.manager_based.crawl import video_flags

from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    # the unified entrypoint builds the env config internally, so --video is surfaced to the
    # task config through process-local flags: training-spec playback -- the single
    # world-space viewer camera + 4K recorder of the training clips, 4096 robots, and the
    # full training disturbance/fallen-start population, recording from the spawn moment
    # (see CrawlEnvCfg._setup_training_video + _apply_training_video_play_spec)
    if "--video" in argv:
        video_flags.training_video_enabled = True
        video_flags.play_video_training_spec = True
        print(
            "[INFO] --video: recording with the training-video spec -- single world-space "
            "~3/4 viewer camera over the grid centre, 4096 robots, full disturbance + 50% "
            "fallen starts; one ~40 s 3840x2160 (4K) clip starting at the spawn moment "
            "-> <log_dir>/videos/play/play_overview_*.mp4."
        )
    # --headless compatibility: translate to HEADLESS=1 and strip before dispatch,
    # where it would otherwise crash Hydra's parser (see the module docstring).
    if "--headless" in argv:
        os.environ["HEADLESS"] = "1"
        argv = [a for a in argv if a != "--headless"]
        print(
            "[INFO] --headless: running headless. Isaac Lab 3.0 removed the --headless "
            "flag from the launcher CLI (headless is the default unless a visualizer is "
            "requested); it was translated to HEADLESS=1."
        )
    sys.exit(run_play_cli(argv))
