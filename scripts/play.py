# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained zombie crawl checkpoint with RSL-RL.

Thin wrapper around the Isaac Lab 3.0 unified RL play entrypoint
(``isaaclab_rl.entrypoints.run_play_cli``). Playback applies the env's
``play_mode()`` overrides (deterministic reset, no curriculum) automatically:

    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --video
"""

import sys

import zombie_crawl.tasks  # noqa: F401  # register the task with the gym registry
from zombie_crawl.tasks.manager_based.crawl import video_flags

from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    # the unified entrypoint builds the env config internally, so --video is surfaced to the
    # task config through the process-local flag: it adds the scene cameras and the
    # sensor-based video recorders (fresh headless RTX frames) instead of the broken
    # headless Kit-visualizer viewport source
    if "--video" in sys.argv[1:]:
        video_flags.sensor_video_enabled = True
    sys.exit(run_play_cli(sys.argv[1:]))
