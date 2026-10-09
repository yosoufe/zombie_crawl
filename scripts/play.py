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
"""

import os
import sys

import zombie_crawl.tasks  # noqa: F401  # register the task with the gym registry
from zombie_crawl.tasks.manager_based.crawl import video_flags

from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    # the unified entrypoint builds the env config internally, so --video is surfaced to the
    # task config through the process-local flag: it adds the scene cameras and the
    # sensor-based video recorders (fresh headless RTX frames) instead of the broken
    # headless Kit-visualizer viewport source
    if "--video" in argv:
        video_flags.sensor_video_enabled = True
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
