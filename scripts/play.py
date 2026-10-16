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
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --dynamic_heading
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --viz kit
    python scripts/play.py --task Zombie-Crawl-SO100-v0 --checkpoint latest --disturbance

``--disturbance`` (alias ``--disturbances``) re-enables the live external disturbance that
deterministic playback disables: random force/torque wrenches on EVERY rigid body at full
training strength (the same ``events.push_robot`` a training run uses; the red overlay
arrows track it). Default playback has no pushes. Combine with ``--dynamic_heading`` for
the full robustness demo (pushes + jumping command direction).

The Isaac Sim GUI window requires an explicit visualizer (``--viz kit``): the Isaac Lab 3.0
launcher is headless BY DEFAULT, so omitting ``--headless`` alone opens no window. Keep
``--num_envs`` small for viewport framerate; combine ``--video --viz kit`` to record while
watching.

``--dynamic_heading`` plays with the VARIABLE target direction: each robot's commanded
crawl heading is redrawn +-180 deg mid-episode every ``heading_resample_range`` seconds
(6-12 s default, per env, staggered) -- the cyan overlay arrow jumps and the policy must
re-aim. The play-mode curriculum is frozen, so the command gate counts as fully unlocked
and resampling is live from the spawn moment. Default playback stays deterministic (one
heading per episode). Combine freely with ``--video`` (training-spec clip) / ``--headless``.

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
    # --viz kit -> the env drives the cyan/red vector-arrow overlay every step so the
    # arrows are visible in the live GUI window, with or without --video (recorders then
    # delegate to the env's single overlay; see video_flags.gui_overlay_enabled)
    if video_flags.enable_gui_overlay_from_argv(argv):
        print(
            "[INFO] --viz kit: live vector-arrow overlay ON -- cyan = commanded crawl "
            "direction per robot (jumps mid-episode with --dynamic_heading), red = current "
            "disturbance force per rigid body. Updated every env step in the GUI window."
        )
    # --dynamic_heading: variable (mid-episode resampled) target direction during
    # playback. Translate to the Hydra env override so the crawl_heading command term
    # is built, plus the process-local flag that stops play_mode() from pinning the
    # resampling to one heading per episode (see video_flags.play_dynamic_heading).
    if "--dynamic_heading" in argv:
        video_flags.play_dynamic_heading = True
        argv = [a for a in argv if a != "--dynamic_heading"]
        argv.append("env.use_dynamic_heading=true")
        print(
            "[INFO] --dynamic_heading: playback uses the variable target direction -- "
            "each robot keeps its commanded crawl heading for heading_resample_range "
            "seconds (default 6-12 s, uniform per env, staggered) and then gets a fresh "
            "+-180 deg draw mid-episode; the cyan overlay arrow jumps accordingly. "
            "Without this flag playback is deterministic (one heading per episode)."
        )
    # --disturbance: keep the live external disturbance (push_robot -> disturb_all_bodies
    # random wrenches on every rigid body, full training strength) that deterministic
    # playback otherwise disables. Accepts the --disturbances spelling as an alias.
    if "--disturbance" in argv or "--disturbances" in argv:
        video_flags.play_disturbance_enabled = True
        argv = [a for a in argv if a not in ("--disturbance", "--disturbances")]
        print(
            "[INFO] --disturbance: live disturbances ON during playback -- random "
            "force/torque wrenches on every rigid body at full training strength "
            "(red overlay arrows track them when --viz kit or --video is active). "
            "Deterministic playback (the default) has no pushes; scale via the Hydra "
            "override, e.g. env.events.push_robot.params.force_range=[-20,20]."
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
