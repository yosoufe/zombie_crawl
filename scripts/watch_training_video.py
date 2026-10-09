# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a live rsl_rl run and render a fresh play video from the latest checkpoint.

Runs in the background *alongside* training (does not modify the run). Each render
launches the unified play entrypoint with ``--video`` (single-env, deterministic — the
same path ``render_progression.py`` uses), so clips land in ``<run>/videos/play/`` as
``side_model_<iter>_*.mp4`` / ``top_model_<iter>_*.mp4``. Rendering is idempotent: an
iteration already present there is skipped, so this is safe to run together with
``render_progression.py``.

Why "latest" instead of "every checkpoint": one render costs ~3-4 min (Kit boot + RTX
render of one env, sharing the GPU with training), while 100 training iterations is only
~46 s at ~2 it/s. Rendering literally every 100-iter checkpoint would therefore be
rendering most of the time and starving training. Instead, this renders the *newest*
checkpoint that is a multiple of ``--every-iter`` and at least ``--every-iter`` newer than
the last render, whenever it is free — a fresh clip every ~3-5 min, always the newest.

Usage:
    python scripts/watch_training_video.py --run logs/rsl_rl/zombie_crawl/<run_dir> \
        [--every-iter 100] [--video-length 500] [--cooldown-s 60] [--max-clips 0]

    # background, detached:
    setsid nohup python scripts/watch_training_video.py --run <run_dir> > /tmp/watch.log 2>&1 &
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=str, required=True, help="Run directory with model_<iter>.pt checkpoints.")
parser.add_argument("--task", type=str, default="Zombie-Crawl-SO100-v0")
parser.add_argument("--every-iter", type=int, default=100, help="Minimum iteration gap between renders (and a multiple-of for the pick).")
parser.add_argument("--video-length", type=int, default=500, help="Clip length in env steps (500 = 10 s @ 50 Hz). Lower = faster renders.")
parser.add_argument("--cooldown-s", type=float, default=60.0, help="Min seconds between render launches.")
parser.add_argument("--max-clips", type=int, default=0, help="Stop after this many clips (0 = unlimited).")
parser.add_argument("--stale-min", type=float, default=15.0, help="Stop if no new checkpoint appears for this many minutes (training finished).")
args_cli = parser.parse_args()

run = os.path.abspath(args_cli.run)
play_dir = os.path.join(run, "videos", "play")
os.makedirs(play_dir, exist_ok=True)
play_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play.py")


def latest_ckpt_iter() -> int:
    best = -1
    for f in glob.glob(os.path.join(run, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", f)
        if m:
            best = max(best, int(m.group(1)))
    return best


def rendered_iters() -> set[int]:
    out: set[int] = set()
    for f in glob.glob(os.path.join(play_dir, "side_model_*.mp4")):
        m = re.search(r"side_model_(\d+)_", f)
        if m:
            out.add(int(m.group(1)))
    return out


def pick_candidate(latest: int, rendered: set[int], last_rendered: int) -> int:
    """Newest multiple of every-iter <= latest, not yet rendered, >= every-iter newer than last render."""
    if latest < args_cli.every_iter:
        return -1
    cand = (latest // args_cli.every_iter) * args_cli.every_iter
    while cand >= args_cli.every_iter:
        if cand not in rendered and cand - last_rendered >= args_cli.every_iter:
            return cand
        cand -= args_cli.every_iter
    return -1


def render(ckpt_iter: int) -> int:
    ckpt = os.path.join(run, f"model_{ckpt_iter}.pt")
    logp = os.path.join(play_dir, f"watch_{ckpt_iter:04d}.log")
    cmd = [
        sys.executable, play_script,
        "--task", args_cli.task,
        "--checkpoint", ckpt,
        "--video", "--video_length", str(args_cli.video_length),
    ]
    with open(logp, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return rc


def main() -> None:
    print(f"[watch] run={run} every_iter={args_cli.every_iter} video_length={args_cli.video_length}", flush=True)
    last_rendered = -1
    last_render_time = -1e9
    n_clips = 0
    last_new_iter = -1
    last_new_time = time.time()

    while True:
        latest = latest_ckpt_iter()
        if latest != last_new_iter:
            last_new_iter = latest
            last_new_time = time.time()

        # training finished (no new checkpoints for stale-min) -> stop
        if time.time() - last_new_time > args_cli.stale_min * 60.0:
            print(f"[watch] no new checkpoint for {args_cli.stale_min} min (latest={latest}); stopping.", flush=True)
            break

        cand = pick_candidate(latest, rendered_iters(), last_rendered)
        if cand > 0 and (time.time() - last_render_time) >= args_cli.cooldown_s:
            t0 = time.time()
            print(f"[watch] rendering model_{cand}.pt (latest={latest}) ...", flush=True)
            rc = render(cand)
            dt = time.time() - t0
            last_rendered = cand
            last_render_time = time.time()
            n_clips += 1
            side = glob.glob(os.path.join(play_dir, f"side_model_{cand}_*.mp4"))
            print(f"[watch] done model_{cand} rc={rc} in {dt:.0f}s -> {side[0] if side else 'NO CLIP'}", flush=True)
            if args_cli.max_clips and n_clips >= args_cli.max_clips:
                print(f"[watch] reached max_clips={args_cli.max_clips}; stopping.", flush=True)
                break
        time.sleep(20)


if __name__ == "__main__":
    main()
