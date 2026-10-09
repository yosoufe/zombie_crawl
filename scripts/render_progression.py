# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render one 20 s play video per selected checkpoint for a training-progression review.

Loops over the checkpoints named on the command line, launching the unified play
entrypoint with ``--video`` for each (sequential, single GPU). Videos land in
``<run_dir>/videos/play/`` as ``side_model_<iter>_<n>.mp4`` and ``top_model_<iter>_<n>.mp4``
(scene camera sensors; env 0 is recorded, and the task config renders a single env when
video recording is enabled).

Usage:
    python scripts/render_progression.py \
        --run logs/rsl_rl/zombie_crawl/2026-10-02_23-35-11 \
        --iters 500 1000 1500 2000 2500 3000 \
        --video-length 1000
"""

import argparse
import glob
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=str, required=True, help="Run directory containing model_<iter>.pt checkpoints.")
parser.add_argument("--iters", type=int, nargs="+", required=True, help="Checkpoint iteration numbers to render.")
parser.add_argument("--video-length", type=int, default=1000, help="Clip length in env steps (1000 = 20 s @ 50 Hz).")
args_cli = parser.parse_args()

run = os.path.abspath(args_cli.run)
task = "Zombie-Crawl-SO100-v0"

for it in args_cli.iters:
    ckpt = os.path.join(run, f"model_{it}.pt")
    if not os.path.exists(ckpt):
        print(f"[skip] {ckpt} not found", flush=True)
        continue
    # the recorders name clips side_model_<iter>_<index>.mp4 / top_model_<iter>_<index>.mp4
    play_dir = os.path.join(run, "videos", "play")
    side = glob.glob(os.path.join(play_dir, f"side_model_{it}_*.mp4"))
    top = glob.glob(os.path.join(play_dir, f"top_model_{it}_*.mp4"))
    if side and top:
        print(f"[skip] iter {it} already rendered: {side[0]} {top[0]}", flush=True)
        continue
    print(f"[render] iter {it}  ({ckpt})", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "play.py"),
        "--task", task,
        "--checkpoint", ckpt,
        "--video", "--video_length", str(args_cli.video_length),
    ]
    log_path = os.path.join(run, "videos", "play", f"render_{it:04d}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAILED rc={rc} (see {log_path})"
    print(f"[done]   iter {it}  {dt:.0f}s  {status}", flush=True)

print("[all done]", flush=True)
