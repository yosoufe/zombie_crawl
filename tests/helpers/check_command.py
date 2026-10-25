# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: AGENTS.md pre-launch smoke gate for the phase-2 yaw-rate command.

Checks (all against the REAL env, on GPU):
1. with ``env.extras['yaw_rate_max'] = 0.3`` the draws are ~20% exact zeros, bounded by
   the dial, and cover both signs;
2. with the dial absent/zero every command is exactly 0 (stage-0 lock);
3. ``pin()`` freezes resampling and ``unpin()`` releases it.

Run: ``../.venv/bin/python tests/helpers/check_command.py --num_envs 16 --headless``
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--draws", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--headless", action="store_true", default=False,
    help="Run headless (no GUI window; the Isaac Lab 3.0 default).",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg

cfg = CrawlEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_curriculum = False  # keep the dials out of the way; we drive the command directly
cfg.seed = args_cli.seed
env = gym.make("Zombie-Crawl-SO100-v0", cfg=cfg)
env.reset()

unwrapped = env.unwrapped
term = unwrapped.command_manager.get_term("yaw_rate_cmd")
n = unwrapped.num_envs
draws = args_cli.draws

failures = []

# 1) dial open: zero fraction ~0.2, bounded, both signs
unwrapped.extras["yaw_rate_max"] = 0.3
total, zeros, pos, neg, max_abs = 0, 0, 0, 0, 0.0
for _ in range(draws):
    term._resample_command(slice(None))
    cmd = term.yaw_rate
    total += n
    zeros += int((cmd == 0).sum())
    pos += int((cmd > 0.05).sum())
    neg += int((cmd < -0.05).sum())
    max_abs = max(max_abs, float(cmd.abs().max()))
zero_frac = zeros / total
if not 0.12 <= zero_frac <= 0.28:
    failures.append(f"zero fraction {zero_frac:.3f} outside [0.12, 0.28]")
if max_abs > 0.3 + 1e-6:
    failures.append(f"|cmd| max {max_abs} exceeds dial 0.3")
if pos / total < 0.15 or neg / total < 0.15:
    failures.append(f"signs unbalanced: pos {pos / total:.3f} neg {neg / total:.3f}")

# 2) dial closed -> every command exactly 0
unwrapped.extras["yaw_rate_max"] = 0.0
term._resample_command(slice(None))
if not bool((term.yaw_rate == 0).all()):
    failures.append("non-zero commands while yaw_rate_max == 0")

# 3) pin/unpin
term.pin(-0.1)
for _ in range(50):
    term._resample_command(slice(None))
    term.compute(0.02)
if not bool((term.yaw_rate == -0.1).all()):
    failures.append("pin() did not freeze the command")
term.unpin()
term._resample_command(slice(None))

env.close()

if failures:
    print("[helper] FAILED:", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    simulation_app.close()
    sys.exit(1)
print(f"[helper] OK (zero_frac={zero_frac:.3f}, max|cmd|={max_abs:.3f}, total={total})", flush=True)
simulation_app.close()
