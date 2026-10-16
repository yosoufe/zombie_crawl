# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: check that the reset places the robot above ground and inside the bounding area."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0 removed --headless from the launcher CLI; accept it here for
# Isaac Lab 2.x-style command lines (it flows through AppLauncher to the launcher).
parser.add_argument(
    "--headless", action="store_true", default=False,
    help="Run headless (no GUI window; the Isaac Lab 3.0 default).",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg

cfg = CrawlEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.seed = args_cli.seed
env = gym.make("Zombie-Crawl-SO100-v0", cfg=cfg)

ok = True
for i in range(10):  # 10 rounds of resets
    env.reset()
    state = env.unwrapped.scene["robot"].data.root_state_w
    z = state[:, 2]
    pos = state[:, 0:2]
    origins = env.unwrapped.scene.env_origins[:, 0:2]
    dxy = (pos - origins).norm(dim=1)
    if (z < -0.01).any():
        print(f"[helper] FAIL: base below ground: min z={z.min().item():.4f}")
        ok = False
        break
    if (dxy > 2.5).any():
        print(f"[helper] FAIL: base outside bounding area: max |xy|={dxy.max().item():.4f}")
        ok = False
        break

# print BEFORE Kit shutdown: simulation_app.close() can terminate the process before
# buffered stdout survives when stdout is a pipe/file (pytest capture)
print("[helper] OK" if ok else "[helper] FAILED", flush=True)
env.close()
simulation_app.close()
sys.exit(0 if ok else 1)
