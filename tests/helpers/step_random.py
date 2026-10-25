# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: create the crawl env and step it with random actions; assert no NaNs."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
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

action_dim = env.unwrapped.action_manager.total_action_dim
print(f"[helper] action_dim={action_dim}", flush=True)
obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()

ok = True
for step in range(args_cli.steps):
    actions = torch.randn(args_cli.num_envs, action_dim, device=env.unwrapped.device)
    obs, rewards, terminated, truncated, info = env.step(actions)
    state = env.unwrapped.scene["robot"].data.root_state_w
    if not torch.isfinite(state).all():
        print(f"[helper] FAIL: non-finite root state at step {step}", flush=True)
        ok = False
        break
    if not torch.isfinite(rewards).all():
        print(f"[helper] FAIL: non-finite reward at step {step}", flush=True)
        ok = False
        break

# print BEFORE Kit shutdown: simulation_app.close() can terminate the process before
# buffered stdout survives when stdout is a pipe/file (pytest capture)
print("[helper] OK" if ok else "[helper] FAILED", flush=True)
env.close()
simulation_app.close()
sys.exit(0 if ok else 1)
