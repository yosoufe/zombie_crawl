# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: reset-pose gating (pitfall X1).

Without ``env.extras['fallen_frac']`` EVERY reset must spawn folded (upright-first sanity)
and ``started_fallen`` must be recorded False for all envs. With ``fallen_frac = 1.0``
nearly every env must start fallen.

Run: ``../.venv/bin/python tests/helpers/check_fallen.py --num_envs 32 --headless``
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
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
cfg.use_curriculum = False  # no fallen_frac in extras
cfg.seed = args_cli.seed
env = gym.make("Zombie-Crawl-SO100-v0", cfg=cfg)

unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]
failures = []

for trial in range(3):
    env.reset()
    grav = robot.data.projected_gravity_b.torch
    flat = -grav[:, 2]
    if float(flat.min()) < 0.9:
        failures.append(f"trial {trial}: min -g_z={float(flat.min()):.2f} < 0.9 (spawned fallen!)")
    st = unwrapped.zombie_metrics_state
    if not bool((~st["started_fallen"]).all()):
        failures.append(f"trial {trial}: started_fallen True without fallen_frac")

# with fallen_frac = 1.0 the population must start fallen
unwrapped.extras["fallen_frac"] = 1.0
env.reset()
st = unwrapped.zombie_metrics_state
frac = float(st["started_fallen"].double().mean())
if frac < 0.9:
    failures.append(f"fallen_frac=1.0 but only {frac:.2f} started fallen")

env.close()

if failures:
    print("[helper] FAILED:", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    simulation_app.close()
    sys.exit(1)
print(f"[helper] OK (fallen_frac absent -> all folded; =1.0 -> {frac:.2f} fallen)", flush=True)
simulation_app.close()
