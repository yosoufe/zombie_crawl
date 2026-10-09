# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the joint limits that EMAJointPositionToLimitsAction actually uses.

The action term rescales raw actions in [-1, +1] onto each joint's SOFT position
limits, read live from ``asset.data.soft_joint_pos_limits`` on every step. This
script prints exactly those values (the raw -1 / raw +1 endpoints per joint),
with the hard limits and default pose for reference.

Usage:
    python scripts/print_action_limits.py [--preset servo]
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", type=str, default="servo", choices=["servo", "ideal"])
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg

env_cfg = CrawlEnvCfg()
env_cfg.num_envs = 1
env_cfg.env_spacing = 4.0
env_cfg.actuator_preset = args_cli.preset
env_cfg.use_curriculum = False
env_cfg.seed = args_cli.seed
env_cfg.__post_init__()

env = gym.make("Zombie-Crawl-SO100-v0", cfg=env_cfg)
asset = env.unwrapped.scene["robot"]

# exactly what EMAJointPositionToLimitsAction reads in process_actions (live, per step)
soft = asset.data.soft_joint_pos_limits[0]  # (num_joints, 2) = [low, high]
hard = asset.data.joint_pos_limits[0]  # (num_joints, 2) = [low, high]
q0 = asset.data.default_joint_pos[0]

print(f"Action term: {type(env.unwrapped.action_manager._terms['joint_pos']).__name__}")
print("Raw action [-1, +1] maps onto the SOFT limits below, per joint:\n")
header = f"{'joint':<16} {'raw -1':>9} {'raw +1':>9} {'range rad':>10} {'range deg':>10} | {'hard lo':>9} {'hard hi':>9} {'default':>9}"
print(header)
print("-" * len(header))
for i, name in enumerate(asset.joint_names):
    lo, hi = float(soft[i, 0]), float(soft[i, 1])
    print(
        f"{name:<16} {lo:9.4f} {hi:9.4f} {hi - lo:10.4f} {math.degrees(hi - lo):10.1f} |"
        f" {float(hard[i, 0]):9.4f} {float(hard[i, 1]):9.4f} {float(q0[i]):9.4f}"
    )
print(f"\nsoft_joint_pos_limit_factor = {env_cfg.scene.robot.soft_joint_pos_limit_factor}")

env.close()
simulation_app.close()
