# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the SO-ARM100 asset in simulation.

Spawns one robot, prints joint names/order/limits/defaults, actuator settings, link
masses, and contact shapes, then (with ``--settle``) drops the arm onto the ground and
steps 5 s with zero action, reporting stability metrics.

Usage:
    python scripts/inspect_asset.py [--actuator-preset servo] [--settle] [--drop-height 0.3]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--actuator-preset", type=str, default="servo", choices=["servo", "ideal"])
parser.add_argument("--settle", action="store_true", help="Run the 5 s zero-action settle test.")
parser.add_argument("--drop-height", type=float, default=0.3, help="Drop height for the settle test [m].")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.assets import JOINT_NAMES
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg, START_POSES

env_cfg = CrawlEnvCfg()
# set the top-level fields first, then re-run __post_init__ so the scene/curriculum
# switches (actuator preset, num envs, curriculum terms) actually take effect
env_cfg.num_envs = 1
env_cfg.env_spacing = 4.0
env_cfg.actuator_preset = args_cli.actuator_preset
env_cfg.use_curriculum = False
env_cfg.seed = args_cli.seed
env_cfg.__post_init__()

env = gym.make("Zombie-Crawl-SO100-v0", cfg=env_cfg)

asset = env.unwrapped.scene["robot"]

print("=" * 70)
print("SO-ARM100 asset inspection")
print("=" * 70)
print(f"joint order ({asset.num_joints}): {asset.joint_names}")
assert tuple(asset.joint_names) == tuple(JOINT_NAMES), "joint order mismatch!"
q0 = asset.data.default_joint_pos[0]
# Isaac Lab 3.0 limits layout: (num_envs, num_joints, 2) = [low, high]
soft = asset.data.soft_joint_pos_limits[0]
hard = asset.data.joint_pos_limits[0]
lower, upper = soft[:, 0], soft[:, 1]
hard_lower, hard_upper = hard[:, 0], hard[:, 1]
print(f"{'joint':<16} {'default':>9} {'soft low':>9} {'soft high':>9} {'hard low':>9} {'hard high':>9}")
for i, name in enumerate(asset.joint_names):
    print(
        f"{name:<16} {q0[i]:9.3f} {lower[i]:9.3f} {upper[i]:9.3f} {hard_lower[i]:9.3f} {hard_upper[i]:9.3f}"
    )
total_mass = float(asset.data.body_mass[0].sum())
print(f"\nper-env total mass: {total_mass:.3f} kg")

# actuator settings
act = asset.actuators["arm"]
print(f"\nactuator preset: {args_cli.actuator_preset}")
print(f"  effort_limit={act.effort_limit} velocity_limit={act.velocity_limit}")
print(f"  (stiffness/damping from make_actuator_cfg: k=1.0, d=0.1)")

root_state = asset.data.root_state_w[0]
print(f"\nroot state after reset: pos={root_state[0:3].tolist()} quat(x,y,z,w)={root_state[3:7].tolist()}")

if args_cli.settle:
    print("\n" + "=" * 70)
    print(f"Settle test: drop from z={args_cli.drop_height} m, zero action, 5 s")
    print("=" * 70)
    # 3.0 gym wrapper: reset() is required before step()
    env.reset()
    # deterministic start: flat base at default pose, dropped from above
    device = env.unwrapped.device
    env_cfg_play_pose = START_POSES["folded"]
    positions = torch.tensor(
        [env_cfg_play_pose["pos"][0], env_cfg_play_pose["pos"][1], args_cli.drop_height], device=device
    )
    positions = positions.repeat(1, 1) + env.unwrapped.scene.env_origins[0].to(device)
    from isaaclab.utils.math import quat_from_euler_xyz

    rpy = torch.tensor(env_cfg_play_pose["rot"], device=device)
    quats = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2]).repeat(1, 1)
    env_ids = torch.tensor([0], device=device)
    # note: Isaac Lab 3.0 write_*_to_sim_index APIs are keyword-only with renamed params
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)
    q_pose = torch.zeros(1, asset.num_joints, device=device)
    for name, value in env_cfg_play_pose["joint_pos"].items():
        q_pose[0, asset.joint_names.index(name)] = value
    asset.write_joint_position_to_sim_index(position=q_pose, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_pose), env_ids=env_ids)

    zero_actions = torch.zeros(1, 6, device=env.unwrapped.device)
    steps = int(5.0 / env.unwrapped.step_dt)
    max_z, min_z = -1e9, 1e9
    nan_seen = False
    start_pos = None
    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(zero_actions)
        z = asset.data.root_state_w[0, 2].item()
        max_z = max(max_z, z)
        min_z = min(min_z, z)
        state = asset.data.root_state_w[0]
        if not torch.isfinite(state).all() or not torch.isfinite(asset.data.joint_pos[0]).all():
            nan_seen = True
            print(f"[settle] NaN detected at step {step}")
            break
        if step == 0:
            start_pos = asset.data.root_state_w[0, 0:2].clone()
    final_pos = asset.data.root_state_w[0, 0:2]
    drift = (final_pos - start_pos).norm().item() if start_pos is not None else float("nan")
    final_z = asset.data.root_state_w[0, 2].item()
    print(f"  max base z      : {max_z:.4f} m")
    print(f"  min base z      : {min_z:.4f} m")
    print(f"  final base z    : {final_z:.4f} m")
    print(f"  xy drift (5 s)  : {drift:.4f} m")
    print(f"  NaNs            : {nan_seen}")
    ok = (not nan_seen) and max_z < 0.45 and drift < 0.1
    print(f"  SETTLE TEST     : {'PASS' if ok else 'FAIL'}")

env.close()
simulation_app.close()
