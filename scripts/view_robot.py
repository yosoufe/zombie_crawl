# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTOR.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open the Kit GUI with the SO-ARM100 loaded and let it idle on the ground.

Spawns a single robot in a named start pose, steps the simulation with zero action
so it settles naturally, and keeps running until interrupted (Ctrl+C) or the
process is killed -- the GUI window stays open the whole time so the scene can be
inspected, orbited, and zoomed.

Usage:
    python scripts/view_robot.py --viz kit [--pose folded] [--preset servo]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", type=str, default="servo", choices=["servo", "ideal"])
parser.add_argument("--pose", type=str, default="folded")
parser.add_argument("--friction", type=float, default=0.8, help="Ground static friction.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

from isaaclab.utils.math import quat_from_euler_xyz

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg, START_POSES

env_cfg = CrawlEnvCfg()
env_cfg.num_envs = 1
env_cfg.env_spacing = 4.0
env_cfg.actuator_preset = args_cli.preset
env_cfg.use_curriculum = False
env_cfg.seed = args_cli.seed
# very long episodes: the viewer never wants a timeout-reset jumping the robot
env_cfg.episode_length_s = 1.0e6
env_cfg.__post_init__()
# freeze friction at the requested value (robot multiplier 1.0 * ground friction)
env_cfg.events.randomize_ground_friction.params["friction_range"] = (1.0, 1.0)
env_cfg.scene.ground.spawn.physics_material.static_friction = args_cli.friction
env_cfg.scene.ground.spawn.physics_material.dynamic_friction = 0.75 * args_cli.friction

env = gym.make("Zombie-Crawl-SO100-v0", cfg=env_cfg)
asset = env.unwrapped.scene["robot"]
env.reset()


def write_start_pose() -> None:
    device = env.unwrapped.device
    pose = START_POSES[args_cli.pose]
    env_ids = torch.tensor([0], device=device)
    rpy = torch.tensor(pose["rot"], device=device)
    positions = torch.tensor(pose["pos"], device=device).repeat(1, 1) + env.unwrapped.scene.env_origins[0].to(device)
    quats = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2]).repeat(1, 1)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)
    q_pose = torch.zeros(1, asset.num_joints, device=device)
    for name, value in pose["joint_pos"].items():
        q_pose[0, asset.joint_names.index(name)] = value
    asset.write_joint_position_to_sim_index(position=q_pose, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_pose), env_ids=env_ids)


def main() -> None:
    write_start_pose()
    print(f"[view_robot] SO-ARM100 loaded: pose={args_cli.pose} preset={args_cli.preset} friction={args_cli.friction}")
    print("[view_robot] stepping with zero action; close the GUI window or kill this process to exit")
    zero_actions = torch.zeros(1, asset.num_joints, device=env.unwrapped.device)
    step_dt = env.unwrapped.step_dt
    status_every = int(2.0 / step_dt)
    step = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(zero_actions)
        state = asset.data.root_state_w[0]
        if not torch.isfinite(state).all():
            print("[view_robot] NaN state detected; rewriting start pose")
            write_start_pose()
        elif terminated[0]:
            print(f"[view_robot] terminated at step {step}; rewriting start pose")
            write_start_pose()
        if step > 0 and step % status_every == 0:
            x, y, z = state[0].item(), state[1].item(), state[2].item()
            print(f"[view_robot] t={step * step_dt:7.1f}s  base xyz=({x:.3f}, {y:.3f}, {z:.3f})")
        step += 1


try:
    main()
except KeyboardInterrupt:
    print("\n[view_robot] interrupted, closing")
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    env.close()
    simulation_app.close()
