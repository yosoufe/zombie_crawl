# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop scripted crawl feasibility test (prompt.md section 7.2).

Drives the arm with a hand-authored multi-sinusoid joint trajectory (no policy) and
measures the net forward displacement over 20 s. If this fails, the task is not
physically solvable with the current friction/preset/geometry and the parameters must
be tuned before starting RL.

The trajectory modulates each joint around the default pose:
    q(t) = q0 + A_i * sin(2*pi*f*t + phi_i)
with per-joint amplitude/phase chosen so the arm hooks the ground and pulls (see the
``TRAJ`` table below).

Usage:
    python scripts/scripted_crawl.py [--preset servo] [--friction 0.8] [--duration 20]
                                      [--frequency 0.25] [--amplitude 0.3]
                                      [--pose folded] [--log-csv out.csv]
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", type=str, default="servo", choices=["servo", "ideal"])
parser.add_argument("--friction", type=float, default=0.8, help="Ground static friction (robot multiplier = 1.0).")
parser.add_argument("--duration", type=float, default=20.0, help="Episode length [s].")
parser.add_argument("--frequency", type=float, default=0.25, help="Crawl cycle frequency [Hz].")
parser.add_argument("--amplitude", type=float, default=0.3, help="Global joint amplitude [rad].")
parser.add_argument("--pose", type=str, default="folded", help="Start pose name.")
parser.add_argument("--log-csv", type=str, default=None, help="Log per-step base state to CSV.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.assets import JOINT_NAMES
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg, START_POSES

# per-joint phase offsets (rad) for a hook-pull-release swing.
# index follows JOINT_NAMES: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
PHASES = {
    "shoulder_pan": 0.0,
    "shoulder_lift": math.pi * 0.5,  # reach forward
    "elbow_flex": math.pi,  # then hook & pull
    "wrist_flex": math.pi * 0.5,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
# per-joint amplitude scale (gripper swings harder, yaw stays put)
AMPLITUDE_SCALE = {
    "shoulder_pan": 0.25,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 0.75,
    "wrist_roll": 0.0,
    "gripper": 1.5,
}


def main() -> int:
    env_cfg = CrawlEnvCfg()
    # set the top-level fields first, then re-run __post_init__ so the scene/curriculum
    # switches (actuator preset, num envs, curriculum terms) actually take effect
    env_cfg.num_envs = 1
    env_cfg.env_spacing = 4.0
    env_cfg.actuator_preset = args_cli.preset
    env_cfg.use_curriculum = False
    env_cfg.seed = args_cli.seed
    env_cfg.__post_init__()
    # freeze friction at the requested value (robot multiplier 1.0 * ground friction)
    env_cfg.events.randomize_ground_friction.params["friction_range"] = (1.0, 1.0)
    env_cfg.scene.ground.spawn.physics_material.static_friction = args_cli.friction
    env_cfg.scene.ground.spawn.physics_material.dynamic_friction = 0.75 * args_cli.friction

    env = gym.make("Zombie-Crawl-SO100-v0", cfg=env_cfg)
    asset = env.unwrapped.scene["robot"]
    # 3.0 gym wrapper: reset() is required before step(); run it first so the
    # deterministic pose written below is not overwritten by reset randomization
    env.reset()

    # deterministic start pose
    device = env.unwrapped.device
    pose = START_POSES[args_cli.pose]
    env_ids = torch.tensor([0], device=device)
    rpy = torch.tensor(pose["rot"], device=device)
    positions = torch.tensor(pose["pos"], device=device).repeat(1, 1) + env.unwrapped.scene.env_origins[0].to(device)
    quats = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2]).repeat(1, 1)
    # note: Isaac Lab 3.0 write_*_to_sim_index APIs are keyword-only with renamed params
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)
    q_pose = torch.zeros(1, asset.num_joints, device=device)
    for name, value in pose["joint_pos"].items():
        q_pose[0, asset.joint_names.index(name)] = value
    asset.write_joint_position_to_sim_index(position=q_pose, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_pose), env_ids=env_ids)
    q0 = q_pose.clone()

    # joint phase/amplitude tensors
    # note: the action term applies target = q0 + scale * action with scale = 0.5,
    # so to realize an absolute offset of A_i * sin(...) we pass 2 * A_i * sin(...)
    phases = torch.tensor([PHASES[n] for n in asset.joint_names], device=env.unwrapped.device)
    amps = torch.tensor(
        [2.0 * args_cli.amplitude * AMPLITUDE_SCALE[n] for n in asset.joint_names], device=env.unwrapped.device
    )

    steps = int(args_cli.duration / env.unwrapped.step_dt)
    csv_file = open(args_cli.log_csv, "w", newline="") if args_cli.log_csv else None
    writer = None
    if csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "t", "x", "y", "z", "vx", "vy", "vz", "yaw", "q0", "q1", "q2", "q3", "q4", "q5"])

    max_z, min_z = -1e9, 1e9
    nan_seen = False
    terminated = False
    last_x = None
    for step in range(steps):
        # capture the true pre-step x: at the final (truncating) step the env resets
        # the robot DURING env.step(), so reading root_state_w afterwards would report
        # the post-reset spawn position instead of where the crawl actually ended
        last_x = float(asset.data.root_state_w.torch[0, 0])
        t = step * env.unwrapped.step_dt
        action = (amps * torch.sin(2.0 * math.pi * args_cli.frequency * t + phases))[None, :]
        obs, reward, terminated_flags, truncated, info = env.step(action)
        state = asset.data.root_state_w[0]
        x, y, z = state[0].item(), state[1].item(), state[2].item()
        vx, vy, vz = state[7].item(), state[8].item(), state[9].item()
        _, _, yaw = euler_xyz_from_quat(state[3:7].unsqueeze(0))
        yaw = float(yaw)
        max_z = max(max_z, z)
        min_z = min(min_z, z)
        if not torch.isfinite(state).all():
            nan_seen = True
            print(f"[scripted_crawl] NaN at step {step} (t={t:.2f}s)")
            break
        if writer:
            q = asset.data.joint_pos[0]
            writer.writerow([step, f"{t:.3f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                             f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}", f"{yaw:.3f}"] + [f"{float(qi):.4f}" for qi in q])
        if terminated_flags[0]:
            print(f"[scripted_crawl] terminated at t={t:.2f}s")
            terminated = True
            break

    if csv_file:
        csv_file.close()

    total_x = last_x if last_x is not None else float(asset.data.root_state_w.torch[0, 0])
    start_x = env.unwrapped.scene.env_origins[0, 0].item() + pose["pos"][0]
    displacement = total_x - start_x
    mean_speed = displacement / max(args_cli.duration, 1e-9)
    print("=" * 70)
    print("Scripted crawl feasibility test")
    print("=" * 70)
    print(f"  preset              : {args_cli.preset}")
    print(f"  ground friction     : {args_cli.friction}")
    print(f"  start pose          : {args_cli.pose}")
    print(f"  frequency / amplitude: {args_cli.frequency} Hz / {args_cli.amplitude} rad")
    print(f"  duration            : {args_cli.duration} s")
    print(f"  net +X displacement : {displacement:.4f} m")
    print(f"  mean forward speed  : {mean_speed:.4f} m/s")
    print(f"  base z range        : [{min_z:.4f}, {max_z:.4f}] m")
    print(f"  NaN / terminated    : {nan_seen} / {terminated}")
    ok = (not nan_seen) and not terminated and displacement >= 0.2
    print(f"  FEASIBILITY TEST    : {'PASS' if ok else 'FAIL'} (threshold: >= 0.2 m in 20 s)")
    env.close()
    return 0 if ok else 1


main_ret = main()
simulation_app.close()
sys.exit(main_ret)
