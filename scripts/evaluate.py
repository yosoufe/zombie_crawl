# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained zombie-crawl policy (prompt.md section 9).

Runs the policy in N parallel deterministic environments (default 100) for full 20 s
episodes and reports:
    - mean / median / std forward displacement along +X [m]
    - mean forward speed [m/s] and fraction of time within +/-50 % of v_target
    - mean absolute yaw drift [deg]
    - fraction of episodes terminated early
    - mean peak base height [m]

The friction sweep (``--friction 0.5 --friction 0.8 --friction 1.2``) re-runs the
evaluation at each ground static friction value.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_policy.pt
                               [--num_envs 100] [--episode-seconds 20]
                               [--friction 0.5 --friction 0.8 --friction 1.2]
                               [--v-target 0.10] [--json out.json]
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel evaluation environments.")
parser.add_argument("--episode-seconds", type=float, default=20.0, help="Episode length [s].")
parser.add_argument("--friction", type=float, action="append", default=[0.8], help="Ground static friction values to sweep.")
parser.add_argument("--v-target", type=float, default=0.10, help="Forward velocity target [m/s].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--json", type=str, default=None, help="Write the results as JSON.")
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import add_rsl_rl_args

add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import update_rsl_rl_cfg
from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector

import zombie_crawl.tasks  # noqa: F401

TASK_ID = "Zombie-Crawl-SO100-Play-v0"


def main() -> None:
    # resolve task + agent configs (play-mode overrides applied for the -Play task)
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", play_mode=True)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    # resolve checkpoint: path, or the Isaac Lab 3.0 "latest"/"best" selectors
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    if args_cli.checkpoint:
        if args_cli.checkpoint in CHECKPOINT_SELECTORS:
            resume_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="rsl_rl",
                task=TASK_ID.replace("-Play", ""),
                checkpoint_pattern=r"model_.*\.pt",
            )
        else:
            resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(os.path.abspath(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    results = {}
    for mu in args_cli.friction:
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.v_target = args_cli.v_target
        env_cfg.scene.ground.spawn.physics_material.static_friction = mu
        env_cfg.scene.ground.spawn.physics_material.dynamic_friction = 0.75 * mu
        env_cfg.seed = args_cli.seed
        env_cfg.episode_length_s = args_cli.episode_seconds

        env = gym.make(TASK_ID, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        steps_per_episode = int(env.unwrapped.cfg.episode_length_s / env.unwrapped.step_dt)
        # we evaluate one full episode per env; envs that terminate early are restarted
        total_steps = steps_per_episode

        robot = env.unwrapped.scene["robot"]
        num_envs = args_cli.num_envs
        device = env.unwrapped.device

        # per-episode accumulators
        # Isaac Lab 3.0 gym wrapper: reset() is required before step()
        env.reset()
        start_x = robot.data.root_pos_w.torch[:, 0].clone()
        episode_peak_z = torch.zeros(num_envs, device=device)
        early_terminations = torch.zeros(num_envs, dtype=torch.int64, device=device)
        within_band = torch.zeros(total_steps, dtype=torch.float64)
        yaw_abs_sum = torch.zeros(total_steps, dtype=torch.float64)

        obs = env.get_observations()
        last_x = start_x.clone()
        with torch.inference_mode():
            for step in range(total_steps):
                # capture pre-step x: on the truncating final step the env resets the
                # robot DURING env.step(), so the post-step root state is the spawn pose
                last_x = robot.data.root_pos_w.torch[:, 0].clone()
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                # the MLP actor has no recurrent state to reset; guard for recurrent policies
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                state = robot.data.root_state_w.torch
                # Isaac Lab 3.0 root_state_w layout: pos(3), quat(4), lin_vel(3), ang_vel(3)
                z = state[:, 2]
                vx = state[:, 7]
                quat = state[:, 3:7]
                _, _, yaw = euler_xyz_from_quat(quat)
                episode_peak_z = torch.maximum(episode_peak_z, z)
                within_band[step] = (
                    (vx >= 0.5 * args_cli.v_target) & (vx <= 1.5 * args_cli.v_target)
                ).double().mean().cpu()
                yaw_abs_sum[step] = torch.abs(yaw).double().mean().cpu()
                # envs that just ended early are restarted by the env; mark them
                if (dones > 0).any():
                    ended = (dones > 0).nonzero().flatten()
                    for eid in ended.tolist():
                        if step + 1 < steps_per_episode:
                            early_terminations[eid] += 1
                    start_x[ended] = robot.data.root_pos_w.torch[ended, 0]

        final_x = last_x
        displacement = final_x - start_x
        # envs that ended early: displacement from the (re)start; keep as-is (documented)
        displacements = displacement.cpu().numpy()
        displacements[early_terminations.cpu().numpy() > 0] = np.nan  # exclude interrupted episodes from displacement stats

        episode_seconds = total_steps * env.unwrapped.step_dt
        report = {
            "num_envs": num_envs,
            "episode_seconds": episode_seconds,
            "v_target": args_cli.v_target,
            "displacement_m": {
                "mean": float(np.nanmean(displacements)),
                "median": float(np.nanmedian(displacements)),
                "std": float(np.nanstd(displacements)),
            },
            "mean_forward_speed_m_s": float(np.nanmean(displacements) / episode_seconds),
            "fraction_time_within_50pct_vtarget": float(within_band.mean()),
            "mean_abs_yaw_drift_deg": float(np.degrees(yaw_abs_sum.mean())),
            "fraction_episodes_terminated_early": float((early_terminations > 0).float().mean().cpu()),
            "mean_peak_base_height_m": float(episode_peak_z.mean()),
            "success_1m_in_20s": float(np.nanmean((displacements > 1.0))),
        }
        key = f"friction_{mu:.2f}"
        results[key] = report
        print(f"\n[INFO] Evaluation @ ground friction {mu:.2f}")
        print_dict(report, nesting=4)
        env.close()

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Wrote results to {args_cli.json}")

    print("\n" + "=" * 70)
    print("Phase-1 success criteria (v_target = %.2f m/s):" % args_cli.v_target)
    ref = results.get(f"friction_{args_cli.friction[0]:.2f}")
    if ref:
        speed_ok = abs(ref["mean_forward_speed_m_s"] - args_cli.v_target) <= 0.3 * args_cli.v_target
        print(f"  mean speed within +/-30 % of v_target : {speed_ok}")
        print(f"  >=90 % episodes advance > 1 m           : {ref['success_1m_in_20s'] >= 0.9}")
        print(f"  no early terminations                   : {ref['fraction_episodes_terminated_early'] == 0.0}")
    print("=" * 70)


main()
simulation_app.close()
