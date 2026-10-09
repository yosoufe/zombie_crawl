# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose whether the play/video pipeline actually reflects the loaded policy.

Loads each requested checkpoint into the deterministic play env, runs a fixed
number of steps, and prints the root-x trajectory plus action statistics.
Different checkpoints MUST produce different trajectories; if they do not,
the policy is not driving the simulated motion.

Usage:
    python scripts/diag_video.py \
        --checkpoint logs/rsl_rl/zombie_crawl/<run>/model_500.pt \
        --checkpoint logs/rsl_rl/zombie_crawl/<run>/model_3499.pt \
        [--steps 500]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=500, help="Steps to run per checkpoint.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", type=str, action="append", required=True, help="Checkpoint paths to compare.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
_ = app_launcher.app

"""Rest everything follows."""

import torch

import gymnasium as gym

import zombie_crawl.tasks  # noqa: F401  # register tasks
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config

TASK_ID = "Zombie-Crawl-SO100-Play-v0"


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", play_mode=True)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed

    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    robot = env.unwrapped.scene["robot"]

    for ckpt in args_cli.checkpoint:
        if ckpt == "none":
            policy = None
            print("== none  (zero actions, physics-only baseline)")
        else:
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f"== {os.path.basename(ckpt)}  (iter={runner.current_learning_iteration})")

        env.reset()
        x0 = robot.data.root_pos_w.torch[0, 0].item()
        q0 = robot.data.joint_pos.torch[0].clone()
        obs = env.get_observations()
        acts = []
        xs = [x0]
        qs = [q0.clone()]
        with torch.inference_mode():
            for i in range(args_cli.steps):
                a = torch.zeros(1, env.num_actions, device=env.unwrapped.device) if policy is None else policy(obs)
                acts.append(a)
                obs, _, dones, _ = env.step(a)
                if policy is not None and hasattr(policy, "reset"):
                    policy.reset(dones)
                if (i + 1) % 50 == 0:
                    xs.append(robot.data.root_pos_w.torch[0, 0].item())
                    qs.append(robot.data.joint_pos.torch[0].clone())
        acts = torch.stack(acts).reshape(-1, env.num_actions)
        traj = [round(v - x0, 4) for v in xs]
        print(f"   root-x @50 steps (rel m): {traj}")
        print(f"   total dx = {traj[-1]:+.4f} m over {args_cli.steps} steps ({args_cli.steps * env.unwrapped.step_dt:.1f} s)")
        print(f"   action mean={acts.mean().item():+.4f} std={acts.std().item():.4f} max|a|={acts.abs().max().item():.4f}")
        print(f"   action per-joint mean: {[round(float(v), 3) for v in acts.mean(dim=0).tolist()]}")
        qs = torch.stack(qs)
        dq = (qs - q0).abs().max(dim=0).values
        print(f"   joint max |dq| @50 steps: {[round(float(v), 3) for v in dq.tolist()]}")
    env.close()


main()
