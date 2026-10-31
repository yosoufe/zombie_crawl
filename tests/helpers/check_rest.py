# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: P3-3 rest-pose feasibility check (offline, on GPU).

Pins M4 (zero command) on every env, commands ZERO actions (the EMA mapping's neutral IS
the folded rest pose), starts FOLDED, and after a settle window measures the three
rest_pose_ok clauses: up >= rest_up_thresh, |z - h_rest| <= rest_height_tol,
max-joint |q - q_rest| <= rest_joint_tol.

Verdicts:
  OK       -> the rest pose is a stable zero-action sink and measurable: M4 learning is
              a reward-shaping problem, not an asset/height-measurement problem.
  Z-FAIL   -> h_rest (env.cfg.rest_height) is wrong for the settled contact state.
  Q-FAIL   -> the zero action does NOT hold the folded pose (actuator/gait leak).

Run: HEADLESS=1 ../.venv/bin/python tests/helpers/check_rest.py --num_envs 8
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle-steps", type=int, default=400)
parser.add_argument("--push-scale", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true", default=False)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg
from zombie_crawl.tasks.manager_based.crawl.mdp.rewards import flatness_gate

cfg = CrawlEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_curriculum = False
cfg.seed = args_cli.seed
env = gym.make("Zombie-Crawl-SO100-v0", cfg=cfg)
env.reset()
unwrapped = env.unwrapped

term = unwrapped.command_manager.get_term("yaw_rate_cmd")
term.pin_mode(4, 0.0, 0.0)
unwrapped.extras["push_scale"] = args_cli.push_scale  # 0.0 = pure feasibility, 1.0 = training push

act = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
qtrack = []
for i in range(args_cli.settle_steps):
    env.step(act)
    if i >= 100:  # skip initial settling transient
        qtrack.append(
            (unwrapped.scene["robot"].data.joint_pos.torch - unwrapped.scene["robot"].data.default_joint_pos.torch)
            .abs().amax(dim=1).cpu()
        )

asset = unwrapped.scene["robot"]
up = flatness_gate(unwrapped)
z = asset.data.root_pos_w.torch[:, 2]
q_err = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch).abs()
h_rest = float(cfg.rest_height)
ok_up = up >= cfg.rest_up_thresh
ok_z = (z - h_rest).abs() <= cfg.rest_height_tol
ok_q = (q_err.amax(dim=1) <= cfg.rest_joint_tol)
all_ok = (ok_up & ok_z & ok_q).float().mean().item()
qall = torch.cat(qtrack)
print(
    f"[helper] qerr quantiles (push {args_cli.push_scale}): "
    f"p50={qall.quantile(0.5):.3f} p90={qall.quantile(0.9):.3f} p99={qall.quantile(0.99):.3f} "
    f"frac<=0.5={float((qall <= 0.5).float().mean()):.3f}",
    flush=True,
)
print(
    f"[helper] rest feasibility: ok_up={ok_up.float().mean():.2f} ok_z={ok_z.float().mean():.2f} "
    f"ok_q={ok_q.float().mean():.2f} ALL={all_ok:.2f} | z={z.mean():.4f} (h_rest {h_rest}) "
    f"qerr_max={q_err.amax(dim=1).mean():.3f} gripper_q={q_err[:, -1].mean():.3f}",
    flush=True,
)
env.close()
if all_ok >= 0.75:
    print("[helper] OK", flush=True)
    simulation_app.close()
    sys.exit(0)
if not bool(ok_z.any()):
    print("[helper] Z-FAIL: rest_height mis-measured (settled z differs by > tol)", flush=True)
elif not bool(ok_q.any()):
    print("[helper] Q-FAIL: zero action does not hold the folded pose", flush=True)
else:
    print("[helper] PARTIAL: clauses individually reachable but never simultaneously", flush=True)
simulation_app.close()
sys.exit(1)
