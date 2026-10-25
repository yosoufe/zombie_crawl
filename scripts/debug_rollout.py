# Debug harness: official rsl_rl play path + per-step v_c probes (bisect evaluate.py).
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl import mdp

TASK_ID = "Zombie-Crawl-SO100-Play-v0"


def main():
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", play_mode=True)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = 16
    mu = float(os.environ.get("DEBUG_MU", "1.0"))
    if os.environ.get("DEBUG_EVALMUT"):
        env_cfg.scene.num_envs = 100
        env_cfg.v_target = 0.1
        env_cfg.v_target_start = 0.1
        env_cfg.seed = 0
        env_cfg.episode_length_s = 10.0
        env_cfg.events.reset_start_pose.params["pose_weights"] = (1.0, 0.0, 0.0)
    if mu != 1.0:
        env_cfg.scene.ground.spawn.physics_material.static_friction = mu
        env_cfg.scene.ground.spawn.physics_material.dynamic_friction = 0.75 * mu
    env_cfg.sim.device = "cuda:0"
    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    term = env.unwrapped.command_manager.get_term("yaw_rate_cmd")
    term.pin(0.0)
    with torch.inference_mode():
        for step in range(300):
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)
            if step % 25 == 0 or step < 6:
                v_c = mdp.metrics._claw_v(env.unwrapped)
                print(
                    f"[roll {step}] v_c={float(v_c.mean()):.4f} rew={float(rew.mean()):.3f} "
                    f"|act|={float(actions.abs().mean()):.3f} "
                    f"obs={[round(float(x),2) for x in (obs["policy"] if "policy" in obs.keys() else next(iter(obs.values()))).mean(0)]} "
                    f"z={float(robot.data.root_pos_w.torch[:, 2].mean()):.4f}",
                    flush=True,
                )
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
