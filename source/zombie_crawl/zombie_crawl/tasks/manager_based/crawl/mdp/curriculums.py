# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the zombie crawl task.

The curriculum advances two knobs, both stored on the environment:

* ``env.v_target`` -- per-environment forward velocity target (tensor, shape (num_envs,)).
* ``env.extras["friction_scale"]`` -- multiplier applied on top of the per-env friction
  sample; starts above 1 (grippier) and decays to 1.

Both advance when the mean forward velocity (and mean episode length) pass thresholds,
so a weak policy gets an easier task and the difficulty creeps up as it learns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv

from .rewards import get_v_target


def _proj_vel_ema(env: "ManagerBasedEnv", tau: float = 2.0) -> float:
    """Population mean of an EMA (~``tau`` s) of the reference-projected base velocity.

    Both curriculum gates used to read ONE instantaneous frame of velocity; a flailing
    population's instantaneous mean spikes on every stand-up/topple kick and advanced the
    difficulty on bogus evidence. A short EMA gates on sustained forward motion instead.
    """
    from .rewards import get_ref_dir

    dt = float(env.step_dt)
    v = env.scene["robot"].data.root_lin_vel_w.torch
    cos_ref, sin_ref = get_ref_dir(env)
    v_proj = v[:, 0] * cos_ref + v[:, 1] * sin_ref
    ema = getattr(env, "zombie_v_ema", None)
    if ema is None or ema.shape[0] != env.num_envs:
        env.zombie_v_ema = v_proj.clone()
        return float(env.zombie_v_ema.mean())
    ema += (v_proj - ema) * min(1.0, dt / tau)
    return float(env.zombie_v_ema.mean())


def v_target_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    v_start: float = 0.06,
    v_final: float = 0.10,
    step_size: float = 0.005,
    min_fraction: float = 0.6,
    min_episode_fraction: float = 0.5,
    check_interval: int = 100,
) -> float:
    """Advance ``env.v_target`` toward ``v_final`` as the population learns to crawl."""
    if "zombie_curriculum_step" not in env.extras:
        env.extras["zombie_curriculum_step"] = 0
    env.extras["zombie_curriculum_step"] += 1
    # lazily initialize env.v_target (curriculum runs at reset, before any reward is computed)
    v_target = get_v_target(env)

    if env.extras["zombie_curriculum_step"] % check_interval != 0:
        return float(v_target[0])

    mean_vx = _proj_vel_ema(env)  # sustained speed along the reference (jaw) axis
    mean_episode_len = env.episode_length_buf.float().mean().item()
    max_len = env.cfg.episode_length_s / env.cfg.decimation / env.cfg.sim.dt
    if mean_vx > min_fraction * v_target[0] and mean_episode_len > min_episode_fraction * max_len:
        if not getattr(env.cfg, "use_velocity_commands", False):
            env.v_target = torch.clamp(env.v_target + step_size, max=v_final)
            v_target = env.v_target
    return float(v_target[0])


def friction_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scale_start: float = 1.5,
    scale_final: float = 1.0,
    step_size: float = 0.05,
    min_fraction: float = 0.6,
    check_interval: int = 200,
) -> float:
    """Decay the extra friction multiplier toward 1.0 as the population learns to crawl."""
    if "zombie_friction_step" not in env.extras:
        env.extras["zombie_friction_step"] = 0
        env.extras["friction_scale"] = torch.tensor(
            scale_start, device=env.device
        )
    env.extras["zombie_friction_step"] += 1
    scale = env.extras["friction_scale"]

    if env.extras["zombie_friction_step"] % check_interval != 0:
        return float(scale)

    mean_vx = _proj_vel_ema(env)  # sustained speed along the reference (jaw) axis
    mean_episode_len = env.episode_length_buf.float().mean().item()
    if mean_vx > min_fraction * get_v_target(env).mean() and mean_episode_len > 0.5 * env.max_episode_length:
        scale = torch.clamp(scale - step_size, min=scale_final)
        env.extras["friction_scale"] = scale
    return float(scale)


def difficulty_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    start_after_v: float = 0.09,
    step_size: float = 0.05,
    check_interval: int = 100,
    min_fraction: float = 0.6,
    min_episode_fraction: float = 0.5,
    fallen_max: float = 0.5,
    push_max: float = 1.0,
) -> float:
    """Ramp the recovery/robustness difficulty 0 -> 1, ONLY after the crawl is established.

    This is the single "difficulty" dial the recovery curriculum turns: it drives two env knobs
    consumed elsewhere:

    * ``env.extras["fallen_frac"]`` -- fraction of resets that start toppled (on_side/face_down),
      read by :func:`reset_start_pose` to recompute its pose_weights.
    * ``env.extras["push_scale"]`` -- multiplier on the external-force disturbance, read by
      :func:`disturb_all_bodies`.

    The whole point is STAGING that the from-scratch run showed was what the manual resume-chain was
    doing by hand: the fragile forward gait must appear FIRST, so difficulty only advances once
    ``v_target`` has essentially reached its final value (the robot is crawling well). Before that
    the policy gets no fallen starts and no disturbance, exactly like the early phase of the
    checkpoint chain. ``start_after_v`` is that gate.
    """
    if "difficulty" not in env.extras:
        env.extras["difficulty"] = 0.0
        env.extras["fallen_frac"] = 0.0
        env.extras["push_scale"] = 0.0
        env.extras["zombie_difficulty_step"] = 0
        # one-shot sanity print: proves which zombie_crawl source tree this run imported
        import zombie_crawl

        print(
            f"[CURRICULUM] difficulty term ACTIVE -- zombie_crawl imported from: "
            f"{zombie_crawl.__file__}",
            flush=True,
        )
    env.extras["zombie_difficulty_step"] += 1

    if env.extras["zombie_difficulty_step"] % check_interval != 0:
        return float(env.extras["difficulty"])

    v_now = float(get_v_target(env).mean())
    # gate: only start turning up difficulty once the robot is genuinely crawling
    if v_now < start_after_v:
        return float(env.extras["difficulty"])

    mean_vx = _proj_vel_ema(env)
    mean_episode_len = env.episode_length_buf.float().mean().item()
    max_len = env.cfg.episode_length_s / env.cfg.decimation / env.cfg.sim.dt
    if mean_vx > min_fraction * v_now and mean_episode_len > min_episode_fraction * max_len:
        d = min(1.0, env.extras["difficulty"] + step_size)
        env.extras["difficulty"] = d
        env.extras["fallen_frac"] = fallen_max * d
        env.extras["push_scale"] = push_max * d
    return float(env.extras["difficulty"])
