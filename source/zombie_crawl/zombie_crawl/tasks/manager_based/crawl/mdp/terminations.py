# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the zombie crawl task.

Note: the robot is *not* terminated for tipping over -- a crawler is allowed to flip
itself. Only numerical failure, being flung into the air, or leaving the bounding
area terminate the episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def base_nan(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """True if any root or joint state entry is NaN or infinite."""
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.root_state_w.torch
    joint_pos = asset.data.joint_pos.torch
    joint_vel = asset.data.joint_vel.torch
    return ~torch.isfinite(root_state).all(dim=1) | ~torch.isfinite(joint_pos).all(dim=1) | ~torch.isfinite(
        joint_vel
    ).all(dim=1)


def base_too_high(
    env: ManagerBasedEnv,
    height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True if the base origin is flung well above its resting height (physics exploit guard)."""
    z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
    return z > height


def out_of_bounds(
    env: ManagerBasedEnv,
    x_range: float = 2.5,
    y_range: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True if the robot leaves its bounding area around the environment origin."""
    asset = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w.torch
    origins = env.scene.env_origins.to(pos_w.device)
    dx = (pos_w[:, 0] - origins[:, 0]).abs()
    dy = (pos_w[:, 1] - origins[:, 1]).abs()
    return (dx > x_range) | (dy > y_range)


def stalled(
    env: ManagerBasedEnv,
    min_displacement: float = 0.05,
    stall_time_s: float = 12.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True if the base has not crawled ``min_displacement`` along +X within ``stall_time_s``.

    Records each env's start x on the first step after reset (when ``episode_length_buf``
    is still <= 1, i.e. the reset pose has just been applied) and terminates episodes that
    stall out of their start position. This is a *failure* termination (not a timeout), so
    the value function treats "doing nothing" as bad rather than as a free ride.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w.torch[:, 0]
    start_x = getattr(env, "zombie_start_x", None)
    if start_x is None or start_x.shape[0] != x.shape[0]:
        env.zombie_start_x = x.clone()
    else:
        just_reset = env.episode_length_buf <= 1
        if just_reset.any():
            env.zombie_start_x[just_reset] = x[just_reset]
    stall_steps = int(round(stall_time_s / (env.cfg.decimation * env.cfg.sim.dt)))
    return (env.episode_length_buf > stall_steps) & ((x - env.zombie_start_x).abs() < min_displacement)
