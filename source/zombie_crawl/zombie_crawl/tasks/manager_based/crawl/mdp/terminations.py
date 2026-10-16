# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the zombie crawl task.

Note: the robot is *not* terminated for tipping over -- a crawler is allowed to flip
itself and the gait is learned freely. Only numerical failure, being flung into the air,
or leaving the bounding area terminate the episode.
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
    """True if the base has not crawled ``min_displacement`` along its reference (jaw)
    axis within ``stall_time_s``.

    Records each env's start position on the first step after reset (when
    ``episode_length_buf`` is still <= 1) and terminates episodes that stall out of their
    start position. Displacement is measured along the frozen reference axis, not world
    +X: with uniform ±180deg spawn yaw a world-axis test kills successful crawlers whose
    heading happens to be perpendicular to +X. This is a *failure* termination (not a
    timeout), so the value function treats "doing nothing" as bad rather than as a free ride.
    """
    from .rewards import crawl_heading_resample_flag, get_ref_dir

    asset = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w.torch[:, :2]
    el = env.episode_length_buf
    start_xy = getattr(env, "zombie_start_xy", None)
    if start_xy is None or start_xy.shape[0] != pos_xy.shape[0]:
        env.zombie_start_xy = pos_xy.clone()
        # episode step at which each env's current stall window opened (episode_length_buf
        # is env-owned and drives time_out; we keep our own anchor instead of touching it)
        env.zombie_stall_anchor = torch.zeros_like(el)
    else:
        if not hasattr(env, "zombie_stall_anchor") or env.zombie_stall_anchor.shape != el.shape:
            env.zombie_stall_anchor = torch.zeros_like(el)
        just_reset = el <= 1
        if just_reset.any():
            env.zombie_start_xy[just_reset] = pos_xy[just_reset]
            env.zombie_stall_anchor[just_reset] = el[just_reset]
        # with a dynamic commanded heading, displacement along the OLD axis says nothing
        # about the NEW command: re-anchor position AND restart the stall window on resample
        resampled = crawl_heading_resample_flag(env)
        if resampled is not None and resampled.any():
            env.zombie_start_xy[resampled] = pos_xy[resampled]
            env.zombie_stall_anchor[resampled] = el[resampled]
    cos_ref, sin_ref = get_ref_dir(env)
    dx = (pos_xy[:, 0] - env.zombie_start_xy[:, 0]) * cos_ref + (
        pos_xy[:, 1] - env.zombie_start_xy[:, 1]
    ) * sin_ref
    stall_steps = int(round(stall_time_s / (env.cfg.decimation * env.cfg.sim.dt)))
    return (el - env.zombie_stall_anchor > stall_steps) & (dx.abs() < min_displacement)
