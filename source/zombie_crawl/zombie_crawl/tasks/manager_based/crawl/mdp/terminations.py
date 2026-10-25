# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the zombie crawl task.

Note: the robot is *not* terminated for tipping over -- a crawler is allowed to flip
itself and the gait is learned freely. Only numerical failure or being flung into the air
terminate the episode. The position box (``out_of_bounds``) was removed from the phase-2
termination config (P2-7): success is a logged metric (``crawl_success``) that drives the
curriculum; the function is kept only so ``use_box_termination`` can bisect (§A.11).
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
    """True if the robot leaves its bounding area around the environment origin.

    Removed from the termination config in phase 2 (P2-7); re-enabled only via
    ``env.use_box_termination=true`` for diagnostic runs.
    """
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
    """True if the base has not moved ``min_displacement`` (EUCLIDEAN, horizontal) from the
    window anchor within ``stall_time_s`` (P2-8).

    Axis-free by construction: the distance is Euclidean, so a rotating claw axis cannot
    fake progress (pitfall X3), and it does not care which way the robot was commanded --
    it applies to fallen starts too. The anchor (position + ``episode_length_buf`` step)
    is recorded by ``mdp.reset_start_pose`` on every reset (provably runs); the old
    ``episode_length_buf <= 1`` heuristic and the mid-episode command-resample re-anchor
    are gone -- this term does NOT re-anchor on command changes. At each check past the
    window the env either terminates (no movement -> failure, non-bootstrapped, so
    "doing nothing" is bad rather than a free ride) or re-anchors (anchor = now) and
    continues, giving a rolling 12 s window.
    """
    asset = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w.torch[:, :2]
    el = env.episode_length_buf
    anchor_xy = getattr(env, "zombie_stall_anchor_xy", None)
    if anchor_xy is None or anchor_xy.shape != pos_xy.shape or env.zombie_stall_anchor_step.shape != el.shape:
        # first steps before the initial reset event populated the anchors: adopt now
        env.zombie_stall_anchor_xy = pos_xy.clone()
        env.zombie_stall_anchor_step = el.clone()
        return torch.zeros(pos_xy.shape[0], dtype=torch.bool, device=pos_xy.device)
    stall_steps = int(round(stall_time_s / (env.cfg.decimation * env.cfg.sim.dt)))
    disp = (pos_xy - anchor_xy).norm(dim=1)
    elapsed = el - env.zombie_stall_anchor_step
    past = elapsed > stall_steps
    terminate = past & (disp < min_displacement)
    re_anchor = past & ~terminate
    if bool(re_anchor.any()):
        anchor_xy[re_anchor] = pos_xy[re_anchor]
        env.zombie_stall_anchor_step[re_anchor] = el[re_anchor]
    return terminate
