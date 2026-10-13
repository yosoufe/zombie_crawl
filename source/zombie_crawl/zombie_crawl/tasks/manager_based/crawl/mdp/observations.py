# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTOR.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the zombie crawl task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def ground_friction(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-environment effective ground friction (static) sampled at reset/startup.

    The value is stored by the ``randomize_ground_friction`` event on
    ``env.extras["ground_friction"]``.
    """
    if "ground_friction" not in env.extras:
        env.extras["ground_friction"] = torch.ones(env.num_envs, device=env.device)
    return env.extras["ground_friction"].unsqueeze(-1)


def base_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Height of the base origin above the ground plane (z = 0)."""
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2:3]


def gait_phase_clock(env: ManagerBasedEnv, frequency: float = 0.5) -> torch.Tensor:
    """Sinusoidal phase clock (sin, cos) advanced by the policy step rate.

    Optionally appended to the actor observations to encourage rhythmic motion.
    The phase advances deterministically every environment step (no randomness),
    so all envs share the same phase -- it acts as a metronome, not as state.
    """
    state = getattr(env, "zombie_phase", None)
    if state is None:
        env.zombie_phase = torch.zeros(env.num_envs, device=env.device)
        env.zombie_phase_freq = frequency
    env.zombie_phase = (env.zombie_phase + 2.0 * math.pi * env.zombie_phase_freq * env.step_dt) % (
        2.0 * math.pi
    )
    return torch.stack([torch.sin(env.zombie_phase), torch.cos(env.zombie_phase)], dim=1)


def heading_error_wrist(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("wrist",)),
) -> torch.Tensor:
    """Commanded-heading observation: (cos, sin) of the angle between where the robot POINTS
    and the commanded direction.

    Pointing direction = world-frame horizontal projection of the arrow from the base origin
    to the WRIST body (down the arm toward the jaws). ``env.zombie_ref_yaw`` (drawn by
    ``mdp.reset_start_pose``) is a uniform +-180 deg COMMAND per episode, so this term IS the
    command signal for the actor: (1, 0) means "point where you must crawl". It matches what
    the ``wrist_pointing_error`` reward measures. Returns (num_envs, 2); cos/sin are wrap-invariant.
    """
    asset = env.scene[asset_cfg.name]
    wrist_ids = getattr(env, "zombie_wrist_body_ids", None)
    if wrist_ids is None:
        ids, _names = asset.find_bodies(list(asset_cfg.body_names), as_proxy=True)
        wrist_ids = ids.torch.reshape(-1)
        if wrist_ids.numel() == 0:
            raise ValueError(f"heading_error_wrist: no body matching {asset_cfg.body_names}")
        env.zombie_wrist_body_ids = wrist_ids
    wrist = asset.data.body_pos_w.torch[:, wrist_ids[0]]
    base = asset.data.root_pos_w.torch
    arrow_yaw = torch.atan2(wrist[:, 1] - base[:, 1], wrist[:, 0] - base[:, 0])
    ref = getattr(env, "zombie_ref_yaw", None)
    if ref is None or ref.shape[0] != base.shape[0]:
        ref = torch.zeros(base.shape[0], device=base.device)
    d = arrow_yaw - ref
    return torch.stack([torch.cos(d), torch.sin(d)], dim=-1)
