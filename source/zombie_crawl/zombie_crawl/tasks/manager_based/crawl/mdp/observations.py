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
