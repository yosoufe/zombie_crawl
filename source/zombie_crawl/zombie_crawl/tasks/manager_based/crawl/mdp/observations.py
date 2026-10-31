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


def yaw_rate_cmd(env: ManagerBasedEnv, scale: float = 1.0 / 0.3) -> torch.Tensor:
    """The commanded yaw rate [rad/s] scaled by 1/0.3 (the actor's ONLY command input).

    Shape ``(num_envs, 1)``. This replaces the phase-1 2-D ``heading_error_wrist`` term:
    a rate command is hardware-issuable (an operator/handicapper sends rad/s), needs no
    absolute heading reference, and no measured heading/heading error/heading rate is
    ever fed to the actor (the QST QMI8658 IMU has no magnetometer). The scaling matches
    the provisional ``yaw_rate_max_final = 0.3`` rad/s so the input sits in ~[-1, 1].
    """
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        return cm.get_command("yaw_rate_cmd") * scale
    return torch.zeros(env.num_envs, 1, device=env.device)


def vx_cmd(env: ManagerBasedEnv, scale: float = 1.0 / 0.15) -> torch.Tensor:
    """The commanded base forward speed [m/s] scaled by 1/0.15 (phase 3, B.6).

    Shape ``(num_envs, 1)``. With ``ModeMixCommand`` active this is the second command
    input; the policy group grows 25-D -> 26-D. Falls back to zeros when the command
    term is the legacy phase-2 one (keeps ``check_dims`` runs of old trees alive).
    """
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        term = cm.get_term("yaw_rate_cmd")
        if hasattr(term, "vx_command"):
            return term.vx_command * scale
    return torch.zeros(env.num_envs, 1, device=env.device)


def mode_onehot_priv(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged critic input: the current behavior-mode one-hot (M1..M4), ``(N, 4)``.

    From :class:`~.commands.ModeMixCommand`; MUST NOT enter the policy group (the real
    robot's command stream is ``(vx, yawrate)``, not a mode label -- the mode is implied
    by which channel is deadbanded).
    """
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        term = cm.get_term("yaw_rate_cmd")
        if hasattr(term, "mode_onehot"):
            return term.mode_onehot
    return torch.zeros(env.num_envs, 4, device=env.device)


def claw_heading_rate_priv(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Privileged critic input: the claw heading rate ``r`` [rad/s] (``env.zombie_r_ema``).

    Simulation-only measurement maintained by ``mdp._update_claw_heading_rate``; MUST NOT
    enter the policy group (the real robot cannot observe the claw heading rate without
    external localization).
    """
    from .rewards import claw_heading_rate

    return claw_heading_rate(env).unsqueeze(-1)
