# Copyright (2026). Custom action term for the zombie crawl task.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the zombie crawl task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class OffsetEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """EMA joint-position action where zero = a NEUTRAL angle and ±1 = the joint's soft limits.

    The stock limits-relative mapping sends a zero action to the limit MIDPOINT, which for
    this task's one-sided joints (lift [0, 3.5], elbow [-3.14, 0]) is far from the folded
    spawn -- every episode opened with a stand-up kick. A limits-relative OFFSET fixes the
    neutral point but, being affine inside one shared [-1, 1] window, clips away ~half of
    every one-sided joint's range (measured: lift/elbow 47% coverage), which disabled the
    rearward claw-drag gait and made a sideways pan-yawing exploit the optimal policy.

    This term instead maps BOTH action halves asymmetrically around the neutral angle:

        target = neutral + a * (high - neutral)    for a >= 0
        target = neutral + a * (neutral - low)     for a < 0

    so a=0 commands the neutral pose and a=±1 reach the joint's full soft limits. The EMA
    smoothing and soft-limit clamping of the parent are reproduced exactly.
    """

    cfg: "OffsetEMAJointPositionToLimitsActionCfg"

    def __init__(self, cfg: "OffsetEMAJointPositionToLimitsActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        neutral = torch.zeros(1, self.action_dim, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.neutral, self._joint_names, preserve_order=cfg.preserve_order
        )
        neutral[0, index_list] = torch.tensor(value_list, device=self.device)
        limits = self._asset.data.soft_joint_pos_limits.torch
        low = limits[:, self._joint_ids, 0]
        high = limits[:, self._joint_ids, 1]
        self._neutral = neutral
        self._up_span = (high - neutral).clamp_min(0.0)
        self._down_span = (neutral - low).clamp_min(0.0)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped = actions.clamp(-1.0, 1.0)
        target = self._neutral + torch.where(
            clipped >= 0, clipped * self._up_span, clipped * self._down_span
        )
        alpha = self._alpha
        # phase-3 (B-12): under a zero command the joint-position target is low-passed
        # HARDER (ema_alpha_zero). rsl-rl's policy noise is one global per-dim scalar, so
        # the gait's exploration sigma (~1.1 on the +/-1 clip) physically prevents rest
        # (needs ~0.05): five full reward-shaping configs, 3000 iters each, left
        # m4_qerr_max frozen at 2.76. The command filter attenuates SAMPLED noise by the
        # IIR gain a/(2-a) while the policy's MEAN still sets the steady-state target --
        # rest becomes reachable without muting what the policy commands (and real
        # actuators have command filters, so it is sim2real-plausible).
        za = getattr(self.cfg, "ema_alpha_zero", None)
        if za is not None:
            term = self._env.command_manager.get_term("yaw_rate_cmd")
            zero = getattr(term, "zero_cmd_raw", None)
            if zero is not None:
                base = torch.as_tensor(self._alpha, device=self.device).expand(self.num_envs, 1)
                alpha = torch.where(
                    zero.unsqueeze(-1) > 0.5, torch.full_like(base, float(za)), base
                )
        ema = alpha * target + (1.0 - alpha) * self._prev_applied_actions
        limits = self._asset.data.soft_joint_pos_limits.torch
        self._processed_actions[:] = torch.clamp(
            ema, limits[:, self._joint_ids, 0], limits[:, self._joint_ids, 1]
        )
        self._prev_applied_actions[:] = self._processed_actions[:]


@configclass
class OffsetEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    """EMA limits-relative position action with a neutral-angle two-sided map."""

    class_type: type | str = (
        "zombie_crawl.tasks.manager_based.crawl.mdp.actions:OffsetEMAJointPositionToLimitsAction"
    )

    neutral: dict[str, float] = None
    """Absolute joint positions (rad) that a zero action should command."""

    ema_alpha_zero: float | None = None
    """Extra EMA weight used while the command is zero (M4 rest). None = same as alpha.
    Small values low-pass the SAMPLED exploration noise under zero command so rest is
    physically reachable despite the global policy std (see process_actions, B-12)."""
