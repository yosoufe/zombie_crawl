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
