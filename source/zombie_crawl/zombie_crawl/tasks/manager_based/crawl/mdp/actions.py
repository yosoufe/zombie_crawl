# Copyright (2026). Custom action term for the zombie crawl task.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the zombie crawl task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class OffsetEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """Limits-relative EMA action with a NEUTRAL command that is not the midpoint.

    The stock mapping sends a zero action to the (soft) joint-limit midpoint. The task's
    only spawn pose -- URDF zero -- sits at the LIMIT extremes of the asymmetric joints,
    so a zero action at spawn meant a stand-up kick within ~0.25 s of every reset. This
    term shifts the raw action by a limits-relative offset computed from the actuator
    soft limits so that ``a = 0`` commands the configured neutral angles (absolute rad).
    """

    cfg: "OffsetEMAJointPositionToLimitsActionCfg"

    def __init__(self, cfg: "OffsetEMAJointPositionToLimitsActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # resolve per-joint neutral angles (absolute rad) in this term's joint order
        neutral = torch.zeros(1, self.action_dim, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.neutral, self._joint_names, preserve_order=cfg.preserve_order
        )
        neutral[0, index_list] = torch.tensor(value_list, device=self.device)
        # limits-relative offset: target = mid + (a + o) * half  ==  neutral  when a == 0
        limits = self._asset.data.soft_joint_pos_limits.torch
        low = limits[:, self._joint_ids, 0]
        high = limits[:, self._joint_ids, 1]
        mid = 0.5 * (low + high)
        half = (0.5 * (high - low)).clamp_min(1e-6)
        self._offset_limits_rel = (neutral - mid) / half
        # NOTE: the term's internal _prev_applied_actions (seeded at reset from the
        # current joint positions) and the soft-limit clamp in process_actions both keep
        # the EMA history consistent with the shifted action coordinates.

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions + self._offset_limits_rel)


@configclass
class OffsetEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    """EMA limits-relative position action with a neutral-angle offset."""

    class_type: type | str = (
        "zombie_crawl.tasks.manager_based.crawl.mdp.actions:OffsetEMAJointPositionToLimitsAction"
    )

    neutral: dict[str, float] = None
    """Absolute joint positions (rad) that a zero action should command."""
