# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the zombie crawl task: EMA-smoothed joint position targets."""

import torch

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class SmoothedJointPositionAction(JointPositionAction):
    """Joint position action with an EMA low-pass filter on the scaled raw actions.

    The filter state is the *scaled* action (before the offset), so the target remains
    ``default + scale * a_smooth``. The filter state is zeroed on environment reset.
    """

    def __init__(self, cfg: "SmoothedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._ema_state = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        scaled = self._raw_actions * self._scale
        alpha = self.cfg.ema_alpha
        if alpha >= 1.0:
            smoothed = scaled
        else:
            smoothed = alpha * scaled + (1.0 - alpha) * self._ema_state
        self._ema_state[:] = smoothed
        self._processed_actions = smoothed + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: "torch.Tensor | None" = None) -> None:
        super().reset(env_ids)
        if env_ids is not None:
            self._ema_state[env_ids] = 0.0
        else:
            self._ema_state.zero_()


@configclass
class SmoothedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`SmoothedJointPositionAction`."""

    class_type: type = SmoothedJointPositionAction

    ema_alpha: float = 0.3
    """EMA coefficient in (0, 1]; 1.0 disables smoothing, smaller = smoother (heavier)."""
