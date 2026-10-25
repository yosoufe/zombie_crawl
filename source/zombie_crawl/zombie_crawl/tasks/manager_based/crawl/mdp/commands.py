# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the zombie crawl task.

``UniformYawRateCommand`` is the phase-2 command: a per-environment commanded YAW RATE
(rad/s) that the actor must track with the claw heading rate ``r``
(``mdp.claw_heading_rate``). Unlike the removed heading-error input, a rate command is
proprioception-only -- the robot never needs external localization or an absolute yaw
reference (the QMI8658 IMU has no magnetometer, so absolute yaw is unobservable on
hardware anyway).

Every (re-)sample: with probability ``zero_prob`` the command is exactly 0, otherwise
it is drawn uniformly from ``(-r_max, +r_max)`` with ``r_max =
env.extras["yaw_rate_max"]`` (a dial owned by the yaw-rate stage curriculum,
:func:`~.curriculums.yaw_rate_stage_curriculum`). While ``yaw_rate_max`` is 0 every
command is 0, i.e. the steering stage is locked off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformYawRateCommand(CommandTerm):
    """Per-env commanded yaw rate (rad/s), resampled uniformly mid-episode.

    ``command`` is ``(num_envs, 1)``. ``just_resampled`` marks the envs whose command
    changed on the most recent ``compute()`` step: ``yaw_rate_tracking`` restarts its
    post-resample mask timer for those envs (the filtered claw heading rate lags a fresh
    command, so the tracking reward must not pay before the filter settles).

    ``pin(value)`` freezes the command at ``value`` for every env (used by ``play.py
    --yaw_rate`` and the evaluation script); ``unpin()`` restores stochastic resampling.
    """

    def __init__(self, cfg: UniformYawRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pinned: float | None = None

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Commanded yaw rate [rad/s]. Shape is (num_envs, 1)."""
        return self.yaw_rate.unsqueeze(-1)

    """
    Operations.
    """

    def compute(self, dt: float):
        # the flag describes only THIS step's resamples; mid-episode resamples are set
        # here, episode resets are re-anchored directly by mdp.reset_start_pose
        self.just_resampled.zero_()
        if self._pinned is not None:
            # pinned: keep the countdown armed-but-unfiring so no stochastic resample
            # can overwrite the constant command
            self._update_metrics()
            self.time_left[:] = float("inf")
            return
        super().compute(dt)

    def pin(self, value: float):
        """Freeze the command at ``value`` rad/s for every environment."""
        self._pinned = float(value)
        self.yaw_rate[:] = self._pinned
        self.time_left[:] = float("inf")

    def unpin(self):
        """Resume stochastic resampling with staggered countdowns."""
        self._pinned = None
        lo, hi = self.cfg.resampling_time_range
        self.time_left.uniform_(lo, hi)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # CommandManager.reset logs mean of each metric per episode
        # (Metrics/yaw_rate_cmd/abs, Metrics/yaw_rate_cmd/resamples_per_episode)
        self.metrics["abs"] = self.yaw_rate.abs()
        self.metrics["resamples_per_episode"] = self.command_counter.to(self.yaw_rate.dtype)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if self._pinned is not None:
            self.yaw_rate[env_ids] = self._pinned
            self.just_resampled[env_ids] = True
            return
        r_max = float(getattr(self._env, "extras", {}).get("yaw_rate_max", 0.0))
        cmd = math_utils.sample_uniform(-r_max, r_max, (num_envs,), device=self.device)
        zero_draw = torch.rand(num_envs, device=self.device) < self.cfg.zero_prob
        cmd[zero_draw] = 0.0
        cmd[r_max <= 0.0] = 0.0
        self.yaw_rate[env_ids] = cmd
        self.just_resampled[env_ids] = True

    def _update_command(self):
        pass


@configclass
class UniformYawRateCommandCfg(CommandTermCfg):
    """Config for the uniform yaw-rate command term."""

    class_type: type[CommandTerm] = UniformYawRateCommand

    zero_prob: float = 0.2
    """Probability that a (re-)sampled command is exactly 0 (stationary / straight target)."""
