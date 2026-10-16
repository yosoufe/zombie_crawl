# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the zombie crawl task.

``UniformCrawlHeadingCommand`` is the framework-native replacement for the frozen
``env.zombie_ref_yaw`` direction drawn by ``mdp.reset_start_pose``: the commanded crawl
heading is a per-environment uniform +-180 deg draw that, unlike the frozen version, is
RESAMPLED mid-episode (per env, staggered) every ``resampling_time_range`` seconds by the
Isaac Lab :class:`~isaaclab.managers.command_manager.CommandManager`.

Everything directional (rewards, stall gate, curriculum gates, the cyan video arrow) reads
the command through ``mdp.get_ref_dir``, which prefers this term when it is active, so the
policy sees the NEW target on the very next observation without any extra channel: the
``heading_error_wrist`` term is (cos, sin) of claw-minus-command and simply moves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformCrawlHeadingCommand(CommandTerm):
    """Per-env commanded crawl heading (rad), resampled uniformly mid-episode.

    ``command`` is ``(num_envs, 1)`` yaw. ``just_resampled`` marks the envs whose command
    changed on the most recent ``compute()`` step (or episode reset): the stateful terms
    that accumulate progress along the command axis (``forward_progress_lpf``, ``stalled``,
    the curriculum velocity EMA) use it to re-anchor for those envs instead of silently
    mixing two different axes.
    """

    def __init__(self, cfg: UniformCrawlHeadingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.heading = torch.zeros(self.num_envs, device=self.device)
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_open = False

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Commanded heading [rad]. Shape is (num_envs, 1)."""
        return self.heading.unsqueeze(-1)

    """
    Operations.
    """

    def compute(self, dt: float):
        # the flag describes only THIS step's resamples; reset() also flips it (episode
        # restart = new axis), which the stateful readers handle as a re-anchor
        self.just_resampled.zero_()
        if not self._gate_open and not self._curriculum_gate_open():
            # early curriculum: hold each episode's heading frozen -- pin the countdown to
            # inf so the base never fires; reset() still draws a fresh heading per episode,
            # which is exactly the frozen (pre-dynamic) behaviour
            self._update_metrics()
            self.time_left[:] = float("inf")
            return
        if not self._gate_open:
            # gate just opened: start staggered countdowns so the first mid-episode jumps
            # spread over [lo, hi] seconds rather than firing simultaneously
            self._gate_open = True
            lo, hi = self.cfg.resampling_time_range
            self.time_left.uniform_(lo, hi)
        super().compute(dt)

    def _curriculum_gate_open(self) -> bool:
        """Mid-episode resampling unlocks at ``curriculum_gate_frac`` of the global difficulty.

        Staging mirrors :func:`~...curriculums.difficulty_curriculum` (fallen spawns fill
        over [0, 0.5], pushes over [0.5, 1]): the command may only start JUMPING once the
        policy already crawls reliably under disturbances. A missing ``difficulty`` extra
        (curriculum disabled) counts as fully trained, so ``use_curriculum=False`` runs get
        the dynamic heading immediately.
        """
        frac = float(getattr(self.cfg, "curriculum_gate_frac", 0.0))
        if frac <= 0.0:
            return True
        extras = getattr(self._env, "extras", {}) or {}
        try:
            return float(extras.get("difficulty", 1.0)) >= frac
        except (TypeError, ValueError):
            return True

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # CommandManager.reset logs mean of each metric per episode (Metrics/crawl_heading/*)
        self.metrics["heading_abs"] = self.heading.abs()
        self.metrics["resamples_per_episode"] = self.command_counter.to(self.heading.dtype)
        self.metrics["gate_open"] = torch.full(
            (self.num_envs,), float(self._gate_open), device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int] | slice):
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        self.heading[env_ids] = math_utils.sample_uniform(
            self.cfg.heading_range[0], self.cfg.heading_range[1], (num_envs,), device=self.device
        )
        self.just_resampled[env_ids] = True

    def _update_command(self):
        pass


@configclass
class UniformCrawlHeadingCommandCfg(CommandTermCfg):
    """Config for the uniform crawl-heading command term."""

    class_type: type[CommandTerm] = UniformCrawlHeadingCommand

    heading_range: tuple[float, float] = (-math.pi, math.pi)
    """Uniform range [rad] the commanded heading is drawn from on every (re-)sample."""

    curriculum_gate_frac: float = 0.75
    """Global-curriculum difficulty in [0, 1] at which mid-episode resampling unlocks.

    Below it the term behaves exactly like the frozen per-episode command (one heading
    per episode, drawn at reset). 0.0 disables the gate. Staggered after the difficulty
    curriculum's other channels (fallen spawns [0, 0.5], pushes [0.5, 1]).
    """
