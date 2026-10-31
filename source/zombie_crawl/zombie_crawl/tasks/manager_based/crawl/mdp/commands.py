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


# ----------------------------------------------------------------------------------------
# Phase 3 (spec B.1/B.7): mode-mix commands (vx_cmd, yaw_rate_cmd) over M1..M4.
# ----------------------------------------------------------------------------------------

MODE_NAMES: tuple[str, ...] = ("M1", "M2", "M3", "M4")
"""M1 forward, M2 forward+turn, M3 rotate in place, M4 rest."""


class ModeMixCommand(CommandTerm):
    """Per-env mode-mix command ``(vx_cmd, yaw_rate_cmd)`` (P3-2).

    Instantiated under the phase-2 term NAME ``yaw_rate_cmd`` so every existing consumer
    (rewards, metrics, observations, evaluation) keeps reading ``command`` = the yaw rate
    (``num_envs, 1``); the forward command, integer mode and deadband indicators are extra
    accessors. Each (re-)sample draws a behavior mode from ``env.extras["mode_mix"]``
    (owned by the mode-stage curriculum; falls back to ``cfg.mode_mix``) and fills:

    ====  ======================  ==========================================
    M     (vx_cmd, yaw_rate_cmd)  duration
    ====  ======================  ==========================================
    M1    (v_fwd, ~0+noise)       uniform ``resampling_time_range`` (5-8 s)
    M2    (v_fwd, ω), |ω|∈[lo,cap]  uniform ``resampling_time_range``
    M3    (0+noise, ω), |ω|∈[m3_lo,cap]  uniform ``env.extras["rotate_segment_s"]``
    M4    (0+noise, ~0+noise)     uniform ``resampling_time_range``
    ====  ======================  ==========================================

    ``v_fwd`` is the phase-2 ``v_target`` dial (per-env); ``cap`` is the phase-2
    ``yaw_rate_max`` dial (still the |ω| ceiling). Small noise near zero
    (``|noise| < deadband``) keeps the deadband indicators exercised. The integer ``mode``
    is stored per env for logging/metrics; indicators ``fwd_on`` / ``is_zero_cmd`` /
    ``is_rotate_only`` derive from the cfg deadbands (P3-1). 12 ordered transition pairs
    are counted per resample.

    ``pin``/``unpin`` keep the phase-2 play/eval API: ``pin(ω)`` freezes a FORWARD command
    with that yaw rate (M1/M2 semantics), ``pin_mode(mode, ...)`` freezes an arbitrary one.
    """

    def __init__(self, cfg: ModeMixCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.yaw_rate = torch.zeros(n, device=self.device)
        self.vx = torch.zeros(n, device=self.device)
        self.mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.prev_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.just_resampled = torch.zeros(n, dtype=torch.bool, device=self.device)
        # transitions observed at command resamples (excluding the initial draw): [from, to]
        self.transition_counts = torch.zeros(4, 4, dtype=torch.long, device=self.device)
        self._pinned: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Commanded yaw rate [rad/s] -- phase-2 compatible shape (num_envs, 1)."""
        return self.yaw_rate.unsqueeze(-1)

    @property
    def vx_command(self) -> torch.Tensor:
        """Commanded forward speed [m/s] along the claw axis. Shape (num_envs, 1)."""
        return self.vx.unsqueeze(-1)

    @property
    def mode_onehot(self) -> torch.Tensor:
        """Privileged one-hot of the current mode. Shape (num_envs, 4).

        Stored ids are 1..4 (0 = pre-first-draw), so the 5-wide one-hot drops column 0;
        unassigned envs get an all-zero row.
        """
        oh = torch.nn.functional.one_hot(self.mode, 5).to(self.vx.dtype)[:, 1:]
        return torch.where((self.mode >= 1).unsqueeze(-1), oh, torch.zeros_like(oh))

    @property
    def fwd_on(self) -> torch.Tensor:
        """Raw indicator |vx_cmd| > deadband (1.0/0.0)."""
        return (self.vx > self.cfg.vx_deadband).to(self.vx.dtype)

    @property
    def zero_cmd_raw(self) -> torch.Tensor:
        """Raw indicator: |vx_cmd| <= deadband AND |yaw_rate_cmd| <= yaw deadband."""
        return ((self.vx.abs() <= self.cfg.vx_deadband) & (self.yaw_rate.abs() <= self.cfg.yaw_deadband)).to(
            self.vx.dtype
        )

    @property
    def rotate_raw(self) -> torch.Tensor:
        """Raw indicator: |vx_cmd| <= deadband AND |yaw_rate_cmd| > yaw deadband."""
        return ((self.vx.abs() <= self.cfg.vx_deadband) & (self.yaw_rate.abs() > self.cfg.yaw_deadband)).to(
            self.vx.dtype
        )

    """
    Operations.
    """

    def compute(self, dt: float):
        self.just_resampled.zero_()
        if self._pinned is not None:
            vx, yaw, mode = self._pinned
            self.vx[:] = vx
            self.yaw_rate[:] = yaw
            self.mode[:] = mode
            self._update_metrics()
            self.time_left[:] = float("inf")
            return
        super().compute(dt)

    def pin(self, value: float):
        """Phase-2 compatible: freeze a FORWARD command (vx = v_fwd) with this yaw rate."""
        from .rewards import get_v_target

        vx = get_v_target(self._env).clone()
        yaw = torch.full_like(vx, float(value))
        mode = torch.where(yaw.abs() > self.cfg.yaw_deadband, 2, 1)  # M2 / M1 (1-based ids)
        self._pinned = (vx, yaw, mode)
        self.vx[:], self.yaw_rate[:], self.mode[:] = vx, yaw, mode
        self.time_left[:] = float("inf")

    def pin_mode(self, mode: int, vx: float | None = None, yaw: float | None = None):
        """Freeze every env into one behavior; ``vx``/``yaw``=None take the mode's canonical
        values (M1/M2 forward at v_fwd, M2/M3 |ω| from the arguments or the dial cap)."""
        from .rewards import get_v_target

        v_fwd = get_v_target(self._env)
        assert 1 <= int(mode) <= 4, f"mode must be 1..4 (M1..M4), got {mode}"
        mode_t = torch.full_like(v_fwd, int(mode), dtype=torch.long)
        if vx is None:
            vx_t = torch.where(mode_t <= 2, v_fwd, torch.zeros_like(v_fwd))
        else:
            vx_t = torch.full_like(v_fwd, float(vx))
        yaw_t = torch.full_like(v_fwd, 0.0 if yaw is None else float(yaw))
        self._pinned = (vx_t, yaw_t, mode_t)
        self.vx[:], self.yaw_rate[:], self.mode[:] = vx_t, yaw_t, mode_t
        self.time_left[:] = float("inf")

    def unpin(self):
        """Resume stochastic mode mixing."""
        self._pinned = None
        lo, hi = self.cfg.resampling_time_range
        self.time_left.uniform_(lo, hi)

    """
    Implementation specific functions.
    """

    def _dial(self, key: str, default):
        return getattr(self._env, "extras", {}).get(key, default)

    def _resample(self, env_ids):
        """Per-env: draw mode, command values, and a per-mode segment duration (base
        ``_resample`` would uniform-fill time_left from one global range)."""
        num = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num == 0:
            return
        self._sample(env_ids, num)
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        # values were written by _sample (single place); the flag describes this step
        self.just_resampled[env_ids] = True

    def _sample(self, env_ids, num: int):
        from .rewards import get_v_target

        dev = self.device
        cap = max(0.0, float(self._dial("yaw_rate_max", 0.0)))
        m3_lo = float(self._dial("m3_omega_min", self.cfg.omega_min_rotate))
        seg_lo, seg_hi = self._dial("rotate_segment_s", tuple(self.cfg.rotate_segment_s))
        mix = self._dial("mode_mix", tuple(self.cfg.mode_mix))
        mix_t = torch.tensor(mix, device=dev, dtype=torch.float32).clamp(min=0.0)
        if float(mix_t.sum()) <= 0.0:
            mix_t = torch.tensor((1.0, 0.0, 0.0, 0.0), device=dev)
        mode0 = torch.multinomial(mix_t / mix_t.sum(), num, replacement=True)  # 0..3
        mode = mode0 + 1  # stored ids: 1..4 (0 = unassigned)
        # rest-start envs (reset_start_pose) draw M4 as their FIRST segment so the folded
        # spawn is actually graded; consumed once, leftover ids survive to later calls
        forced = self._env.extras.get("force_m4_ids")
        if forced is not None and len(forced):
            env_idx = torch.arange(self.num_envs, device=dev)[env_ids]
            hit = torch.isin(env_idx, forced.to(dev))
            if bool(hit.any()):
                mode[hit] = 4
            keep = forced.to(dev)[~torch.isin(forced.to(dev), env_idx)]
            self._env.extras["force_m4_ids"] = keep if keep.numel() else None
        v_fwd = get_v_target(self._env)[env_ids]
        # forward command: v_fwd in M1/M2, 0 + sub-deadband noise in M3/M4
        vx = torch.where(mode <= 2, v_fwd, torch.zeros_like(v_fwd))
        noise_vx = (torch.rand(num, device=dev) - 0.5) * 2.0 * min(self.cfg.vx_noise, self.cfg.vx_deadband)
        vx = vx + torch.where(mode <= 2, torch.zeros_like(vx), noise_vx)
        # yaw command: |omega| per mode, sign uniform, extra sub-deadband noise for M1/M4
        sign = torch.where(torch.rand(num, device=dev) < 0.5, -1.0, 1.0)
        u = torch.rand(num, device=dev)
        mag = torch.zeros_like(vx)
        m2 = mode == 2
        m3 = mode == 3
        lo2 = min(self.cfg.omega_min_forward, cap)
        lo3 = min(m3_lo, cap)
        mag = torch.where(m2, lo2 + u * max(0.0, cap - lo2), mag)
        mag = torch.where(m3, lo3 + u * max(0.0, cap - lo3), mag)
        noise_yaw = (torch.rand(num, device=dev) - 0.5) * 2.0 * min(self.cfg.yaw_noise, self.cfg.yaw_deadband)
        yaw = sign * mag + torch.where((mode == 1) | (mode == 4), noise_yaw, torch.zeros_like(mag))
        # |omega| can never exceed the phase-2 dial
        yaw = torch.clamp(yaw, -cap, cap)
        # per-mode segment duration
        hold_lo, hold_hi = self.cfg.resampling_time_range
        # M4 gets longer windows than B.7's 5-8 s (owner calibration): the 2 s tracking
        # mask + 2 s settle ramp would otherwise leave only 1-2 s of graded time per
        # rest segment -- rest is unlearnable. Dial: env.extras["m4_segment_s"]
        m4_lo, m4_hi = self._dial("m4_segment_s", tuple(self.cfg.m4_segment_s))
        dur_m4 = m4_lo + u * max(0.0, m4_hi - m4_lo)
        dur = torch.where(mode == 3, seg_lo + u * max(0.0, seg_hi - seg_lo),
                          torch.where(mode == 4, dur_m4,
                                      hold_lo + u * max(0.0, hold_hi - hold_lo)))
        self.vx[env_ids] = vx
        self.yaw_rate[env_ids] = yaw
        self.prev_mode[env_ids] = self.mode[env_ids]
        self.mode[env_ids] = mode
        self.time_left[env_ids] = dur
        # 12-pair transition coverage: count mode CHANGES only (self-transitions excluded;
        # prev_mode == 0 is the first draw of an episode, not a transition)
        off = (
            (self.prev_mode[env_ids] != self.mode[env_ids]) & (self.prev_mode[env_ids] > 0)
        ).nonzero().flatten()
        if off.numel() > 0:
            f = self.prev_mode[env_ids][off] - 1
            t = self.mode[env_ids][off] - 1
            self.transition_counts.reshape(-1).index_add_(
                0, (f * 4 + t).to(torch.long), torch.ones(off.numel(), dtype=torch.long, device=dev)
            )

    def _update_metrics(self):
        self.metrics["abs"] = self.yaw_rate.abs()
        self.metrics["vx_abs"] = self.vx.abs()
        self.metrics["resamples_per_episode"] = self.command_counter.to(self.yaw_rate.dtype)
        for i, name in enumerate(MODE_NAMES):
            self.metrics[f"frac_{name}"] = (self.mode == i + 1).to(self.yaw_rate.dtype)

    def _update_command(self):
        pass


@configclass
class ModeMixCommandCfg(CommandTermCfg):
    """Config for the phase-3 mode-mix command term (P3-1/P3-2/B.7)."""

    class_type: type[CommandTerm] = ModeMixCommand

    vx_deadband: float = 0.02
    """|vx_cmd| at or below this counts as zero [m/s] (P3-1)."""
    yaw_deadband: float = 0.05
    """|yaw_rate_cmd| at or below this counts as zero [rad/s] (P3-1)."""
    omega_min_forward: float = 0.05
    """M2 |omega| range lower bound [rad/s] (B.7)."""
    omega_min_rotate: float = 0.10
    """M3 |omega| range lower bound [rad/s] before the final stage (B.7)."""
    vx_noise: float = 0.015
    """Upper bound of the sub-deadband vx noise applied in M3/M4 [m/s] (B.7)."""
    yaw_noise: float = 0.03
    """Upper bound of the sub-deadband yaw noise applied in M1/M4 [rad/s] (B.7)."""
    mode_mix: tuple[float, float, float, float] = (0.5, 0.5, 0.0, 0.0)
    """Default (M1, M2, M3, M4) sampling mix -- stage 0; the mode-stage curriculum owns
    ``env.extras["mode_mix"]`` during training (falls back to this when absent)."""
    rotate_segment_s: tuple[float, float] = (5.0, 8.0)
    """Default M3 segment duration range [s]; the curriculum dial is
    ``env.extras["rotate_segment_s"]`` (B.7, P3-4)."""
    m4_segment_s: tuple[float, float] = (8.0, 14.0)
    """M4 (rest) segment duration range [s] -- longer than B.7's 5-8 s so the 2 s mask +
    2 s settle ramp leave real graded rest time (owner calibration; dial
    ``env.extras["m4_segment_s"]``)."""
