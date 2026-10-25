# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the zombie crawl task.

Convention: all functions return a tensor of shape ``(num_envs,)``. The environment
must expose ``env.v_target`` (tensor, shape ``(num_envs,)``) -- the current forward
velocity target -- which is created in ``CrawlEnvCfg.__post_init__`` and optionally
driven by the curriculum.

Phase 2 (yaw-rate command):

* all directional terms are scored on the CLAW axis (``get_claw_dir``), never on a
  world-frame command direction -- no term projects an absolute world position onto a
  rotating axis (pitfall X3);
* the claw heading rate ``r`` (``env.zombie_r_ema``) is a simulation-only measurement
  maintained by :func:`_update_claw_heading_rate`, which runs exactly once per env step
  inside :func:`yaw_rate_tracking` (every consumer reads it after that); it is NEVER an
  actor input and is never computed on the robot.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _base_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_w.torch


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def yaw_rate_command(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-env commanded yaw rate [rad/s] from the ``yaw_rate_cmd`` CommandManager term."""
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        return cm.get_command("yaw_rate_cmd")[:, 0]
    return torch.zeros(env.num_envs, device=env.device)


def yaw_rate_resample_flag(env: ManagerBasedEnv) -> torch.Tensor | None:
    """Bool mask (num_envs,) of envs whose yaw-rate command changed on the latest command
    ``compute()`` (mid-episode resample), or ``None`` when the term is not active.

    The stateful claw-rate filter restarts its post-resample mask timer for these envs.
    Episode resets are re-anchored separately (``mdp.reset_start_pose`` sets the reseed
    flag), because the CommandManager's reset resample is not observable here (the flag
    is cleared by the same step's ``compute()``).
    """
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        return cm.get_term("yaw_rate_cmd").just_resampled
    return None


def _claw_yaw_and_valid(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("wrist",))
) -> tuple[torch.Tensor, torch.Tensor]:
    """(psi, valid): yaw [rad] of the base->wrist (claw) arrow and its validity mask.

    Valid means the horizontal arrow is longer than 0.01 m; when invalid the arrow
    degenerates (arm curled) and consumers fall back to the base axis ``yaw - pi/2``.
    """
    asset = env.scene[asset_cfg.name]
    wrist_ids = getattr(env, "zombie_wrist_body_ids", None)
    if wrist_ids is None:
        ids, _names = asset.find_bodies(list(asset_cfg.body_names), as_proxy=True)
        wrist_ids = ids.torch.reshape(-1)
        if wrist_ids.numel() == 0:
            raise ValueError(f"claw arrow: no body matching {asset_cfg.body_names}")
        env.zombie_wrist_body_ids = wrist_ids
    wrist = asset.data.body_pos_w.torch[:, wrist_ids[0]]
    base = asset.data.root_pos_w.torch
    dx = wrist[:, 0] - base[:, 0]
    dy = wrist[:, 1] - base[:, 1]
    arrow_yaw = torch.atan2(dy, dx)
    quat_w = asset.data.root_quat_w.torch  # Isaac Lab 3.0: x, y, z, w
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
    fallback_yaw = yaw - 0.5 * math.pi
    valid = torch.hypot(dx, dy) > 0.01
    claw_yaw = torch.where(valid, arrow_yaw, fallback_yaw)
    return claw_yaw, valid


def get_claw_dir(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("wrist",))
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of the base->wrist (claw) arrow heading -- the axis *forward velocity* is scored on.

    This is the locomotion frame: the robot must drag its base toward wherever the claw points, so
    the claw leads and a backward slide scores negative. When the arm is curled the arrow degenerates,
    so it falls back to the base's own forward axis (base -Y = ``yaw - pi/2``), which is always defined
    and cannot be farmed by tucking the arm. Under the yaw-rate command policy every directional
    reward/termination/curriculum term projects on this axis; there is no world-frame command
    direction anymore (the removed ``get_ref_dir``).
    """
    claw_yaw, _valid = _claw_yaw_and_valid(env, asset_cfg)
    return torch.cos(claw_yaw), torch.sin(claw_yaw)


def upright_bonus(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Recovery shaping: ~1 when the base is level/upright, ~0 when tipped onto its side.

    ``clamp(-projected_gravity_b[:, 2], 0, 1)``: +1 flat, 0 at ~90 deg (parked on the side), clipped
    to 0 when inverted. This is a dense gradient that pays simply for RIGHTING the base, so a robot
    knocked onto its side is rewarded for climbing back toward level instead of freezing there in the
    folded crouch. It is the antidote to the old reward where lying down was ~free (``base_height``
    targeted 5 mm above the floor, so a low/collapsed posture barely cost anything). It does NOT
    reward standing tall -- a level crawling crouch scores the same ~1 as a level base -- so it is
    compatible with the low claw-drag gait and cannot be farmed by rearing up and freezing.
    """
    grav = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return torch.clamp(-grav[:, 2], min=0.0, max=1.0)


def flatness_gate(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Posture gate ``up``: ``clamp(1 - (g_x^2 + g_y^2), 0, 1)`` from projected gravity."""
    grav = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    tilt = grav[:, 0] * grav[:, 0] + grav[:, 1] * grav[:, 1]
    return torch.clamp(1.0 - tilt, min=0.0, max=1.0)


def get_v_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-environment forward velocity target (lazily initialized on ``env.v_target``).

    With the curriculum enabled it starts at ``cfg.v_target_start`` and eases toward
    ``cfg.v_target``; otherwise it is constant at ``cfg.v_target``.
    """
    if not hasattr(env, "v_target") or env.v_target is None:
        cfg = env.cfg
        start = cfg.v_target_start if getattr(cfg, "use_curriculum", False) else cfg.v_target
        # Isaac Lab 3.0: ManagerBasedRLEnv has no ``unwrapped`` property; use num_envs directly
        env.v_target = torch.full((env.num_envs,), start, device=env.device, dtype=torch.float32)
    return env.v_target


def claw_heading_rate(env: ManagerBasedEnv) -> torch.Tensor:
    """The claw heading rate ``r`` [rad/s]: per-env EMA buffer (read-only accessor).

    The value is maintained by :func:`_update_claw_heading_rate`, which runs exactly once
    per env step inside :func:`yaw_rate_tracking`. Consumers (critic observation, metrics,
    curriculum, evaluation) read this buffer after the reward step. It is a
    SIMULATION-ONLY measurement: never an actor input, never computed on the robot.
    """
    r = getattr(env, "zombie_r_ema", None)
    if r is None or r.shape[0] != env.num_envs:
        r = torch.zeros(env.num_envs, device=env.device)
        env.zombie_r_ema = r
    return r


def _update_claw_heading_rate(env: ManagerBasedEnv, dt: float, tau_r: float) -> torch.Tensor:
    """Advance the claw heading rate EMA one env step. MUST be called exactly once per step.

    ``psi_t`` = claw heading (atan2 of the claw unit vector); ``dpsi = wrap(psi_t - psi_prev)``.
    The increment is dropped (``dpsi = 0``) when the arrow validity flipped between the two
    steps or ``|dpsi| > 0.5`` rad, so a curling arm snapping the arrow onto the base-axis
    fallback cannot inject a spike. Then ``r += (dt / tau_r) * (dpsi / dt - r)``: the EMA
    (``tau_r`` s, a fixed constant) averages out the base sway of the gait.

    Reseed (``r = 0``, ``psi_prev = psi``, timer = 0) is requested by
    ``mdp.reset_start_pose`` via ``env.zombie_rate_reseed``; the timer (only) restarts when
    the command term reports ``just_resampled``.
    """
    psi, valid = _claw_yaw_and_valid(env)
    r = getattr(env, "zombie_r_ema", None)
    if r is None or r.shape[0] != env.num_envs:
        env.zombie_r_ema = torch.zeros(env.num_envs, device=env.device)
        env.zombie_psi_prev = psi.clone()
        env.zombie_arrow_valid_prev = valid.clone()
        env.zombie_time_since_resample = torch.zeros(env.num_envs, device=env.device)
        env.zombie_rate_reseed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return env.zombie_r_ema
    reseed = env.zombie_rate_reseed
    if bool(reseed.any()):
        env.zombie_r_ema[reseed] = 0.0
        env.zombie_psi_prev[reseed] = psi[reseed]
        env.zombie_arrow_valid_prev[reseed] = valid[reseed]
        env.zombie_time_since_resample[reseed] = 0.0
        reseed[reseed] = False
    resampled = yaw_rate_resample_flag(env)
    if resampled is not None and bool(resampled.any()):
        env.zombie_time_since_resample[resampled] = 0.0
    dpsi = _wrap_to_pi(psi - env.zombie_psi_prev)
    bad = (valid != env.zombie_arrow_valid_prev) | (dpsi.abs() > 0.5)
    dpsi = torch.where(bad, torch.zeros_like(dpsi), dpsi)
    env.zombie_r_ema.add_((dpsi / max(dt, 1e-9) - env.zombie_r_ema) * min(1.0, dt / tau_r))
    env.zombie_psi_prev = psi
    env.zombie_arrow_valid_prev = valid
    env.zombie_time_since_resample += dt
    return env.zombie_r_ema


def forward_velocity_exp(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_at_target: bool = False,
) -> torch.Tensor:
    """Forward-velocity reward: ``tanh(v_c / v_target)``, posture-gated. "Move along your claw."

    ``v_c`` is the base horizontal velocity projected onto the CURRENT claw axis
    (``get_claw_dir``). Phase 2 changed only the axis: measuring on the claw axis is what
    makes the claw LEAD -- advancing against the arrow pays negative, so the tail-first
    exploit the removed ``wrist_pointing`` term used to block is dominated directly by
    this term (watch item W2 logs any residual backward crawling).

    The forward half is multiplied by the flatness gate ``up`` (a toppled robot earns
    nothing until level); the backward half stays the plain negative ``tanh``. ``tanh`` is
    linear from rest (a gradient at a dead stop) -- a Gaussian bell has none and MUST NOT
    replace it (pitfall X4).
    """
    v = _base_lin_vel_w(env, asset_cfg)
    cos_u, sin_u = get_claw_dir(env)
    v_c = v[:, 0] * cos_u + v[:, 1] * sin_u
    target = torch.clamp(get_v_target(env), min=1e-3)
    up = flatness_gate(env, asset_cfg)
    # cap_at_target flattens the reward above v_target (no gradient for overshooting);
    # the backward branch stays uncapped so reversing keeps paying its full negative.
    x_pos = torch.clamp(v_c / target, max=1.0) if cap_at_target else v_c / target
    return torch.where(v_c > 0, up * torch.tanh(x_pos), torch.tanh(v_c / target))


def forward_progress_lpf(
    env: ManagerBasedEnv,
    tau_s: float = 1.0,
    normalize: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap: bool = False,
) -> torch.Tensor:
    """Exponentially filtered forward progress along the claw axis (a ``tau_s``-second window).

    Same unit as velocity: filtered dx / filtered dt. Unlike the instantaneous per-step dx,
    it cannot be inflated by jerky oscillation (reciprocal motion averages to zero over the
    window), so it directly shapes the end-of-episode displacement the task is evaluated on.

    Axis-free bookkeeping (pitfall X3): the per-step increment is
    ``dx = (p_t - p_{t-1}) . u_t`` -- the base DISPLACEMENT dotted with the current claw
    unit vector -- never the absolute world position (which carries the env-origin offset,
    up to ~45 m at 4096 envs) projected on a rotating axis. The jump guard zeroes the
    increment window when ``|p_t - p_{t-1}| > 3 * v_target * tau_s`` (episode-boundary
    teleport); ``mdp.reset_start_pose`` additionally reseeds the cached previous position.

    With ``normalize`` the filtered claw-axis speed is divided by ``v_target`` so this term
    sits on the SAME [0,1]-ish scale as the normalized ``forward_velocity`` term.
    """
    asset = env.scene[asset_cfg.name]
    dt = float(env.step_dt)
    pos = asset.data.root_pos_w.torch[:, :2]
    if not hasattr(env, "zombie_prev_xy") or env.zombie_prev_xy.shape != pos.shape:
        env.zombie_prev_xy = pos.clone()
        env.zombie_lp_dx = torch.zeros(env.num_envs, device=env.device)
        env.zombie_lp_dt = torch.zeros(env.num_envs, device=env.device)
    d = pos - env.zombie_prev_xy
    env.zombie_prev_xy = pos.clone()
    cos_u, sin_u = get_claw_dir(env)
    dx = d[:, 0] * cos_u + d[:, 1] * sin_u
    jump = d.norm(dim=1) > 3.0 * float(getattr(env.cfg, "v_target", 0.1)) * tau_s
    env.zombie_lp_dx = torch.where(jump, torch.zeros_like(dx), env.zombie_lp_dx + dx)
    env.zombie_lp_dt = torch.where(jump, torch.zeros_like(dx), env.zombie_lp_dt)
    decay = math.exp(-dt / tau_s)
    env.zombie_lp_dx = env.zombie_lp_dx * decay
    env.zombie_lp_dt = env.zombie_lp_dt * decay + dt
    out = env.zombie_lp_dx / env.zombie_lp_dt
    if normalize:
        out = out / torch.clamp(get_v_target(env), min=1e-3)
    if cap:
        out = torch.clamp(out, max=1.0)
    return out


def yaw_rate_tracking(
    env: ManagerBasedEnv,
    sigma: float = 0.1,
    tau_r: float = 1.5,
    yaw_mask_s: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Claw-heading-rate tracking: ``up * clamp(tanh(v_c/v_target), 0, 1) * mask * bump``.

    ``bump = exp(-((r - yaw_rate_cmd) / sigma)^2)`` is the ONLY exception to the additive
    -terms rule (P2-4): these factors are gates on this one term, not products of separate
    rewards.

    * ``up`` -- flatness gate: no tracking credit while toppled (recovery first).
    * ``clamp(tanh(v_c / v_target), 0, 1)`` -- pays only while crawling FORWARD, so standing
      still cannot earn it (an ungated bump under a zero command pays its maximum for doing
      nothing and collapses PPO to sitting).
    * ``mask`` -- fades 0 -> 1 linearly over ``yaw_mask_s`` after every resample/reset,
      because the filtered rate ``r`` lags a fresh command.
    * Returns 0 while ``yaw_rate_max == 0`` (steering stage locked off).

    Also advances the per-step bookkeeping: the claw-rate filter (exactly once per env
    step, inside this term -- every consumer reads ``env.zombie_r_ema`` afterwards) and the
    success-metric episode accumulators (:mod:`.metrics`).
    """
    from . import metrics

    dt = float(env.step_dt)
    _update_claw_heading_rate(env, dt, tau_r)
    metrics.note_step(env, dt)
    r_max = float(getattr(env, "extras", {}).get("yaw_rate_max", 0.0))
    if r_max <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    cmd = yaw_rate_command(env)
    v = _base_lin_vel_w(env, asset_cfg)
    cos_u, sin_u = get_claw_dir(env)
    v_c = v[:, 0] * cos_u + v[:, 1] * sin_u
    target = torch.clamp(get_v_target(env), min=1e-3)
    up = flatness_gate(env, asset_cfg)
    fwd = torch.clamp(torch.tanh(v_c / target), min=0.0, max=1.0)
    mask = torch.clamp(env.zombie_time_since_resample / yaw_mask_s, min=0.0, max=1.0)
    bump = torch.exp(-((env.zombie_r_ema - cmd) / sigma) ** 2)
    return up * fwd * mask * bump


def lateral_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared base velocity perpendicular to the claw (locomotion) axis."""
    v = _base_lin_vel_w(env, asset_cfg)
    cos_ref, sin_ref = get_claw_dir(env)
    vy = -v[:, 0] * sin_ref + v[:, 1] * cos_ref
    return vy**2


def overspeed_penalty(
    env: ManagerBasedEnv,
    factor: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty for velocity along the claw (locomotion) axis above ``factor * v_target``."""
    v = _base_lin_vel_w(env, asset_cfg)
    cos_ref, sin_ref = get_claw_dir(env)
    vx = v[:, 0] * cos_ref + v[:, 1] * sin_ref
    excess = torch.relu(vx - factor * get_v_target(env))
    return excess**2


def vertical_velocity_l2(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared world-frame vertical base velocity (anti-hop)."""
    vz = _base_lin_vel_w(env, asset_cfg)[:, 2]
    return vz**2


def base_height_l2(
    env: ManagerBasedEnv,
    target_height: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared distance of the base origin height from its resting value (anti-launch)."""
    z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
    return (z - target_height) ** 2


def joint_deviation_l2(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Very weak pull toward the default joint configuration."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos.torch
    q0 = asset.data.default_joint_pos.torch
    return ((q - q0) ** 2).sum(dim=1)


def low_ground_contact_bonus(
    env: ManagerBasedEnv,
    height_threshold: float = 0.08,
    decay: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Optional zombie-style term: reward keeping the base low above the ground."""
    z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
    above = torch.relu(z - height_threshold)
    return torch.exp(-above / decay)


def gait_asymmetry_bonus(
    env: ManagerBasedEnv,
    history: int = 50,
    joint_index: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Optional zombie-style term (experimental): reward cyclic joint activity correlated
    with forward motion. Measures the time-average of ``sign(qdot) * sign(vx)`` over a
    sliding window; a pull-and-recover gait yields a value near zero while flailing or
    monotonic spinning yields larger absolute values, so the term rewards *alternation*
    by penalizing sustained same-sign activity.
    """
    asset = env.scene[asset_cfg.name]
    qdot = asset.data.joint_vel.torch[:, joint_index]
    vx = _base_lin_vel_w(env, asset_cfg)[:, 0]
    prod = torch.sign(qdot) * torch.sign(vx)
    n = env.num_envs
    state = getattr(env, "zombie_asym_buffer", None)
    if state is None or state.shape[0] != n:
        env.zombie_asym_buffer = torch.zeros(n, history, device=env.device)
        env.zombie_asym_ptr = 0
        return torch.zeros(n, device=env.device)
    env.zombie_asym_buffer[:, env.zombie_asym_ptr] = prod
    env.zombie_asym_ptr = (env.zombie_asym_ptr + 1) % history
    mean_abs = env.zombie_asym_buffer.abs().mean(dim=1)
    # 1 at perfect alternation (mean |x| -> 0 with x != 0), 0 at sustained activity
    active = (env.zombie_asym_buffer.abs() > 0.5).any(dim=1).float()
    return active * (1.0 - mean_abs.clamp(max=1.0))
