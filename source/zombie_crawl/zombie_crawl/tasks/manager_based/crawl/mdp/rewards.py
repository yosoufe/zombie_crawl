# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the zombie crawl task.

Convention: all functions return a tensor of shape ``(num_envs,)``. The environment
must expose ``env.v_target`` (tensor, shape ``(num_envs,)``) -- the current forward
velocity target -- which is created in ``CrawlEnvCfg.__post_init__`` and optionally
driven by the curriculum.
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


def get_ref_dir(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of each env's commanded heading, drawn by ``reset_start_pose``.

    The command is uniform +-180 deg, INDEPENDENT of the spawn facing, frozen for the whole
    episode. All directional rewards/curricula project world velocities onto it; the
    ``heading_error_wrist`` observation shows the actor where the claw points relative to it.
    It is a commanded direction, not a heading feedback loop -- it never moves to follow the
    robot, so turning cannot farm it. Falls back to world +X before the first reset.
    """
    yaw = getattr(env, "zombie_ref_yaw", None)
    if yaw is None or yaw.shape[0] != env.num_envs:
        ones = torch.ones(env.num_envs, device=env.device)
        return ones, torch.zeros_like(ones)
    return torch.cos(yaw), torch.sin(yaw)


def get_claw_dir(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("wrist",))
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of the base->wrist (claw) arrow heading -- the axis *forward velocity* is scored on.

    This is the locomotion frame: the robot must drag its base toward wherever the claw points, so
    the claw leads and a backward slide scores negative. It is the SAME horizontal base->wrist arrow
    ``wrist_pointing_error`` (reward) and ``heading_error_wrist`` (observation) measure, so the actor
    is steered by exactly what the reward pays. When the arm is curled the arrow degenerates, so it
    falls back to the base's own forward axis (base -Y = ``yaw - pi/2``), which is always defined and
    cannot be farmed by tucking the arm. Distinct from ``get_ref_dir`` (the world command), which is
    kept only for net *progress* toward the goal and the stall/curriculum gates.
    """
    asset = env.scene[asset_cfg.name]
    wrist_ids = getattr(env, "zombie_wrist_body_ids", None)
    if wrist_ids is None:
        ids, _names = asset.find_bodies(list(asset_cfg.body_names), as_proxy=True)
        wrist_ids = ids.torch.reshape(-1)
        if wrist_ids.numel() == 0:
            raise ValueError(f"get_claw_dir: no body matching {asset_cfg.body_names}")
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


def get_v_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-environment forward velocity target (lazily initialized on ``env.v_target``).

    With the curriculum enabled it starts at ``cfg.v_target_start`` and eases toward
    ``cfg.v_target``; otherwise it is constant at ``cfg.v_target``. If velocity-command
    tracking is enabled (phase-2 stub) the commanded vx replaces the fixed target.
    """
    if not hasattr(env, "v_target") or env.v_target is None:
        cfg = env.cfg
        start = cfg.v_target_start if getattr(cfg, "use_curriculum", False) else cfg.v_target
        # Isaac Lab 3.0: ManagerBasedRLEnv has no ``unwrapped`` property; use num_envs directly
        env.v_target = torch.full((env.num_envs,), start, device=env.device, dtype=torch.float32)
    if getattr(env.cfg, "use_velocity_commands", False) and env.command_manager is not None:
        cmd = env.command_manager.get_command("base_velocity")
        return cmd[:, 0]
    return env.v_target


def forward_velocity_exp(
    env: ManagerBasedEnv,
    std: float | None = None,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward-velocity reward: ``tanh(v_claw / v_target)``, no speed floor, posture-gated.

    ``v_claw`` is the base velocity projected onto the CLAW (base->wrist) axis via ``get_claw_dir``,
    so the reward pays only for dragging the base toward wherever the claw points -- the claw must
    lead, and a backward slide scores negative. (The old version projected onto the world command,
    which let the robot earn full forward reward by crawling backward with the claw trailing.) tanh
    is linear at low speed and flattens past the target, so a stationary base scores 0 and there is
    no upper clip to farm; overspeed beyond 2x target is punished by the overspeed term.

    A *posture* gate multiplies only the positive half: ~1 when the base is upright and ->0 as it
    tilts, so a face-plant topple no longer pays. Net progress toward the commanded *goal* is scored
    separately by ``forward_progress_lpf`` (which stays on ``get_ref_dir``), so banking both terms at
    once requires claw-aligned-to-command and crawling that way -- the intended zombie gait.
    """
    vx = _base_lin_vel_w(env, asset_cfg)[:, 0]
    vy = _base_lin_vel_w(env, asset_cfg)[:, 1]
    cos_ref, sin_ref = get_claw_dir(env)
    vx = vx * cos_ref + vy * sin_ref  # velocity along the claw (locomotion) axis
    target = get_v_target(env)
    # posture gate: squared horizontal tilt of projected gravity (0 = upright/flat, 1 = sideways).
    # A crawling base stays near-flat so this is ~1; a face-plant drives it toward 0, killing the
    # reward for the topple itself. Only the forward half is gated (backward is always a penalty).
    grav = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    tilt = grav[:, 0] * grav[:, 0] + grav[:, 1] * grav[:, 1]
    up = torch.clamp(1.0 - tilt, min=0.0, max=1.0)
    return torch.where(vx > 0, up * torch.tanh(vx / target), torch.tanh(vx / target))


def forward_progress(
    env: ManagerBasedEnv,
    clip: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-step displacement along the reference (jaw) axis (clipped to ±clip).

    Displacements larger than 3*clip are treated as episode-boundary jumps (the env was
    reset between the two measurements) and zeroed.
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch
    cos_ref, sin_ref = get_ref_dir(env)
    x = pos[:, 0] * cos_ref + pos[:, 1] * sin_ref  # position along the commanded heading
    if not hasattr(env, "zombie_prev_x") or env.zombie_prev_x.shape[0] != x.shape[0]:
        env.zombie_prev_x = x.clone()
        return torch.zeros(env.num_envs, device=env.device)
    dx = x - env.zombie_prev_x
    env.zombie_prev_x = x.clone()
    jump_mask = dx.abs() > 3.0 * clip
    if jump_mask.any():
        dx[jump_mask] = 0.0
    return torch.clamp(dx, min=-clip, max=clip)


def forward_progress_lpf(
    env: ManagerBasedEnv,
    tau_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponentially filtered forward progress (displacement over a ``tau_s``-second window).

    Same unit as velocity: filtered dx / filtered dt. Unlike the instantaneous per-step dx,
    it cannot be inflated by jerky oscillation (reciprocal motion averages to zero over the
    window), so it directly shapes the end-of-episode displacement the task is evaluated on.
    """
    asset = env.scene[asset_cfg.name]
    dt = float(env.step_dt)
    pos = asset.data.root_pos_w.torch
    cos_ref, sin_ref = get_ref_dir(env)
    x = pos[:, 0] * cos_ref + pos[:, 1] * sin_ref  # position along the commanded heading
    if not hasattr(env, "zombie_lp_dx") or env.zombie_lp_dx.shape[0] != x.shape[0]:
        env.zombie_lp_dx = torch.zeros_like(x)
        env.zombie_lp_dt = torch.zeros_like(x)
        env.zombie_lp_x0 = x.clone()
    dx = x - env.zombie_lp_x0
    jump = dx.abs() > 3.0 * float(getattr(env.cfg, "v_target", 0.1)) * tau_s
    env.zombie_lp_dx = torch.where(jump, torch.zeros_like(dx), env.zombie_lp_dx + dx)
    env.zombie_lp_dt = torch.where(jump, torch.zeros_like(dx), env.zombie_lp_dt)
    env.zombie_lp_x0 = x.clone()
    decay = math.exp(-dt / tau_s)
    env.zombie_lp_dx = env.zombie_lp_dx * decay
    env.zombie_lp_dt = env.zombie_lp_dt * decay + dt
    return env.zombie_lp_dx / env.zombie_lp_dt


def wrist_pointing_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("wrist",)),
) -> torch.Tensor:
    """1 - cos of the angle between the base->wrist arrow and the COMMAND (0 = aligned, 2 = reversed).

    The pointing vector is the horizontal (XY-projected) arrow from the base origin to the
    WRIST body origin -- the real claw-pointing direction, unlike the old base-yaw axis, which
    ignored shoulder_pan (a robot crawling with the arm twisted off its base scored as aligned).
    The commanded direction is ``zombie_ref_yaw`` (the target velocity is ``v_target`` along it,
    and the speed magnitude cancels in the cos comparison). Matches the ``heading_error_wrist``
    observation exactly, so the actor observes what the reward measures.

    Degeneracy guard: with the arm curled the horizontal arrow is near zero and its direction is
    atan2 noise; there the penalty falls back to the old base-yaw axis (yaw - pi/2), which is
    always defined and cannot be farmed by curling the arm.
    """
    asset = env.scene[asset_cfg.name]
    wrist_ids = getattr(env, "zombie_wrist_body_ids", None)
    if wrist_ids is None:
        ids, _names = asset.find_bodies(list(asset_cfg.body_names), as_proxy=True)
        wrist_ids = ids.torch.reshape(-1)
        if wrist_ids.numel() == 0:
            raise ValueError(f"wrist_pointing_error: no body matching {asset_cfg.body_names}")
        env.zombie_wrist_body_ids = wrist_ids
    wrist = asset.data.body_pos_w.torch[:, wrist_ids[0]]
    base = asset.data.root_pos_w.torch
    dx = wrist[:, 0] - base[:, 0]
    dy = wrist[:, 1] - base[:, 1]
    arrow_yaw = torch.atan2(dy, dx)
    # fallback axis: base yaw - pi/2 (body -Y, the claw side of the base plate)
    quat_w = asset.data.root_quat_w.torch  # Isaac Lab 3.0: x, y, z, w
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
    fallback_yaw = yaw - 0.5 * math.pi
    valid = torch.hypot(dx, dy) > 0.01
    pointing_yaw = torch.where(valid, arrow_yaw, fallback_yaw)
    ref = getattr(env, "zombie_ref_yaw", None)
    if ref is None or ref.shape[0] != pointing_yaw.shape[0]:
        ref = torch.zeros_like(pointing_yaw)
    return 1.0 - torch.cos(_wrap_to_pi(pointing_yaw - ref))


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
    command_name: str | None = None,
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
