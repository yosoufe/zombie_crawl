# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and randomization events for the zombie crawl task (Isaac Lab 3.0 API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv


def _normalize_env_ids(
    env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None, device: torch.device | str
) -> torch.Tensor:
    """Resolve ``None`` / full slices (Isaac Lab 3.0 passes ``slice(None)`` for whole-env resets)
    to the full env id range on *device*; convert partial slices to index tensors."""
    num_envs = env.num_envs
    if env_ids is None or (
        isinstance(env_ids, slice) and env_ids.start is None and env_ids.stop is None and env_ids.step is None
    ):
        return torch.arange(num_envs, device=device, dtype=torch.int32)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs)[env_ids].to(device=device, dtype=torch.int32)
    return torch.as_tensor(env_ids, device=device, dtype=torch.int32)


def reset_start_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    poses: dict[str, dict],
    pose_weights: tuple[float, ...] | None = None,
    joint_noise: float = 0.1,
    xy_offset: float = 0.05,
    yaw_range: tuple[float, float] = (-3.14, 3.14),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    fallen_frac_override: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot into one of the named start poses, sampled per environment.

    Each pose entry maps a name to ``{"pos": (x, y, z), "rot": (roll, pitch, yaw), "joint_pos": {name: rad}}``.
    On top of the named pose the function adds: a uniform yaw in ``yaw_range``, a small x/y
    offset, uniform joint noise, and zero velocity.

    Start-pose mixing (pitfall X1): fallen starts come ONLY from the difficulty
    curriculum's ``env.extras["fallen_frac"]`` (or an explicit ``fallen_frac_override``,
    used by the training-spec play clip and the evaluation script). When that key is
    absent the configured ``pose_weights`` are used as-is; with no explicit weights the
    robot spawns FOLDED ONLY -- the old implicit 0.5/0.25/0.25 toppled fallback is removed
    so a replaced/disabled curriculum can never silently start half the population
    toppled from iteration 0.

    This term is the single mandated place where every per-episode stateful bookkeeping
    buffer is re-anchored: the claw-axis progress LPF (``zombie_prev_xy`` + window), the
    claw-heading-rate filter reseed request, the stall-window anchor (position + step 0),
    and ``started_fallen``.
    """
    from . import metrics

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    env_ids = _normalize_env_ids(env, env_ids, device)
    n = len(env_ids)

    names = list(poses.keys())
    # dynamic fallen-start fraction from the difficulty curriculum: with >=3 poses ordered
    # (folded, on_side, face_down) and an explicit fallen fraction set, override the baked
    # pose_weights so the toppled fraction grows 0 -> fallen_max as difficulty ramps.
    f = fallen_frac_override
    if f is None:
        f = env.extras.get("fallen_frac", None)
    # phase 3 (B.7): rest-pose starts (folded at rest height, zero command drawn normally)
    # come from env.extras["rest_frac"] (set by the mode_stage curriculum; 0 = phase-2)
    r = env.extras.get("rest_frac", None)
    if f is not None and len(names) >= 3:
        f = float(f)
        if len(names) >= 4 and r is not None:
            r = float(min(max(float(r), 0.0), 1.0 - f))
            weights_list = [1.0 - f - r, f / 2.0, f / 2.0, r]
        else:
            weights_list = [1.0 - f, f / 2.0, f / 2.0]
        # phase 3 added the 4th (rest) pose: legacy 3-entry lists pad with zero weight
        weights_list += [0.0] * (len(names) - len(weights_list))
        weights = torch.tensor(weights_list[: len(names)], device=device)
    elif pose_weights is not None:
        pw = list(pose_weights) + [0.0] * (len(names) - len(pose_weights))
        weights = torch.tensor(pw[: len(names)], device=device)
    elif len(names) >= 3:
        # X1: no curriculum fraction and no explicit weights -> folded only, ALWAYS
        weights = torch.tensor([1.0] + [0.0] * (len(names) - 1), device=device)
    else:
        weights = torch.full((len(names),), 1.0 / len(names), device=device)
    weights = weights / weights.sum()
    pose_ids = torch.multinomial(weights, n, replacement=True)
    # phase 3: rest-start envs must actually be graded at rest -- queue their first
    # command segment for M4 (consumed by ModeMixCommand._sample)
    if len(names) >= 4:
        rest_rows = env_ids[pose_ids == 3]
        if rest_rows.numel():
            prev = env.extras.get("force_m4_ids")
            env.extras["force_m4_ids"] = (
                rest_rows.clone() if prev is None or not len(prev) else torch.cat([prev, rest_rows])
            )

    # base pose
    positions = torch.zeros(n, 3, device=device)
    orientations_rpy = torch.zeros(n, 3, device=device)
    joint_pos = torch.zeros(n, asset.num_joints, device=device)
    for i, pid in enumerate(pose_ids.tolist()):
        pose = poses[names[pid]]
        positions[i] = torch.tensor(pose["pos"], device=device)
        orientations_rpy[i] = torch.tensor(pose.get("rot", (0.0, 0.0, 0.0)), device=device)
        for jname, jval in pose["joint_pos"].items():
            idx = asset.joint_names.index(jname)
            joint_pos[i, idx] = jval
    # record per-env start class for the success metrics (upright vs fallen episodes).
    # Only on_side/face_down are fallen -- the phase-3 "rest" start (id 3) is UPRIGHT.
    metrics.set_started_fallen(env, env_ids, (pose_ids == 1) | (pose_ids == 2))

    # randomization: yaw, xy offset, joint noise (kept at the phase-1 values; the old
    # zombie_ref_yaw command draw is gone -- the command is a yaw RATE now, drawn by the
    # command term, so nothing directional is sampled here)
    # note: Isaac Lab 3.0 sample_uniform signature is (lower, upper, size, device)
    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], n, device=device)
    orientations_rpy[:, 2] += yaw
    positions[:, 0] += math_utils.sample_uniform(-xy_offset, xy_offset, n, device=device)
    positions[:, 1] += math_utils.sample_uniform(-xy_offset, xy_offset, n, device=device)
    if joint_noise > 0:
        joint_pos += math_utils.sample_uniform(
            -joint_noise, joint_noise, (n, asset.num_joints), device=device
        )
        # soft limits layout in 3.0: (num_envs, num_joints, 2) = [low, high]
        limits = asset.data.soft_joint_pos_limits.torch
        joint_pos = joint_pos.clamp_(limits[0, :, 0], limits[0, :, 1])
    # phase 3 (B.7): rest starts are EXACT (no joint noise) so rest_pose_ok is reachable
    # immediately and the M4 metric measures the policy, not spawn jitter
    if len(names) >= 4:
        rest_rows = pose_ids == 3
        if bool(rest_rows.any()):
            joint_pos[rest_rows] = 0.0

    # Isaac Lab 3.0: quaternions are XYZW; write_*_to_sim is superseded by the *_index variants
    orientations = math_utils.quat_from_euler_xyz(orientations_rpy[:, 0], orientations_rpy[:, 1], orientations_rpy[:, 2])
    # positions are relative to the environment origin
    # (env_origins is a CPU tensor on the no-terrain path; move it to the sim device)
    positions = positions + env.scene.env_origins[env_ids].to(device)

    # ---- per-episode stateful bookkeeping re-anchors (P2-5/P2-6/P2-8) ----------------
    # claw-axis progress LPF: cache the SPAWN position as "previous position" and drop the
    # accumulated window (replaces the old axis-projected zombie_lp_x0/zombie_prev_x seeding)
    prev_xy = getattr(env, "zombie_prev_xy", None)
    if prev_xy is None or prev_xy.shape[0] != env.num_envs:
        env.zombie_prev_xy = positions[:, :2].clone()
        env.zombie_lp_dx = torch.zeros(env.num_envs, device=device)
        env.zombie_lp_dt = torch.zeros(env.num_envs, device=device)
    else:
        env.zombie_prev_xy[env_ids] = positions[:, :2]
        env.zombie_lp_dx[env_ids] = 0.0
        env.zombie_lp_dt[env_ids] = 0.0
    # claw heading rate filter: request reseed (r=0, psi_prev=current, timer=0) on the
    # first reward step of the new episode -- the live claw yaw is not readable yet.
    # The five rate-filter buffers are created TOGETHER (any lazy creator may run first),
    # so the presence of the reseed flag is the single source of truth for "initialized".
    if (
        getattr(env, "zombie_rate_reseed", None) is None
        or env.zombie_rate_reseed.shape[0] != env.num_envs
    ):
        env.zombie_r_ema = torch.zeros(env.num_envs, device=device)
        env.zombie_psi_prev = torch.zeros(env.num_envs, device=device)
        env.zombie_arrow_valid_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env.zombie_time_since_resample = torch.zeros(env.num_envs, device=device)
        env.zombie_rate_reseed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    else:
        env.zombie_rate_reseed[env_ids] = True
    # stall window anchor: position = spawn, step = 0 (episode_length_buf is zeroed later
    # in the same reset step by the env manager, so "now" is the new episode's origin)
    if getattr(env, "zombie_stall_anchor_xy", None) is None or env.zombie_stall_anchor_xy.shape[0] != env.num_envs:
        env.zombie_stall_anchor_xy = positions[:, :2].clone()
        env.zombie_stall_anchor_step = torch.zeros_like(env.episode_length_buf)
    else:
        env.zombie_stall_anchor_xy[env_ids] = positions[:, :2]
        env.zombie_stall_anchor_step[env_ids] = 0

    # Isaac Lab 3.0: write_*_to_sim_index APIs are keyword-only with renamed params
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)

    velocities = torch.zeros(n, 6, device=device)
    if velocity_range[1] > velocity_range[0]:
        velocities += math_utils.sample_uniform(velocity_range[0], velocity_range[1], (n, 6), device=device)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)


def randomize_ground_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float] = (0.625, 1.5),
    num_buckets: int = 64,
    ground_static_friction: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a per-environment friction multiplier and apply it to the robot's shapes.

    The ground material has static friction ``ground_static_friction`` (0.8) and
    ``friction_combine_mode="multiply"``, so the *effective* pair friction is
    ``ground_static_friction * x`` with ``x`` uniform in ``friction_range``
    (defaults give an effective range of 0.5-1.2). The sampled multiplier is stored on
    ``env.extras["ground_friction"]`` for privileged observations.

    ``env_ids=None`` (startup mode) selects all environments.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _normalize_env_ids(env, env_ids, "cpu")
    n = len(env_ids)
    x = math_utils.sample_uniform(friction_range[0], friction_range[1], (n,), device="cpu")

    # Isaac Lab 3.0: the PhysX view moved from ``root_physx_view`` to ``root_view`` and
    # speaks warp arrays (material properties in (static, dynamic, restitution) layout).
    total_num_shapes = asset.root_view.max_shapes
    materials = wp.to_torch(asset.root_view.get_material_properties())
    # static friction = x, dynamic friction = 0.75 * x (same 0.8/0.6 ratio as the ground), restitution 0
    samples = torch.zeros(n, total_num_shapes, 3, device="cpu")
    samples[:, :, 0] = x[:, None]
    samples[:, :, 1] = 0.75 * x[:, None]
    materials[env_ids, :] = samples
    asset.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
    )

    if "ground_friction" not in env.extras:
        env.extras["ground_friction"] = torch.ones(env.num_envs, device=env.device)
    env.extras["ground_friction"][env_ids] = x.to(env.device)
    # store the unscaled per-env samples so the curriculum can rescale the materials
    if "friction_base_materials" not in env.extras:
        env.extras["friction_base_materials"] = torch.zeros(
            env.num_envs, total_num_shapes, 2, device="cpu"
        )
    env.extras["friction_base_materials"][env_ids] = samples[:, :, :2]


def apply_friction_scale(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Interval event: rescale the robot friction materials by ``env.extras["friction_scale"]``.

    Called by the curriculum; a no-op while the scale is 1.0 (curriculum done or disabled).
    Note: the signature must be exactly ``(env, env_ids)`` -- Isaac Lab 3.0 injects
    defaulted arguments into the term params (colliding with the manager's positional
    ``env_ids`` call) and its static check treats ``**kwargs`` as a required parameter.
    """
    if "friction_scale" not in env.extras or "friction_base_materials" not in env.extras:
        return
    scale = float(env.extras["friction_scale"])
    if abs(scale - 1.0) < 1e-9:
        return
    asset: Articulation = env.scene["robot"]
    base = env.extras["friction_base_materials"]  # (num_envs, num_shapes, 2) on CPU
    materials = wp.to_torch(asset.root_view.get_material_properties())
    all_env_ids = torch.arange(env.num_envs, device="cpu", dtype=torch.int32)
    materials[all_env_ids, :, 0] = base[:, :, 0] * scale
    materials[all_env_ids, :, 1] = base[:, :, 1] * scale
    asset.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.from_torch(all_env_ids, dtype=wp.int32)
    )
    # update the privileged observation (effective static friction of the pair)
    ground_mu = 0.8
    env.extras["ground_friction"][:] = (base[:, :, 0].mean(dim=1) * scale * ground_mu).to(env.device)


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stiffness_range: tuple[float, float] = (1.0, 1.0),
    damping_range: tuple[float, float] = (0.1, 0.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize the PD gains of the actuated joints (experimental, off by default).

    Isaac Lab 3.0 exposes the solver drive gains through the articulation data
    (``joint_stiffness`` / ``joint_damping``) as write-through buffers, so the sampled
    gains are written there directly. Only active when the ranges are non-degenerate
    (min < max).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if stiffness_range[1] <= stiffness_range[0] and damping_range[1] <= damping_range[0]:
        return
    device = asset.device
    env_ids = _normalize_env_ids(env, env_ids, device)
    n = len(env_ids)
    # Isaac Lab 3.0: joint_stiffness/damping are write-through ProxyArrays over the solver buffers
    if stiffness_range[1] > stiffness_range[0]:
        sample = math_utils.sample_uniform(stiffness_range[0], stiffness_range[1], (n, asset.num_joints), device=device)
        asset.data.joint_stiffness.torch[env_ids] = sample
    if damping_range[1] > damping_range[0]:
        sample = math_utils.sample_uniform(damping_range[0], damping_range[1], (n, asset.num_joints), device=device)
        asset.data.joint_damping.torch[env_ids] = sample


def disturb_all_bodies(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    force_range: tuple[float, float] = (-0.4, 0.4),
    torque_range: tuple[float, float] = (-0.03, 0.03),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_key: str = "push_scale",
) -> None:
    """Random per-body force + torque disturbance, magnitude scaled by the difficulty curriculum.

    Like ``mdp.apply_external_force_torque`` but multiplies the sampled force/torque ranges by
    ``env.extras["scale_key"]`` (set 0 -> 1 by :func:`difficulty_curriculum`), so the robot is NOT
    disturbed at all at the start of training and gets progressively poked on every link as it
    learns to crawl. When the scale is 0 the wrench is explicitly cleared (so no residual force
    lingers). Forces are applied at each body COM and propagate through the joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    scale = float(env.extras.get(scale_key, 1.0))
    num_bodies = asset.num_bodies if isinstance(asset_cfg.body_ids, slice) else len(asset_cfg.body_ids)
    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
    size = (num_envs, num_bodies, 3)
    if scale <= 0.0:
        forces = torch.zeros(size, device=asset.device)
        torques = torch.zeros(size, device=asset.device)
    else:
        forces = math_utils.sample_uniform(
            force_range[0] * scale, force_range[1] * scale, size, asset.device
        )
        torques = math_utils.sample_uniform(
            torque_range[0] * scale, torque_range[1] * scale, size, asset.device
        )
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=torques, body_ids=asset_cfg.body_ids, env_ids=env_ids
    )
    # remember the wrench currently applied per (env, body) so the video overlay can draw a
    # red disturbance arrow at each body COM; the permanent composer holds these exact
    # values until the next interval fire, so this buffer mirrors the live disturbance
    buf = env.extras.get("disturb_forces")
    if buf is None or buf.shape != (env.num_envs, asset.num_bodies, 3):
        buf = torch.zeros(env.num_envs, asset.num_bodies, 3, device=asset.device)
        env.extras["disturb_forces"] = buf
    env_idx = _normalize_env_ids(env, env_ids, asset.device)
    body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, slice) else slice(None)
    buf[env_idx, body_ids, :] = forces
    # per-axis force ceiling actually in effect, in the worst-case-|f| units the red video
    # arrow normalizes by (sqrt(3) * axis max), so the arrow length stays meaningful no
    # matter how the config scales the ranges
    env.extras["disturb_force_ref"] = max(1e-6, scale * max(abs(force_range[0]), abs(force_range[1])) * math.sqrt(3))
