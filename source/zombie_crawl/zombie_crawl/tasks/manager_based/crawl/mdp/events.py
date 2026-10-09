# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and randomization events for the zombie crawl task (Isaac Lab 3.0 API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot into one of the named start poses, sampled per environment.

    Each pose entry maps a name to ``{"pos": (x, y, z), "rot": (roll, pitch, yaw), "joint_pos": {name: rad}}``.
    On top of the named pose the function adds: a uniform yaw in ``yaw_range`` (the robot
    must find +X itself), a small x/y offset, uniform joint noise, and zero velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    env_ids = _normalize_env_ids(env, env_ids, device)
    n = len(env_ids)

    names = list(poses.keys())
    weights = torch.tensor(pose_weights if pose_weights is not None else [1.0 / len(names)] * len(names), device=device)
    weights = weights / weights.sum()
    pose_ids = torch.multinomial(weights, n, replacement=True)

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

    # randomization: yaw, xy offset, joint noise
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

    # Isaac Lab 3.0: quaternions are XYZW; write_*_to_sim is superseded by the *_index variants
    orientations = math_utils.quat_from_euler_xyz(orientations_rpy[:, 0], orientations_rpy[:, 1], orientations_rpy[:, 2])
    # positions are relative to the environment origin
    # (env_origins is a CPU tensor on the no-terrain path; move it to the sim device)
    positions = positions + env.scene.env_origins[env_ids].to(device)
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
