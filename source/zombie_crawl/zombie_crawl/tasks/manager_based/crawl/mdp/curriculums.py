# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the zombie crawl task.

The curriculum advances two knobs, both stored on the environment:

* ``env.v_target`` -- per-environment forward velocity target (tensor, shape (num_envs,)).
* ``env.extras["friction_scale"]`` -- multiplier applied on top of the per-env friction
  sample; starts above 1 (grippier) and decays to 1.

Both advance when the mean forward velocity (and mean episode length) pass thresholds,
so a weak policy gets an easier task and the difficulty creeps up as it learns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv

from .rewards import get_v_target


def _proj_vel_ema(env: "ManagerBasedEnv", tau: float = 2.0) -> float:
    """Population mean of an EMA (~``tau`` s) of the reference-projected base velocity.

    Both curriculum gates used to read ONE instantaneous frame of velocity; a flailing
    population's instantaneous mean spikes on every stand-up/topple kick and advanced the
    difficulty on bogus evidence. A short EMA gates on sustained forward motion instead.
    """
    from .rewards import get_ref_dir

    dt = float(env.step_dt)
    v = env.scene["robot"].data.root_lin_vel_w.torch
    cos_ref, sin_ref = get_ref_dir(env)
    v_proj = v[:, 0] * cos_ref + v[:, 1] * sin_ref
    ema = getattr(env, "zombie_v_ema", None)
    if ema is None or ema.shape[0] != env.num_envs:
        env.zombie_v_ema = v_proj.clone()
        return float(env.zombie_v_ema.mean())
    ema += (v_proj - ema) * min(1.0, dt / tau)
    return float(env.zombie_v_ema.mean())


def _escape_rate(env: "ManagerBasedEnv", term: str = "out_of_bounds") -> float:
    """Fraction of the population whose last episode ended by escaping the box.

    This is the *real* "it can crawl" signal: the robot reaches the wall and leaves the
    play field. We read the value the TerminationManager already logs every step
    (``Episode_Termination/<term>``, the mean last-episode activation over all envs).

    Why not mean episode length: a good crawler escapes the box *early*, so its episodes
    get SHORT -- gating progress on "survive a long time" is self-defeating, because
    success actively prevents the gate from passing (this is what froze the v_target ladder).
    """
    log = env.extras.get("log", {}) if hasattr(env, "extras") else {}
    val = log.get(f"Episode_Termination/{term}", 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def v_target_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    v_start: float = 0.06,
    v_final: float = 0.10,
    step_size: float = 0.005,
    min_fraction: float = 0.6,
    min_escape: float = 0.5,
    check_interval: int = 100,
) -> float:
    """Advance ``env.v_target`` toward ``v_final`` as the population learns to crawl.

    Gate: sustained forward speed (``mean_vx``) plus a sustained box-escape rate
    (``out_of_bounds >= min_escape``). We deliberately do NOT gate on episode length --
    see :func:`_escape_rate`; a good crawler escapes early, so long episodes are the wrong
    success signal and froze the ladder at ``v_start``.
    """
    if "zombie_curriculum_step" not in env.extras:
        env.extras["zombie_curriculum_step"] = 0
    env.extras["zombie_curriculum_step"] += 1
    # lazily initialize env.v_target (curriculum runs at reset, before any reward is computed)
    v_target = get_v_target(env)

    if env.extras["zombie_curriculum_step"] % check_interval != 0:
        return float(v_target[0])

    mean_vx = _proj_vel_ema(env)  # sustained speed along the reference (jaw) axis
    escape = _escape_rate(env)  # fraction escaping the box last step
    if mean_vx > min_fraction * v_target[0] and escape >= min_escape:
        if not getattr(env.cfg, "use_velocity_commands", False):
            env.v_target = torch.clamp(env.v_target + step_size, max=v_final)
            v_target = env.v_target
    return float(v_target[0])


def friction_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scale_start: float = 1.5,
    scale_final: float = 1.0,
    step_size: float = 0.05,
    min_fraction: float = 0.6,
    check_interval: int = 200,
) -> float:
    """Decay the extra friction multiplier toward 1.0 as the population learns to crawl."""
    if "zombie_friction_step" not in env.extras:
        env.extras["zombie_friction_step"] = 0
        env.extras["friction_scale"] = torch.tensor(
            scale_start, device=env.device
        )
    env.extras["zombie_friction_step"] += 1
    scale = env.extras["friction_scale"]

    if env.extras["zombie_friction_step"] % check_interval != 0:
        return float(scale)

    mean_vx = _proj_vel_ema(env)  # sustained speed along the reference (jaw) axis
    mean_episode_len = env.episode_length_buf.float().mean().item()
    if mean_vx > min_fraction * get_v_target(env).mean() and mean_episode_len > 0.5 * env.max_episode_length:
        scale = torch.clamp(scale - step_size, min=scale_final)
        env.extras["friction_scale"] = scale
    return float(scale)


def difficulty_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    start_after_v: float = 0.09,
    step_size: float = 0.05,
    check_interval: int = 100,
    min_fraction: float = 0.6,
    min_escape: float = 0.5,
    fallen_max: float = 0.5,
    push_max: float = 1.0,
) -> dict:
    """Ramp the recovery/robustness difficulty 0 -> 1, ONLY after the crawl is established.

    This is the single "difficulty" dial the recovery curriculum turns: it drives two env knobs
    consumed elsewhere:

    * ``env.extras["fallen_frac"]`` -- fraction of resets that start toppled (on_side/face_down),
      read by :func:`reset_start_pose` to recompute its pose_weights.
    * ``env.extras["push_scale"]`` -- multiplier on the external-force disturbance, read by
      :func:`disturb_all_bodies`.

    Staging: the fragile forward gait must appear FIRST. Onset is gated on a sustained
    box-escape rate (``out_of_bounds >= min_escape``) -- the true "it reliably crawls out"
    signal -- instead of the old ``v_target >= start_after_v`` gate, which could never fire
    because the v_target ladder itself was stuck (see :func:`_escape_rate`).

    Once difficulty has begun (> 0), it keeps ramping as long as the population still crawls
    forward (``mean_vx > min_fraction * v_target``). We intentionally do NOT re-check the escape
    rate after onset: fallen spawns and pushes deliberately lower the escape rate, so re-checking
    it would lock the ramp at its first step. ``mean_vx`` is the self-limiting signal -- if the
    robot can no longer make forward progress under the difficulty, ramping pauses.
    """
    if "difficulty" not in env.extras:
        env.extras["difficulty"] = 0.0
        env.extras["fallen_frac"] = 0.0
        env.extras["push_scale"] = 0.0
        env.extras["zombie_difficulty_step"] = 0
        # one-shot sanity print: proves which zombie_crawl source tree this run imported
        import zombie_crawl

        print(
            f"[CURRICULUM] difficulty term ACTIVE -- zombie_crawl imported from: "
            f"{zombie_crawl.__file__}",
            flush=True,
        )
    env.extras["zombie_difficulty_step"] += 1

    # returned as a dict so the CurriculumManager logs each dial as its own TensorBoard card:
    # Curriculum/difficulty/difficulty, /fallen_frac, /push_scale (manager wraps keys under the term name)
    def _state() -> dict:
        return {
            "difficulty": float(env.extras["difficulty"]),
            "fallen_frac": float(env.extras["fallen_frac"]),
            "push_scale": float(env.extras["push_scale"]),
        }

    if env.extras["zombie_difficulty_step"] % check_interval != 0:
        return _state()

    v_now = float(get_v_target(env).mean())
    started = float(env.extras["difficulty"]) > 0.0
    escape = _escape_rate(env)
    # onset: only begin turning up difficulty once the population reliably crawls out of the box
    if not started and escape < min_escape:
        return _state()

    mean_vx = _proj_vel_ema(env)
    # continuation: keep ramping while the population still makes forward progress under the difficulty
    if mean_vx > min_fraction * v_now:
        d = min(1.0, env.extras["difficulty"] + step_size)
        env.extras["difficulty"] = d
        env.extras["fallen_frac"] = fallen_max * d
        env.extras["push_scale"] = push_max * d
    return _state()
