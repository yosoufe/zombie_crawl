# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the zombie crawl task (phase 2).

The curriculum advances five dials, all consumed elsewhere:

* ``env.v_target`` -- per-environment forward velocity target (tensor, (num_envs,)).
* ``env.extras["friction_scale"]`` -- extra friction multiplier, decays 1.5 -> 1.0.
* ``env.extras["difficulty"]`` (+ ``fallen_frac`` over [0, 0.5], ``push_scale`` over
  [0.5, 1]) -- recovery/robustness staging.
* ``env.extras["yaw_rate_max"]`` -- the yaw-rate command stage dial (P2-12): 0 -> 0.10
  -> up to 0.30 rad/s, monotonic, gated on difficulty and steering success.

The phase-2 gates read the *success metrics* (:mod:`.metrics`), not the removed position
box: ``escape`` became ``crawl_success`` (upright-start episodes only -- pitfall X2), and
``mean_vx`` is now the claw-axis-projected velocity EMA (pitfall X3: velocity is
instantaneous, so projecting on the rotating claw axis is harmless and needs no
re-anchoring). The curriculum terms run at the START of the reset step, before the reset
events re-initialize the per-episode accumulators (verified against the installed Isaac
Lab: ``ManagerBasedRLEnv._reset_idx`` calls ``curriculum_manager.compute`` first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # imported lazily: pulling the concrete env class in at module level drags in
    # ``pxr`` before SimulationApp is created, which breaks Kit extension loading
    from isaaclab.envs import ManagerBasedEnv

from . import metrics
from .rewards import get_v_target


def _proj_vel_ema(env: "ManagerBasedEnv", tau: float = 2.0) -> float:
    """Population mean of an EMA (~``tau`` s) of the CLAW-axis-projected base velocity.

    Gating on instantaneous velocity let a flailing population's stand-up/topple kicks
    advance difficulty on bogus evidence; the EMA gates on sustained forward motion. The
    projection is instantaneous, so rotating the claw axis between steps is harmless
    (pitfall X3) and -- unlike the removed world-frame command axis -- no resample
    re-anchoring is needed.
    """
    from .rewards import get_claw_dir

    dt = float(env.step_dt)
    v = env.scene["robot"].data.root_lin_vel_w.torch
    cos_u, sin_u = get_claw_dir(env)
    v_proj = v[:, 0] * cos_u + v[:, 1] * sin_u
    ema = getattr(env, "zombie_v_ema", None)
    if ema is None or ema.shape[0] != env.num_envs:
        env.zombie_v_ema = v_proj.clone()
        return float(env.zombie_v_ema.mean())
    ema += (v_proj - ema) * min(1.0, dt / tau)
    return float(env.zombie_v_ema.mean())


def crawl_metrics_curriculum(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> dict:
    """Success-metrics term (P2-10): scores the just-finished episodes and logs the EMAs.

    Runs at the start of every reset step (before the reset events), reads the per-episode
    accumulators :mod:`.metrics` maintained during the episode, folds them into the
    population EMAs (``crawl_success`` on upright starts, ``recovery_success`` +
    ``time_to_upright`` on fallen starts, ``steer_success`` per command segment), and
    returns everything (plus W2/W3 watch diagnostics) so the CurriculumManager logs each
    value as its own TensorBoard card.
    """
    metrics.score_finished_episodes(env, env_ids)
    return metrics.metrics_snapshot(env)


def v_target_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    v_start: float = 0.06,
    v_final: float = 0.10,
    step_size: float = 0.005,
    min_fraction: float = 0.6,
    min_crawl: float = 0.5,
    check_interval: int = 100,
) -> float:
    """Advance ``env.v_target`` toward ``v_final`` as the population learns to crawl.

    Gate (P2-11): sustained claw-axis forward speed (``mean_v_c EMA > min_fraction *
    v_target``) AND ``crawl_success >= min_crawl`` (upright-start episodes only). The
    phase-1 box-escape gate is gone: ``crawl_success`` is the real "it can crawl" signal.
    """
    if "zombie_curriculum_step" not in env.extras:
        env.extras["zombie_curriculum_step"] = 0
    env.extras["zombie_curriculum_step"] += 1
    # lazily initialize env.v_target (curriculum runs at reset, before any reward is computed)
    v_target = get_v_target(env)

    if env.extras["zombie_curriculum_step"] % check_interval != 0:
        return float(v_target[0])

    mean_v_c = _proj_vel_ema(env)  # sustained speed along the claw axis
    crawl = metrics.get_metric(env, "crawl_success")
    if mean_v_c > min_fraction * v_target[0] and crawl >= min_crawl:
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
    min_crawl: float = 0.5,
    check_interval: int = 200,
) -> float:
    """Decay the extra friction multiplier toward 1.0 as the population learns to crawl.

    Same success gate as ``v_target_curriculum`` (P2-11): claw-axis speed EMA +
    ``crawl_success``.
    """
    if "zombie_friction_step" not in env.extras:
        env.extras["zombie_friction_step"] = 0
        env.extras["friction_scale"] = torch.tensor(
            scale_start, device=env.device
        )
    env.extras["zombie_friction_step"] += 1
    scale = env.extras["friction_scale"]

    if env.extras["zombie_friction_step"] % check_interval != 0:
        return float(scale)

    mean_v_c = _proj_vel_ema(env)
    crawl = metrics.get_metric(env, "crawl_success")
    if mean_v_c > min_fraction * get_v_target(env).mean() and crawl >= min_crawl:
        scale = torch.clamp(scale - step_size, min=scale_final)
        env.extras["friction_scale"] = scale
    return float(scale)


def difficulty_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    step_size: float = 0.05,
    check_interval: int = 100,
    min_fraction: float = 0.6,
    min_crawl: float = 0.5,
    fallen_max: float = 0.5,
    push_max: float = 1.0,
    push_start_frac: float = 0.5,
) -> dict:
    """Ramp the recovery/robustness difficulty 0 -> 1, ONLY after the crawl is established.

    This is the single "difficulty" dial the recovery curriculum turns: it drives two env knobs
    consumed elsewhere:

    * ``env.extras["fallen_frac"]`` -- fraction of resets that start toppled (on_side/face_down),
      read by :func:`reset_start_pose` to recompute its pose_weights (P2-9: fallen starts come
      ONLY from this key -- never from the baked weights).
    * ``env.extras["push_scale"]`` -- multiplier on the external-force disturbance, read by
      :func:`disturb_all_bodies`.

    Staging: the fragile forward gait must appear FIRST. Onset is gated on
    ``crawl_success >= min_crawl`` (P2-11; computed on upright-start episodes only --
    pitfall X2). Once difficulty has begun (> 0), it keeps ramping while the population
    still crawls forward (``mean_v_c EMA > min_fraction * v_now``); re-checking onset
    would lock the ramp at its first step, because fallen spawns and pushes deliberately
    lower the success rate. The dial never moves backward.

    Within phase 2 the two sub-channels are STAGGERED by ``push_start_frac``: the robot
    learns to get UP from fallen spawns first (``fallen_frac`` fills over ``difficulty`` in
    ``[0, f]``), and only once it can do that does the external disturbance turn on
    (``push_scale`` fills over ``difficulty`` in ``[f, 1]``).
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
    crawl = metrics.get_metric(env, "crawl_success")
    # onset: only begin turning up difficulty once upright-start episodes reliably crawl
    if not started and crawl < min_crawl:
        return _state()

    mean_v_c = _proj_vel_ema(env)
    # continuation: keep ramping while the population still makes forward progress under the difficulty
    if mean_v_c > min_fraction * v_now:
        d = min(1.0, env.extras["difficulty"] + step_size)
        env.extras["difficulty"] = d
        # staggered: fallen spawns fill over [0, f], pushes only fill over [f, 1]
        f = min(max(push_start_frac, 1e-6), 1.0 - 1e-6)
        env.extras["fallen_frac"] = fallen_max * min(1.0, d / f)
        env.extras["push_scale"] = push_max * max(0.0, min(1.0, (d - f) / (1.0 - f)))
    return _state()


def yaw_rate_stage_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    unlock_difficulty: float = 0.75,
    stage1_rate: float = 0.10,
    step_size: float = 0.05,
    min_steer: float = 0.7,
    check_interval: int = 100,
) -> dict:
    """Yaw-rate command stages (P2-12), stored on ``env.extras["yaw_rate_max"]``.

    ====  ==========================  ==========================================
    St  dial                          unlock
    ====  ==========================  ==========================================
    0     0 (all commands 0)          start
    1     ``stage1_rate`` (0.10)      ``difficulty >= unlock_difficulty`` (0.75,
                                      the point phase 1 used for its dynamic heading)
    2     +step_size per check to     ``steer_success >= min_steer`` (0.7), checked
        ``env.cfg.yaw_rate_max_final`` every ``check_interval`` curriculum calls
        (0.30 rad/s provisional)
    ====  ==========================  ==========================================

    Monotonic -- a dial never moves backward. ``env.cfg.yaw_rate_stage_lock`` (debug,
    read live so Hydra can set it post-construction) forces a stage: 0 keeps steering
    locked off (R1), >=1 jumps straight to that stage ignoring the gates (single-stage
    debugging). The reachable maximum is discovered by RL: the final cap is provisional.
    """
    if "yaw_rate_max" not in env.extras:
        env.extras["yaw_rate_max"] = 0.0
        env.extras["yaw_rate_stage"] = 0
        env.extras["zombie_yaw_stage_step"] = 0
        import zombie_crawl

        print(
            f"[CURRICULUM] yaw_rate_stage term ACTIVE -- zombie_crawl imported from: "
            f"{zombie_crawl.__file__}",
            flush=True,
        )
    env.extras["zombie_yaw_stage_step"] += 1

    def _state() -> dict:
        return {
            "yaw_rate_max": float(env.extras["yaw_rate_max"]),
            "stage": float(env.extras["yaw_rate_stage"]),
        }

    if env.extras["zombie_yaw_stage_step"] % check_interval != 0:
        return _state()

    cap_final = float(getattr(env.cfg, "yaw_rate_max_final", 0.30))
    lock = getattr(env.cfg, "yaw_rate_stage_lock", None)
    if lock is not None:
        # single-stage debug: force the dial to the requested stage, ignore the gates
        stage = int(lock)
        env.extras["yaw_rate_stage"] = stage
        env.extras["yaw_rate_max"] = 0.0 if stage <= 0 else (stage1_rate if stage == 1 else cap_final)
        return _state()

    stage = int(env.extras["yaw_rate_stage"])
    if stage == 0:
        if float(env.extras["difficulty"]) >= unlock_difficulty:
            stage = 1
            env.extras["yaw_rate_max"] = stage1_rate
    if stage == 1:
        if metrics.get_metric(env, "steer_success") >= min_steer:
            stage = 2
    if stage >= 2:
        stage = 2
        # climb toward the provisional cap only while the population actually tracks
        if metrics.get_metric(env, "steer_success") >= min_steer:
            env.extras["yaw_rate_max"] = min(cap_final, float(env.extras["yaw_rate_max"]) + step_size)
    env.extras["yaw_rate_stage"] = stage
    return _state()
