# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success metrics for the zombie crawl task (spec P2-10).

Self-contained per-episode bookkeeping. :func:`note_step` runs once per env step
(driven by the ``yaw_rate_tracking`` reward -- the single per-step hook) and integrates
the episode accumulators; :func:`score_finished_episodes` runs from the ``crawl_metrics``
curriculum term at the START of the reset step (before the reset events re-initialize
anything, but with the per-step accumulators still holding the just-finished episode) and
folds the episode results into population EMAs. Accumulators re-initialize themselves on
the first step of the next episode via ``episode_length_buf`` wrap detection, so the
module plugs into the phase-1 tree unchanged (R0 reference run) as well as phase 2.

Definitions (upright-start = spawned ``folded``; fallen-start = ``on_side``/``face_down``):

* ``crawl_success``   -- upright starts: ``sum(v_c*dt) / 20s >= 0.6 * v_target``
  (a failure termination shortens the episode, so it scores low; never dropped).
* ``recovery_success``-- fallen starts: ``-projected_gravity_b.z >= 0.8`` reached within
  10 s of episode start; ``time_to_upright`` logged over the recovered ones.
* ``steer_success``   -- per command segment (between resamples, or resample -> episode
  end) with ``|cmd| > 0.05`` and >= ``YAW_MASK_S + 1`` s of unmasked time:
  ``|mean(r over unmasked part) - cmd| <= max(0.35*|cmd|, 0.03)``. A failure termination
  inside the segment counts as a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# --- thresholds (initial values from the spec; tune only via the experiment log) -----
YAW_MASK_S = 2.0  # post-resample mask ramp [s]; must match yaw_rate_tracking's param
SEG_MIN_UNMASKED_S = YAW_MASK_S + 1.0  # minimum scored unmasked span per segment [s]
SEG_MIN_CMD = 0.05  # segments with |yaw_rate_cmd| below this are not scored [rad/s]
STEER_TOL_REL = 0.35
STEER_TOL_ABS = 0.03  # [rad/s]
CRAWL_SUCCESS_FRAC = 0.6  # fraction of v_target the claw-axis integral must reach
RECOVERY_UPRIGHT_THRESH = 0.8  # -projected_gravity_b.z threshold
RECOVERY_WINDOW_S = 10.0  # [s] to reach upright from a fallen start
BACK_VEL_THRESH = -0.02  # [m/s]; "moving against the claw arrow" watch item W2
_EMA_WINDOW = 2000  # ~episodes in the population EMAs
_STEP_EMA_TAU_S = 4.0  # EMA time constant for the per-step population diagnostics


def _state(env: ManagerBasedEnv, num_envs: int, device) -> dict:
    st = getattr(env, "zombie_metrics_state", None)
    if st is not None and st["v_c_int"].shape[0] == num_envs:
        return st
    st = {
        "v_c_int": torch.zeros(num_envs, device=device),
        "steps": torch.zeros(num_envs, device=device),
        "back": torch.zeros(num_envs, device=device),
        "upright_t": torch.full((num_envs,), -1.0, device=device),
        "prev_el": torch.zeros(num_envs, device=device, dtype=torch.long),
        "started_fallen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "fallen_known": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "seg_cmd": torch.zeros(num_envs, device=device),
        "seg_rsum": torch.zeros(num_envs, device=device),
        "seg_rn": torch.zeros(num_envs, device=device),
        "seg_scored": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    env.zombie_metrics_state = st
    return st


def _metrics(env: ManagerBasedEnv) -> dict:
    m = getattr(env, "zombie_metrics", None)
    if m is None:
        m = {}
        env.zombie_metrics = m
    return m


def _ema(env: ManagerBasedEnv, key: str, k: float, mean_new: float) -> None:
    """Episode-count EMA with effective window ~``_EMA_WINDOW`` finished episodes."""
    if k <= 0:
        return
    m = _metrics(env)
    cur = m.get(key)
    w = k / (k + _EMA_WINDOW)
    m[key] = mean_new if cur is None else cur + w * (mean_new - cur)


def get_metric(env: ManagerBasedEnv, key: str, default: float = 0.0) -> float:
    """Current population EMA for a gate input (0.0 before the first finished episode)."""
    return float(_metrics(env).get(key, default))


def set_started_fallen(env: ManagerBasedEnv, env_ids: torch.Tensor, fallen: torch.Tensor) -> None:
    """Record ``started_fallen`` for a reset batch (called from ``mdp.reset_start_pose``)."""
    st = _state(env, env.num_envs, env.device)
    st["started_fallen"][env_ids] = fallen.to(st["started_fallen"])
    st["fallen_known"][env_ids] = True


def _claw_v(env: ManagerBasedEnv):
    from .rewards import get_claw_dir

    v = env.scene["robot"].data.root_lin_vel_w.torch
    cos_u, sin_u = get_claw_dir(env)
    return v[:, 0] * cos_u + v[:, 1] * sin_u


def note_step(env: ManagerBasedEnv, dt: float) -> None:
    """One hook per env step: integrate episode accumulators + steering segments.

    Called from ``mdp.yaw_rate_tracking`` (the mandated single per-step hook). Detects
    episode restarts by ``episode_length_buf`` wrap and re-initializes accumulators.
    """
    num_envs = env.num_envs
    device = env.device
    st = _state(env, num_envs, device)
    el = env.episode_length_buf
    wrapped = el < st["prev_el"]
    if bool(wrapped.any()):
        st["v_c_int"][wrapped] = 0.0
        st["steps"][wrapped] = 0.0
        st["back"][wrapped] = 0.0
        st["upright_t"][wrapped] = -1.0
        st["fallen_known"][wrapped] = False
        st["seg_rsum"][wrapped] = 0.0
        st["seg_rn"][wrapped] = 0.0
        st["seg_scored"][wrapped] = False

    v_c = _claw_v(env)
    st["v_c_int"] += v_c * dt
    st["steps"] += 1.0
    st["back"] += (v_c < BACK_VEL_THRESH).double()
    st["prev_el"] = el

    # fallen-start bookkeeping: derive from the pose at the first step when
    # reset_start_pose did not record it (phase-1 backport path); phase 2 sets it there.
    first_step = el <= 1
    unknown_first = first_step & ~st["fallen_known"]
    if bool(unknown_first.any()):
        grav = env.scene["robot"].data.projected_gravity_b.torch
        st["started_fallen"][unknown_first] = -grav[unknown_first, 2] < RECOVERY_UPRIGHT_THRESH
        st["fallen_known"][unknown_first] = True

    # recovery timing: first moment -g_z >= 0.8
    grav = env.scene["robot"].data.projected_gravity_b.torch
    pending = (st["upright_t"] < 0.0) & (-grav[:, 2] >= RECOVERY_UPRIGHT_THRESH)
    if bool(pending.any()):
        st["upright_t"][pending] = el[pending].to(st["upright_t"].dtype) * dt

    # per-step population diagnostics (W2/W3): EMA over ~_STEP_EMA_TAU_S seconds
    alpha = min(1.0, dt / _STEP_EMA_TAU_S)
    m = _metrics(env)
    back_now = float(st["back"].sum()) / max(1.0, float(st["steps"].sum()))
    m["back_frac"] = back_now if "back_frac" not in m else m["back_frac"] + alpha * (back_now - m["back_frac"])
    base_yaw = env.scene["robot"].data.root_ang_vel_w.torch[:, 2]
    yaw_now = float(base_yaw.abs().mean())
    m["base_yaw_rate_abs"] = yaw_now if "base_yaw_rate_abs" not in m else m["base_yaw_rate_abs"] + alpha * (
        yaw_now - m["base_yaw_rate_abs"]
    )
    v_now = float(v_c.mean())
    m["mean_v_c"] = v_now if "mean_v_c" not in m else m["mean_v_c"] + alpha * (v_now - m["mean_v_c"])

    # steering segments (phase-2 only: no r buffer in the phase-1 backport -> exit here,
    # so the yaw-rate reward imports below never run against a phase-1 tree)
    r = getattr(env, "zombie_r_ema", None)
    if r is None or r.shape[0] != num_envs:
        return
    from .rewards import yaw_rate_command, yaw_rate_resample_flag

    r_now = float(r.abs().mean())
    m["r_abs"] = r_now if "r_abs" not in m else m["r_abs"] + alpha * (r_now - m["r_abs"])
    cmd = yaw_rate_command(env)
    timer = getattr(env, "zombie_time_since_resample", None)
    if timer is None or timer.shape[0] != num_envs:
        return
    resampled = yaw_rate_resample_flag(env)
    if resampled is not None and bool(resampled.any()):
        _score_segments(env, st, resampled, failed=None)
        st["seg_cmd"][resampled] = cmd[resampled]
        st["seg_rsum"][resampled] = 0.0
        st["seg_rn"][resampled] = 0.0
        st["seg_scored"][resampled] = False
    unmasked = timer >= YAW_MASK_S
    counting = unmasked & ~st["seg_scored"] & (st["seg_cmd"].abs() > SEG_MIN_CMD)
    st["seg_rsum"][counting] += r[counting]
    st["seg_rn"][counting] += 1.0


def _score_segments(env: ManagerBasedEnv, st: dict, env_ids: torch.Tensor, failed: torch.Tensor | None) -> None:
    """Fold the segments of ``env_ids`` (that have not been scored yet) into steer_success."""
    if env_ids.dtype == torch.bool:  # callers pass either a boolean mask or an index tensor
        env_ids = env_ids.nonzero().flatten()
    eligible = (
        ~st["seg_scored"][env_ids]
        & (st["seg_cmd"][env_ids].abs() > SEG_MIN_CMD)
        & (st["seg_rn"][env_ids] * env.step_dt >= SEG_MIN_UNMASKED_S)
    )
    if not bool(eligible.any()):
        return
    ids = env_ids[eligible]
    mean_r = st["seg_rsum"][ids] / st["seg_rn"][ids].clamp(min=1.0)
    cmd = st["seg_cmd"][ids]
    tol = torch.clamp(STEER_TOL_REL * cmd.abs(), min=STEER_TOL_ABS)
    ok = (mean_r - cmd).abs() <= tol
    if failed is not None:
        ok = ok & ~failed[ids]
    successes = float(ok.sum())
    _ema(env, "steer_success", float(ids.numel()), successes / float(ids.numel()))
    st["seg_scored"][ids] = True


def score_finished_episodes(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Fold the just-finished episodes of ``env_ids`` into the population EMAs.

    Called by the ``crawl_metrics`` curriculum term at the start of the reset step, while
    ``episode_length_buf`` still holds the finished episode length and
    ``reset_terminated`` marks failure terminations (truncations are not failures).
    """
    from .rewards import get_v_target

    num_envs = env.num_envs
    if env_ids is None or (
        isinstance(env_ids, slice) and env_ids.start is None and env_ids.stop is None
    ):
        env_ids = torch.arange(num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    st = _state(env, num_envs, env.device)
    if env_ids.numel() == 0:
        return
    stepped = st["steps"][env_ids] > 0
    env_ids = env_ids[stepped]
    if env_ids.numel() == 0:
        return

    nominal_t = float(getattr(env.cfg, "episode_length_s", 20.0))
    v_target_mean = float(get_v_target(env).mean())
    upright = st["upright_t"][env_ids]
    fallen = st["started_fallen"][env_ids]
    not_fallen = ~fallen

    # crawl_success: upright-start episodes only (pitfall X2: fallen starts can never
    # succeed early, so a gate that includes them can never open)
    if bool(not_fallen.any()):
        frac = st["v_c_int"][env_ids][not_fallen] / max(nominal_t, 1e-6)
        ok = frac >= CRAWL_SUCCESS_FRAC * max(v_target_mean, 1e-6)
        k = float(not_fallen.sum())
        _ema(env, "crawl_success", k, float(ok.sum()) / k)

    # recovery_success + time_to_upright: fallen-start episodes only
    if bool(fallen.any()):
        up = upright[fallen]
        ok = (up >= 0.0) & (up <= RECOVERY_WINDOW_S)
        k = float(fallen.sum())
        _ema(env, "recovery_success", k, float(ok.sum()) / k)
        recovered = up[ok]
        if recovered.numel() > 0:
            _ema(env, "time_to_upright", float(recovered.numel()), float(recovered.mean()))

    # open steering segment: a failure termination inside it fails the segment
    terminated = getattr(env, "reset_terminated", None)
    failed = None
    if terminated is not None and terminated.shape[0] == num_envs:
        failed = terminated
    _score_segments(env, st, env_ids, failed)


def metrics_snapshot(env: ManagerBasedEnv) -> dict:
    """Scalar dict for TensorBoard logging (every dial-adjacent metric, per iteration)."""
    m = _metrics(env)
    return {
        "crawl_success": float(m.get("crawl_success", 0.0)),
        "recovery_success": float(m.get("recovery_success", 0.0)),
        "steer_success": float(m.get("steer_success", 0.0)),
        "time_to_upright": float(m.get("time_to_upright", 0.0)),
        "back_frac": float(m.get("back_frac", 0.0)),
        "base_yaw_rate_abs": float(m.get("base_yaw_rate_abs", 0.0)),
        "r_abs": float(m.get("r_abs", 0.0)),
        "mean_v_c": float(m.get("mean_v_c", 0.0)),
    }
