# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success metrics for the zombie crawl task (spec P2-10, extended by B.8).

Self-contained per-episode bookkeeping. :func:`note_step` runs once per env step
(driven by the ``yaw_rate_tracking`` reward -- the single per-step hook) and integrates
the episode accumulators; :func:`score_finished_episodes` runs from the ``crawl_metrics``
curriculum term at the START of the reset step (before the reset events re-initialize
anything, but with the per-step accumulators still holding the just-finished episode) and
folds the episode results into population EMAs. Accumulators re-initialize themselves on
the first step of the next episode via ``episode_length_buf`` wrap detection, so the
module plugs into the phase-1 tree unchanged (R0 reference run) as well as phases 2/3.

Definitions (upright-start = spawned ``folded``/rest; fallen-start = ``on_side``/``face_down``):

* ``crawl_success``   -- upright starts: ``sum(v_c*dt) / episode_t >= 0.6 * v_target``
  (a failure termination shortens the episode, so it scores low; never dropped).
* ``recovery_success``-- fallen starts: ``-projected_gravity_b.z >= 0.8`` reached within
  10 s of episode start; ``time_to_upright`` logged over the recovered ones.
* ``steer_success``   -- per command segment (between resamples, or resample -> episode
  end) with ``|yaw_rate_cmd| > 0.05`` and >= ``YAW_MASK_S + 1`` s of unmasked time:
  ``|mean(r over the unmasked part) - cmd| <= max(0.35*|cmd|, 0.03)``. A failure termination
  inside the segment counts as a failure. (Phase-2 metric; kept -- the yaw-rate stage gate
  reads it. Phase 3 adds the per-mode metrics below.)

Phase 3 (B.8) per-mode segment metrics, scored per command segment with >=
``SEG_MIN_UNMASKED_S`` of unmasked time (mode M read from the command term; A FAILURE
termination inside the segment fails it):

* ``success/M1`` -- mean claw-axis speed within +-30 % of the commanded ``vx``; |mean r| < 0.05.
* ``success/M2`` -- speed within +-35 %; |mean r - omega| <= max(35 % |omega|, 0.06).
* ``success/M3`` -- |mean r - omega| <= max(30 % |omega|, 0.06) over the WHOLE post-mask
  segment; mean base translation speed < 0.03 m/s.
* ``success/M4`` -- mean base speed < 0.03 m/s; ``rest_pose_ok`` over >= 90 % of the
  post-mask steps.

Owner relaxation (phase-2 A.12-8 lesson applied up front): the ABSOLUTE yaw tolerance is
0.06 rad/s (not the spec's 0.03) because the robot's mechanical drift makes tighter
in-rollout tracking impossible; logged deviation. Extra diagnostics: ``trans_pairs_seen``
(of 12 ordered mode pairs), ``trans_min_frac``, ``settle_mean_s`` (mean time-to-settle
across resamples), ``base_turn_fraction`` (base |yaw| change over claw |yaw| change on
scored M3 segments), ``pan_at_limit_fraction`` (share of M3 steps with the pan joint
within 0.05 rad of a soft limit), ``m4_creep_m``.
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
STEER_TOL_ABS = 0.06  # [rad/s] raised from phase-2's 0.03 per the owner's phase-3
# tolerance-floor decision (M2/M3): this legacy EMA now also feeds the mode-ladder
# p2-ceilings check (yaw_rate_stage gate), where 0.03 proved unreachable (plateau 0.28)
MODE_TOL_ABS = 0.06  # [rad/s] per-mode yaw tolerance floor (owner: achievable-value relaxation)
# Owner decision (phase-3 acceptance, docs/phase3_report.md): the spec's +-30/35 % relative
# bands are UNREACHABLE on the low-power robot -- straight crawl measures 0.146-0.161 m/s
# at vx_cmd=0.10 (+46..61 %), so a relative band would pin success/M1 at 0 and never open
# the stage gate. Band = [low_rel * vx, vx + headroom], calibrated to the owner's
# [0.07, 0.17] m/s at vx=0.10 and scaled with the v_target dial.
M1_SPEED_LOW_REL = 0.70  # M1 claw-speed lower bound (relative to vx_cmd)
M2_SPEED_LOW_REL = 0.65  # M2 claw-speed lower bound (relative to vx_cmd)
M_SPEED_HEADROOM = 0.07  # [m/s] absolute headroom above vx_cmd (both M1 and M2)
M2_YAW_REL = 0.35  # M2 |mean r - omega| relative tolerance
M3_YAW_REL = 0.30  # M3 |mean r - omega| relative tolerance (spec B.8: whole segment)
M3_SPEED_MAX = 0.03  # [m/s] mean base translation speed limit in M3
M4_SPEED_MAX = 0.03  # [m/s] mean base speed limit in M4
M4_REST_FRAC = 0.90  # fraction of post-mask steps that must satisfy rest_pose_ok
CRAWL_SUCCESS_FRAC = 0.6  # fraction of v_target the claw-axis integral must reach
RECOVERY_UPRIGHT_THRESH = 0.8  # -projected_gravity_b.z threshold
RECOVERY_WINDOW_S = 10.0  # [s] to reach upright from a fallen start
BACK_VEL_THRESH = -0.02  # [m/s]; "moving against the claw arrow" watch item W2
PAN_LIMIT_MARGIN = 0.05  # [rad]; "at the pan soft limit" window (V1)
_EMA_WINDOW = 2000  # ~episodes/segments in the population EMAs
_STEP_EMA_TAU_S = 4.0  # EMA time constant for the per-step population diagnostics


def _in_speed_band(v: torch.Tensor, vx: torch.Tensor, low_rel: float) -> torch.Tensor:
    """M1/M2 claw-speed band [low_rel*vx, vx + M_SPEED_HEADROOM] (owner calibration)."""
    vx = vx.clamp(min=1e-3)
    return (v >= low_rel * vx) & (v <= vx + M_SPEED_HEADROOM)


def _state(env: ManagerBasedEnv, num_envs: int, device) -> dict:
    st = getattr(env, "zombie_metrics_state", None)
    if st is not None and st["v_c_int"].shape[0] == num_envs:
        return st
    z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    st = {
        "v_c_int": z(),
        "steps": z(),
        "back": z(),
        "upright_t": torch.full((num_envs,), -1.0, device=device),
        "prev_el": torch.zeros(num_envs, device=device, dtype=torch.long),
        "started_fallen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "fallen_known": torch.zeros(num_envs, dtype=torch.bool, device=device),
        # steering segment (phase-2 steer_success)
        "seg_cmd": z(),
        "seg_rsum": z(),
        "seg_rn": z(),
        "seg_scored": torch.zeros(num_envs, dtype=torch.bool, device=device),
        # phase-3 per-mode segment accumulators
        "seg_mode": torch.zeros(num_envs, dtype=torch.long, device=device),
        "seg_prev_mode": torch.zeros(num_envs, dtype=torch.long, device=device),
        "seg_vx": z(),
        "seg_vsum": z(),  # integral of v_c over unmasked time
        "seg_unmasked_t": z(),  # unmasked duration
        "seg_speed_int": z(),  # integral of ||v_xy|| over all segment time
        "seg_t": z(),  # total segment duration
        "seg_rest_ok": z(),  # rest_pose_ok step count (post-mask)
        "seg_rest_n": z(),  # post-mask step count
        "seg_base_dpsi": z(),  # summed |base yaw delta| (M3 turn accounting)
        "seg_claw_dpsi": z(),  # summed |claw heading delta|
        "seg_xy0": torch.zeros(num_envs, 2, device=device),
        "seg_settle_t": torch.full((num_envs,), -1.0, device=device),
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


def _base_xy_speed(env: ManagerBasedEnv) -> torch.Tensor:
    v = env.scene["robot"].data.root_lin_vel_w.torch
    return torch.hypot(v[:, 0], v[:, 1])


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
        for k in ("seg_rsum", "seg_rn", "seg_vsum", "seg_unmasked_t", "seg_speed_int", "seg_t",
                  "seg_rest_ok", "seg_rest_n", "seg_base_dpsi", "seg_claw_dpsi"):
            st[k][wrapped] = 0.0
        st["seg_scored"][wrapped] = False
        st["seg_settle_t"][wrapped] = -1.0

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

    # steering / per-mode segments: need the claw-rate filter (absent in phase-1 backport)
    r = getattr(env, "zombie_r_ema", None)
    if r is None or r.shape[0] != num_envs:
        return
    from .rewards import mode_term, yaw_rate_command, yaw_rate_resample_flag

    r_now = float(r.abs().mean())
    m["r_abs"] = r_now if "r_abs" not in m else m["r_abs"] + alpha * (r_now - m["r_abs"])
    cmd = yaw_rate_command(env)
    timer = getattr(env, "zombie_time_since_resample", None)
    if timer is None or timer.shape[0] != num_envs:
        return

    term = mode_term(env)
    has_mode = term is not None and hasattr(term, "mode")

    resampled = yaw_rate_resample_flag(env)
    if resampled is not None and bool(resampled.any()):
        _score_segments(env, st, resampled, failed=None)
        _begin_segment(env, st, resampled, cmd, term)
    elif bool(wrapped.any()) and has_mode:
        # opening segment of a new episode: the reset resample already wrote fresh values
        _begin_segment(env, st, wrapped.nonzero().flatten(), cmd, term)

    unmasked = timer >= YAW_MASK_S
    # the |cmd| floor is a TRACKING-mode gate (M2/M3 need a non-zero omega to score
    # against). M1/M4 commands ARE zero -- gating them by SEG_MIN_CMD made every M1
    # segment ineligible (phase-3 fix; phase 2's steer_success keeps the original gate)
    cmd_ok = (
        (st["seg_cmd"].abs() > SEG_MIN_CMD - 0.01)
        | (st["seg_mode"] == 1)
        | (st["seg_mode"] == 4)
    )
    counting = unmasked & ~st["seg_scored"] & cmd_ok
    st["seg_rsum"][counting] += r[counting]
    st["seg_rn"][counting] += 1.0

    if not has_mode:
        return

    # ---- phase-3 per-mode accumulators (B.8) --------------------------------------
    speed = _base_xy_speed(env)
    live = ~st["seg_scored"]
    st["seg_speed_int"][live] += speed[live] * dt
    st["seg_t"][live] += dt
    st["seg_vsum"][counting] += v_c[counting] * dt
    st["seg_unmasked_t"][counting] += dt
    # rest_pose_ok accounting for M4 (post-mask steps only)
    from .rewards import rest_pose_ok

    is_m4 = live & (st["seg_mode"] == 4)
    # per-clause telemetry of rest_pose_ok over the M4 population (which clause blocks?)
    if bool(is_m4.any()):
        from .rewards import flatness_gate

        a4 = env.scene["robot"]
        up4 = flatness_gate(env)
        z4 = a4.data.root_pos_w.torch[:, 2]
        qerr4 = (a4.data.joint_pos.torch - a4.data.default_joint_pos.torch).abs().amax(dim=1)
        h4 = float(getattr(env.cfg, "rest_height", 0.0037))
        cu = (up4 >= float(getattr(env.cfg, "rest_up_thresh", 0.95)))[is_m4].to(torch.float32).mean()
        cz = ((z4 - h4).abs() <= float(getattr(env.cfg, "rest_height_tol", 0.01)))[is_m4].to(torch.float32).mean()
        cq = (qerr4 <= float(getattr(env.cfg, "rest_joint_tol", 0.15)))[is_m4].to(torch.float32).mean()
        _ema(env, "m4_ok_up", 1.0, float(cu))
        _ema(env, "m4_ok_z", 1.0, float(cz))
        _ema(env, "m4_ok_q", 1.0, float(cq))
        _ema(env, "m4_qerr_max", 1.0, float(qerr4[is_m4].mean()))
    ok4 = rest_pose_ok(env)
    st["seg_rest_n"][is_m4] += unmasked[is_m4].to(st["seg_rest_n"].dtype)
    st["seg_rest_ok"][is_m4] += (unmasked & ok4.bool())[is_m4].to(st["seg_rest_ok"].dtype)
    # M3 turn bookkeeping: base |yaw| delta vs claw |heading| delta
    from isaaclab.utils import math as math_utils

    quat = env.scene["robot"].data.root_quat_w.torch
    _, _, byaw = math_utils.euler_xyz_from_quat(quat)
    prev_byaw = getattr(env, "zombie_base_yaw_prev", None)
    if prev_byaw is None or prev_byaw.shape[0] != num_envs:
        env.zombie_base_yaw_prev = byaw.clone()
        d_base = torch.zeros(num_envs, device=device)
    else:
        d_base = torch.atan2(torch.sin(byaw - prev_byaw), torch.cos(byaw - prev_byaw))
        env.zombie_base_yaw_prev = byaw
        d_base = torch.where(d_base.abs() > 0.5, torch.zeros_like(d_base), d_base)
    d_claw = getattr(env, "zombie_dpsi_last", None)
    if d_claw is not None and d_claw.shape[0] == num_envs:
        is_m3 = live & (st["seg_mode"] == 3)
        st["seg_base_dpsi"][is_m3] += d_base[is_m3].abs()
        st["seg_claw_dpsi"][is_m3] += d_claw[is_m3].abs()
    # instantaneous settle criterion (time recorded at first satisfaction)
    pending_settle = live & (st["seg_settle_t"] < 0.0)
    if bool(pending_settle.any()):
        ids = pending_settle.nonzero().flatten()
        md = st["seg_mode"][ids]
        om = st["seg_cmd"][ids]
        vx = st["seg_vx"][ids]
        tol3 = torch.clamp(M3_YAW_REL * om.abs(), min=MODE_TOL_ABS)
        cond_m1 = (md == 1) & _in_speed_band(v_c[ids], vx, M1_SPEED_LOW_REL)
        cond_m2 = (md == 2) & _in_speed_band(v_c[ids], vx, M2_SPEED_LOW_REL)
        cond_m3 = (md == 3) & ((r[ids] - om).abs() <= tol3) & (speed[ids] < 0.05)
        cond_m4 = (md == 4) & (speed[ids] < M4_SPEED_MAX)
        newly = torch.zeros_like(pending_settle)
        newly[ids] = cond_m1 | cond_m2 | cond_m3 | cond_m4
        if bool(newly.any()):
            st["seg_settle_t"][newly] = timer[newly]
    # pan-at-limit watch (V1): EMA over M3-population steps
    rot_now = (term.rotate_raw > 0.5) if has_mode else torch.zeros_like(speed, dtype=torch.bool)
    if bool(rot_now.any()):
        asset = env.scene["robot"]
        pan = asset.data.joint_pos.torch[:, 0]
        lim = asset.data.soft_joint_pos_limits.torch[0, 0]
        at_lim = ((pan - lim[1]).abs() <= PAN_LIMIT_MARGIN) | ((pan - lim[0]).abs() <= PAN_LIMIT_MARGIN)
        frac = float(at_lim[rot_now].to(torch.float32).mean())
        m["pan_at_limit_fraction"] = frac if "pan_at_limit_fraction" not in m else m[
            "pan_at_limit_fraction"
        ] + alpha * (frac - m["pan_at_limit_fraction"])


def _begin_segment(env: ManagerBasedEnv, st: dict, ids: torch.Tensor, cmd: torch.Tensor, term) -> None:
    """(Re)initialize the segment accumulators for ``ids`` after a resample/reset."""
    st["seg_cmd"][ids] = cmd[ids]
    st["seg_rsum"][ids] = 0.0
    st["seg_rn"][ids] = 0.0
    st["seg_scored"][ids] = False
    st["seg_vsum"][ids] = 0.0
    st["seg_unmasked_t"][ids] = 0.0
    st["seg_speed_int"][ids] = 0.0
    st["seg_t"][ids] = 0.0
    st["seg_rest_ok"][ids] = 0.0
    st["seg_rest_n"][ids] = 0.0
    st["seg_base_dpsi"][ids] = 0.0
    st["seg_claw_dpsi"][ids] = 0.0
    st["seg_settle_t"][ids] = -1.0
    if term is not None and hasattr(term, "mode"):
        st["seg_prev_mode"][ids] = st["seg_mode"][ids]
        st["seg_mode"][ids] = term.mode[ids]
        st["seg_vx"][ids] = term.vx[ids]
        pos = env.scene["robot"].data.root_pos_w.torch[:, :2]
        st["seg_xy0"][ids] = pos[ids]


def _score_segments(env: ManagerBasedEnv, st: dict, env_ids: torch.Tensor, failed: torch.Tensor | None) -> None:
    """Fold the segments of ``env_ids`` (that have not been scored yet) into the EMAs."""
    if env_ids.dtype == torch.bool:  # callers pass either a boolean mask or an index tensor
        env_ids = env_ids.nonzero().flatten()
    ids_all = env_ids
    # phase-2 steer_success (unchanged eligibility)
    eligible = (
        ~st["seg_scored"][ids_all]
        & (st["seg_cmd"][ids_all].abs() > SEG_MIN_CMD)
        & (st["seg_rn"][ids_all] * env.step_dt >= SEG_MIN_UNMASKED_S)
    )
    if bool(eligible.any()):
        ids = ids_all[eligible]
        mean_r = st["seg_rsum"][ids] / st["seg_rn"][ids].clamp(min=1.0)
        cmd = st["seg_cmd"][ids]
        tol = torch.clamp(STEER_TOL_REL * cmd.abs(), min=STEER_TOL_ABS)
        ok = (mean_r - cmd).abs() <= tol
        if failed is not None:
            ok = ok & ~failed[ids]
        _ema(env, "steer_success", float(ids.numel()), float(ok.sum()) / float(ids.numel()))
    # phase-3 per-mode scoring (B.8)
    mode = st["seg_mode"][ids_all]
    has_mode = bool((mode > 0).any())
    if has_mode:
        unmasked_ok = st["seg_unmasked_t"][ids_all] >= SEG_MIN_UNMASKED_S
        live = ~st["seg_scored"][ids_all] & unmasked_ok & (mode > 0)
        failed_ids = failed[ids_all] if failed is not None else torch.zeros_like(live)
        m1 = live & (mode == 1)
        m2 = live & (mode == 2)
        m3 = live & (mode == 3)
        m4 = live & (mode == 4)
        for mask, key in ((m1, "success/M1"), (m2, "success/M2"), (m3, "success/M3"), (m4, "success/M4")):
            ids = ids_all[mask]
            if ids.numel() == 0:
                continue
            mean_r = st["seg_rsum"][ids] / st["seg_rn"][ids].clamp(min=1.0)
            om = st["seg_cmd"][ids]
            vx = st["seg_vx"][ids]
            v_mean = st["seg_vsum"][ids] / st["seg_unmasked_t"][ids].clamp(min=1e-6)
            speed_mean = st["seg_speed_int"][ids] / st["seg_t"][ids].clamp(min=1e-6)
            tol = torch.clamp(M2_YAW_REL * om.abs(), min=MODE_TOL_ABS)
            tol3 = torch.clamp(M3_YAW_REL * om.abs(), min=MODE_TOL_ABS)
            if key == "success/M1":
                # |mean r| limit is cfg-overridable (Hydra env.m1_r_max) and the LIVE
                # population mean is logged as m1_abs_mean_r: owner rule = calibrate the
                # limit from measured +20 %, not from the spec's 0.05 guess
                m1_r_max = float(getattr(env.cfg, "m1_r_max", 0.05))
                _ema(env, "m1_abs_mean_r", float(ids.numel()), float(mean_r.abs().mean()))
                ok = _in_speed_band(v_mean, vx, M1_SPEED_LOW_REL) & (mean_r.abs() < m1_r_max)
            elif key == "success/M2":
                ok = _in_speed_band(v_mean, vx, M2_SPEED_LOW_REL) & ((mean_r - om).abs() <= tol)
            elif key == "success/M3":
                ok = ((mean_r - om).abs() <= tol3) & (speed_mean < M3_SPEED_MAX)
            else:
                rest_frac = st["seg_rest_ok"][ids] / st["seg_rest_n"][ids].clamp(min=1.0)
                _ema(env, "m4_rest_frac", float(ids.numel()), float(rest_frac.mean()))
                ok = (speed_mean < M4_SPEED_MAX) & (rest_frac >= M4_REST_FRAC) & (
                    st["seg_rest_n"][ids] >= 50.0  # at least 1 s of graded post-mask window
                )
            if failed is not None:
                ok = ok & ~failed[ids]
            _ema(env, key, float(ids.numel()), float(ok.sum()) / float(ids.numel()))
        # M3 turn accounting + M4 creep + settle table (scored segments of any mode)
        scored = live
        ids = ids_all[scored]
        if ids.numel() > 0:
            md = st["seg_mode"][ids]
            m3s = ids[(md == 3) & (st["seg_claw_dpsi"][ids] > 0.15)]
            if m3s.numel() > 0:
                ratio = st["seg_base_dpsi"][m3s] / st["seg_claw_dpsi"][m3s].clamp(min=1e-6)
                _ema(env, "base_turn_fraction", float(m3s.numel()), float(ratio.mean()))
            m4s = ids[md == 4]
            if m4s.numel() > 0:
                pos = env.scene["robot"].data.root_pos_w.torch[:, :2]
                creep = (pos[m4s] - st["seg_xy0"][m4s]).norm(dim=1)
                _ema(env, "m4_creep_m", float(m4s.numel()), float(creep.mean()))
            settled = ids[st["seg_settle_t"][ids] >= 0.0]
            if settled.numel() > 0:
                _ema(env, "settle_mean_s", float(settled.numel()), float(st["seg_settle_t"][settled].mean()))
    st["seg_scored"][ids_all] = True


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
    out = {
        "crawl_success": float(m.get("crawl_success", 0.0)),
        "recovery_success": float(m.get("recovery_success", 0.0)),
        "steer_success": float(m.get("steer_success", 0.0)),
        "time_to_upright": float(m.get("time_to_upright", 0.0)),
        "back_frac": float(m.get("back_frac", 0.0)),
        "base_yaw_rate_abs": float(m.get("base_yaw_rate_abs", 0.0)),
        "r_abs": float(m.get("r_abs", 0.0)),
        "mean_v_c": float(m.get("mean_v_c", 0.0)),
        "success/M1": float(m.get("success/M1", 0.0)),
        "m1_abs_mean_r": float(m.get("m1_abs_mean_r", 0.0)),
        "m4_rest_frac": float(m.get("m4_rest_frac", 0.0)),
        "m4_ok_up": float(m.get("m4_ok_up", 0.0)),
        "m4_ok_z": float(m.get("m4_ok_z", 0.0)),
        "m4_ok_q": float(m.get("m4_ok_q", 0.0)),
        "m4_qerr_max": float(m.get("m4_qerr_max", 0.0)),
        "success/M2": float(m.get("success/M2", 0.0)),
        "success/M3": float(m.get("success/M3", 0.0)),
        "success/M4": float(m.get("success/M4", 0.0)),
        "base_turn_fraction": float(m.get("base_turn_fraction", 0.0)),
        "pan_at_limit_fraction": float(m.get("pan_at_limit_fraction", 0.0)),
        "m4_creep_m": float(m.get("m4_creep_m", 0.0)),
        "settle_mean_s": float(m.get("settle_mean_s", 0.0)),
    }
    term = None
    cm = getattr(env, "command_manager", None)
    if cm is not None and "yaw_rate_cmd" in cm.active_terms:
        term = cm.get_term("yaw_rate_cmd")
    if term is not None and hasattr(term, "transition_counts"):
        tc = term.transition_counts
        off = tc.clone()
        off.fill_diagonal_(0)
        total = float(off.sum())
        pairs = int((off > 0).sum())
        out["trans_pairs_seen"] = float(pairs)
        # min fraction over pairs observed at all (0.0 while the mix starves a pair)
        nz = off[off > 0]
        out["trans_min_frac"] = float((nz / max(total, 1.0)).min()) if nz.numel() else 0.0
    return out
