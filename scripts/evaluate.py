# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained zombie-crawl policy (phase 2, spec §A.13).

Runs the DETERMINISTIC policy path (the -Play config: no observation corruption, no
sensor-noise model) in N parallel environments (default 100) for full episodes. Metrics
project on the CLAW axis and use the filtered claw heading rate ``r``
(``env.zombie_r_ema``). Episodes ended by a FAILURE termination count as failures and are
NEVER dropped from any statistic (early termination only shortens the measured episode).

Scenarios (--scenarios, comma separated):

    straight       yaw_rate_cmd = 0, upright (folded) starts.
                   Accept: mean claw speed within +-30 % of v_target; |mean r| < 0.05.
    steered        pinned +- commands per --steer-cmds (default 0.1,-0.1,0.2,-0.2),
                   upright starts (append the highest stage reached).
                   Accept: speed within +-35 % and |mean r - cmd| <= max(35 %, 0.03)
                   in >= 70 % of segments.
    fallen         on_side and face_down starts (pose_weights forced per pose).
                   Accept: recovery_success >= 0.9x R0; time_to_upright reported.
    cmd_change     +0.1 -> -0.1 mid-episode: r must settle within 3 s of the unmasked
                   period after the switch.

The friction sweep (--friction, default 0.5 0.8 1.2) re-runs every scenario per ground
static friction.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_policy.pt \\
        [--num_envs 100] [--v-target 0.10] [--scenarios straight,steered] \\
        [--steer-cmds 0.1,-0.1,0.2,-0.2,0.3] [--friction 0.5 --friction 0.8 --friction 1.2] \\
        [--json out.json]
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import os as _os
_EVAL_SAMPLE = bool(_os.environ.get("EVAL_SAMPLE"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel evaluation environments.")
parser.add_argument("--episode-seconds", type=float, default=20.0, help="Episode length [s].")
parser.add_argument(
    "--friction", type=float, action="append", default=None,
    help="Ground static friction values to sweep (spec §A.13: 0.5 / 0.8 / 1.2). "
    "Each use REPLACES the default sweep (pass once per value).",
)
parser.add_argument("--v-target", type=float, default=0.10, help="Forward velocity target [m/s].")
parser.add_argument(
    "--scenarios", type=str, default="straight,steered,fallen,cmd_change",
    help="Comma-separated: straight,steered,fallen,cmd_change",
)
parser.add_argument(
    "--steer-cmds", type=str, default="0.1,-0.1,0.2,-0.2",
    help="Comma-separated pinned yaw-rate commands [rad/s] for the steered scenario "
    "(append the highest stage reached, e.g. 0.3).",
)
parser.add_argument("--cmd-change-at", type=float, default=10.0, help="cmd_change: switch time [s].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--disturbance", action="store_true", default=False,
    help="Keep the training-style live pushes (mdp.disturb_all_bodies) during evaluation."
)
parser.add_argument("--json", type=str, default=None, help="Write the results as JSON.")
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import add_rsl_rl_args

add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import re

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import update_rsl_rl_cfg
from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl import mdp

TASK_ID = "Zombie-Crawl-SO100-Play-v0"

RECOVERY_UPRIGHT_THRESH = 0.8
RECOVERY_WINDOW_S = 10.0
YAW_MASK_S = 2.0  # post-command-change mask before r may be judged (matches training)
STEER_TOL_REL = 0.35
STEER_TOL_ABS = 0.03


def _steer_tol(cmd: float) -> float:
    return max(STEER_TOL_REL * abs(cmd), STEER_TOL_ABS)


def _run_episode(
    env,
    policy,
    steps_per_episode: int,
    cmd: float,
    switch_at_step: int | None,
    new_cmd: float | None,
    fallen_start: bool,
) -> dict:
    """One batch of parallel episodes; returns per-scenario statistics (never drops failures)."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    num_envs = unwrapped.num_envs
    device = unwrapped.device
    dt = unwrapped.step_dt
    term_mgr = unwrapped.termination_manager

    env.reset()
    term = unwrapped.command_manager.get_term("yaw_rate_cmd")
    term.pin(cmd)

    start_xy = robot.data.root_pos_w.torch[:, :2].clone()
    v_int = torch.zeros(num_envs, device=device)  # claw-axis velocity integral
    step_count = torch.zeros(num_envs, device=device)
    peak_z = torch.zeros(num_envs, device=device)
    upright_t = torch.full((num_envs,), -1.0, device=device)
    early = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # steering-window accumulators (post-mask r; for cmd_change only the post-switch window)
    r_sum = torch.zeros(num_envs, device=device)
    r_n = torch.zeros(num_envs, device=device)
    r_abs_sum = torch.zeros(num_envs, device=device)
    r_abs_n = torch.zeros(num_envs, device=device)
    # once an env's episode ends (time_out or failure) its accumulators FREEZE; re-measuring
    # the auto-reset sequel episode zeroed them after the final step and reported speed 0.0
    frozen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask_steps = int(round(YAW_MASK_S / dt))
    settle_steps = int(round(3.0 / dt))
    judged_cmd = cmd if new_cmd is None else new_cmd

    obs, _ = env.reset()
    with torch.inference_mode():
        for step in range(steps_per_episode):
            if switch_at_step is not None and step == switch_at_step and new_cmd is not None:
                term.pin(new_cmd)
                # restart the tracking mask for the new command (same semantics as a
                # resample in training: r is judged only after the mask has fully faded)
                unwrapped.zombie_time_since_resample.zero_()
            actions = policy(obs, stochastic_output=True) if _EVAL_SAMPLE else policy(obs)
            if os.environ.get("EVAL_DEBUG"):
                _s = robot.data.root_state_w.torch
                _p = obs["policy"] if isinstance(obs, dict) else obs
                if hasattr(_p, "values"):  # TensorDict
                    _p = _p["policy"] if "policy" in _p.keys() else next(iter(_p.values()))
                print(
                    f"[dbg step {step}] |act|={float(actions.abs().mean()):.3f} "
                    f"obs_mean={[round(float(x), 2) for x in _p.mean(0)]} "
                    f"z={float(_s[:, 2].mean()):.4f} v={float(_s[:, 7:10].abs().mean()):.4f}",
                    flush=True,
                )
            obs, _, dones, _ = env.step(actions)
            if os.environ.get("EVAL_DEBUG") and step < 6:
                _st = robot.data.root_state_w.torch
                _cu, _su = mdp.get_claw_dir(unwrapped)
                _vc = _st[:, 7] * _cu + _st[:, 8] * _su
                print(
                    f"[dbg post {step}] v_c={float(_vc.mean()):.4f} "
                    f"envclaw={float(mdp.metrics._claw_v(unwrapped).mean()):.4f} "
                    f"dones={int((dones > 0).sum())}",
                    flush=True,
                )
            if hasattr(policy, "reset"):
                policy.reset(dones)
            # r is maintained by the yaw_rate_tracking reward inside step()
            r = unwrapped.zombie_r_ema
            timer = unwrapped.zombie_time_since_resample
            state = robot.data.root_state_w.torch
            v = state[:, 7:10]
            cos_u, sin_u = mdp.get_claw_dir(unwrapped)
            v_c = v[:, 0] * cos_u + v[:, 1] * sin_u
            grav = robot.data.projected_gravity_b.torch

            active = ~frozen
            v_int += v_c * dt * active
            step_count += active.to(step_count.dtype)
            peak_z = torch.where(active, torch.maximum(peak_z, state[:, 2]), peak_z)
            t_now = (step + 1) * dt
            pending = active & (upright_t < 0.0) & (-grav[:, 2] >= RECOVERY_UPRIGHT_THRESH)
            upright_t = torch.where(pending, torch.full_like(upright_t, t_now), upright_t)
            r_abs_sum += r.abs() * active
            r_abs_n += active.to(r_abs_n.dtype)
            if switch_at_step is None:
                unmasked = (timer >= YAW_MASK_S) & active
            else:
                in_window = (step + 1 >= switch_at_step + mask_steps) and (
                    step + 1 < switch_at_step + mask_steps + settle_steps
                )
                unmasked = torch.full((num_envs,), bool(in_window), device=device) & active
            r_sum[unmasked] += r[unmasked]
            r_n[unmasked] += 1.0
            if (dones > 0).any():
                ended = (dones > 0).nonzero().flatten()
                failed_here = term_mgr.terminated[ended]
                early[ended[failed_here]] = True
                frozen[ended] = True

    nominal_t = steps_per_episode * dt
    ep_seconds = torch.clamp(step_count * dt, min=1e-6)
    mean_r = torch.where(r_n > 0, r_sum / r_n.clamp(min=1.0), torch.zeros_like(r_sum))
    v_int_per_nominal = v_int / nominal_t
    speed = v_int / ep_seconds

    if os.environ.get("EVAL_DEBUG"):
        print(
            f"[dbg final] mine_speed={float(speed.mean()):.4f} "
            f"env_mean_v_c={mdp.metrics.get_metric(unwrapped, 'mean_v_c'):.4f} "
            f"env_crawl_success={mdp.metrics.get_metric(unwrapped, 'crawl_success'):.4f}",
            flush=True,
        )
    report = {
        "command": cmd if new_cmd is None else f"{cmd}->{new_cmd}@{switch_at_step * dt:.1f}s",
        "num_envs": int(num_envs),
        "mean_claw_axis_speed_m_s": float(speed.mean()),
        "mean_abs_r_rad_s": float((r_abs_sum / r_abs_n.clamp(min=1.0)).mean()),
        "mean_r_judged_window_rad_s": float(mean_r.mean()),
        "fraction_time_within_50pct_vtarget": float(
            ((v_int_per_nominal >= 0.5 * args_cli.v_target) & (v_int_per_nominal <= 1.5 * args_cli.v_target))
            .double().mean()
        ),
        "crawl_success": float((v_int_per_nominal >= 0.6 * args_cli.v_target).double().mean()),
        "fraction_episodes_terminated_early": float(early.double().mean()),
        "mean_peak_base_height_m": float(peak_z.mean()),
        "success_1m_in_20s": float((v_int >= 0.6 * args_cli.v_target * nominal_t).double().mean()),
    }
    if fallen_start:
        recovered = (upright_t >= 0.0) & (upright_t <= RECOVERY_WINDOW_S)
        report["recovery_success"] = float(recovered.double().mean())
        report["time_to_upright_s"] = float(upright_t[recovered].mean()) if bool(recovered.any()) else None
    else:
        tol = _steer_tol(judged_cmd)
        # a segment needs enough judged samples to count; a failure termination inside
        # the segment counts as a failure
        enough = r_n >= max(2.0, (YAW_MASK_S + 1.0) / dt * 0.5)
        segment_ok = ((mean_r - judged_cmd).abs() <= tol) & ~early & enough
        report["segment_steer_success"] = float(segment_ok.double().mean())
    if switch_at_step is not None and new_cmd is not None:
        report["cmd_change_settle_window_s"] = [
            switch_at_step * dt + YAW_MASK_S,
            switch_at_step * dt + YAW_MASK_S + 3.0,
        ]
    return report


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", play_mode=True)
    if _os.environ.get("EVAL_TRAINCFG"):
        env_cfg, agent_cfg = resolve_task_config(
            TASK_ID.replace("-Play", ""), "rsl_rl_cfg_entry_point", play_mode=False
        )
        env_cfg.use_curriculum = False
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.record_video = False
    if _os.environ.get("EVAL_CORRUPT"):
        env_cfg.observations.policy.enable_corruption = True
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    if args_cli.checkpoint:
        if args_cli.checkpoint in CHECKPOINT_SELECTORS:
            resume_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="rsl_rl",
                task=TASK_ID.replace("-Play", ""),
                checkpoint_pattern=r"model_.*\.pt",
            )
        else:
            resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(os.path.abspath(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if args_cli.friction is None:
        args_cli.friction = [0.5, 0.8, 1.2]
    from zombie_crawl.tasks.manager_based.crawl import video_flags
    video_flags.play_disturbance_enabled = bool(args_cli.disturbance)
    video_flags.play_video_training_spec = bool(os.environ.get("EVAL_TRAIN_SPEC"))
    scenarios = [s.strip() for s in args_cli.scenarios.split(",") if s.strip()]
    steer_cmds = [float(x) for x in re.split(r"[,\s]+", args_cli.steer_cmds.strip()) if x]

    steps_per_episode = int(round(args_cli.episode_seconds / 0.02))

    results = {}
    for mu in args_cli.friction:
        for scen in scenarios:
            cmds = [None] if scen != "steered" else steer_cmds
            for cmd in cmds:
                env_cfg.scene.num_envs = args_cli.num_envs
                env_cfg.v_target = args_cli.v_target
                env_cfg.v_target_start = args_cli.v_target
                env_cfg.scene.ground.spawn.physics_material.static_friction = mu
                env_cfg.scene.ground.spawn.physics_material.dynamic_friction = 0.75 * mu
                env_cfg.seed = args_cli.seed
                env_cfg.episode_length_s = args_cli.episode_seconds
                # start-pose scenarios are set EXPLICITLY (P2-9: no implicit fallen fallback)
                pose_weights = (1.0, 0.0, 0.0)
                if scen == "fallen":
                    pose_weights = None  # set per pose below
                env_cfg.events.reset_start_pose.params["pose_weights"] = pose_weights or (1.0, 0.0, 0.0)

                poses = [None] if scen != "fallen" else ["on_side", "face_down"]
                for pose in poses:
                    if pose is not None:
                        env_cfg.events.reset_start_pose.params["pose_weights"] = {
                            "on_side": (0.0, 1.0, 0.0),
                            "face_down": (0.0, 0.0, 1.0),
                        }[pose]
                    env = gym.make(TASK_ID, cfg=env_cfg)
                    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
                    runner.load(resume_path)
                    policy = runner.get_inference_policy(device=env.unwrapped.device)

                    pinned = 0.0 if cmd is None else cmd
                    switch = None
                    new_cmd = None
                    if scen == "cmd_change":
                        switch = int(round(args_cli.cmd_change_at / 0.02))
                        new_cmd = -pinned
                    report = _run_episode(
                        env,
                        policy,
                        steps_per_episode,
                        cmd=pinned,
                        switch_at_step=switch,
                        new_cmd=new_cmd,
                        fallen_start=(pose is not None),
                    )
                    report["scenario"] = scen
                    report["pose"] = pose or "folded"
                    report["friction"] = mu
                    report["v_target"] = args_cli.v_target
                    key = f"friction_{mu:.2f}/{scen}"
                    if pose is not None:
                        key += f"/{pose}"
                    elif cmd is not None:
                        key += f"/cmd_{cmd:+.2f}"
                    results[key] = report
                    print(f"\n[INFO] Evaluation @ friction {mu:.2f} :: {key}")
                    print_dict(report, nesting=4)
                    env.close()

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Wrote results to {args_cli.json}")

    print("\n" + "=" * 70)
    print(f"Phase-2 acceptance summary (v_target = {args_cli.v_target:.2f} m/s):")
    ref = results.get(f"friction_{args_cli.friction[0]:.2f}/straight")
    if ref:
        ok = abs(ref["mean_claw_axis_speed_m_s"] - args_cli.v_target) <= 0.3 * args_cli.v_target
        print(f"  straight: claw speed within +/-30 % of v_target : {ok}")
        print(f"  straight: |mean r| < 0.05 rad/s                  : {ref['mean_abs_r_rad_s'] < 0.05}")
        print(f"  straight: no early terminations                  : {ref['fraction_episodes_terminated_early'] == 0.0}")
    for key, rep in results.items():
        if "/steered/cmd_" in key and "segment_steer_success" in rep:
            cmd_s = key.rsplit("cmd_", 1)[-1]
            speed_ok = abs(rep["mean_claw_axis_speed_m_s"] - args_cli.v_target) <= 0.35 * args_cli.v_target
            print(
                f"  steered cmd={cmd_s}: speed +-35 %: {speed_ok} | segment steer_success "
                f"{rep['segment_steer_success']:.2f} (need >= 0.70)"
            )
    for key, rep in results.items():
        if "recovery_success" in rep:
            print(
                f"  {key}: recovery_success {rep['recovery_success']:.2f} "
                f"time_to_upright {rep['time_to_upright_s']}"
            )
    print("=" * 70)


main()
simulation_app.close()
