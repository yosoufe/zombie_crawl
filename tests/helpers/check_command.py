# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper: AGENTS.md pre-launch smoke gate for the phase-3 mode-mix command.

Checks (all against the REAL env, on GPU):
1. motion-only mix (stage-0/1 dial): zero-command fraction is EXACTLY 0.0 (AGENTS.md),
   M1 is deadband-quiet on yaw with vx == v_target, M2 draws |omega| in [0.05, cap]
   with both signs;
2. full mix: all four modes appear, M3 respects the m3_omega_min + rotate_segment_s
   dials, M4 is fully inside both deadbands;
3. yaw_rate_max == 0 closes every |omega| below the yaw deadband (stage lock);
4. ``pin()``/``unpin()`` (phase-2 compat) and ``pin_mode()`` freeze the stream;
5. transition coverage: all 12 ordered pairs appear over many draws (B.7).

Run: ``HEADLESS=1 ../.venv/bin/python tests/helpers/check_command.py --num_envs 64``
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--draws", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--headless", action="store_true", default=False,
    help="Run headless (no GUI window; the Isaac Lab 3.0 default).",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg

cfg = CrawlEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.use_curriculum = False  # keep the dials out of the way; we drive the command directly
cfg.seed = args_cli.seed
env = gym.make("Zombie-Crawl-SO100-v0", cfg=cfg)
env.reset()

unwrapped = env.unwrapped
term = unwrapped.command_manager.get_term("yaw_rate_cmd")
n = unwrapped.num_envs
draws = args_cli.draws
cap = 0.3

failures = []

# 1) motion-only mix: zero fraction EXACTLY 0, M1 quiet, M2 in-band both signs
unwrapped.extras["yaw_rate_max"] = cap
unwrapped.extras["mode_mix"] = (0.5, 0.5, 0.0, 0.0)
total, zeros, m1, m2, pos, neg, max_abs = 0, 0, 0, 0, 0, 0, 0.0
for _ in range(draws):
    term._resample(slice(None))
    total += n
    zeros += int((term.mode == 0).sum())
    m1 += int((term.mode == 1).sum())
    m2 += int((term.mode == 2).sum())
    pos += int(((term.mode == 2) & (term.yaw_rate > 0.05)).sum())
    neg += int(((term.mode == 2) & (term.yaw_rate < -0.05)).sum())
    max_abs = max(max_abs, float(term.yaw_rate.abs().max()))
if m1 + m2 != total:
    failures.append(f"motion-only mix produced {total - m1 - m2} non-forward samples")
if zeros:
    failures.append(f"zero-command fraction {zeros / total:.4f} != 0.0 at the motion stage")
if not 0.35 <= m1 / total <= 0.65:
    failures.append(f"M1 fraction {m1 / total:.3f} outside [0.35, 0.65]")
if max_abs > cap + 1e-6:
    failures.append(f"|omega| max {max_abs} exceeds dial {cap}")
if pos / m2 < 0.35 or neg / m2 < 0.35:
    failures.append(f"M2 signs unbalanced: pos {pos / m2:.3f} neg {neg / m2:.3f}")
v_fwd = float(unwrapped.v_target.mean()) if hasattr(unwrapped, "v_target") else float(cfg.v_target)
term._resample(slice(None))
if not bool((term.vx[term.mode == 1] >= v_fwd - 1e-6).all()):
    failures.append("M1 vx below v_fwd")
if not bool((term.yaw_rate[term.mode == 1].abs() <= 0.05).all()):
    failures.append("M1 yaw outside the deadband")
if not bool((term.yaw_rate[term.mode == 2].abs() >= 0.05 - 1e-6).all()):
    failures.append("M2 |omega| below 0.05")

# 2) full mix: all four modes, M3 dial-respecting, M4 fully deadbanded
unwrapped.extras["mode_mix"] = (0.30, 0.25, 0.25, 0.20)
unwrapped.extras["m3_omega_min"] = 0.20
unwrapped.extras["rotate_segment_s"] = (30.0, 40.0)
seen = torch.zeros(5, dtype=torch.long, device=term.mode.device)
for _ in range(draws):
    term._resample(slice(None))
    seen += torch.bincount(term.mode, minlength=5)
if (seen[1:] == 0).any():
    failures.append(f"full mix never drew some mode: {seen.tolist()}")
if not bool((term.vx[term.mode == 3].abs() <= 0.02).all()):
    failures.append("M3 vx outside the vx deadband")
if not bool((term.yaw_rate[term.mode == 3].abs() >= 0.20 - 1e-6).all()):
    failures.append("M3 |omega| below m3_omega_min dial")
if not bool(((term.time_left[term.mode == 3] >= 30.0) & (term.time_left[term.mode == 3] <= 40.0)).all()):
    failures.append("M3 segment duration outside the rotate_segment_s dial")
if not bool((term.vx[term.mode == 4].abs() <= 0.02).all()):
    failures.append("M4 vx outside the vx deadband")
if not bool((term.yaw_rate[term.mode == 4].abs() <= 0.05).all()):
    failures.append("M4 yaw outside the yaw deadband")

# 3) yaw dial closed: no |omega| above the deadband anywhere
unwrapped.extras["yaw_rate_max"] = 0.0
term._resample(slice(None))
if not bool((term.yaw_rate.abs() <= 0.05).all()):
    failures.append("omega above deadband while yaw_rate_max == 0")
unwrapped.extras["yaw_rate_max"] = cap

# 4) pin/unpin (phase-2 compat) and pin_mode
term.pin(-0.1)
for _ in range(50):
    term._resample(slice(None))
    term.compute(0.02)
if not bool((term.yaw_rate == -0.1).all()) or not bool((term.vx > 0).all()):
    failures.append("pin() did not freeze a forward+turn command")
term.unpin()
term.pin_mode(4, 0.0, 0.0)
for _ in range(50):
    term._resample(slice(None))
    term.compute(0.02)
if not bool((term.zero_cmd_raw == 1).all()):
    failures.append("pin_mode(4) did not present a zero command")
term.unpin()
term._resample(slice(None))

# 5) transition coverage: all 12 ordered pairs over many draws
term.transition_counts.zero_()
for _ in range(2000):
    term._resample(slice(None))
off_diag = ~torch.eye(4, dtype=torch.bool, device=term.transition_counts.device)
tc = term.transition_counts.to(torch.float32)
if bool((tc[off_diag] == 0).any()):
    failures.append(f"missing ordered mode pairs: {tc[off_diag].tolist()}")
else:
    frac = tc / tc[off_diag].sum()
    if float(frac[off_diag].min()) < 0.02:
        failures.append(f"min pair fraction {float(frac[off_diag].min()):.4f} < 2%")

env.close()

if failures:
    print("[helper] FAILED:", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    simulation_app.close()
    sys.exit(1)
print(
    f"[helper] OK (motion zero_frac=0, m1={m1 / total:.3f}, max|omega|={max_abs:.3f}, "
    f"trans_min_frac={float(frac[off_diag].min()):.4f}, total={total})",
    flush=True,
)
simulation_app.close()
