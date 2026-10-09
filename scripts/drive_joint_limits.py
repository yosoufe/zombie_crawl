# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive each SO-ARM100 joint through its full range (max -> min) and report the achieved range of motion.

Loads one robot (Kit GUI), settles it in a start pose, then for each joint in
turn commands it to its SOFT upper limit, lets the PD actuators drive it there,
records the achieved position, commands the SOFT lower limit, records again,
and returns to the default pose before moving to the next joint.

The training action term (``EMAJointPositionToLimitsAction``) rescales raw
actions in [-1, 1] onto each joint's SOFT position limits (95% of the hard
range about the default pose) and clamps there, so the soft limits are the
outermost commandable targets. Joints are driven through the normal action
path (``env.step``) with ``raw = 2 * (target - soft_low) / (soft_high - soft_low) - 1``.


If a joint cannot be held at the limit (servo torque ceiling, gravity sag,
ground/self contact), the achieved position will fall short of the commanded
one -- that shortfall is the useful diagnostic.

Usage:
    python scripts/drive_joint_limits.py --viz kit [--preset servo] [--phase 3.0]
    python scripts/drive_joint_limits.py --no-idle   # headless: exit after the summary
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", type=str, default="servo", choices=["servo", "ideal"])
parser.add_argument("--pose", type=str, default="folded")
parser.add_argument("--phase", type=float, default=3.0, help="Seconds to hold at each limit.")
parser.add_argument("--settle", type=float, default=2.0, help="Seconds of zero-action settle at start.")
parser.add_argument("--no-idle", action="store_true", help="Exit after the summary instead of idling (headless use).")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

from isaaclab.utils.math import quat_from_euler_xyz

import zombie_crawl.tasks  # noqa: F401
from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg, START_POSES

env_cfg = CrawlEnvCfg()
env_cfg.num_envs = 1
env_cfg.env_spacing = 4.0
env_cfg.actuator_preset = args_cli.preset
env_cfg.use_curriculum = False
env_cfg.seed = args_cli.seed
# never let a timeout reset jump the robot mid-sweep
env_cfg.episode_length_s = 1.0e6
env_cfg.__post_init__()
# freeze friction at the ground default
env_cfg.events.randomize_ground_friction.params["friction_range"] = (1.0, 1.0)

env = gym.make("Zombie-Crawl-SO100-v0", cfg=env_cfg)
asset = env.unwrapped.scene["robot"]
device = env.unwrapped.device
step_dt = env.unwrapped.step_dt
zero = torch.zeros(1, asset.num_joints, device=device)


def write_start_pose() -> None:
    pose = START_POSES[args_cli.pose]
    env_ids = torch.tensor([0], device=device)
    rpy = torch.tensor(pose["rot"], device=device)
    positions = torch.tensor(pose["pos"], device=device).repeat(1, 1) + env.unwrapped.scene.env_origins[0].to(device)
    quats = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2]).repeat(1, 1)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)
    q_pose = torch.zeros(1, asset.num_joints, device=device)
    for name, value in pose["joint_pos"].items():
        q_pose[0, asset.joint_names.index(name)] = value
    asset.write_joint_position_to_sim_index(position=q_pose, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_pose), env_ids=env_ids)


def step_seconds(seconds: float, action: torch.Tensor) -> None:
    """Step the env for `seconds` with a constant action; restore the start pose if the episode terminates."""
    n = max(1, int(seconds / step_dt))
    for _ in range(n):
        obs, reward, terminated, truncated, info = env.step(action)
        state = asset.data.root_state_w[0]
        if not torch.isfinite(state).all() or terminated[0]:
            print("    [reset] episode terminated / NaN state during drive; rewriting start pose")
            write_start_pose()


def hold(target: torch.Tensor, seconds: float) -> torch.Tensor:
    """Drive joint i to `target` (absolute positions, other joints at default) for `seconds`; return achieved q."""
    # limits-relative raw action, matching EMAJointPositionToLimitsAction (crawl_env_cfg.py
    # CrawlActionsCfg): the term rescales raw in [-1, 1] onto the SOFT limits per joint
    soft = asset.data.soft_joint_pos_limits[0]  # (num_joints, 2) = [low, high]
    raw = (2.0 * (target - soft[:, 0]) / (soft[:, 1] - soft[:, 0]) - 1.0).clamp(-1.0, 1.0)
    step_seconds(seconds, raw.reshape(1, -1))
    return asset.data.joint_pos[0].clone()


def main() -> None:
    env.reset()
    write_start_pose()
    print(f"[drive] SO-ARM100 loaded: pose={args_cli.pose} preset={args_cli.preset} phase={args_cli.phase:.1f}s")
    print(f"[drive] settling {args_cli.settle:.1f}s with zero action (relaxing toward the soft-range midpoint)...")
    step_seconds(args_cli.settle, zero)

    names = asset.joint_names
    q0 = asset.data.default_joint_pos[0]
    hard = asset.data.joint_pos_limits[0]
    soft = asset.data.soft_joint_pos_limits[0]

    print("\n" + "=" * 100)
    print("Driving each joint max -> min (soft limits = outermost commandable via the action term).")
    print("raw action = 2*(target-soft_low)/(soft_high-soft_low)-1, no +/-1 clip (that clip is RL-only).")
    print("=" * 100)

    results = []
    for i, name in enumerate(names):
        up = asset.data.default_joint_pos[0].clone()
        up[i] = soft[i, 1]
        lo = asset.data.default_joint_pos[0].clone()
        lo[i] = soft[i, 0]
        print(f"\n[drive] {name}: -> soft UPPER {soft[i, 1]:+.4f} rad ({math.degrees(soft[i, 1]):.1f} deg), holding {args_cli.phase:.1f}s")
        q_hi = hold(up, args_cli.phase)
        print(f"[drive] {name}: -> soft LOWER {soft[i, 0]:+.4f} rad ({math.degrees(soft[i, 0]):.1f} deg), holding {args_cli.phase:.1f}s")
        q_lo = hold(lo, args_cli.phase)
        # relax back (zero action) before the next joint
        step_seconds(0.5, zero)

        hi, lo_ach = q_hi[i].item(), q_lo[i].item()
        rom = hi - lo_ach
        results.append((name, hard[i, 0], hard[i, 1], soft[i, 0], soft[i, 1], q0[i].item(), lo_ach, hi, rom))
        sag_hi = soft[i, 1] - hi
        sag_lo = lo_ach - soft[i, 0]
        print(
            f"    {name}: achieved [{lo_ach:+.4f}, {hi:+.4f}] rad  ROM={rom:.4f} rad ({math.degrees(rom):.1f} deg)"
            f"  | shortfall vs soft: upper {sag_hi:+.4f} rad, lower {sag_lo:+.4f} rad"
        )

    print("\n" + "=" * 100)
    print("SUMMARY -- achieved range of motion (preset: " + args_cli.preset + ")")
    print("=" * 100)
    header = f"{'joint':<16} {'soft lo':>8} {'soft hi':>8} {'hard lo':>8} {'hard hi':>8} {'ach lo':>8} {'ach hi':>8} {'ROM deg':>8} {'%soft':>6} {'%hard':>6}"
    print(header)
    print("-" * len(header))
    for name, hlo, hhi, slo, shi, q0v, lo_ach, hi, rom in results:
        print(
            f"{name:<16} {slo:8.3f} {shi:8.3f} {hlo:8.3f} {hhi:8.3f} {lo_ach:8.3f} {hi:8.3f} {math.degrees(rom):8.1f}"
            f" {100.0 * rom / (shi - slo):6.1f} {100.0 * rom / (hhi - hlo):6.1f}"
        )
    if args_cli.no_idle:
        print("\n[drive] done (--no-idle): exiting.")
        return
    print(
        "\n[drive] done. The robot now idles at its default pose -- orbit/zoom in the GUI,\n"
        "        or run again with --preset ideal to see the torque-limited joints un-stall.\n"
        "        Press Ctrl+C here (or close the window) to exit."
    )
    while True:
        obs, reward, terminated, truncated, info = env.step(zero)
        state = asset.data.root_state_w[0]
        if not torch.isfinite(state).all() or terminated[0]:
            write_start_pose()


try:
    main()
except KeyboardInterrupt:
    print("\n[drive] interrupted, closing")
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    env.close()
    simulation_app.close()
