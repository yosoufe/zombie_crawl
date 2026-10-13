# Render the spawn start pose(s) two ways:
#   frozen  -- gravity zeroed right after env.reset(), before the first physics step: the frame
#              shows the EXACTLY authored joint angles (no falling, no sag).
#   settled -- gravity on, pose held as the action for a few seconds: what the robot looks
#              like once the episode has settled (soft-PD sag under real gravity).
#
# The sensor video camera (source="sensor:video_camera") is the only camera that renders live
# physics in headless mode (the KitViewport render_rgb_array path returns frozen frames).
#
# Usage:  source scripts/env.sh && python scripts/render_spawn_poses.py [calib]
#         calib = measure each pose's contact-resting spawn z (gravity settle from a deep spawn)
# Output: docs/start_poses/{<pose>.png, <pose>_settled.png} for every pose in START_POSES
#         intermediate single-frame clips under logs/render_spawn_poses/

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "zombie_crawl"))
os.environ["HEADLESS"] = "1"
CALIB = "calib" in sys.argv[1:]
POSE_ARGS = tuple(a for a in sys.argv[1:] if a != "calib") or ("folded",)
sys.argv = [sys.argv[0]]

import numpy as np
import torch

from isaaclab.app import AppLauncher

al = AppLauncher(enable_cameras=True)
app = al.app

import omni.replicator.core  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from zombie_crawl.tasks.manager_based.crawl import video_flags

video_flags.sensor_video_enabled = True

from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg, START_POSES

WORK = REPO / "logs" / "render_spawn_poses"
OUT = REPO / "docs" / "start_poses"
MODE_STEPS = {"frozen": 30, "settled": 150}
EYES = (1.0, -1.0, 0.26)
TARGET = (0.05, 0.0, 0.11)


def _one_hot(name: str) -> tuple[float, ...]:
    return tuple(1.0 if n == name else 0.0 for n in START_POSES)


def _hold_action(env: ManagerBasedRLEnv, asset, name: str) -> torch.Tensor:
    q = torch.zeros(1, asset.num_joints, device=env.device)
    for jn, jv in START_POSES[name]["joint_pos"].items():
        q[0, asset.joint_names.index(jn)] = jv
    lim = asset.data.soft_joint_pos_limits[0]
    lo, hi = lim[:, 0], lim[:, 1]
    mid = 0.5 * (lo + hi)
    half = torch.clamp(0.5 * (hi - lo), min=1e-6)
    return torch.clamp((q[0] - mid) / half, -1.0, 1.0).unsqueeze(0)


def _make_cfg(name: str, output_dir: str | None):
    cfg = CrawlEnvCfg()
    cfg.scene.num_envs = 1
    p = cfg.events.reset_start_pose.params
    p["pose_weights"] = _one_hot(name)
    p["joint_noise"] = 0.0
    p["xy_offset"] = 0.0
    p["yaw_range"] = (0.0, 0.0)
    if output_dir is None:
        cfg.video_recorders = []
    return cfg


def render_pose(name: str, mode: str) -> Path:
    clip_dir = WORK / mode / name
    if clip_dir.exists():
        shutil.rmtree(clip_dir)  # recorder continues numbering over old clips; start clean
    cfg = _make_cfg(name, str(clip_dir))
    r_v2w = create_rotation_matrix_from_view(
        torch.tensor(EYES).unsqueeze(0), torch.tensor(TARGET).unsqueeze(0), up_axis="Z"
    )
    quat = quat_from_matrix(r_v2w)[0]
    cfg.scene.video_camera.offset = CameraCfg.OffsetCfg(pos=EYES, rot=quat, convention="opengl")
    cfg.video_recorders = [
        VideoRecorderCfg(
            source="sensor:video_camera",
            output_dir=str(clip_dir),
            video_length=40,
            video_interval=1,
            step_offset=0,
            fps=20,
            output_filename_prefix="z",
            keep_last_n_clips=None,
        )
    ]
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    asset = env.scene["robot"]
    env.reset()
    if mode == "frozen":
        # zero gravity before the first physics step: authored pose stays exact, nothing falls
        from pxr import UsdPhysics

        UsdPhysics.Scene(
            env.sim.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path)
        ).GetGravityMagnitudeAttr().Set(0.0)
    hold = _hold_action(env, asset, name)
    for _ in range(MODE_STEPS[mode]):
        env.step(hold)
    z = asset.data.root_pos_w[0].cpu().numpy()
    print(f"{mode} {name}: base z after {MODE_STEPS[mode]} steps: {z}", flush=True)
    clip = clip_dir / f"z_{MODE_STEPS[mode] - 1:04d}.mp4"
    env.close()
    return clip


def calib_pose(name: str) -> None:
    """Measure the contact-resting base z: spawn deep (z=-0.03) WITH gravity (the only thing
    that damps the ejection into a real rest), hold the authored pose, average base z at the end."""
    cfg = _make_cfg(name, None)
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    asset = env.scene["robot"]
    env.reset()
    q = torch.zeros(1, asset.num_joints, device=env.device)
    for jn, jv in START_POSES[name]["joint_pos"].items():
        q[0, asset.joint_names.index(jn)] = jv
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, 2] = -0.03
    pose[0, 6] = 1.0
    env_ids = torch.arange(1, device=env.device)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(1, 6, device=env.device), env_ids=env_ids
    )
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
    hold = _hold_action(env, asset, name)
    zs = []
    for k in range(400):
        env.step(hold)
        if k >= 300:
            zs.append(asset.data.root_pos_w[0, 2].item())
    print(
        f"calib {name}: resting z = {np.mean(zs):.4f} (spread {np.max(zs) - np.min(zs):.4f})",
        flush=True,
    )
    env.close()


def last_frame(clip: Path, png: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-frames:v", "1", str(png)],
        check=True,
    )


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if CALIB:
        for name in POSE_ARGS:
            calib_pose(name)
        app.close()
        return
    for mode, suffix in (("frozen", ""), ("settled", "_settled")):
        for name in POSE_ARGS:
            clip = render_pose(name, mode)
            last_frame(clip, OUT / f"{name}{suffix}.png")
    app.close()


if __name__ == "__main__":
    main()
