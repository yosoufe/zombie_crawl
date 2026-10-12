# Render the robot in its URDF zero position (all joints at 0; the URDF defines no <default>),
# frozen (zero-g, exact authored angles) and settled (gravity on, PD holding zeros).
# Usage: source scripts/env.sh && python scripts/render_urdf_zero.py
# Output: docs/start_poses/urdf_zero.png, docs/start_poses/urdf_zero_settled.png

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "zombie_crawl"))
os.environ["HEADLESS"] = "1"
sys.argv = ["render_urdf_zero"]

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

from zombie_crawl.tasks.manager_based.crawl.crawl_env_cfg import CrawlEnvCfg

WORK = REPO / "logs" / "render_spawn_poses"
OUT = REPO / "docs" / "start_poses"
EYES = (1.0, -1.0, 0.26)
TARGET = (0.05, 0.0, 0.11)
MODE_STEPS = {"frozen": 30, "settled": 150}


def render(mode: str) -> Path:
    clip_dir = WORK / mode / "urdf_zero"
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    cfg = CrawlEnvCfg()
    cfg.scene.num_envs = 1
    p = cfg.events.reset_start_pose.params
    p["joint_noise"] = 0.0
    p["xy_offset"] = 0.0
    p["yaw_range"] = (0.0, 0.0)
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
    q = torch.zeros(1, asset.num_joints, device=env.device)  # URDF zero position
    asset.write_joint_position_to_sim_index(
        position=q, env_ids=torch.arange(1, device=env.device)
    )
    if mode == "frozen":
        from pxr import UsdPhysics

        UsdPhysics.Scene(
            env.sim.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path)
        ).GetGravityMagnitudeAttr().Set(0.0)
    lim = asset.data.soft_joint_pos_limits[0]
    lo, hi = lim[:, 0], lim[:, 1]
    mid = 0.5 * (lo + hi)
    half = torch.clamp(0.5 * (hi - lo), min=1e-6)
    hold = torch.clamp((q[0] - mid) / half, -1.0, 1.0).unsqueeze(0)
    for _ in range(MODE_STEPS[mode]):
        env.step(hold)
    env.close()
    return clip_dir / f"z_{MODE_STEPS[mode] - 1:04d}.mp4"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, suffix in (("frozen", ""), ("settled", "_settled")):
        clip = render(mode)
        png = OUT / f"urdf_zero{suffix}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-frames:v", "1", str(png)],
            check=True,
        )
        print("wrote", png)
    app.close()


if __name__ == "__main__":
    main()
