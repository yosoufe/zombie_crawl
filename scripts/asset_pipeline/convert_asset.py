# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the prepared SO-ARM100 URDF to USD with Isaac Lab's UrdfConverter.

Settings (see prompt.md section 3):
    - floating base            (fix_base=False)
    - position joint drives
    - convex-hull collision    (simplified, stable for thin links)
    - self-collision enabled

The PD gains passed here only seed the USD joint drives; the runtime gains come from
the actuator preset configured in ``zombie_crawl.assets.SoArm100Cfg``.

Usage:
    python scripts/asset_pipeline/convert_asset.py [--force]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert the prepared SO-ARM100 URDF to USD.")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so100_source import repo_root  # noqa: E402

REPO_ROOT = repo_root()
parser.add_argument(
    "--input",
    type=str,
    default=os.path.join(REPO_ROOT, "source", "zombie_crawl", "zombie_crawl", "assets", "urdf", "so_arm100_float.urdf"),
    help="Path to the prepared URDF.",
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(REPO_ROOT, "source", "zombie_crawl", "zombie_crawl", "assets", "usd", "so_arm100_float.usd"),
    help="Path to store the USD file.",
)
parser.add_argument(
    "--joint-stiffness", type=float, default=1.0, help="Seed PD stiffness for the USD joint drives [N*m/rad]."
)
parser.add_argument("--joint-damping", type=float, default=0.1, help="Seed PD damping [N*m*s/rad].")
parser.add_argument("--force", action="store_true", help="Force re-conversion even if the USD exists.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import UrdfConverter, UrdfConverterCfg

# obtain output directory
if os.path.dirname(args_cli.output) == "":
    output_dir = os.getcwd()
else:
    output_dir = os.path.dirname(args_cli.output)
os.makedirs(output_dir, exist_ok=True)

converter_cfg = UrdfConverterCfg(
    asset_path=os.path.abspath(args_cli.input),
    usd_dir=output_dir,
    usd_file_name=os.path.basename(args_cli.output),
    fix_base=False,  # floating base -- the arm crawls on the ground
    merge_fixed_joints=True,
    collision_from_visuals=False,
    collision_type="Convex Hull",  # simplified collision for speed & stability (3.0 field name/value)
    self_collision=True,
    force_usd_conversion=args_cli.force,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=args_cli.joint_stiffness, damping=args_cli.joint_damping
        ),
        target_type="position",
    ),
)
urdf_converter = UrdfConverter(converter_cfg)
print(f"Generated USD at: {urdf_converter.usd_path}")

# close sim app
simulation_app.close()
