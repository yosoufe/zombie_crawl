# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the SO-ARM100 URDF for floating-base Isaac Lab conversion.

Fixes applied to the stock ``so100.urdf`` (see docs/so_arm100_urdf_analysis.md):

1. **Base link mass/inertia** -- the stock URDF uses a 1.0 kg placeholder with an
   "approximate" isotropic inertia. The physical base is a printed mount; we set the
   mass to a documented estimate (default 0.15 kg) and derive the inertia from a
   hollow-box approximation of the ``Base.stl`` bounding box.
2. **Gripper/jaw collision geometry** -- the stock URDF has *no* collision on the
   ``gripper`` and ``jaw`` links. We add convex-collision entries reusing the existing
   ``Fixed_Jaw.stl`` / ``Moving_Jaw.stl`` meshes so the (actuated) gripper can interact
   with the ground.

The stock URDF is read from the SO-ARM100 upstream clone (``SO_ARM100_DIR`` env var,
else ``<repo>/../third_party/SO-ARM100``; see so100_source.py).

Usage:
    python scripts/asset_pipeline/prepare_urdf.py [--base-mass 0.15] [--base-fill 0.5]
"""

import argparse
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so100_source import repo_root, require_so100, so100_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = repo_root()
SO100 = so100_dir()
DEFAULT_INPUT = os.path.join(SO100, "Simulation", "SO100", "so100.urdf")
# the URDF references meshes as "assets/*.stl" relative to the URDF's own directory
DEFAULT_MESH_DIR = os.path.join(SO100, "Simulation", "SO100")
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "source", "zombie_crawl", "zombie_crawl", "assets", "urdf", "so_arm100_float.urdf"
)

# collision mesh to add per link (mesh file relative to the mesh dir, origin = identity)
COLLISION_ADDITIONS = {
    "gripper": "assets/Fixed_Jaw.stl",
    "jaw": "assets/Moving_Jaw.stl",
}


def stl_bbox(path: str):
    with open(path, "rb") as f:
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
        mn = [1e9, 1e9, 1e9]
        mx = [-1e9, -1e9, -1e9]
        for _ in range(n):
            d = struct.unpack("<12f", f.read(48))
            f.read(2)
            for i in range(3):
                x, y, z = d[3 + 3 * i : 6 + 3 * i]
                for k, v in enumerate((x, y, z)):
                    mn[k] = min(mn[k], v)
                    mx[k] = max(mx[k], v)
    return mn, mx


def box_inertia(mass: float, sx: float, sy: float, sz: float, fill: float):
    return (
        fill * mass * (sy**2 + sz**2) / 12.0,
        fill * mass * (sx**2 + sz**2) / 12.0,
        fill * mass * (sx**2 + sy**2) / 12.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input URDF path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output URDF path")
    parser.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR, help="Directory containing the STL meshes")
    parser.add_argument("--base-mass", type=float, default=0.15, help="Mass of the base link [kg]")
    parser.add_argument(
        "--base-fill", type=float, default=0.5, help="Hollow-fill factor of the base box inertia (0-1]"
    )
    args = parser.parse_args()

    input_path = os.path.normpath(args.input)
    output_path = os.path.normpath(args.output)
    if not os.path.isfile(input_path):
        require_so100()  # prints clone instructions and exits
        raise SystemExit(f"[prepare_urdf] input URDF not found: {input_path}")
    tree = ET.parse(input_path)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}

    # 1) base link: replace mass/inertia with the documented estimate
    base = links["base"]
    base_stl = os.path.join(args.mesh_dir, "assets", "Base.stl")
    mn, mx = stl_bbox(base_stl)
    sx, sy, sz = (mx[i] - mn[i] for i in range(3))
    com = tuple((mn[i] + mx[i]) / 2.0 for i in range(3))
    ixx, iyy, izz = box_inertia(args.base_mass, sx, sy, sz, args.base_fill)

    mass_el = base.find("inertial/mass")
    mass_el.set("value", f"{args.base_mass:.6f}")
    origin_el = base.find("inertial/origin")
    origin_el.set("xyz", f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    inertia_el = base.find("inertial/inertia")
    inertia_el.set("ixx", f"{ixx:.8f}")
    inertia_el.set("iyy", f"{iyy:.8f}")
    inertia_el.set("izz", f"{izz:.8f}")
    print(
        f"[prepare_urdf] base: mass={args.base_mass} kg, com={tuple(round(c, 4) for c in com)}, "
        f"bbox=({sx:.4f}, {sy:.4f}, {sz:.4f}) m, I=({ixx:.2e}, {iyy:.2e}, {izz:.2e}) kg*m^2 (fill={args.base_fill})"
    )

    # 2) gripper/jaw: add collision geometry
    for link_name, mesh in COLLISION_ADDITIONS.items():
        link = links[link_name]
        if link.find("collision") is not None:
            print(f"[prepare_urdf] {link_name}: collision already present, skipping")
            continue
        mesh_path = os.path.join(args.mesh_dir, mesh)
        if not os.path.isfile(mesh_path):
            print(f"[prepare_urdf] WARNING: mesh not found: {mesh_path}")
            continue
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        mesh_el = ET.SubElement(geometry, "mesh")
        mesh_el.set("filename", mesh)
        print(f"[prepare_urdf] {link_name}: added collision from {mesh}")

    # fix mesh paths: point at an absolute directory so the converter finds the meshes
    # (the URDF keeps the same relative layout; the converter is run with the mesh dir
    #  as the asset root by copying the meshes next to the output URDF)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    # copy the meshes next to the prepared URDF so conversion is self-contained
    out_mesh_dir = os.path.join(os.path.dirname(output_path), "meshes")
    os.makedirs(out_mesh_dir, exist_ok=True)
    for el in root.iter("mesh"):
        fname = el.get("filename")
        src = os.path.join(args.mesh_dir, fname)
        dst = os.path.join(out_mesh_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        elif not os.path.isfile(src):
            print(f"[prepare_urdf] WARNING: mesh not found: {src}")
    # make the URDF reference the copied meshes
    tree2 = ET.parse(output_path)
    for el in tree2.getroot().iter("mesh"):
        el.set("filename", os.path.join("meshes", el.get("filename")))
    tree2.write(output_path, xml_declaration=True, encoding="utf-8")

    total_mass = sum(
        float(l.find("inertial/mass").get("value")) for l in root.findall("link")
    )
    print(f"[prepare_urdf] wrote {output_path}")
    print(f"[prepare_urdf] total robot mass: {total_mass:.3f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
