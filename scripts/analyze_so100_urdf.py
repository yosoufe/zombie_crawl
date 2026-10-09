#!/usr/bin/env python3
"""Analyze SO-ARM100 (SO-100) URDF: joints, links, meshes, mass, reach, PhysX flags."""
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = "/home/yousof/robotics/experimental/third_party/SO-ARM100"
URDF = os.path.join(REPO, "Simulation/SO100/so100.urdf")
REL = "/home/yousof/robotics/experimental"


def rel(p):
    return os.path.relpath(p, REL)


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    robot_name = root.get("name")
    links = {}
    for l in root.findall("link"):
        name = l.get("name")
        info = {"name": name, "inertial": None, "visual": [], "collision": []}
        inf = l.find("inertial")
        if inf is not None:
            m = inf.find("mass")
            o = inf.find("origin")
            i = inf.find("inertia")
            info["inertial"] = {
                "mass": float(m.get("value")) if m is not None else None,
                "origin": o.get("xyz", "0 0 0").split() if o is not None else ["0", "0", "0"],
                "origin_rpy": o.get("rpy", "0 0 0").split() if o is not None else ["0", "0", "0"],
                "inertia": None if i is None else {k: float(i.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
            }
        for v in l.findall("visual"):
            g = v.find("geometry")
            mesh = g.find("mesh") if g is not None else None
            geo = mesh.get("filename") if mesh is not None else (g.find("box").get("size") if g is not None and g.find("box") is not None else g.find("cylinder").get("radius") if g is not None and g.find("cylinder") is not None else "unknown")
            if mesh is not None:
                geo = mesh.get("filename")
                sc = mesh.get("scale")
                if sc:
                    geo += " (scale %s)" % sc
            o = v.find("origin")
            info["visual"].append({"geometry": geo, "origin": o.get("xyz") if o is not None else "0 0 0"})
        for c in l.findall("collision"):
            g = c.find("geometry")
            mesh = g.find("mesh") if g is not None else None
            geo = mesh.get("filename") if mesh is not None else "non-mesh"
            o = c.find("origin")
            info["collision"].append({"geometry": geo, "origin": o.get("xyz") if o is not None else "0 0 0"})
        links[name] = info
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        jt = j.get("type")
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        o = j.find("origin")
        a = j.find("axis")
        lim = j.find("limit")
        info = {
            "name": name, "type": jt, "parent": p, "child": c,
            "origin_xyz": o.get("xyz", "0 0 0").split(), "origin_rpy": o.get("rpy", "0 0 0").split(),
            "axis": a.get("xyz", "0 0 0").split() if a is not None else None,
            "limit": None,
        }
        if lim is not None:
            info["limit"] = {
                "lower": float(lim.get("lower")), "upper": float(lim.get("upper")),
                "effort": float(lim.get("effort")), "velocity": float(lim.get("velocity")),
            }
        joints[name] = info
    return robot_name, links, joints


def stl_info(path):
    """Return (kind, bbox or None, ntri or None)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(5)
    is_ascii = head[:1] in (b"s", b"S") or (head[:4] in (b"solid",) )
    # binary files: 80-byte header + uint32 count + count*50 bytes
    ntri = None
    bbox = None
    binary_ok = False
    if size >= 84:
        with open(path, "rb") as f:
            f.seek(80)
            (count,) = struct.unpack("<I", f.read(4))
        if size == 84 + count * 50 and count < 10_000_000:
            binary_ok = True
            ntri = count
    kind = "binary STL" if binary_ok else "ASCII STL"
    if binary_ok:
        pts = []
        with open(path, "rb") as f:
            f.seek(84)
            for _ in range(ntri):
                data = f.read(50)
                vals = struct.unpack("<12f", data[:48])
                pts.extend(vals[3:12])
        arr = np.array(pts).reshape(-1, 3)
        bbox = (arr.min(axis=0), arr.max(axis=0))
    else:
        n = 0
        vmin = np.full(3, np.inf)
        vmax = np.full(3, -np.inf)
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("vertex"):
                    n += 1
                    _, x, y, z = line.split()
                    p = np.array([float(x), float(y), float(z)])
                    vmin = np.minimum(vmin, p)
                    vmax = np.maximum(vmax, p)
        ntri = max(0, n // 3)
        if n:
            bbox = (vmin, vmax)
    return kind, bbox, ntri, size


def fk_grid(joints, links):
    """Coarse grid FK: max distance from base origin to gripper & jaw frame origins."""
    order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    # build transforms in order; chain: base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper -> jaw
    chain = [
        (joints["shoulder_pan"], "base", "shoulder"),
        (joints["shoulder_lift"], "shoulder", "upper_arm"),
        (joints["elbow_flex"], "upper_arm", "lower_arm"),
        (joints["wrist_flex"], "lower_arm", "wrist"),
        (joints["wrist_roll"], "wrist", "gripper"),
        (joints["gripper"], "gripper", "jaw"),
    ]
    limits = {}
    for jn in order:
        lo, hi = joints[jn]["limit"]["lower"], joints[jn]["limit"]["upper"]
        limits[jn] = (lo, hi)
    # precompute base transforms (origin + rpy), joint rotation axis applied about child frame?
    # URDF joint model: child_frame = parent_frame * origin_transform * Rot(axis, q)
    base_T = []
    for j, _, _ in chain:
        x = j["origin_xyz"]
        rpy = [float(v) for v in j["origin_rpy"]]
        t = rpy_to_matrix(rpy)
        p = np.array([float(v) for v in x])
        base_T.append((t, p))
    axes = [np.array([float(v) for v in j["axis"]], dtype=float) for j, _, _ in chain]

    def rot_axis(a, ang):
        c, s = math.cos(ang), math.sin(ang)
        a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    def fk(angs):
        T = np.eye(4)
        pos = []
        for i in range(len(chain)):
            t, p = base_T[i]
            Rj = rot_axis(axes[i], angs[i])
            # child frame origin in parent frame:
            T = T @ np.block([[t @ Rj, t @ p], [np.zeros(3), 1]])
            pos.append(T[:3, 3].copy())
        return pos

    N = 10
    grids = [np.linspace(lo, hi, N) for lo, hi in limits.values()]
    G = np.array(np.meshgrid(*grids, indexing="ij")).reshape(6, -1).T  # (M, 6)
    M = G.shape[0]
    R_all = np.empty((M, 6, 3, 3))
    for i in range(6):
        c = np.cos(G[:, i])
        s = np.sin(G[:, i])
        a = axes[i]
        n = np.linalg.norm(a)
        a = a / n
        ax, ay, az = a
        K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
        K2 = K @ K
        R_all[:, i] = np.broadcast_to(np.eye(3), (M, 3, 3)) + s[:, None, None] * K[None] + (1 - c)[:, None, None] * K2[None]
    R = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
    p = np.zeros((M, 3))
    pos = []
    for i in range(6):
        t, p_o = base_T[i]
        Rt = t @ R_all[:, i]  # (M,3,3)
        pt = t @ p_o  # (3,), constant per joint (t, p_o do not depend on q)
        p = (R @ pt[:, None]).squeeze(-1) + p
        R = R @ Rt
        pos.append(p.copy())
    dg = np.linalg.norm(pos[4], axis=1)
    dj = np.linalg.norm(pos[5], axis=1)
    ig = int(np.argmax(dg))
    best_g = float(dg[ig])
    best_jaw = float(dj.max())
    best_cfg = G[ig]
    return best_g, best_jaw, best_cfg, chain, N, M


def main():
    robot_name, links, joints = parse_urdf(URDF)
    print("robot name:", robot_name)
    print("\n== LINKS ==")
    total_mass = 0.0
    for name, info in links.items():
        inf = info["inertial"]
        m = inf["mass"] if inf else None
        if m is not None:
            total_mass += m
        print(f"\nlink {name}: mass={m}")
        if inf and inf["inertia"]:
            ii = inf["inertia"]
            print(f"  inertial origin xyz=({inf['origin'][0]}, {inf['origin'][1]}, {inf['origin'][2]}) rpy=({inf['origin_rpy'][0]},{inf['origin_rpy'][1]},{inf['origin_rpy'][2]})")
            print(f"  ixx={ii['ixx']} iyy={ii['iyy']} izz={ii['izz']} ixy={ii['ixy']} ixz={ii['ixz']} iyz={ii['iyz']}")
        else:
            print("  NO INERTIAL")
        for v in info["visual"]:
            print(f"  visual: {v['geometry']} @ {v['origin']}")
        for c in info["collision"]:
            print(f"  collision: {c['geometry']} @ {c['origin']}")
        if not info["collision"]:
            print("  ** NO COLLISION **")
    print("\nTOTAL MASS:", total_mass)

    print("\n== JOINTS ==")
    for name, j in joints.items():
        lim = j["limit"]
        extra = ""
        if lim:
            extra = f"limits [{lim['lower']}, {lim['upper']}] rad = [{math.degrees(lim['lower'])}, {math.degrees(lim['upper'])}] deg; effort {lim['effort']} N.m; vel {lim['velocity']} rad/s"
        print(f"\njoint {name} type={j['type']} {j['parent']} -> {j['child']}")
        print(f"  origin xyz=({j['origin_xyz'][0]},{j['origin_xyz'][1]},{j['origin_xyz'][2]}) rpy=({j['origin_rpy'][0]},{j['origin_rpy'][1]},{j['origin_rpy'][2]})")
        print(f"  axis=({j['axis'][0]},{j['axis'][1]},{j['axis'][2]})")
        if lim:
            print(f"  {extra}")
        else:
            print("  ** NO LIMITS **")

    # segment chain lengths (translation magnitudes along chain)
    print("\n== CHAIN LENGTHS ==")
    chain_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    total_chain = 0.0
    for jn in chain_names:
        j = joints[jn]
        p = np.array([float(v) for v in j["origin_xyz"]])
        L = float(np.linalg.norm(p))
        total_chain += L
        print(f"  {jn}: |origin| = {L:.4f} m")
    print(f"  sum of chain translations (upper bound on reach) = {total_chain:.4f} m")

    bg, bj, cfg, _, N, M = fk_grid(joints, links)
    print(f"\nFK grid search ({N} steps/joint, {M} configs): max dist base->gripper frame = {bg:.4f} m; base->jaw frame = {bj:.4f} m")
    print(f"  gripper-max config (rad): {np.round(cfg, 3)}")

    print("\n== MESHES (Simulation/SO100/assets) ==")
    assets = os.path.join(REPO, "Simulation/SO100/assets")
    for f in sorted(os.listdir(assets)):
        p = os.path.join(assets, f)
        kind, bbox, ntri, size = stl_info(p)
        bb = ""
        if bbox:
            vmin, vmax = bbox
            dims = vmax - vmin
            bb = f"bbox mm-equiv x:[{vmin[0]*1e3:.1f},{vmax[0]*1e3:.1f}] y:[{vmin[1]*1e3:.1f},{vmax[1]*1e3:.1f}] z:[{vmin[2]*1e3:.1f},{vmax[2]*1e3:.1f}] dims mm=({dims[0]*1e3:.1f} x {dims[1]*1e3:.1f} x {dims[2]*1e3:.1f})"
        print(f"  {f}: {size} bytes, {kind}, ntri={ntri}, {bb}")

    print("\n== referenced meshes exist? ==")
    for name, info in links.items():
        for key in ("visual", "collision"):
            for g in info[key]:
                fn = g["geometry"].split(" (")[0]
                if fn.startswith("assets/"):
                    p = os.path.join(REPO, "Simulation/SO100", fn)
                    print(f"  {name} {key}: {fn} -> {'OK' if os.path.exists(p) else 'MISSING!'} ({rel(p)})")


if __name__ == "__main__":
    main()
