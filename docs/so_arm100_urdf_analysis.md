# SO-ARM100 (SO-100) URDF Analysis

Repo: https://github.com/TheRobotStudio/SO-ARM100 (cloned `--depth 1`, commit `5f6d2b8`, dated 2026-09-23, "Add SO-101 leader arm URDF")
Local clone: `<workspace>/third_party/SO-ARM100`

## 1. File locations

| Item | Path (relative to `<workspace>`) |
|---|---|
| **URDF (primary, SO-100 / SO-ARM100 follower)** | `third_party/SO-ARM100/Simulation/SO100/so100.urdf` |
| SO-100 simulation meshes | `third_party/SO-ARM100/Simulation/SO100/assets/*.stl` (12 files) |
| SO-100 URDF preview scene (Rerun) | `third_party/SO-ARM100/Simulation/SO100/so100.rrd` |
| SO-101 URDF variants (for comparison, §7) | `third_party/SO-ARM100/Simulation/SO101/so101_{new_calib,old_calib,new_calib_camera,leader_new_calib}.urdf` |
| SO-101 meshes | `third_party/SO-ARM100/Simulation/SO101/assets/*.stl` (18 files) |
| CAD source (not used by the URDF) | `third_party/SO-ARM100/STEP/**/*.step` (47 MB, 30 files) |
| Printable STLs (not used by the URDF) | `third_party/SO-ARM100/STL/**` |
| Analysis script (this report) | `zombie_crawl/scripts/asset_pipeline/analyze_so100_urdf.py` |

Robot name in URDF: `so_arm100`. **7 links, 6 joints, all joints revolute.** No `<state>` tags, no `<dynamics>`/`<damping>` on joints, no fixed base joint (the `base` link is simply the root of the kinematic tree).

## 2. Joints (6, all revolute)

URDF `effort` is in N·m (SI), so 35 N·m as written. Limits given in rad and degrees.

| Joint | Parent → Child | Axis | Origin xyz (m) | Origin rpy (rad) | Lower | Upper | Lower (°) | Upper (°) | Effort (N·m) | Velocity (rad/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `shoulder_pan` | base → shoulder | (0, 1, 0) | (0, −0.0452, 0.0165) | (1.57079, 0, 0) | −2.0 | 2.0 | −114.6 | 114.6 | 35 | 1 |
| `shoulder_lift` | shoulder → upper_arm | (1, 0, 0) | (0, 0.1025, 0.0306) | (−1.8, 0, 0) | 0 | 3.5 | 0 | 200.5 | 35 | 1 |
| `elbow_flex` | upper_arm → lower_arm | (1, 0, 0) | (0, 0.11257, 0.028) | (1.57079, 0, 0) | −3.14158 | 0 | −180.0 | 0 | 35 | 1 |
| `wrist_flex` | lower_arm → wrist | (1, 0, 0) | (0, 0.0052, 0.1349) | (−1, 0, 0) | −2.5 | 1.2 | −143.2 | 68.8 | 35 | 1 |
| `wrist_roll` | wrist → gripper | (0, 1, 0) | (0, −0.0601, 0) | (0, 1.57079, 0) | −3.14158 | 3.14158 | −180.0 | 180.0 | 35 | 1 |
| `gripper` | gripper → jaw | (0, 0, 1) | (−0.0202, −0.0244, 0) | (0, 3.14158, 0) | −0.2 | 2.0 | −11.5 | 114.6 | 35 | 1 |

Every joint has position limits and an effort/velocity limit (nothing is unlimited).

Notes:
- Naming collision: the **link** `gripper` and the **joint** `gripper` share a name (legal in URDF — separate namespaces — but confusing).
- `shoulder_lift` range 0→3.5 rad (0→200.5°) is unusual for a servo (real STS3215 range ≈ ±120°); it is a modeling choice that lets the arm fold.
- 35 N·m is ~10–20× the real stall torque of the Feetech STS3215 servos (≈16.5 kg·cm ≈ 1.6 N·m at 6 V; ≈30 kg·cm ≈ 3 N·m for the 12 V version, per the repo README). PhysX will accept it, but it is not physically representative.
- No friction/damping entries on any joint (defaults to 0).

## 3. Links (7)

All 7 links have a complete `<inertial>` (mass + full inertia tensor + origin). All inertias are diagonal (off-diagonals 0) and satisfy the triangle inequalities (ixx+iyy>izz etc.), so PhysX accepts them.

| Link | Mass (kg) | Inertial origin xyz (m) | ixx | iyy | izz (kg·m²) | Visual mesh(es) | Collision mesh(es) |
|---|---|---|---|---|---|---|---|
| `base` | **1.0** (marked "Approximate" in the file) | (0, 0, 0) | 0.01 | 0.01 | 0.01 | `Base.stl`, `Base_Motor.stl` | `Base.stl` only |
| `shoulder` | 0.119226 | (−9.08e−05, 0.059097, 0.031089) | 5.94278e−05 | 5.89975e−05 | 3.13712e−05 | `Rotation_Pitch.stl`, `Rotation_Pitch_Motor.stl` | `Rotation_Pitch.stl` only |
| `upper_arm` | 0.162409 | (−1.72e−05, 0.070180, 0.003105) | 2.13312e−04 | 1.67164e−04 | 7.01522e−05 | `Upper_Arm.stl`, `Upper_Arm_Motor.stl` | `Upper_Arm.stl` only |
| `lower_arm` | 0.147968 | (−0.003396, 0.001378, 0.076801) | 1.38803e−04 | 1.07748e−04 | 4.84242e−05 | `Lower_Arm.stl`, `Lower_Arm_Motor.stl` | `Lower_Arm.stl` only |
| `wrist` | 0.0661321 | (−0.008527, −0.035228, −2.35e−05) | 3.45403e−05 | 2.39041e−05 | 1.94704e−05 | `Wrist_Pitch_Roll.stl`, `Wrist_Pitch_Roll_Motor.stl` | `Wrist_Pitch_Roll.stl` only |
| `gripper` | 0.0929859 | (0.005524, −0.028017, 0.000484) | 5.03136e−05 | 4.64098e−05 | 2.72961e−05 | `Fixed_Jaw.stl`, `Fixed_Jaw_Motor.stl` | **none** |
| `jaw` | 0.0202444 | (−0.001617, −0.030347, 0.000450) | 1.11265e−05 | 8.99651e−06 | 2.99548e−06 | `Moving_Jaw.stl` | **none** |

All visual/collision origins are identity (0 0 0, rpy 0 0 0); the pose is baked into the joint origins.

## 4. Mesh inventory (`Simulation/SO100/assets/`)

All 12 referenced meshes exist and are **binary STL triangle meshes** (PhysX-compatible as-is; no conversion needed). Units are **meters** — verified by bounding boxes (e.g. `Base.stl` spans ≈111×96×84 mm, matching the physical base plate). No STEP/DAE/OBJ are referenced from this URDF (the repo's `STEP/` directory is CAD source only).

| Mesh (in `Simulation/SO100/assets/`) | Size (bytes) | Format | Tris | Bounding box (m) |
|---|---|---|---|---|
| `Base.stl` | 443,484 | binary STL | 8,868 | 0.1109 × 0.0959 × 0.0835 |
| `Base_Motor.stl` | 154,284 | binary STL | 3,084 | 0.0248 × 0.0454 × 0.0396 |
| `Rotation_Pitch.stl` | 347,584 | binary STL | 6,950 | 0.0524 × 0.0863 × 0.0628 |
| `Rotation_Pitch_Motor.stl` | 154,284 | binary STL | 3,084 | 0.0396 × 0.0454 × 0.0248 |
| `Upper_Arm.stl` | 215,184 | binary STL | 4,302 | 0.0700 × 0.1421 × 0.0280 |
| `Upper_Arm_Motor.stl` | 152,284 | binary STL | 3,044 | 0.0396 × 0.0248 × 0.0454 |
| `Lower_Arm.stl` | 347,184 | binary STL | 6,942 | 0.0612 × 0.0369 × 0.1307 |
| `Lower_Arm_Motor.stl` | 151,884 | binary STL | 3,036 | 0.0396 × 0.0248 × 0.0454 |
| `Wrist_Pitch_Roll.stl` | 341,684 | binary STL | 6,832 | 0.0633 × 0.0744 × 0.0320 |
| `Wrist_Pitch_Roll_Motor.stl` | 133,684 | binary STL | 2,672 | 0.0454 × 0.0396 × 0.0248 |
| `Fixed_Jaw.stl` | 264,584 | binary STL | 5,290 | 0.0652 × 0.1064 × 0.0480 |
| `Fixed_Jaw_Motor.stl` | 151,884 | binary STL | 3,036 | 0.0454 × 0.0248 × 0.0396 |
| `Moving_Jaw.stl` | 177,184 | binary STL | 3,542 | 0.0223 × 0.0920 × 0.0480 |

Total ≈ 3.0 MB, max 8,868 triangles per mesh — trivial for PhysX convex-hull or triangle-mesh collision.

## 5. Base link and gripper

- **Base link:** `base` is the kinematic root (no parent joint; normally the fixed/mounted part). It is the only link with a placeholder inertial (mass 1.0 kg, isotropic 0.01 kg·m², commented "Approximate inertial values"). If the robot is made floating, this link is the free-floating body; its 1.0 kg dominates the total mass (62 %).
- **Gripper:** joint `gripper` (gripper → jaw) is a **revolute** joint, not prismatic — the SO-100 gripper is a rotating jaw. Axis (0,0,1) in the child frame, origin offset (−0.0202, −0.0244, 0) with rpy (0, π, 0), limits **−0.2 … 2.0 rad (−11.5° … 114.6°)**, effort 35 N·m, velocity 1 rad/s.
- **Finger collision:** the `gripper` link (fixed jaw + servo) and the `jaw` link (moving jaw) have **no collision geometry at all** — see §8.

## 6. Mass and reach

**Total robot mass (sum of 7 link masses): 1.6089654 kg ≈ 1.61 kg**
(1.0 + 0.119226 + 0.162409 + 0.147968 + 0.0661321 + 0.0929859 + 0.0202444)

**Reach** — chain of joint-origin translations from base to gripper:

| Segment (joint) | |origin| (m) |
|---|---|
| base → shoulder (`shoulder_pan`) | 0.0481 |
| shoulder → upper_arm (`shoulder_lift`) | 0.1070 |
| upper_arm → lower_arm (`elbow_flex`) | 0.1160 |
| lower_arm → wrist (`wrist_flex`) | 0.1350 |
| wrist → gripper (`wrist_roll`) | 0.0601 |
| gripper → jaw (`gripper`) | 0.0317 |
| **Sum (geometric upper bound)** | **0.4979 m** |

Forward-kinematics grid search over all joint limits (10 steps/joint, 1,000,000 configs) gives the maximum distance from the base frame origin:
- to the **gripper frame**: **0.4078 m** (at q ≈ [−2.0, 3.5, 0, −1.68, −3.14, −0.2] rad)
- to the **jaw frame**: **0.4352 m**

The moving jaw extends up to ≈0.05 m beyond the jaw frame (its mesh is 92 mm long), so the practical fingertip reach is **≈ 0.45–0.47 m**. Rule of thumb: **≈ 0.5 m chain length, ≈ 0.45 m effective reach.**

## 7. SO-101 comparison

**The standalone repo `https://github.com/TheRobotStudio/SO-101` does not exist** (GitHub 404, clone fails with auth prompt). There is no separate public SO-101 repository to compare against.

However, the SO-ARM100 repo itself contains full SO-101 simulation files in `third_party/SO-ARM100/Simulation/SO101/` (4 URDFs, 18 STL meshes, MuJoCo files, per-part metadata `.part` files). Comparing `so101_new_calib.urdf` with `so100.urdf`:

| Aspect | SO-100 (`so100.urdf`) | SO-101 (`so101_new_calib.urdf`) |
|---|---|---|
| Links / joints | 7 / 6 (all revolute) | 8 / 7 (6 revolute + 1 fixed `gripper_frame_joint` adding a `gripper_frame` tool link) |
| Collision coverage | `gripper` and `jaw` links have **no collision** | **Every part has both visual and collision** (incl. the moving jaw) |
| Inertia tensors | Hand-written, diagonal only | Generated from CAD via `onshape-to-robot`, full 3×3 (non-zero off-diagonals) |
| Effort limits | 35 N·m (unrealistic) | 10 N·m (still high, but closer) |
| Joint limits | Un-calibrated, wide (e.g. shoulder_lift 0→200°) | Calibrated variants with notes ("new calibration", 5° offset comment) |
| Variants | 1 | 4: `old_calib`, `new_calib`, `new_calib_camera` (+wrist cam), `leader_new_calib` (teleop arm) |
| Maintenance | SO-100 docs are marked **deprecated** in the README | Actively developed — the repo's latest commit (2026-09-23) *is* the SO-101 leader URDF |
| Mass | 1.609 kg (incl. 1.0 kg placeholder base) | ≈0.632 kg (no placeholder base mass) |

Caveat for SO-101 → PhysX: its `gripper_frame_link` has mass 1e-9 and **zero inertia** — a PhysX hazard as-is (it must be merged into the parent link or given a small real inertia during import).

**Verdict on the comparison:** the SO-101 description in this same repo is the more complete, CAD-accurate, and more recently maintained URDF (full collision coverage, computed inertias, calibrated limits, multiple variants). The SO-100 URDF is a simpler, older model. Per the task requirement we use SO-ARM100/SO-100 (`so100.urdf`), but if the physical arm is actually an SO-101 (the current product), `Simulation/SO101/so101_new_calib.urdf` is the better starting point.

## 8. PhysX concerns & recommendations

1. **No collision on `gripper` and `jaw` links (major).** The gripper/finger links are visual-only. In PhysX the fingers will pass through everything — the arm cannot grasp or push objects with the jaw. **Fix:** add `<collision>` entries (either reuse `Fixed_Jaw.stl` / `Moving_Jaw.stl`, or box approximations) to both links. If grasping isn't needed, at least add the fixed-jaw collision.
2. **Placeholder base inertia (moderate).** `base` has mass 1.0 kg with an "approximate" isotropic 0.01 kg·m², and it is 62 % of total mass. For a **floating** robot this dominates the base dynamics. Recommend re-deriving the base mass/inertia (CAD export or measured) or at least recentering the inertial origin; if the base is only a mounting flange, dropping it to ≈0.1–0.2 kg changes the floating-base behavior a lot.
3. **Motors excluded from collision (minor, intentional).** Each link's collision uses only the printed part; the servo body mesh is visual-only. This reduces self-collision risk at joints — keep as-is, but know the collision hulls don't fully represent the robot envelope.
4. **No joint friction/damping (minor).** All 6 joints default to 0 friction/0 damping. Fine for stability, but a small damping (e.g. 0.05–0.2 N·m·s) on the 5 arm joints and on the gripper helps mimic servo behavior and settles contact.
5. **Unrealistic effort limits (cosmetic for sim).** 35 N·m vs ~1.6 N·m actual servo torque. PhysX doesn't validate it, but any controller relying on the limit will be wrong. Consider 1.6–3 N·m.
6. **`shoulder_lift` 0→200.5° range (check intent).** Wider than the servo's real ±120°. Keep if the arm geometry needs it for folding; otherwise clamp to the real range.
7. **All else is clean for PhysX:** every link has positive mass + a valid (triangle-inequality-satisfying) inertia tensor; all joints are limited; meshes are binary STL triangle meshes in **meters** (no unit inconsistency, no STEP in the URDF, no missing files); smallest mass is 20.2 g (`jaw`) — acceptable, no near-zero-mass links; triangle counts are small.
8. **Naming:** joint and link both named `gripper` — harmless but rename the link (e.g. `gripper_base`) when copying for Isaac/PhysX if your importer dislikes it.

## 9. Method

- Cloned `TheRobotStudio/SO-ARM100` with `git clone --depth 1`.
- Parsed the URDF with `xml.etree` (no ROS/urdfdom used); script: `zombie_crawl/scripts/asset_pipeline/analyze_so100_urdf.py` (system python3 + numpy).
- STL format detected via binary size formula (84 + 50·N bytes) and full vertex parse for bounding boxes; reach computed by vectorized forward kinematics on a 10-step-per-joint grid (10⁶ configurations) using the standard URDF model `child = parent · origin · Rot(axis, q)`.
