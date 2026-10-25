# Policy I/O contract — Zombie Crawl (phase 2)

Hardware-facing contract for deploying the phase-2 actor. The actor consumes ONLY
proprioception an SO-ARM100 can actually produce (joint encoders + IMU without
magnetometer) plus one operator-issued scalar. It never needs localization, a camera,
or an absolute heading reference.

## Actor observation (25-D, float32, concatenated in this exact order)

| Slice | Dim | Term | Units / range | Scaling applied | Train-time noise (drop at deploy) |
|---|---|---|---|---|---|
| 0:3 | 3 | `projected_gravity` (body frame) | unit vector | none | U(-0.05, 0.05) per component |
| 3:4 | 1 | `yaw_rate_cmd` | rad/s | x (1 / 0.3) → ~[-1, 1] | none |
| 4:7 | 3 | `base_ang_vel` (body frame gyro) | rad/s | none | U(-0.2, 0.2) |
| 7:13 | 6 | `joint_pos_rel` = q − q_default | rad | none (q_default = 0, folded spawn) | U(-0.01, 0.01) |
| 13:19 | 6 | `joint_vel` | rad/s | none | U(-0.5, 0.5) |
| 19:25 | 6 | `last_action` (raw pre-clip action) | [-1, 1] | none | none |

Joint order (SO-ARM100 URDF): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`.

`use_gait_clock` would insert a 1-D metronome after `projected_gravity` (27-D); it is
off by default and MUST stay off for the 25-D contract.

Deploy note: `enable_corruption` noise exists only at training time. Evaluation and
deployment use the noise-free observation path (`evaluate.py` already does).

## Command semantics

`yaw_rate_cmd` is a commanded BASE yaw rate [rad/s] (positive = counter-clockwise seen
from above), issued by the operator/handicapper; it is NOT a heading and requires no
heading measurement at the robot. The sim draws it uniformly from
[−`yaw_rate_max`, +`yaw_rate_max`] per env, re-drawn every 5–8 s; 20 % of draws are an
exact zero (stop). During training `yaw_rate_max` is a curriculum dial starting at 0
(motion-first: stage 0 = straight crawl only) and climbing to the measured reachable
range; 0.30 rad/s is the provisional cap.

**Measured reachable `yaw_rate_cmd` range (phase-2 run):** TBD — filled from the R2/R3
ladder (highest `yaw_rate_max` the population keeps tracking with
`steer_success ≥ 0.7`).

Deploy recipe: hold a constant command for ≥ 5 s segments; after a command CHANGE the
policy needs ~2 s of mask before `r` converges (expect a transient during the first
~2–3 s, do not treat it as failure).

## Joint-velocity recipe (deployment)

`joint_vel` must match the training distribution: finite-difference the encoder joint
positions at the policy rate (50 Hz, `dt = 0.02 s`), i.e.
`qd_t = (q_t − q_{t−1}) / 0.02`, optionally light low-pass (~τ 20–40 ms) to suppress
encoder quantization noise. Do NOT use drive-internal velocity estimates at a different
rate — the network expects the 50 Hz finite-difference scale. Seed the first step with
zeros.

## Action space (6-D)

Raw action a ∈ [−1, 1]⁶ per joint (same order), clipped to ±1 (`clip_actions=1.0`),
then mapped by `OffsetEMAJointPositionToLimitsAction`: a = 0 → neutral = folded spawn
(all joints 0); a = +1 → joint high soft limit; a = −1 → low soft limit (piecewise,
`rescale_to_limits=True`). An EMA (α = 0.3) smooths the joint-position targets; on
episode start the EMA state is initialized at the current joint positions.

## Network (fixed; must match exported ONNX/TorchScript)

Asymmetric actor-critic, MLPs `elu`: actor 25 → [256, 128, 64] → 6 (deterministic
mean at deploy); critic input 25 + 6 privileged (base_lin_vel 3, base_height 1,
ground_friction 1, claw heading rate `r` 1) → [256, 128, 64] → 1. The privileged group
(including `r`, the filtered claw heading rate) is simulation-only and never an actor
input.
