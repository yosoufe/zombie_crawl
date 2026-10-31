# Rest pose & pan span (Phase 3, P3-3 / P3-4)

## Chosen rest pose (M4)

**Folded = URDF zero position** (all six joints at 0 rad), base plate flat on the ground.

| property | value | source |
|---|---|---|
| joint targets `q_rest` | all joints 0 rad (= `DEFAULT_JOINT_POS` = URDF zero) | `assets/so_arm100.py`, `START_POSES["folded"]` |
| resting base height `h_rest` | **0.0037 m** (base origin z above ground plane) | repo-measured folded contact-resting spawn height (`START_POSES["folded"].pos.z`, in use since phase 1); validated by `tests/helpers/check_rest.py` (spawn folded, settle 2 s, assert \|z−h_rest\|<0.01) |
| holding | servos hold position | zero action of `OffsetEMAJointPositionToLimitsAction` (neutral = 0 everywhere) commands exactly this pose under the `servo` preset |

Reason (P3-3): folded is simultaneously the zero-action pose, the spawn pose, and the
observation/action zero anchor, so M4 requires no new pose, no action bias, and no
observation offset. The lower candidate (`shoulder_lift 0.3, wrist_flex 0.6, gripper 0.5`)
also holds but would need its own action offset; naive mid-joint tucks collapse.

`rest_pose_ok` (rewards, metrics, acceptance) = flatness `up >= 0.95` **and**
`|base_z − h_rest| <= 0.01` **and** every joint within `0.15 rad` of `q_rest`.
Implemented in `mdp/rewards.py::rest_pose_ok`; thresholds are `CrawlEnvCfg` fields
(`rest_up_thresh`, `rest_height_tol`, `rest_joint_tol`).

Note: spec §P3-3 quotes `h_rest ≈ 0.033 m` for the folded rest; the repo's measured
folded resting height is 0.0037 m (the base plate lies flat; the spec value likely came
from the crouched non-contact estimate). The MEASURED value is authoritative here —
`rest_pose_ok` uses `h_rest` from the asset, not the spec prose.

## `S_pan` — shoulder_pan usable span (P3-4)

| quantity | value | source |
|---|---|---|
| URDF hard limits | lower −2.0, upper +2.0 rad | `assets/urdf/so_arm100_float.urdf` (`shoulder_pan` `<limit>`) |
| `soft_joint_pos_limit_factor` | 1.0 | `assets/so_arm100.py` (folded spawn sits at the one-sided joints' stops; shrinking would cut the commanded range) |
| **`S_pan`** | **4.0 rad** (soft limits = hard limits) | `asset.data.soft_joint_pos_limits`, asserted in `tests/helpers/check_rest.py` |

P3-4 requirement: every final-stage M3 segment must satisfy `|ω|·T_seg >= 1.5·S_pan = 6.0 rad`
at the LOWEST `|ω|` actually sampled. The mode-mix sampler enforces this at the final
`rotate_segment_s` stage: `T_seg ~ U(30, 40) s` with `|ω| ∈ [0.20, 0.30] rad/s` →
min product `0.20·30 = 6.0 rad` ✔ (the spec's own M3 evaluation uses ω=0.2, 30 s for
exactly this reason). Intermediate stages keep `|ω| ∈ [0.10, 0.30]` with growing
`T_seg` (5–8 → 10–15 → 18–25 → 30–40 s), and episode length is 40 s (§B.7).
