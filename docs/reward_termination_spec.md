# Current Reward / Penalty / Termination Spec

Snapshot of the reward, penalty, and termination terms as currently wired in
`source/zombie_crawl/zombie_crawl/tasks/manager_based/crawl/crawl_env_cfg.py`.
Per-step weights are nominal — the reward manager scales them by the step time, so read
them as relative influence.

All base/vertical/height/orientation terms measure the **base link** (the robot's root
body = the chassis that sits on the floor). Only `wrist_pointing` involves the wrist.

## Rewards (pay)

| Term | Weight | Quantity measured | Body/frame | Current behavior |
|---|---|---|---|---|
| `forward_velocity` | +2.5 | `tanh(v_proj / speed_target)` where `v_proj` = base linear velocity projected onto the command axis, x `(1 - (gx^2+gy^2))` of base projected gravity | base root body (world) | +2.5 at target speed when base flat, ~0 when tipped, negative if moving backward |
| `forward_progress` | +1.0 | 1 s EMA of base position displacement along the command axis | base root body (world) | net ground covered, not fakeable by jitter |
| `alive` | +0.05 | 1 while episode runs | — | small survival bonus |
| `low_ground_contact` | 0.0 | off | — | disabled |
| `gait_asymmetry` | 0.0 | off | — | disabled |

## Penalties (cost)

| Term | Weight | Quantity measured | Body/frame | Current behavior |
|---|---|---|---|---|
| `wrist_pointing` | -0.01 | `1 - cos(angle)` between the base->wrist arrow and the command | wrist body vs base (world XY) | near-off tie-breaker (was -1.0); kept only to break alignment ties, no longer a real attractor |
| `lateral_velocity` | -1.0 | base linear velocity component perpendicular to the command, squared | base root body (world) | punishes crabbing |
| `overspeed` | -2.0 | base horizontal speed beyond 2x target | base root body (world) | punishes bolting |
| `vertical_velocity` | -2.0 | base linear-velocity Z component squared | base root body (world Z) | punishes base rising/falling (hop) |
| `base_height` | -1.0 | `(base Z - 0.005 m)^2` | base root body (world Z) | target base Z = 0.005 m (approx. lying flat on floor); ~0 at rest |
| `orientation` | -0.5 | `gx^2 + gy^2` of base projected gravity (tilt from flat) | base root body (body frame) | favors base lying flat |
| `joint_limits` | 0.0 | joints pushed into soft limits | all actuated joints | **disabled** |
| `action_rate` | -0.01 | change in consecutive actions | actions | mild smoothness |
| `joint_deviation` | 0.0 | drift from nominal joint angles | all actuated joints | **disabled** |
| `joint_acc` | -0.0001 | joint acceleration | all actuated joints | negligible |
| `joint_torques` | -0.00001 | joint effort | all actuated joints | negligible |

Specifically: **height** is the base chassis's world Z (target 0.005 m above ground);
**vertical velocity** is the base chassis's world vertical speed. Both are on the base,
not the claw/wrist. The only term that touches the wrist is `wrist_pointing`.

## Terminations (episode ends)

| Term | Type | Triggers | Measured on |
|---|---|---|---|
| `time_out` | success (bootstrapped) | 20 s elapsed | clock |
| `base_nan` | failure | physics -> NaN | base |
| `base_too_high` | failure | base Z > 0.45 m | base world Z |
| `out_of_bounds` | failure | outside \|x\|>2.5 m or \|y\|>1.5 m | base world XY |
| `stalled` | failure | after 12 s, base moved <5 cm along the command axis | base world XY projected on command |

## Other live config (not rewards/terminations)
- Command direction: random +/-180 deg each reset, independent of spawn facing.
- Curriculum `speed_target`: 0.06 (start) -> 0.10, frozen.
- Curriculum `friction_scale`: 1.5 (start) -> 1.0, frozen.
