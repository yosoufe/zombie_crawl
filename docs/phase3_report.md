# Phase 3 experiment log (spec §B)

Owner decisions carried in: ALL mode-stage gates = **0.3** (spec 0.7 prose untouched,
evidence-based deviation per A.12-6/7); acceptance thresholds looser-but-measured
(M1 speed band [0.07, 0.17] m/s, M2/M3 absolute yaw tolerance floor 0.06 rad/s);
2 concurrent trainers allowed for Phase 3 only (L6 seeds).

## Rung registry

| Rung | Scope | Run dir | Status |
|---|---|---|---|
| L1 | `vx_cmd` input + M1/M2-only (mode ladder locked at stage 0), full phase-2 recipe | `logs/rsl_rl/phase03_l1/2026-10-26_22-18-01` | stopped by owner at iter 2194 (`model_2150.pt` last): "I happy with the recovery success" — crawl 1.0, steer EMA 0.45 (R2_s2: 0.37), recovery EMA 0.22 (R2_s2: 0.47, accepted per owner; W-P3-1 closed) |
| L2 | ladder UNLOCKED (auto stage 0→4 on 0.3 gates), same recipe | `logs/rsl_rl/phase03_l2/2026-10-27_00-02-40` | launched 2026-10-27 00:02 PDT, 3000 iters, video policy active |
| L3..L6 | seeds / final acceptance | — | pending |

L1 launch command (proven pattern + v2-full reward bundle):
`HEADLESS=1 ... scripts/train.py --task Zombie-Crawl-SO100-v0 --rl_library rsl_rl
--video --video_start_iters 100 --video_length 1250 --experiment_name phase03_l1
env.mode_stage_lock=0 agent.max_iterations=3000
env.rewards.yaw_rate_tracking.params.sigma=0.05 env.rewards.yaw_rate_tracking.weight=2.5
env.rewards.forward_velocity.params.cap_at_target=true
env.rewards.forward_progress.params.cap=true env.rewards.overspeed.params.factor=1.0`

## L1 live watch log (iter-matched vs R2_s2)

| iter | metric | L1 | R2_s2 | verdict |
|---|---|---|---|---|
| 330 | crawl / recovery / steer EMA / meanR | 0.98 / 0.06 / 0.24 / 36 | 1.00 / 0.73 / 0.00 / 34 | recovery dip at fallen-onset |
| 682 | " | 1.00 / 0.20 / 0.26 / 71 | 1.00 / 0.48 / 0.21 / 40 | recovery lagging |
| 1500 | " | 1.00 / 0.22 / 0.39 / 117 | 1.00 / 0.47 / 0.27 / 65 | steer AHEAD (L1 cap already 0.3; R2_s2 was still stage 1 @1500) |

Findings so far:
- L1's yaw-rate ladder ran AHEAD of phase 2 (stage 2 / cap 0.3 by ~iter 320 vs ~1900 in
  R2_s2): the vx_cmd channel + genuinely-active sigma=0.05 make steering easier to pick
  up — steer EMA at 1500 is 0.388 vs R2_s2's 0.268. Motion-first ordering is intact.
- **Watch item W-P3-1**: recovery_success EMA plateaued at ~0.21–0.25 vs R2_s2's ~0.47
  at the same cap-0.3 regime (R2_s2's own recovery fell 0.73→0.49 as its cap climbed;
  its DETERMINISTIC eval was 1.0). Stall terminations are 0.0 in BOTH runs, so the
  new stall re-anchoring is NOT the cause. Decision: run L1 to 3000 and judge with the
  R2_s2 deterministic eval protocol (recovery from fallen + steer bands); the EMA is
  push-noise-pessimistic. If deterministic recovery < 0.9 → re-tune recovery weight /
  stagger the yaw ladder behind recovery, log as A-style note.

- **B-2 (metrics calibration, caused first L2 restart at ~iter 502):** the spec's initial
  M1/M2 speed bands (±30/35 % relative) pin `success/M1` at 0.000 / `success/M2` at 0.03
  on this robot — straight crawl measures 0.146–0.161 m/s at vx=0.10 (+46..61 %), so the
  stage-0 gate could never fire and the run would stuck-abort at 600. Replaced with the
  owner-approved absolute band [0.07, 0.17] @ vx=0.10, expressed as
  [low_rel·vx, vx + 0.07] (low_rel 0.70/0.65) so it scales with the `v_target` dial.
  First L2 dir `2026-10-27_00-02-40` kept on disk; relaunched same day with the fix.

- **B-3 (metrics bug, caused further L2 restarts at ~iter 440/466):** per-mode segment
  accumulation was gated by the phase-2 steering floor `|seg_cmd| > SEG_MIN_CMD`, so M1
  (whose omega IS zero) never accumulated r/speed/unmasked time -> zero M1 segments were
  ever scored. Fixed: the floor now applies only to tracking modes M2/M3; M1/M4 always
  accumulate. Also added diagnostic `Curriculum/crawl_metrics/m1_abs_mean_r`.
- **B-4 (owner-rule calibration):** measured straight-crawl |mean r| = **0.101 rad/s**
  (2x the spec's 0.05 guess) -> launch override `env.m1_r_max=0.12` (measured +20 %).
  `m1_r_max` is an env cfg field, Hydra-settable, and its value + the live measurement
  are printed by the stuck-abort.
- **B-5:** stage-0 legitimately lives ~1500 iters (EMA gates need the policy to mature,
  cf. L1 steer EMA 0.45 @1945) -> `env.mode_stuck_iters=2000` at launch (spec's 600
  targets a DEAD ladder, not slow EMA rise; kept enforced).
- **B-6 (hydra-after-post_init trap):** curriculum gate params were pre-injected into
  the CurrTerm params in `__post_init__`, but hydra applies `env.<field>=` overrides
  AFTER `__post_init__` -> `env.mode_stuck_iters=2000` silently stayed 600 and the run
  self-aborted at iter 602 with the (correct, healthy) gate print:
  `M1 0.273, M2 0.156, m1_abs_mean_r 0.074, all dials at ceilings` — the ladder was
  merely SLOW, not dead. Fixed: `mode_stage_curriculum` re-reads `mode_gate_min`,
  `mode_hold`, `mode_stuck_iters`, `rest_start_frac` LIVE from `env.cfg` every call.
  (Class of bug: anything consumed at construction from cfg fields is stale; anything
  read via getattr at runtime works — that is why m1_r_max=0.12 took effect.)
- **B-7 (M4 rest discovery, weight/kernel calibration):** after stage 1 unlocked
  (iter 1094, mix_M4 -> 0.20), `success/M4` stayed **0.0** and `Episode_Reward/rest_pose`
  was EXACTLY 0: the spec-kernel `exp(-||q-q_rest||^2/sigma^2)` with my sigma=0.3 pays
  e^-22 for typical crawl poses -> zero gradient, rest undiscoverable. First fix
  (sigma=1.5) produced reward 1e-4 (gradient exists, value ~0); relaunched from
  `03-58-25/model_2550.pt` with `rest_pose weight=4.0, sigma=3.0` (owner's
  "looser, achievable" rule; M4 threshold stays the spec 0.3 until evidence says
  otherwise). Stillness/settle diagnostics are healthy: settle_mean_s 0.12,
  m4_creep 0.064 m, stillness_base ~0 — the robot IS quiet in M4, it just never folds.
- **B-8 (legacy steer gate re-blocked the ladder):** after the B-7 resume the mode ladder
  sat at stage 0 because `p2_ceilings` demands `yaw_rate_stage == 2`, whose gate is the
  LEGACY `steer_success` EMA — with phase-2's 0.03 rad/s tolerance floor it plateaued at
  0.283 (max over 1100 checks) forever. Owner's tolerance-floor decision (0.06, the same
  one behind `MODE_TOL_ABS`) applied to `STEER_TOL_ABS`; unit tests stay green; resumed
  from `04-17-17/model_3650.pt`. Note: resumes re-climb the whole ladder (curriculum
  state is env-side, not in the checkpoint) — ~500-1000 iters per re-climb, strong
  policy makes the gates re-pass quickly. No further restarts planned for this rung.
- **B-9 (M4 windows unlearnably short):** even with tol relaxed (rest_joint_tol 0.15→0.25,
  rest_up_thresh 0.95→0.90, owner calibration) `m4_rest_frac` stayed 0 — the policy never
  passes rest_pose_ok in M4. B.7's 5-8 s segments minus 2 s mask minus 2 s settle ramp
  leave 1-2 s of graded time per rest segment. Added `m4_segment_s` dial (default 8-14 s,
  spec deviation logged) + per-clause telemetry `m4_ok_up/m4_ok_z/m4_ok_q/m4_qerr_max` to
  pinpoint the blocking clause. Resumed `06-05-23/model_6050.pt`; if a clause is still 0
  after stage-1 re-unlock, next levers: M4 hold time 20 s, rest_pose weight 8, or start
  the rest fraction (rest_frac) higher at stage 1 so folded starts seed the behavior.
  Executed ladder: w 4→10, sigma 3→4, rest_start_frac 0.125→0.25 (raw kernel mean
  0.047→0.057 = real folding emerging), rest_joint_tol 0.25→0.5 (telemetry: `m4_ok_q`
  stayed exactly 0.0 at tol 0.25; push-jitter theory then DISPROVED by
  `check_rest.py --push-scale 1.0`: joint swing at rest under full push is tiny
  (p50 0.056, p99 0.173) -> tol 0.5 is generous. Real failure mode: the propped-up
  posture (z settles at 0.04, ok_z 0) -- the policy uncurls the folded spawn via its
  actions instead of resting, and sigma 4 paid 0.80 at qerr 0.9 (flat gradient).
  Added `force_m4_ids` (rest-start envs' FIRST segment forced to M4 via extras queue in
  `reset_start_pose` -> `ModeMixCommand._sample`) and final kernel sigma 2.0 (rest 10.0
  vs partial-crawl 2.1 = real slope). If still 0 after re-unlock: escalate to
  always-folded M4 probes (rest_start_frac 1.0 gated to M4 windows) or per-DOF rest
  targets (fold joints only, gripper free).
- **B-10 (THE M4 blocker was a policy-std blow-up, not reward design):** `Policy/mean_std`
  grew monotonically from ~4 at iter 700 (when M4/rest first entered the mix) to **530 by
  iter 11000** on the ±1 `clip_actions` space -- every sampled action was clip-saturated
  white noise. Root cause: the rest kernel (w=10) pays 10 for a pose reachable by random
  joint flailing, so the entropy bonus (+0.01) rewarded growing the std; PPO happily
  took it (Loss/entropy pinned at the 45.8 ceiling = log-term of std=500). All the M4
  gate failures at 4 restarts trace here. Fixes: `_GaussianDistCfg(std_range=(0.1, 2.0))`
  (rsl-rl clamps `std_param`; the Isaac wrapper doesn't expose the kwarg, passed through
  `resolve_class`), and `model_11000_stdreset.pt` resets `distribution.std_param` to 0.6
  (a plain clamp would leave the 530 param stuck at the upper bound: zero grad outside
  clamp). Phase-2 golden run settled std ~1.0-1.7; cap 2.0 keeps exploration, kills the
  exploit. Restarted from stdreset ckpt; WATCH: `Policy/mean_std` must stay <= 2.0 and
  `m4_rest_frac` must finally leave 0.
- **B-11 (exploration conflict is the root blocker; hetero attempt failed):** with std
  clamped [0.1,2] and every reward lever given its turn (kernel sigma 2 then 4, un-ramped
  payment, soft repulsion, stop_actions -2, lock-1 to skip re-climbs), `m4_qerr_max` sat
  at 2.7618 for five consecutive configs: the gait needs sigma~1.1 (sample actions hit the
  +/-1 clip constantly) while rest needs sigma~0.05, and rsl-rl's Gaussian std is ONE
  global number per dim -- rest is physically unreachable for the sampled actions. Tried
  `HeteroscedasticGaussianDistribution` (state-dependent std; ckpt surgery: final layer
  6->12 rows, std head bias init_std, optimizer_state_dict rebuilt over
  actor+critic params): `Policy/mean_std` saturated at the 2.0 cap within 600 iters
  (LL gradient pushes std up on outliers; rsl-rl has no std control for it) and rest
  still never formed -> reverted to plain clamped Gaussian (cfg keeps the comment).
  Current cook: model_13500_stdreset2 (std 0.6), lock-1 stage 1, sigma-4 w-10 kernel
  (un-ramped), rest_deviation soft, stop_actions -2, M4 45%, forced-M4 rest starts 25%.
  If m4_qerr_max does not move ~1500 iters, the remaining honest options: (a) M4-action
  clamp curriculum (action_manager clip per-mode -- invasive), (b) BC warm-start of the
  M4 response from the P3-3 folded rollouts, (c) per-mode rollout noise (needs rsl-rl
  patch). Owner call on which to fund.
- **B-12 (full-budget verdicts, mode-EMA filter):** two configs got the mandated >=3000
  iteration budgets with hands-off monitoring: (1) clamped Gaussian std 0.6 + un-ramped
  sigma-4 w-10 kernel + rest_deviation + stop_actions -2 + lock-1 (13539->16500, 2961
  iters): success/M4 0.0000 the whole run, m4_qerr_max frozen 2.759, Train/mean_reward
  -263; first-ever log showed the population DID start rested (qerr EMA 0.03) at episode
  0 -- sim can rest, PPO drives it out and back in is unreachable. (2) mode-conditional
  action EMA (ema_alpha_zero=0.05 under zero cmd, B-12 in actions.py -- sampled noise
  attenuated by the IIR gain, policy MEAN untouched) 16544->19499: qerr 2.78, success/M4
  0. Conclusion across ~9k iters and 7 levers: the reward/exposure/filter surface cannot
  move it -- the ACTOR MEAN is saturated at the +/-1 clip in every mode (probe: |mean|=1
  even with zero cmd while the robot visibly slows), and PPO gradient through the clip +
  adaptive-KL keeps it pinned. Next honest lever: offline BC warm-start -- roll the
  current policy under pin_mode(4), regress the actor's mean toward limp (a~0) / folded
  targets on those states (offline supervised, ckpt-level, no rsl-rl fork), then resume
  PPO.
- L2 restart history (all dirs kept): `00-02-40` (pre-B2), `00-31-13` (post-B2, killed
  for B3), `00-58-16` (post-B3 diag, killed for B4/B5), `01-25-15` (first with correct
  semantics but pre-B4/5), final active: the run started with
  `env.m1_r_max=0.12 env.mode_stuck_iters=2000`.

## B.10 smoke gates (all green before L1)

- `check_command.py` (rewritten for phase 3): motion-stage zero-command fraction
  **exactly 0.0**, M1 49.9 %, max|omega| = dial 0.300, M3 honors `m3_omega_min` +
  `rotate_segment_s` dials, M4 fully deadbanded, `pin()`/`pin_mode(4)` freeze, all 12
  ordered pairs with min fraction 6.7 % (>= 2 % floor).
- `check_dims.py`: policy **26**, critic **10**, action 6.
- `test_unit.py` 11 passed; `test_smoke.py` 8 passed (training-smoke test updated:
  expects `ModeMixCommand`, `agent.max_iterations=3` via hydra, |omega| <= yaw
  deadband under the closed dial, mode ladder stays at stage 0).
- 400-step full-mix stress (random actions): no crashes, `trans_pairs_seen=12`.

## Tuning / bug notes

- **B-1 (pre-existing phase-2 bug, now fixed): reward param overrides were silently
  discarded.** `CrawlEnvCfg.__post_init__` re-assigned
  `rewards.yaw_rate_tracking.params` wholesale, so
  `env.rewards.yaw_rate_tracking.params.sigma=0.05` never applied (hydra applies
  overrides as constructor args BEFORE `__post_init__`). Phase-2 A.12-5b "v2-full"
  gains therefore came from the top-level `weight=2.5` + the forward/overspeed CAPS
  (those params dicts were not clobbered), NOT from sigma 0.05. Fixed with
  `setdefault`-style injection; L1 now genuinely runs sigma=0.05 (verified in
  `params/env.yaml`). Re-baseline note: L1 vs phase-2 R2_s2 differs by sigma too.
- `ModeMixCommand` stored ids are 1..4 (0 = pre-first-draw): `mode_onehot` uses a
  5-wide one-hot with column 0 dropped (a 4-wide one of the mode id overflows at 4);
  transition counting excludes `prev_mode == 0` (negative `index_add_` indices
  tripped a device-side assert).
- `reset_start_pose`: `pose_weights`/fallen lists now PAD to the number of named poses
  (phase 3 adds the 4th "rest" entry); rest starts are written with ZERO joint noise
  and count as UPRIGHT for the started_fallen metric.
- Command resample contract: `ModeMixCommand` samples inside `_resample` (per-mode
  durations); `_resample_command` only flags `just_resampled`. Test helpers must drive
  `term._resample(...)`.
- B.7 sampler bias (spec: bias draws if any ordered pair < 2 %): not needed —
  min pair fraction 6.7 % with the uniform mode draws; `trans_min_frac` is logged,
  bias will be added only if a training run ever shows a starved pair.
- Spec deviations (logged, per owner): gate 0.3 everywhere; M1 speed band widened to
  the measured [0.07, 0.17]; M3/M4 thresholds start at spec values and will be
  calibrated on first evidence.

## Deliverable status (spec §B deliverables)

| Item | Status |
|---|---|
| code on top of phase 2 | done (mdp mode-mix stack, wired + gated) |
| `play.py --vx/--yaw_rate/--mode` | pending (after L1) |
| evaluate scenarios M1-M4 + 12-pair schedule | pending (after L1) |
| checkpoint + ONNX/TorchScript | pending (final rung) |
| TensorBoard logs | live (`tensorboard --logdir logs/rsl_rl/phase03_l1`) |
| videos | policy live (every 200 iters from 100, 25 s) |
| `docs/rest_pose.md` | DONE (h_rest 0.0037, S_pan 4.0; P3-4 via |omega|>=0.20 x T>=30 s) |
| `policy_io.md` 26-D update | pending (after training recipe frozen) |
| `docs/phase3_report.md` | this file |
