# Phase 2 report — zombie crawl

Running log per spec §A.12 (tuning) and the rung registry. Final §A.14 deliverables
are appended at the end of each rung.

## Rung registry

| rung | run dir | status |
|---|---|---|
| R0 | `zombie_crawl_r0/logs/rsl_rl/phase02_r0/2026-10-23_22-02-18` | done — crawl 1.0, recovery 0 (baseline) |
| R1 | `logs/rsl_rl/phase02_r1/2026-10-24_18-37-55` | done — owner-blessed reproduction (identical std fingerprint to `2026-10-23_23-37-26`); deterministic recovery 1.0/1.0; straight gates FAIL (speed 0.227, mean r −0.30) |
| R2 (first, no lock, 3000) | `logs/rsl_rl/phase02_r2/2026-10-24_21-43-40` | stage ladder opened stage 1; **steer_success plateau 0.226 < 0.7** → stop-and-report per §A.11 (cause in A.12-5) |
| R2 v2-full | `logs/rsl_rl/phase02_r2_v2full/2026-10-24_23-40-55` | done 1500 — train steer EMA 0.245; **deterministic steered gates 0.91–1.00 on most (friction,cmd) combos** (fails cmd −0.10 and friction-0.5 row); recovery 1.0/1.0; straight speed 0.14–0.16 (band ≤0.13) |
| R2 track-only | `logs/rsl_rl/phase02_r2_track/2026-10-24_23-45-59` | done 1500 — train steer EMA 0.208; deterministic steer 0.00–0.37 (**worse**); straight speed 0.21–0.24; recovery 1.0/1.0 → control arm shows the SPEED CAPS, not the bump sharpening, are what matters |
| **R2 s2 (gate 0.3, 3000)** | `logs/rsl_rl/phase02_r2_s2/2026-10-25_12-57-23` | **R2 rung PASS (owner gate 0.3)** — stage ladder: stage 1 @322, steer EMA crossed 0.3 → **stage 2 @1946**, dial full 0.30; final train crawl 1.0 / recovery 0.489 (R1 0.527, −7 %) / steer 0.373. Deterministic @model_2999: recovery 1.0/1.0 all frictions (0.18/0.66 s); steered ±0.2/±0.3 track 0.78–1.00 (+0.3 at friction 0.5 = 0.34); small commands weakest (−0.1: 0.00–0.40, tol 0.035 rad/s vs mechanical drift); straight speed 0.146–0.161 still above band ≤0.13 |

Superseded/invalid runs (kept on disk per no-delete rule; do not cite):
`phase02_r1/2026-10-23_22-30-31` (contaminated launch), `phase02_r1_stdclamp`,
`phase02_r1_smooth`, `phase02_r1_retuned`, `phase02_r1_retuned2`,
`phase02_r1_retuned3` (reverted by owner directive back to the §A.10 agent cfg),
and the two aborted `phase02_r2` launches that died before iteration 50
(`logs/phase02_r2_train.log` first attempts).

## §A.12 tuning notes

- **A.12-1..4 (reverted bundle, history):** std clamp `ClampedGaussianDistributionCfg`
  (std_range 0.1–1.0) + entropy 0.003/0.002; `forward_progress_lpf` cap; overspeed
  factor 2.0→1.4→1.0; `zero_cmd_yaw_rate` straight-hold penalty; tracking σ0.05/w2.5;
  forward_velocity tanh cap. All reverted on owner directive back to §A.10 defaults.
  Partial evidence retained: fixed-evaluator results — R1c 0.247 m/s (r +0.12),
  R1d |r| 0.033 pass / 0.198 m/s, R1e |r| 0.015 / 0.162 m/s; recovery never dropped
  below ~0.51 train / 1.0 deterministic.
- **A.12-5 (active, R2 plateau diagnosis):** at stage 1 the straight-speed lane pays
  ~8.5× steering: `forward_velocity` 2.00 + `forward_progress` 1.92 vs
  `yaw_rate_tracking` 0.46 (w=1.5, σ=0.1 on a ±0.1 rad/s command scale; overspeed
  ≈0 until 0.20 m/s). Ignoring commands is the dominant optimum → steer_success
  plateaus at 0.226. Fix bundle (hydra-overridden, defaults unchanged):
  - A.12-5a: `yaw_rate_tracking` σ 0.1→0.05, weight 1.5→2.5 (control arm: track-only).
  - A.12-5b: `forward_velocity_exp(cap_at_target=True)` (flat above v_target; backward
    branch untouched), `forward_progress_lpf(cap=True)` (clamp max 1.0),
    `overspeed_penalty(factor=1.0)`.
  Both arms 1500 iters (`agent.max_iterations=1500`), videos per owner policy
  (start 100, every 200, 25 s via `--video_length 1250`). Ladder/command constants
  untouched.
- **A.12-6 (open finding — steering gate unreachable under §A.10 exploration):**
  v2full's deterministic (§A.13) steered segment_success is 0.91–1.00 on most
  friction×cmd combos, yet the training EMA gate (`steer_success >= 0.7`) never
  exceeds ~0.25. Explanation: the §A.10 agent runs with blown-up Gaussian std
  (~40–90) — sampled actions are ±1 clips of near-noise, so *during training* no
  policy can hold the 0.03–0.05 rad/s tracking tolerance the metric demands, while
  the deterministic mean policy tracks fine. The ladder gate as specified is
  therefore not reachable without taming exploration (std clamp / entropy reduction
  — owner-approved once, later reverted) or scoring the gate from a deterministic
  probe instead of the stochastic rollout EMA.   Decision pending with owner;
  candidate R2 checkpoint = `phase02_r2_v2full/2026-10-24_23-40-55/model_1499.pt`.
- **A.12-7 (owner directive, 2026-10-25): stage-2 gate lowered 0.7 -> 0.3.**
  Rationale (owner): the policy tracks well; disturbances + the robot's mechanics make
  sustained in-rollout yaw-rate tracking very hard, so the stochastic EMA gate at 0.7 is
  effectively unreachable — evidence: v2-full training EMA 0.18@526 -> 0.245@1466 (rising)
  while the deterministic policy's steered segment_success is 0.91–1.00 on most
  friction×cmd combos (A.12-6). Edit: `min_steer` 0.7→0.3 in `CrawlCurriculumCfg.
  yaw_rate_stage` params (crawl_env_cfg.py) + matching default/docstring in
  `mdp/curriculums.py::yaw_rate_stage_curriculum`. Spec text (phase3_spec.md:171)
  intentionally left at 0.7; this log records the deviation. Follow-on run:
  `phase02_r2_s2` — v2-full reward bundle, no lock, 3000 iters, videos per owner policy;
  expected stage-2 unlock ~iter 1800–2400 at gate 0.3. OUTCOME: unlocked at iter 1946,
  dial reached the full 0.30, steer EMA rose to 0.39 under ±0.3 commands — gate 0.3 was
  sufficient; an owner request to lower further to 0.2 was declined as unnecessary
  (evidence above; param is launch-time read anyway).
- **A.12-8 (observations, no action taken yet):** (i) deterministic steering is
  command-magnitude dependent — large commands (±0.2/±0.3) track at 0.78–1.00, small
  ones (±0.1) at 0.00–0.42, because the metric tolerance is max(0.35·|cmd|, 0.03)
  = 0.035 rad/s at |cmd|=0.1 while the robot's natural drift is larger (mechanical
  left/right asymmetry: −0.1 fails hardest at frictions 0.5/1.2). If small-command
  tracking matters for deployment, candidates are a gyro-yaw actor input (spec watch
  item W1) or a deadband-aware tolerance, both requiring owner sign-off. (ii) The
  straight speed band (0.07–0.13) is still missed (0.146–0.161) despite the caps; the
  residual ~20 % overspeed is the gait's free-glide phase under the capped-but-flat
  reward — owner decision pending whether to pursue for R3 or accept.

## Evaluator fixes (not tuning — training unaffected)

- `scripts/evaluate.py`: frozen-accumulator fix (time_out auto-reset zeroed the report
  numerators → false "static robot"); `--disturbance`, `EVAL_DEBUG/EVAL_SAMPLE/
  EVAL_TRAIN_SPEC/EVAL_CORRUPT/EVAL_TRAINCFG` hooks.
- `scripts/debug_rollout.py`: official-play clone used to bisect harness vs policy.
- `scripts/train.py`: `--video_length N` (env steps) handler + `video_flags.
  training_video_length` consumed in `CrawlEnvCfg._setup_training_video`, so launches
  can meet the owner video policy (25 s clips) without editing the 40 s config default.
