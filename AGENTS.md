# Agent rules — zombie_crawl

## Training runs
- ONE training at a time (owner rule): before launching, verify no other
  `scripts/train.py` is running (`pgrep -cf 'scripts/trai[n].py'` must be 0); kill
  the old one first if replacing a run. Never let two trainers share the GPU.
- Launch training ALWAYS detached and never blocking the chat:
  `HEADLESS=1 setsid nohup ../.venv/bin/python scripts/train.py ... > logs/<name>_train.log 2>&1 < /dev/null & disown`
  Then check status with SHORT separate calls only — no long `sleep` polling in the
  launch call (it blocks the chat and the harness kills detached children on timeout).
- `HEADLESS=1` is REQUIRED whenever `--video` is used: the launcher treats the injected
  recording visualizer as GUI intent and opens a window otherwise
  (`IsaacLab3/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py` +
  kit_launcher `_resolve_headless_settings`). No `--headless` CLI flag exists.
- Experiment dir template: `--experiment_name phase02` ->
  `logs/rsl_rl/phase02/<default-datetime>/`. Keep the default datetime folder.
  Do NOT pass `--run_timestamp` (no fixed "latest" folders).
- NEVER delete previous runs. No `rm -rf logs/rsl_rl/...`, no cleanup of old
  `<datetime>/` dirs, ever. Restarts just create new folders alongside old ones.
  TensorBoard runs once: `tensorboard --logdir logs/rsl_rl/phase02`.
- Kill stale trainers with self-safe patterns: `pkill -9 -f 'scripts/trai[n].py'`
  (plain patterns match the wrapper shell and kill the tool call).
- Everything runs through `../.venv/bin/python` (repo-parent venv, isaacsim 6.1, rsl-rl 5.5.1).

## Phase-2 command policy (phase2_v2.md is the spec; phase2.md is frozen — never edit)
- Switch: Hydra `env.use_command_policy=true`; stage ladder in
  `mdp/commands.py::_allowed_modes`: stage 0 = constant forward sanity,
  stage 1 = MOTION ONLY (forward/rotate/forward_turn, zero-command fraction must be
  0.0 — verified in logs), stage >=2 adds stop/sit. Owner directive: motion first,
  sit last.
- Reward design rules (learned the hard way, do not regress):
  - Tracking bumps (`exp(-(v-cmd)^2/sigma^2)`) MUST be gated ON by
    `|cmd| > deadband`; ungated bumps under a zero command pay their maximum for
    doing nothing and PPO collapses to sitting.
  - Penalties saturate (`torch.clamp(val, max=1.0)`); raw `(v/tol)^2` cliffs make
    "never move" the dominant optimum.
  - No phase-1 ref-frame (`zombie_ref_yaw`, `get_ref_dir`) consumers in the
    command-policy path: progress/stall/curriculum use the CLAW axis
    (`get_claw_dir`); ref_yaw is pinned to spawn facing under the command policy.
  - Removed for good: `wrist_pointing_error` reward, non-LPF `forward_progress`,
    all `HEADLESS`/`--headless` repo shims. Do not reintroduce.
- Smoke before launching: `../.venv/bin/python tests/helpers/check_command.py`
  (phase-2 wiring) and `tests/helpers/check_dims.py` (phase-1 26-D regression).

## Code style
- No comments unless non-obvious WHY; match existing mdp/cfg conventions.
