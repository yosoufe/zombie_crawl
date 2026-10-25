# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fast unit tests for the phase-2 mdp terms (no Isaac Sim boot).

The mdp modules import torch-only + isaaclab config classes, so pure-torch tests with
stub envs run in seconds. The spec-mandated cases (A.14.7):

* ``claw_heading_rate``: known ramp (including the +-pi wrap), oscillation + small mean
  recovers the mean, arrow-validity flip produces no spike, reset reseeds;
* ``forward_progress`` axis-free: rotating the axis without moving the base yields ~0;
* ``stalled`` Euclidean test;
* command-sampling statistics (zero fraction ~ 0.2, range obeys yaw_rate_max).

Run: ``../.venv/bin/python -m pytest tests/test_unit.py -v``
"""

import math

import torch

from zombie_crawl.tasks.manager_based.crawl.mdp import commands, rewards, terminations

DEVICE = "cpu"


class _Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class _FindBodiesResult:
    def __init__(self, ids):
        self.torch = ids


class _FakeData:
    def __init__(self, body_pos_w, root_pos_w, root_quat_w, root_lin_vel_w=None):
        # production layout: (num_envs, num_bodies, 3) -- single wrist body here
        self.body_pos_w = _Proxy(body_pos_w.unsqueeze(1))
        self.root_pos_w = _Proxy(root_pos_w)
        self.root_quat_w = _Proxy(root_quat_w)
        self.root_lin_vel_w = _Proxy(
            root_lin_vel_w if root_lin_vel_w is not None else torch.zeros_like(root_pos_w)
        )
        self.root_ang_vel_w = _Proxy(torch.zeros_like(root_pos_w))
        self.projected_gravity_b = _Proxy(torch.tensor([[0.0, 0.0, -1.0]]).repeat(root_pos_w.shape[0], 1))


class _FakeAsset:
    def __init__(self, data):
        self.data = data

    def find_bodies(self, names, as_proxy=True):
        return _FindBodiesResult(torch.tensor([0])), ["wrist"]


class _FakeScene(dict):
    pass


class _FakeCfg:
    v_target = 0.10
    decimation = 4
    use_curriculum = False

    class _Sim:
        dt = 0.005

    sim = _Sim()


class _FakeEnv:
    """Minimal env stand-in for the mdp functions under test."""

    def __init__(self, num_envs=1, claw_yaw=0.0, base_pos=None, quat=None, lin_vel=None):
        self.num_envs = num_envs
        self.device = DEVICE
        self.extras = {}
        self.command_manager = None
        self.cfg = _FakeCfg()
        self.step_dt = 0.02
        n = num_envs
        base = base_pos if base_pos is not None else torch.zeros(n, 3)
        # wrist at 0.1 m along the (horizontal) claw arrow
        wrist = base + torch.stack(
            [0.1 * torch.cos(torch.as_tensor(claw_yaw, dtype=torch.float32).expand(n)),
             0.1 * torch.sin(torch.as_tensor(claw_yaw, dtype=torch.float32).expand(n)),
             torch.zeros(n)],
            dim=-1,
        )
        q = quat if quat is not None else torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)
        self.scene = _FakeScene(robot=_FakeAsset(_FakeData(wrist, base, q, lin_vel)))
        self._base = base
        self._wrist = wrist

    # -- live mutation helpers (write into the SAME tensors the data proxies hold) ----
    def set_claw_yaw(self, yaw: torch.Tensor | float, valid: bool | torch.Tensor = True):
        yaw = torch.as_tensor(yaw, dtype=torch.float32).reshape(-1)
        if yaw.numel() == 1:
            yaw = yaw.repeat(self.num_envs)
        r = 0.1 * (
            torch.ones(self.num_envs)
            if isinstance(valid, bool) and valid
            else (torch.ones(self.num_envs) * (1.0 if valid is True else 0.0))
        )
        if not isinstance(valid, bool):
            r = torch.where(valid, torch.ones(self.num_envs), torch.full((self.num_envs,), 0.005))
        self._wrist[:, 0] = self._base[:, 0] + r * torch.cos(yaw)
        self._wrist[:, 1] = self._base[:, 1] + r * torch.sin(yaw)

    def set_base_xy(self, xy: torch.Tensor):
        # the wrist is a body rigidly attached: it must translate WITH the base,
        # otherwise the claw arrow spuriously flips backwards
        self._wrist[:, 0] += xy[:, 0] - self._base[:, 0]
        self._wrist[:, 1] += xy[:, 1] - self._base[:, 1]
        self._base[:, 0] = xy[:, 0]
        self._base[:, 1] = xy[:, 1]


# ---------------------------------------------------------------- claw_heading_rate ---


def _ramp_env(n=1):
    env = _FakeEnv(num_envs=n)
    return env


def test_claw_heading_rate_known_ramp_with_wrap():
    """Constant +0.1 rad/s claw heading, crossing +-pi: the EMA converges to the ramp rate."""
    dt, tau = 0.02, 1.5
    env = _ramp_env()
    psi0 = math.pi - 0.2  # cross +pi during the run
    psi = psi0
    rewards._update_claw_heading_rate(env, dt, tau)  # init
    for _ in range(1000):  # 20 s at 0.1 rad/s -> 2 rad of heading (wraps +pi)
        psi += 0.1 * dt
        env.set_claw_yaw(torch.tensor(psi))
        rewards._update_claw_heading_rate(env, dt, tau)
    assert abs(float(env.zombie_r_ema) - 0.1) < 5e-3, f"r={float(env.zombie_r_ema)}"


def test_claw_heading_rate_oscillation_recovers_mean():
    """Gait sway (oscillation) on top of a slow mean rotation: r recovers the mean rate."""
    dt, tau = 0.02, 1.5
    env = _ramp_env()
    rewards._update_claw_heading_rate(env, dt, tau)
    drift = 0.04
    for k in range(4000):
        psi = drift * (k * dt) + 0.3 * math.sin(2.0 * math.pi * 1.0 * (k * dt))
        env.set_claw_yaw(torch.tensor(psi))
        rewards._update_claw_heading_rate(env, dt, tau)
    assert abs(float(env.zombie_r_ema) - drift) < 0.02, f"r={float(env.zombie_r_ema)}"


def test_claw_heading_rate_arrow_flip_no_spike():
    """A validity flip (arm curl snapping to the base-axis fallback) must not spike r."""
    dt, tau = 0.02, 1.5
    env = _ramp_env()
    rewards._update_claw_heading_rate(env, dt, tau)
    psi = 0.0
    for _ in range(300):  # settle r ~ 0
        env.set_claw_yaw(torch.tensor(psi))
        rewards._update_claw_heading_rate(env, dt, tau)
    before = float(env.zombie_r_ema)
    # snap: invalid one step (fallback axis yaw-90deg ~ 0 here), then valid again
    env.set_claw_yaw(torch.tensor(psi), valid=torch.tensor([False]))
    rewards._update_claw_heading_rate(env, dt, tau)
    env.set_claw_yaw(torch.tensor(psi + 2.0), valid=torch.tensor([True]))  # huge apparent jump
    rewards._update_claw_heading_rate(env, dt, tau)
    after = float(env.zombie_r_ema)
    assert abs(after - before) < 0.05, f"spike: {before} -> {after}"


def test_claw_heading_rate_reset_reseeds():
    env = _ramp_env()
    dt, tau = 0.02, 1.5
    rewards._update_claw_heading_rate(env, dt, tau)
    for _ in range(200):
        env.set_claw_yaw(torch.tensor(0.1 * 200 * dt))
        rewards._update_claw_heading_rate(env, dt, tau)
    assert float(env.zombie_r_ema) != 0.0
    env.zombie_time_since_resample[:] = 99.0
    env.zombie_rate_reseed[:] = True
    rewards._update_claw_heading_rate(env, dt, tau)
    assert float(env.zombie_r_ema) == 0.0
    assert abs(float(env.zombie_time_since_resample) - dt) < 1e-6  # timer restarted at reset


# ------------------------------------------------------------- forward_progress -----


def test_forward_progress_axis_free_under_rotation():
    """Rotating the claw axis while the base does NOT move yields ~0 progress (pitfall X3).

    The old axis-projected implementation would fake huge progress here whenever the axis
    rotated while the (origin-offset) absolute position was projected on it.
    """
    dt, tau_s = 0.02, 1.0
    env = _ramp_env()
    env.v_target = torch.full((env.num_envs,), 0.10)
    psi = 0.0
    out = 0.0
    for k in range(1500):  # 30 s: spin the axis at 1 rad/s with the base parked
        psi += 1.0 * dt
        env.set_claw_yaw(torch.tensor(psi))
        out = float(rewards.forward_progress_lpf(env, tau_s=tau_s))
    assert abs(out) < 0.15, f"axis-free progress leaked: {out}"  # normalized by v_target=0.1
    # and a genuinely moving base along the CURRENT axis scores positive progress
    env2 = _ramp_env()
    env2.v_target = torch.full((env2.num_envs,), 0.10)
    x = 0.0
    for _ in range(1500):
        x += 0.06 * dt  # 0.06 m/s along +x, claw fixed at psi=0
        env2.set_base_xy(torch.tensor([[x, 0.0]]))
        val = float(rewards.forward_progress_lpf(env2, tau_s=tau_s))
    assert 0.4 < val <= 1.2, f"forward crawl progress wrong: {val}"


# ------------------------------------------------------------------------- stalled ----


def _stall_env(n=1):
    env = _ramp_env(n)
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env.zombie_stall_anchor_xy = torch.zeros(n, 2)
    env.zombie_stall_anchor_step = torch.zeros(n, dtype=torch.long)
    return env


def test_stalled_euclidean():
    dt_policy = 0.02  # decimation 4 x sim dt 0.005
    # (a) parked -> terminates past 12 s
    env = _stall_env()
    for step in range(1, 700):
        env.episode_length_buf[:] = step
        done = terminations.stalled(env, min_displacement=0.05, stall_time_s=12.0)
    assert bool(done[0])
    # (b) genuine crawl -> re-anchors, never terminates
    env = _stall_env()
    for step in range(1, 700):
        env.set_base_xy(torch.tensor([[0.02 * step * dt_policy * 100, 0.0]]))  # 2 m/s
        env.episode_length_buf[:] = step
        done = terminations.stalled(env, min_displacement=0.05, stall_time_s=12.0)
        assert not bool(done[0]), "moving robot was stalled"
    # (c) ran far and came back to the anchor -> still a stall (Euclidean, not path length)
    env = _stall_env()
    for step in range(1, 400):  # 8 s out
        env.set_base_xy(torch.tensor([[0.5 * step * dt_policy, 0.3 * step * dt_policy]]))
        env.episode_length_buf[:] = step
        terminations.stalled(env)
    for step in range(400, 900):  # come back to the anchor, total elapsed > 12 s
        env.set_base_xy(torch.tensor([[0.0, 0.0]]))
        env.episode_length_buf[:] = step
        done = terminations.stalled(env)
    assert bool(done[0])
    # (d) movement perpendicular to the ORIGINAL spawn axis still counts (axis-free)
    env = _stall_env()
    for step in range(1, 700):
        env.set_base_xy(torch.tensor([[0.0, 0.5 * step * dt_policy]]))
        env.episode_length_buf[:] = step
        done = terminations.stalled(env)
    assert not bool(done[0])


# ------------------------------------------------------------------ command term ------


class _CmdEnv(_FakeEnv):
    def __init__(self, num_envs, extras):
        super().__init__(num_envs=num_envs)
        self.extras = extras


def test_command_sampling_statistics():
    torch.manual_seed(0)
    cfg = commands.UniformYawRateCommandCfg(resampling_time_range=(5.0, 8.0))
    term = commands.UniformYawRateCommand(cfg, _CmdEnv(20000, {"yaw_rate_max": 0.3}))
    term._resample_command(slice(None))
    cmd = term.yaw_rate
    zero_frac = float((cmd == 0).double().mean())
    assert abs(zero_frac - 0.2) < 0.03, f"zero fraction {zero_frac}"
    assert float(cmd.abs().max()) <= 0.3 + 1e-6, "range violated yaw_rate_max"
    assert float((cmd > 0.05).double().mean()) > 0.3 and float((cmd < -0.05).double().mean()) > 0.3
    assert bool(term.just_resampled.all())
    # yaw_rate_max == 0 -> every command exactly 0 (steering stage locked)
    term0 = commands.UniformYawRateCommand(cfg, _CmdEnv(4096, {}))
    term0._resample_command(slice(None))
    assert bool((term0.yaw_rate == 0).all())


def test_command_pin_freezes_resampling():
    cfg = commands.UniformYawRateCommandCfg(resampling_time_range=(5.0, 8.0))
    env = _CmdEnv(64, {"yaw_rate_max": 0.3})
    env.step_dt = 0.02
    term = commands.UniformYawRateCommand(cfg, env)
    term.pin(-0.1)
    for _ in range(50):
        term._resample_command(slice(None))
        term.compute(0.02)
    assert bool((term.yaw_rate == -0.1).all())
    term.unpin()
    term._resample_command(slice(None))
    assert not bool((term.yaw_rate == -0.1).all()) or float(term.yaw_rate.abs().max()) > 0


# ------------------------------------------------------------------ episode scoring ---


def test_score_finished_episodes_mixed_fallen():
    """Regression (crashed R1 at iter 204): mixed upright/fallen finished episodes must
    score crawl_success + recovery_success + time_to_upright without shape errors."""
    from zombie_crawl.tasks.manager_based.crawl.mdp import metrics

    env = _ramp_env(n=8)
    env.episode_length_buf = torch.zeros(8, dtype=torch.long)
    env.v_target = torch.full((8,), 0.10)
    # envs 0-3: upright starts crawling at 2x v_target; 4-6: fallen, recovered at 5 s;
    # env 7: fallen, never recovered
    st = metrics._state(env, 8, DEVICE)
    metrics.set_started_fallen(env, torch.arange(8), torch.tensor([False] * 4 + [True] * 4))
    st["v_c_int"][:4] = 0.20 * 20.0
    st["upright_t"] = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, -1.0])
    st["steps"][:] = 1000.0
    metrics.score_finished_episodes(env, slice(None))
    m = env.zombie_metrics
    assert m["crawl_success"] == 1.0
    assert abs(m["recovery_success"] - 0.75) < 1e-6
    assert abs(m["time_to_upright"] - 5.0) < 1e-6


def test_note_step_derives_fallen_from_gravity():
    """Phase-1 backport path: without set_started_fallen, note_step derives started_fallen
    from the gravity vector at the first step (el <= 1)."""
    from zombie_crawl.tasks.manager_based.crawl.mdp import metrics

    env = _ramp_env(n=4)
    env.episode_length_buf = torch.ones(4, dtype=torch.long)  # first policy step
    data = env.scene["robot"].data
    # envs 0-1 flat (-g_z = 1), envs 2-3 on their side (-g_z ~ 0)
    data.projected_gravity_b.torch[:] = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    metrics.note_step(env, dt=0.02)
    st = env.zombie_metrics_state
    assert bool((~st["started_fallen"][:2]).all())
    assert bool(st["started_fallen"][2:].all())
    assert bool(st["fallen_known"].all())


def test_score_segments_accepts_bool_mask():
    """Regression (crashed R2 at iter 261): _score_segments must accept the boolean
    resampled mask note_step passes, not only a resolved index tensor."""
    from zombie_crawl.tasks.manager_based.crawl.mdp import metrics

    env = _ramp_env(n=8)
    env.step_dt = 0.02
    st = metrics._state(env, 8, DEVICE)
    st["seg_cmd"][:] = 0.1
    st["seg_rsum"][:] = 0.1 * 200
    st["seg_rn"][:] = 200.0
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:6] = True
    metrics._score_segments(env, st, mask, failed=None)
    assert bool(st["seg_scored"][:6].all())
    assert not bool(st["seg_scored"][6:].any())
    assert env.zombie_metrics["steer_success"] == 1.0
