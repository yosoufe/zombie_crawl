# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest smoke tests for the zombie crawl project.

Each test drives a small headless Isaac Sim subprocess (the environment must boot the
full simulator). The full suite takes several minutes; the training test is the
slowest. Run with:

    pytest tests/ -v --timeout=3000

or individually, e.g.:

    pytest tests/test_smoke.py::test_env_steps_random_actions -v
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS = REPO_ROOT / "tests" / "helpers"
SCRIPTS = REPO_ROOT / "scripts"


def run_script(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a python script with the current interpreter (must be the project venv)."""
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


@pytest.fixture(scope="module")
def venv_python():
    if "env_isaaclab" not in sys.executable and "zombie_crawl" not in sys.executable:
        pytest.fail(f"Run pytest from the project venv, got interpreter: {sys.executable}")
    return sys.executable


# ---------------------------------------------------------------- test 1: env stepping


def test_env_steps_random_actions(venv_python):
    """Env creates and steps 100 iterations with random actions at num_envs=16, no NaNs."""
    proc = run_script(
        [str(HELPERS / "step_random.py"), "--num_envs", "16", "--steps", "100", "--headless"]
    )
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr}"
    assert "[helper] OK" in proc.stdout


# ---------------------------------------------------------------- test 2: dimensions


def test_obs_action_dimensions(venv_python):
    """Observation/action dimensions match the config."""
    proc = run_script([str(HELPERS / "check_dims.py"), "--num_envs", "4", "--headless"])
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr}"
    dims = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"(obs_group \w+|action_dim) (\d+)", line)
        if m:
            dims[m.group(1)] = int(m.group(2))
    assert dims.get("obs_group policy") == 25, f"unexpected policy obs dim: {dims}"
    assert dims.get("obs_group critic") == 5, f"unexpected critic (privileged) obs dim: {dims}"
    assert dims.get("action_dim") == 6, f"unexpected action dim: {dims}"


# ---------------------------------------------------------------- test 3: reset pose


def test_reset_pose(venv_python):
    """Reset places the robot above ground and inside the bounding area."""
    proc = run_script([str(HELPERS / "check_reset.py"), "--num_envs", "32", "--headless"])
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr}"
    assert "[helper] OK" in proc.stdout


# ---------------------------------------------------------------- test 4: short training

SMOKE_LOG_DIR = REPO_ROOT / "logs" / "rsl_rl" / "smoke_test"


def test_short_training_writes_checkpoint(venv_python):
    """A short training run (few iterations) completes and writes a checkpoint."""
    proc = run_script(
        [
            str(SCRIPTS / "train.py"),
            "--task",
            "Zombie-Crawl-SO100-v0",
            "--num_envs",
            "32",
            "--max_iterations",
            "3",
            "--headless",
            "--seed",
            "0",
            "--experiment_name",
            "smoke_test",
            "--run_name",
            "pytest",
        ],
        timeout=2400,
    )
    assert proc.returncode == 0, f"train failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    checkpoints = list(SMOKE_LOG_DIR.rglob("model_*.pt"))
    assert checkpoints, f"no checkpoint found under {SMOKE_LOG_DIR}"


# ---------------------------------------------------------------- test 5: play loads checkpoint


def test_play_loads_checkpoint(venv_python):
    """play.py loads the smoke-test checkpoint and runs 100 steps."""
    checkpoints = sorted(SMOKE_LOG_DIR.rglob("model_*.pt"))
    assert checkpoints, "smoke-test checkpoint missing (did the training test run?)"
    proc = run_script(
        [
            str(SCRIPTS / "play.py"),
            "--task",
            "Zombie-Crawl-SO100-Play-v0",
            "--num_envs",
            "1",
            "--checkpoint",
            str(checkpoints[0]),
            "--max-steps",
            "100",
            "--headless",
        ],
        timeout=1800,
    )
    assert proc.returncode == 0, f"play failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
