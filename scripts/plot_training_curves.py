# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot training curves (reward + episode length) from the rsl_rl TensorBoard runs.

Merges every run directory under ``logs/rsl_rl/zombie_crawl/`` (resume runs overlap on
iteration, where the later run wins) and writes two PNGs next to this file's project:

    zombie_crawl/training_reward_curve.png          -- Train/mean_reward
    zombie_crawl/training_episode_length_curve.png  -- Train/mean_episode_length

No Isaac Sim / Kit needed: plain tensorboard + matplotlib.

    python scripts/plot_training_curves.py [--smooth 0.05] [--out DIR]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "rsl_rl", "zombie_crawl")

REWARD_TAG = "Train/mean_reward"
LENGTH_TAG = "Train/mean_episode_length"


def load_runs(log_root: str) -> dict[str, np.ndarray]:
    """Read all scalar series from every run dir, merged by iteration (later run wins)."""
    series: dict[str, dict[int, float]] = {}
    run_dirs = sorted(
        d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))
    )
    for run in run_dirs:
        run_path = os.path.join(log_root, run)
        tfevents = [
            os.path.join(run_path, f)
            for f in os.listdir(run_path)
            if f.startswith("events.out.tfevents")
        ]
        if not tfevents:
            continue
        ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
        ea.Reload()
        print(f"[info] run {run}: {len(tfevents)} tfevent file(s), tags={len(ea.Tags()['scalars'])}")
        for tag in ea.Tags()["scalars"]:
            if tag.endswith("/time"):
                continue
            buf = series.setdefault(tag, {})
            for ev in ea.Scalars(tag):
                buf[ev.step] = ev.value
    return {tag: buf for tag, buf in series.items()}


def to_array(buf: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    it = np.array(sorted(buf), dtype=np.int64)
    v = np.array([buf[i] for i in it], dtype=np.float64)
    return it, v


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def plot_scalar(
    it: np.ndarray,
    v: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    smooth_alpha: float,
    hline: float | None = None,
    hline_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    ax.plot(it, v, color="#b8b8b8", linewidth=0.6, label="raw (per-iteration mean)")
    s = ema(v, smooth_alpha)
    ax.plot(it, s, color="#1f77b4", linewidth=1.8, label=f"EMA (α={smooth_alpha})")
    if hline is not None:
        ax.axhline(hline, color="#d62728", linestyle="--", linewidth=1.0, label=hline_label)
    peak_i = int(np.argmax(s))
    ax.annotate(
        f"max {s[peak_i]:.2f}\nit {it[peak_i]}",
        xy=(it[peak_i], s[peak_i]),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
    )
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=float, default=0.05, help="EMA alpha for smoothing")
    ap.add_argument("--out", type=str, default=PROJECT_ROOT, help="output directory for PNGs")
    args = ap.parse_args()

    series = load_runs(LOG_ROOT)

    if REWARD_TAG not in series:
        raise SystemExit(f"{REWARD_TAG} not found in any run under {LOG_ROOT}")
    it_r, v_r = to_array(series[REWARD_TAG])
    it_l, v_l = to_array(series[LENGTH_TAG])

    plot_scalar(
        it_r,
        v_r,
        "Zombie Crawl — mean episode reward during PPO training",
        "mean episode reward",
        os.path.join(args.out, "training_reward_curve.png"),
        args.smooth,
    )
    plot_scalar(
        it_l,
        v_l,
        "Zombie Crawl — mean episode (session) length during PPO training",
        "mean episode length [steps]",
        os.path.join(args.out, "training_episode_length_curve.png"),
        args.smooth,
        hline=1000.0,
        hline_label="episode max = 1000 steps (20 s)",
    )

    # compact numeric summary for the console
    s_r, s_l = ema(v_r, args.smooth), ema(v_l, args.smooth)
    for label, it, s in (("reward", it_r, s_r), ("length", it_l, s_l)):
        idx = [0, len(s) // 4, len(s) // 2, 3 * len(s) // 4, len(s) - 1]
        print(f"[summary] {label}: " + "  ".join(f"it{it[i]}={s[i]:.3f}" for i in idx))


if __name__ == "__main__":
    main()
