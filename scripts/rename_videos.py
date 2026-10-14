# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTERS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rename recorded training clips so their filenames carry the TensorBoard iteration/step range.

The live recorder bakes ``train_overview_it{start}-{end}_step{start}-{end}_{index}.mp4`` into NEW
clips (see ``streaming_recorder.py``). This script retro-fits the same names onto clips that were
recorded by an OLDER run (before that naming existed, or a process still running old code), using
the timing the run itself recorded:

  * ``params/agent.yaml``  -> num_steps_per_env (the real per-iteration step count)
  * ``params/env.yaml``    -> the recorder's step_offset / video_interval / video_length / prefix
  * the run's ``events.*`` -> the minimum logged step == the iteration this run's TensorBoard axis
    starts at (0 for a from-scratch run, the checkpoint's iter for a ``--checkpoint`` resume).

Clip ``k`` (the 4-digit index in the old name) triggered at recorder env-step
``step_offset + 1 + k * video_interval`` and lasted ``video_length`` steps, so it corresponds to
TensorBoard iterations ``[offset + start_step / nsp, offset + end_step / nsp]``.

Usage:
    python scripts/rename_videos.py <run_dir> [--prefix train_overview] [--start-iter N] [--dry-run]

Safe: never deletes or overwrites; refuses to clobber an existing target; idempotent (already
renamed clips are skipped). Pass --dry-run to preview.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import yaml


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that renders python-object tags (tuples, etc.) as plain lists/scalars.

    The dumped ``params/*.yaml`` files carry ``!!python/tuple`` / ``python/object`` tags that the
    default safe loader refuses. We only read plain scalar fields (offsets, lengths, names), so
    collapsing those tags to generic containers is both sufficient and avoids arbitrary-object
    construction.
    """


def _construct_unknown(loader, _suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_unknown)


def _find_recorder(obj, prefix: str):
    """Depth-first search a serialized cfg for the video-recorder dict matching ``prefix``."""
    if isinstance(obj, dict):
        if (
            obj.get("output_filename_prefix") == prefix
            and "step_offset" in obj
            and "video_interval" in obj
            and "video_length" in obj
        ):
            return obj
        for value in obj.values():
            found = _find_recorder(value, prefix)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_recorder(value, prefix)
            if found is not None:
                return found
    return None


def _load_yaml(path: str):
    with open(path) as fh:
        return yaml.load(fh, Loader=_TolerantLoader)


def _num_steps_per_env(run_dir: str) -> int | None:
    agent = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent):
        return None
    data = _load_yaml(agent) or {}
    nsp = data.get("num_steps_per_env")
    return int(nsp) if nsp else None


def _min_logged_step(run_dir: str) -> int | None:
    """Minimum scalar step across the run's TensorBoard event files == its starting iteration."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return None
    try:
        ea = event_accumulator.EventAccumulator(run_dir, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        steps = [s.step for tag in ea.Tags().get("scalars", []) for s in ea.Scalars(tag)]
        return min(steps) if steps else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Training run dir (contains params/ and videos/train/).")
    ap.add_argument("--prefix", default="train_overview", help="Recorder output_filename_prefix.")
    ap.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Override the TensorBoard start iteration (resume offset). Default: min step from the "
        "run's event files; falls back to 0 if unreadable.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview renames without touching files.")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    video_dir = os.path.join(run_dir, "videos", "train")
    if not os.path.isdir(video_dir):
        print(f"[ERR] no videos/train under {run_dir}", file=sys.stderr)
        return 1

    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        print(f"[ERR] missing {env_yaml}", file=sys.stderr)
        return 1
    rec = _find_recorder(_load_yaml(env_yaml), args.prefix)
    if rec is None:
        print(f"[ERR] no recorder with prefix={args.prefix!r} in {env_yaml}", file=sys.stderr)
        return 1

    nsp = _num_steps_per_env(run_dir)
    if not nsp:
        print("[ERR] could not read num_steps_per_env from params/agent.yaml", file=sys.stderr)
        return 1
    step_offset = int(rec["step_offset"])
    video_interval = int(rec["video_interval"])
    video_length = int(rec["video_length"])

    start_iter = args.start_iter
    if start_iter is None:
        start_iter = _min_logged_step(run_dir)
        if start_iter is None:
            print("[WARN] could not read event files; assuming start-iter 0 (from-scratch).")
            start_iter = 0
    print(
        f"[INFO] {video_dir}\n       nsp={nsp} step_offset={step_offset} interval={video_interval} "
        f"length={video_length} start_iter={start_iter}"
    )

    # old plain name: <prefix>_<index>.mp4  (index = the clip counter recorded by the old code)
    old_pat = re.compile(rf"^{re.escape(args.prefix)}_(?P<index>\d+)\.mp4$")
    renamed = 0
    for fn in sorted(os.listdir(video_dir)):
        m = old_pat.match(fn)
        if not m:
            continue  # already extended / unrelated -> idempotent skip
        k = int(m.group("index"))
        start_step = step_offset + 1 + k * video_interval
        end_step = start_step + video_length
        si, ei = start_iter + start_step // nsp, start_iter + end_step // nsp
        label = f"it{si:05d}-{ei:05d}_step{start_step:07d}-{end_step:07d}_{k:04d}"
        new_fn = f"{args.prefix}_{label}.mp4"
        src, dst = os.path.join(video_dir, fn), os.path.join(video_dir, new_fn)
        if os.path.exists(dst):
            print(f"[SKIP] target exists: {new_fn}")
            continue
        print(f"[RENAME] {fn}  ->  {new_fn}   (it {si}-{ei})")
        if not args.dry_run:
            os.rename(src, dst)
        renamed += 1

    print(f"[DONE] {'would rename' if args.dry_run else 'renamed'} {renamed} clip(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
