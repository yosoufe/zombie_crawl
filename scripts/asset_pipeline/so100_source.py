# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate the SO-ARM100 upstream clone used by the asset-pipeline scripts.

Resolution order: ``SO_ARM100_DIR`` env var, else ``<repo>/../third_party/SO-ARM100``
(clone it from the upstream project; the converted assets are already committed, so
only re-running the URDF -> USD pipeline needs the clone).
"""

import os

UPSTREAM_URL = "https://github.com/TheRobotStudio/SO-ARM100.git"


def repo_root() -> str:
    """Repo root, resolved from this file (scripts/asset_pipeline/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def so100_dir() -> str:
    """Where the SO-ARM100 clone is expected (not checked)."""
    return os.environ.get("SO_ARM100_DIR", os.path.join(repo_root(), "..", "third_party", "SO-ARM100"))


def require_so100() -> str:
    """Like :func:`so100_dir` but exits with clone instructions when missing."""
    d = so100_dir()
    if not os.path.isdir(d):
        raise SystemExit(
            f"[so100] SO-ARM100 clone not found at: {d}\n"
            f"[so100] Clone it there (or set SO_ARM100_DIR to an existing clone):\n"
            f"[so100]   mkdir -p '{os.path.dirname(d)}' && git clone {UPSTREAM_URL} '{d}'"
        )
    return d
