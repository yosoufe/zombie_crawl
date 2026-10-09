# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zombie crawl tasks for Isaac Lab (manager-based)."""

from isaaclab_tasks.utils import import_packages

# Blacklist: submodules that are not task registrations (skip them on auto-import).
_BLACKLIST_PKGS = ["utils", ".mdp"]

# The following line auto-imports all submodules (task registrations) in this package
import_packages(__name__, _BLACKLIST_PKGS)
