# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zombie crawl task registration for the SO-ARM100 (Isaac Lab 3.0).

Isaac Lab 3.0 plays via a ``play_mode()`` method on the training env cfg (applied by the
unified play entrypoint), so the ``-Play`` task id is a legacy alias kept for
backwards compatibility.
"""

import gymnasium as gym

from . import agents  # noqa: F401
from . import mdp  # noqa: F401


def register_crawl_envs() -> None:
    """Register the zombie crawl gym environments."""
    task_id = "Zombie-Crawl-SO100-v0"
    gym.register(
        id=task_id,
        # CrawlRLEnv forces the render/Fabric settings on for the single world-space
        # headless training-video viewer (no Camera sensor to flip them), so the
        # recorded clip is fresh rather than static.
        entry_point=f"{__name__}.crawl_env:CrawlRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.crawl_env_cfg:CrawlEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrawlPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
    # play variant: same env without curriculum or randomization (deterministic evaluation)
    gym.register(
        id=task_id.replace("-v0", "-Play-v0"),
        entry_point=f"{__name__}.crawl_env:CrawlPlayRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.crawl_env_cfg:CrawlPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrawlPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )


register_crawl_envs()
