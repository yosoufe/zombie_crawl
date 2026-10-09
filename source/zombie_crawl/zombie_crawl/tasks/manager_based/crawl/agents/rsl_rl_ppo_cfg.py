# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the zombie crawl task (rsl-rl 5.x model-config style).

Derived from Isaac Lab's locomotion defaults, scaled down to a small MLP suited to a
6-DOF lightweight arm: 256/128/64 hidden layers with ELU and empirical observation
normalization.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CrawlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the zombie crawl task."""

    # -- runner --------------------------------------------------------------------
    num_steps_per_env = 24
    """Steps per environment per rollout (24 x 0.02 s = 0.48 s of experience)."""
    max_iterations = 3000
    """Maximum training iterations (~1-3 h on a laptop GPU with 2048-4096 envs)."""
    save_interval = 50
    """Checkpoint every N iterations."""
    experiment_name = "zombie_crawl"
    logger = "tensorboard"
    # asymmetric actor-critic: the critic sees the privileged "critic" group in addition
    # to the actor's "policy" group. rsl-rl 5.x keys observation sets "actor"/"critic".
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    clip_actions = 1.0

    # -- networks --------------------------------------------------------------------
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    # -- algorithm ---------------------------------------------------------------------
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # lowered from 0.005: with the old value the action std grew to ~1.7 while the
        # adaptive LR schedule collapsed, so nearly every output saturated the clip and
        # exploration became uninformative noise. 0.001 lets std settle around a sane value.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # "constant" instead of "adaptive": the adaptive schedule (desired_kl=0.01) decayed
        # the LR from 1e-3 to ~5e-5 over the run, which froze learning mid-task. A constant
        # 1e-3 keeps the policy responsive once the motion-gated reward provides a real signal.
        schedule="constant",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
