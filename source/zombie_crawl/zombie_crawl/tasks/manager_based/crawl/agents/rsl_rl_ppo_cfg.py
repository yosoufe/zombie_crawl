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

# Single source of truth for the runner's rollout length. The training-video timing in
# ``CrawlEnvCfg._setup_training_video`` divides/ multiplies env steps by this SAME value to
# convert between the recorder's raw env-step counter and the TensorBoard iteration axis
# (rsl_rl logs every scalar at ``it``, and ``global_step == it * NUM_STEPS_PER_ENV``). Keep the
# video math importing this constant instead of re-hardcoding 24/40 -- a stale 24 here vs the
# real 40 skewed every training clip ~1.67x relative to TensorBoard.
NUM_STEPS_PER_ENV = 40


@configclass
class _GaussianDistCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """State-DEPENDENT std (phase-3 B-11): the vanilla Gaussian std is one global number
    per action dim, so the gait's exploration (sigma ~1.1 on the +/-1 clip) physically
    prevented rest -- rest needs sigma ~0.05 while crawling needs ~1, in one policy. The
    heteroscedastic variant lets the MLP output a per-state std (state-conditional noise:
    loud during M1/M2 gait, near-deterministic under M4's zero command). ``std_range`` is
    the rsl-rl clamp the Isaac wrapper doesn't expose (passed via ``resolve_class``);
    2.0 bounds the phase-3 noise-exploit (the rest kernel let PPO grow the global std to
    500+). Trained checkpoints are migrated by expanding the final layer (mean rows) and
    initializing the std head at init_std (see ``model_*_hetero.pt``)."""

    # Heteroscedastic was tried (B-11): rsl-rl's PPO let the state-dependent std saturate
    # at the cap within 600 iters (LL gradient on outliers + no std control), and rest
    # still never formed. Plain clamped Gaussian until a std-EMA control exists.
    std_range: tuple[float, float] = (0.1, 2.0)


@configclass
class CrawlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the zombie crawl task."""

    # -- runner --------------------------------------------------------------------
    num_steps_per_env = NUM_STEPS_PER_ENV
    """Steps per environment per rollout (40 x 0.02 s = 0.8 s of experience; lengthened from 24
    to give credit for the turn-then-crawl chain somewhere to propagate)."""
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
        # init_std 0.4 (was 1.0): on the +/-1-clipped action space std=1.0 saturated the clip and,
        # in run 2026-10-09_11-04-48, later blew up 0.08 -> 0.49 at iter ~710 (reward cliff +9 -> -11).
        distribution_cfg=_GaussianDistCfg(init_std=0.6),
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
        # raised 0.001 -> 0.01 to push exploration off the "point-and-sit" zero-reward plateau.
        # The old 0.001 was lowered during the init_std=1.0 blow-up era; with init_std now a sane
        # 0.6 and adaptive-LR KL guard, a standard locomotion entropy bonus is safe and keeps the
        # policy stochastic long enough to discover turning then crawling.
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # "adaptive" (KL guard): rsl-rl 5.5.1 has NO KL early-stop break, so adaptive LR is the only
        # feedback loop that contains an entropy/std blow-up. The old "frozen at ~5e-5" problem came
        # from init_std=1.0 saturating the clip so KL read ~0 and LR decayed to nothing; with a sane
        # init_std the KL signal is meaningful. The 11-04-48 collapse happened because
        # schedule="constant" left the std explosion completely unprotected.
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
