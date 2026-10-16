# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configuration for the SO-ARM100 (SO-100) free-floating crawler.

The USD asset is produced by ``scripts/asset_pipeline/convert_asset.py`` from the
prepared URDF in ``assets/urdf/so_arm100_float.urdf`` (see
``scripts/asset_pipeline/prepare_urdf.py``).

Actuator presets
----------------
``servo`` (default): approximates a Feetech STS3215-class servo (effort ~2.9 N*m,
velocity ~6 rad/s). ``ideal``: same PD gains with a raised effort limit (10 N*m) to
distinguish task-design limitations from torque limitations.
"""

import os
from dataclasses import field

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# Path to this package (assets live next to this file)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# Isaac Lab 3.0 UrdfConverter output: a composition root with a "Physics" variant set
# (default "physics" = backend-neutral PhysX+Newton articulation).
USD_PATH = os.path.join(_PACKAGE_DIR, "usd", "so_arm100_float", "so_arm100_float.usda")
URDF_PATH = os.path.join(_PACKAGE_DIR, "urdf", "so_arm100_float.urdf")

#: Joint order used throughout the task (matches the articulation joint order).
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: Default joint positions in rad: the URDF zero position = the single folded spawn pose.
#: Anchors joint_pos_rel observations, joint_deviation reward, and the soft-limit window,
#: so spawn / observation-zero / action-zero all agree on one pose.
DEFAULT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def make_actuator_cfg(preset: str = "servo") -> ImplicitActuatorCfg:
    """Return the actuator configuration for the given preset.

    Args:
        preset: Either ``"servo"`` (default) or ``"ideal"``.

    Raises:
        ValueError: If the preset is unknown.
    """
    if preset == "servo":
        effort_limit = 2.9
    elif preset == "ideal":
        effort_limit = 10.0
    else:
        raise ValueError(f"Unknown actuator preset: {preset!r}. Use 'servo' or 'ideal'.")

    return ImplicitActuatorCfg(
        joint_names_expr=[f".*{name}" for name in JOINT_NAMES],
        # PD gains: tuned with a step-response test (see docs/asset_report.md).
        # k ~ (2*pi*2.5 Hz)^2 * I_link-class, d ~ critical damping.
        stiffness=1.0,
        damping=0.1,
        effort_limit=effort_limit,
        # The physics solver must also clip at the actuator limit, otherwise the
        # USD effort limit (35 N*m from the URDF) is used by the engine.
        effort_limit_sim=effort_limit,
        velocity_limit=6.0,
        velocity_limit_sim=6.0,
        armature=0.01,
        friction=0.05,
        viscous_friction=0.02,
    )


@configclass
class SoArm100Cfg(ArticulationCfg):
    """SO-ARM100 with a free-floating base, ready for the crawl task.

    Note: this is a plain ``configclass`` (no custom ``__init__``) so that Isaac Lab's
    ``copy()``/``replace()``/``from_dict()`` keep working (they re-instantiate the cfg
    with every field as a kwarg). The actuator preset is a regular field resolved in
    ``__post_init__``.
    """

    # -- crawl-specific field ------------------------------------------------------
    actuator_preset: str = "servo"
    """One of ``"servo"`` (default) or ``"ideal"``; selects the actuator limits."""

    # -- defaults (class-level; configclass wraps mutables in default_factory) -----
    prim_path: str = "{ENV_REGEX_NS}/Robot"
    spawn = sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ],
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.02),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    )
    # give actuators a default so the dataclass __init__ does not require it;
    # the real value is built from the preset in __post_init__.
    actuators: dict[str, ImplicitActuatorCfg] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Build the actuator configuration from the selected preset.
        self.actuators = {"arm": make_actuator_cfg(self.actuator_preset)}
        # 1.0: the folded spawn sits AT the hard stops of the one-sided joints; shrinking
        # the window would push the spawn pose outside the commanded range entirely.
        self.soft_joint_pos_limit_factor = 1.0
