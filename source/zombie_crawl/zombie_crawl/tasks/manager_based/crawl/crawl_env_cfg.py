# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM100 zombie crawl task (manager-based).

Tunables (also overridable with Hydra):
    actuator_preset          "servo" (default) | "ideal"
    v_target                 final forward velocity target [m/s] (0.05-0.20)
    v_target_start           curriculum starting target [m/s]
    use_curriculum           enable v_target + friction curriculum
    use_velocity_commands    velocity-command tracking stub (phase 2, off)
    use_gait_clock           append sinusoidal gait-phase clock to actor obs (off)
    num_envs / env_spacing   parallelism & env isolation
"""

import math

import torch

from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import DomeLightCfg, GroundPlaneCfg, PinholeCameraCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from zombie_crawl.assets import SoArm100Cfg
from zombie_crawl.tasks.manager_based.crawl import mdp
from zombie_crawl.tasks.manager_based.crawl import video_flags


##
# Start poses (named). Poses are given in the base link frame convention: the base plate
# bottom face is z = 0 and spans x in [-0.0555, 0.0555], y in [-0.071, 0.0249].
##

START_POSES: dict[str, dict] = {
    "folded": {
        "pos": (0.0, 0.0, 0.005),
        "rot": (0.0, 0.0, 0.0),
        "joint_pos": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.2,
            "elbow_flex": -1.6,
            "wrist_flex": 0.3,
            "wrist_roll": 0.0,
            "gripper": 0.3,
        },
    },
    "extended": {
        "pos": (0.0, 0.0, 0.005),
        "rot": (0.0, 0.0, 0.0),
        "joint_pos": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.35,
            "elbow_flex": -0.35,
            "wrist_flex": 0.2,
            "wrist_roll": 0.0,
            "gripper": 0.3,
        },
    },
    "on_side": {
        "pos": (0.0, 0.0, 0.048),
        "rot": (math.pi / 2.0, 0.0, 0.0),
        "joint_pos": {
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.0,
            "elbow_flex": -1.4,
            "wrist_flex": 0.3,
            "wrist_roll": 0.0,
            "gripper": 0.3,
        },
    },
}


@configclass
class CrawlSceneCfg(InteractiveSceneCfg):
    """Flat ground + free-floating SO-ARM100 (robot set in CrawlEnvCfg.__post_init__)."""

    num_envs: int = 2048
    env_spacing: float = 4.0

    # ground plane; friction combine mode "multiply" so the per-env robot friction
    # multiplier (randomize_ground_friction event) scales the effective pair friction
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            )
        ),
    )

    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=DomeLightCfg(intensity=750.0))

    robot: SoArm100Cfg = None

    # Video cameras (one instance per env; each frames its own env's robot, and the
    # video recorders read env_0). Populated only when video recording is enabled, see
    # CrawlEnvCfg._setup_video_recording. They are the ``--video`` source for play.py.
    video_camera: CameraCfg = None
    video_camera_top: CameraCfg = None

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        max_contact_data_count_per_prim=32,
    )


@configclass
class CrawlCommandsCfg:
    """Velocity command term -- only active when ``use_velocity_commands`` is set (phase 2 stub)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.05, 0.15),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CrawlActionsCfg:
    """6-D joint position targets via the limits-relative EMA mapping.

    ``EMAJointPositionToLimitsAction`` maps a raw action in [-1, +1] onto each joint's own
    (soft) position limits: -1 = low soft limit, +1 = high soft limit. With the asset's
    ``soft_joint_pos_limit_factor = 0.95`` the reachable range is the middle 95% of each
    joint's hard range (shrunk symmetrically about the range midpoint) -- the full measured
    range of motion, minus a 5% safety margin that also matches the ``joint_limits`` reward.
    This replaces the old fixed +/-0.5 rad
    window around the default pose, which covered only 16-45% of each joint's range.
    The EMA (alpha=0.3) smooths the commanded targets; on reset the EMA state is
    initialized to the current joint positions, so episodes do not start with a jump.
    """

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        rescale_to_limits=True,
        alpha=0.3,
    )


@configclass
class CrawlObservationsCfg:
    """Actor observations (hardware-plausible) + privileged critic extras."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: 25-D (27-D with the optional gait clock)."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # optional metronome (removed in __post_init__ unless use_gait_clock)
        gait_clock = ObsTerm(func=mdp.gait_phase_clock, params={"frequency": 0.5})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged extras: 5-D (concatenated with the policy group by the runner)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=mdp.base_height)
        ground_friction = ObsTerm(func=mdp.ground_friction)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CrawlRewardsCfg:
    """Reward terms. Weights are starting points (see docs/training_report.md for tuning notes)."""

    # main: steady forward motion (gated on actual motion: no reward below 30% of v_target).
    # phase 2: tanh velocity reward (unsaturated gradient at low speed) + 1 Hz low-passed
    # displacement term at weight 1.0, replacing the saturated bell and the token 0.1 progress term.
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_exp, weight=1.5, params={"min_vel_fraction": 0.3}
    )
    forward_progress = RewTerm(func=mdp.forward_progress_lpf, weight=1.0, params={"tau_s": 1.0})
    # keep heading along +X and stay on the line
    yaw_error = RewTerm(func=mdp.yaw_error, weight=-0.5)
    lateral_velocity = RewTerm(func=mdp.lateral_velocity, weight=-1.0)
    # anti-exploit terms
    overspeed = RewTerm(func=mdp.overspeed_penalty, weight=-2.0, params={"factor": 2.0})
    vertical_velocity = RewTerm(func=mdp.vertical_velocity_l2, weight=-2.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.06})
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # smooth, cheap, joint-friendly motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-4)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_deviation = RewTerm(func=mdp.joint_deviation_l2, weight=-1.0e-3)
    # survival
    alive = RewTerm(func=mdp.is_alive, weight=0.05)  # phase 2: halved (flat value contrast problem)
    # optional zombie-style terms (disabled by default; enable by setting weight > 0)
    low_ground_contact = RewTerm(func=mdp.low_ground_contact_bonus, weight=0.0)
    gait_asymmetry = RewTerm(func=mdp.gait_asymmetry_bonus, weight=0.0)


@configclass
class CrawlTerminationsCfg:
    """Termination: only failure / exploit conditions. Tipping over is *allowed*."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_nan = DoneTerm(func=mdp.base_nan)
    base_too_high = DoneTerm(func=mdp.base_too_high, params={"height": 0.45})
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds, params={"x_range": 2.5, "y_range": 1.5})
    # "sit still" exploit guard: after 12 s, an episode that has not crawled 5 cm along +X
    # is ended as a FAILURE (no bootstrap), so doing nothing is no longer a viable strategy
    stalled = DoneTerm(func=mdp.stalled, params={"min_displacement": 0.05, "stall_time_s": 12.0})


@configclass
class CrawlEventsCfg:
    """Reset and randomization events."""

    # every reset: restore the scene (ground, lights) to default
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # every reset: place the robot into a named start pose + randomization
    reset_start_pose = EventTerm(
        func=mdp.reset_start_pose,
        mode="reset",
        params={
            "poses": START_POSES,
            # "on_side" (index 2) starts the robot tipped over and made reorientation the
            # dominant early task (yaw_error stayed ~-0.5 for the whole run); drop it and
            # weight "folded" 2:1 over "extended"
            "pose_weights": (2.0, 1.0, 0.0),
            "joint_noise": 0.1,
            "xy_offset": 0.05,
            # +-pi/2 instead of +-pi: a 180-degree turn-off is too costly early in training
            "yaw_range": (-math.pi / 2, math.pi / 2),
        },
    )
    # once at startup: per-env ground friction multiplier (effective 0.5-1.2)
    randomize_ground_friction = EventTerm(
        func=mdp.randomize_ground_friction,
        mode="startup",
        params={"friction_range": (0.625, 1.5), "ground_static_friction": 0.8},
    )
    # curriculum: apply the current friction multiplier to the physics materials
    # (only kept when use_curriculum is enabled)
    apply_friction_scale = EventTerm(
        func=mdp.apply_friction_scale,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={},
    )
    # off by default: push perturbations (future real-robot phase)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
        },
    )
    # off by default: actuator gain randomization (experimental)
    # note: base-mass randomization from the 2.x config was dropped -- the 3.0
    # ``randomize_rigid_body_mass`` term requires distribution params that the
    # degenerate (0.0, 0.0) "disabled" form does not satisfy.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "stiffness_range": (1.0, 1.0),
            "damping_range": (0.1, 0.1),
        },
    )


@configclass
class CrawlCurriculumCfg:
    """Curriculum: ease v_target in and decay the extra friction multiplier."""

    v_target = CurrTerm(
        func=mdp.v_target_curriculum,
        params={
            "v_start": 0.06,
            "v_final": 0.10,
            "step_size": 0.005,
            "min_fraction": 0.6,
            "min_episode_fraction": 0.5,
            "check_interval": 100,
        },
    )
    friction_scale = CurrTerm(
        func=mdp.friction_curriculum,
        params={
            "scale_start": 1.5,
            "scale_final": 1.0,
            "step_size": 0.05,
            "min_fraction": 0.6,
            "check_interval": 200,
        },
    )


@configclass
class CrawlEnvCfg(ManagerBasedRLEnvCfg):
    """Zombie crawl: drag a free-floating SO-ARM100 forward at ~0.1 m/s."""

    # -- simulation ---------------------------------------------------------------
    # Isaac Lab 3.0: physics backend config moved from ``sim.physx`` to ``sim.physics``.
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        device="cuda:0",
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    # -- managers ------------------------------------------------------------------
    scene: CrawlSceneCfg = CrawlSceneCfg()
    commands: CrawlCommandsCfg = None
    actions: CrawlActionsCfg = CrawlActionsCfg()
    observations: CrawlObservationsCfg = CrawlObservationsCfg()
    rewards: CrawlRewardsCfg = CrawlRewardsCfg()
    terminations: CrawlTerminationsCfg = CrawlTerminationsCfg()
    events: CrawlEventsCfg = CrawlEventsCfg()
    curriculum: CrawlCurriculumCfg = CrawlCurriculumCfg()

    # -- tunables -------------------------------------------------------------------
    actuator_preset: str = "servo"
    """Actuator preset: "servo" (2.9 N*m, default) or "ideal" (10 N*m)."""
    v_target: float = 0.10
    """Final forward velocity target [m/s]. Configurable range 0.05-0.20."""
    v_target_start: float = 0.06
    """Curriculum starting forward velocity target [m/s]."""
    use_curriculum: bool = True
    """Enable the v_target + friction curriculum."""
    use_velocity_commands: bool = False
    """Phase-2 stub: track commanded (vx, vy, yaw_rate) instead of a fixed v_target."""
    use_gait_clock: bool = False
    """Append a sinusoidal gait-phase clock to the actor observations."""
    num_envs: int = 2048
    """Number of parallel environments."""
    env_spacing: float = 1.5
    """Distance between environment origins [m]. Reduced from 4.0 so a top-down overview
    camera frames a cluster of ~4 robots at once (training video). Robots (~0.3 m) stay
    well clear of neighbours at this spacing."""

    record_video: bool = False
    """When True, add the scene video cameras and sensor-based VideoRecorderCfg entries.

    Set by the project scripts (``play.py`` / ``train.py``) from the ``--video`` flag
    before the env config is instantiated (see ``video_flags``); the unified RL
    entrypoints build the config internally, so this is a process-local flag rather
    than a Hydra field.
    """
    video_width: int = 1280
    """Video camera image width [px]."""
    video_height: int = 720
    """Video camera image height [px]."""

    record_training_video: bool = False
    """When True, record live training videos from a SINGLE world-space viewer camera
    (one ``KitVisualizer`` with ``origin_type="world"`` that frames the grid centre, so a
    handful of robots are visible at once) + an interval-based ``VideoRecorderCfg`` (set by
    ``scripts/train.py --video`` through ``video_flags.training_video_enabled``; see
    :meth:`_setup_training_video`).

    The viewer is a visualizer, not a scene ``CameraCfg`` sensor: it is never cloned per env
    (a path without ``{ENV_REGEX_NS}`` is skipped by the cloner), so it costs exactly ONE
    viewport render per captured frame regardless of ``num_envs`` — no per-env camera tiling,
    no OOM at 4096 envs.
    """
    training_video_length: int = 500
    """Clip length in env steps (500 = 10 s at 50 Hz)."""
    training_video_interval_iters: int = 100
    """Start a new clip every N PPO iterations (converted to env steps with the runner's
    ``num_steps_per_env`` = 24; 100 iters x 24 = 2400 env steps)."""
    training_video_cam_height: float = 4.5
    """Height of the world-space viewer camera above the grid centre [m]."""
    training_video_cam_distance: float = 7.0
    """Horizontal distance of the world-space viewer camera from the grid centre [m].
    Together with :attr:`training_video_cam_height` this sets a natural ~3/4 view angle
    (atan(height / distance) down from horizontal, ~33 deg), not a top-down view. The view
    is anchored on the grid centre (world origin), where envs exist in every direction, so
    several robots are in frame at ~env_spacing regardless of the total env count."""
    training_video_width: int = 3840
    """Viewer image width [px]. One viewport render per captured frame, so this no longer
    scales with num_envs."""
    training_video_height: int = 2160
    """Viewer image height [px]."""

    def __post_init__(self):
        """Set up scene, curriculum, and command/obs switches."""
        # physics & timing: 200 Hz physics, 50 Hz policy
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # scene
        self.scene.num_envs = self.num_envs
        self.scene.env_spacing = self.env_spacing
        self.scene.robot = SoArm100Cfg(prim_path="{ENV_REGEX_NS}/Robot", actuator_preset=self.actuator_preset)
        self.scene.contact_forces.update_period = self.sim.dt
        # keep the ground material as the sim default physics material
        self.sim.physics_material = self.scene.ground.spawn.physics_material

        # optional: velocity-command tracking (phase 2)
        if self.use_velocity_commands:
            self.commands = CrawlCommandsCfg()
            # switch the velocity-tracking rewards onto the command
            self.rewards.forward_velocity.params["command_name"] = "base_velocity"
            self.rewards.overspeed.params["command_name"] = "base_velocity"
        else:
            self.commands = None

        # optional: gait clock (set to None so the obs manager skips it; the group
        # iterates instance __dict__ and ignores None terms)
        if not self.use_gait_clock:
            self.observations.policy.gait_clock = None

        # curriculum on/off
        if not self.use_curriculum:
            self.curriculum.v_target = None
            self.curriculum.friction_scale = None
            self.events.apply_friction_scale = None
            # without curriculum the v_target is constant at its final value
            self.v_target_start = self.v_target

        # video recording: add the scene cameras and sensor-based recorders only when
        # enabled (scripts set the process-local flag from ``--video``; see video_flags)
        if self.record_video or video_flags.sensor_video_enabled:
            self.record_video = True
            self._setup_video_recording()
        # live training video: single headless viewport camera + interval-based recorder (train.py --video)
        if self.record_training_video or video_flags.training_video_enabled:
            self.record_training_video = True
            self._setup_training_video()

    @staticmethod
    def _look_at_pose(
        eyes: tuple[float, float, float], targets: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Pose of an OpenGL-convention camera (forward -Z, up +Y) at ``eyes`` looking at ``targets``.

        ``create_rotation_matrix_from_view`` returns the camera orientation in the world frame
        (camera axes as columns, i.e. view -> world), so it feeds ``quat_from_matrix`` directly —
        the same usage as :meth:`KitVisualizer._set_usd_camera_pose`.
        """
        r_v2w = create_rotation_matrix_from_view(
            torch.tensor(eyes, dtype=torch.float32).unsqueeze(0),
            torch.tensor(targets, dtype=torch.float32).unsqueeze(0),
            up_axis="Z",
        )
        quat = quat_from_matrix(r_v2w)[0]
        return eyes, tuple(float(v) for v in quat)

    def _setup_video_recording(self) -> None:
        """Add per-env scene cameras and the sensor-based ``VideoRecorderCfg`` entries.

        Scene-sensor recording drives the headless RTX render pipeline every env step
        (``/isaaclab/render/rtx_sensors`` -> ``physx_headless_video_pump``), producing
        fresh frames. The default headless source instead records the Kit visualizer's
        viewport render product, which does not reliably refresh in pure headless runs and
        yields byte-identical (frozen) clips.

        The 3.0 ``Camera`` sensor requires one camera prim per environment, so the cameras
        are cloned per env (each frames its own robot); the recorders read env_0, which the
        deterministic play/evaluation path resets to its start pose at the world origin.
        """
        # side view: 3/4 view from +Y; target the crawl midpoint so the start pose (x~0) and
        # the crawl (+X) both stay framed
        side_pos, side_rot = self._look_at_pose((1.2, 2.8, 0.9), (0.15, 0.0, 0.1))
        # top view: near-plumb top-down (small +Y bias keeps the look-at off-singular)
        top_pos, top_rot = self._look_at_pose((0.6, 0.7, 3.5), (0.6, 0.0, 0.0))
        self.scene.video_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/video_camera",
            update_period=0.0,
            offset=CameraCfg.OffsetCfg(pos=side_pos, rot=side_rot, convention="opengl"),
            spawn=PinholeCameraCfg(),  # defaults: 24 mm focal, 20.955 mm aperture (~47 deg HFOV)
            width=self.video_width,
            height=self.video_height,
            data_types=["rgb"],
        )
        self.scene.video_camera_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/video_camera_top",
            update_period=0.0,
            offset=CameraCfg.OffsetCfg(pos=top_pos, rot=top_rot, convention="opengl"),
            spawn=PinholeCameraCfg(),
            width=self.video_width,
            height=self.video_height,
            data_types=["rgb"],
        )
        num_steps = int(round(self.episode_length_s / (self.decimation * self.sim.dt)))
        self.video_recorders = [
            VideoRecorderCfg(source="sensor:video_camera", video_length=num_steps, output_filename_prefix="side"),
            VideoRecorderCfg(source="sensor:video_camera_top", video_length=num_steps, output_filename_prefix="top"),
        ]

    def _setup_training_video(self) -> None:
        """Add ONE world-space viewer camera + interval-based training video recorder.

        The frame source is a single ``KitVisualizer`` with ``origin_type="world"`` — a
        world-space view of the whole scene, NOT a scene ``CameraCfg`` sensor. A visualizer
        is never cloned per env (the cloner only clones prims whose path matches the
        ``{ENV_REGEX_NS}`` env template), so there is exactly one camera no matter how many
        envs exist. This is what "one camera for the whole sim" requires, and it costs one
        viewport render per captured frame (not ``num_envs`` renders), so it does not OOM or
        slow training at 4096 envs the way a per-env camera sensor does.

        Framing: a natural ~3/4 view from low and to the -Y side of the grid centre (the
        world origin, where the env grid is centred), looking across it (pitch =
        atan(height / distance)). Because the grid is centred on the origin, envs surround
        the look-at point in every direction, so one frame shows several robots at
        ~env_spacing regardless of the total env count.

        Fresh frames: the video recorder pulls each frame via ``viz.render_rgb_array()``,
        which pumps the Kit app and re-renders the viewport with the current physics state,
        so frames track the moving robots (see :meth:`KitVisualizer.render_rgb_array`).
        Recording is interval-driven (a new clip every ``training_video_interval_iters``
        iterations, lasting ``training_video_length`` env steps).
        """
        # Natural ~3/4 view anchored on the grid centre (world origin): camera low and to
        # the -Y side, looking across the centre. The grid is centred on the origin, so
        # several envs (and their robots) are visible around the look-at point.
        eyes = (0.0, -self.training_video_cam_distance, self.training_video_cam_height)
        lookat = (0.0, 0.0, 0.1)
        self.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                eye=eyes,
                lookat=lookat,
                origin_type="world",
                headless=True,
                window_width=self.training_video_width,
                window_height=self.training_video_height,
            )
        ]
        # 24 = runner num_steps_per_env (see CrawlPPORunnerCfg); keep in sync if changed
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                video_length=self.training_video_length,
                video_interval=self.training_video_interval_iters * 24,
                output_filename_prefix="train_overview",
                fps=30,
                step_offset=5,  # skip RTX warm-up frames at the start of each clip
                keep_last_n_clips=None,  # keep every clip on disk (was 10: the recorder deleted older clips mid-run)
            )
        ]

    def play_mode(self):
        """Deterministic play/evaluation variant: no randomization, no curriculum, single start pose.

        Called by the play entrypoint (Isaac Lab 3.0 replaces the separate ``-Play`` env with
        this method). Also applied by ``scripts/evaluate.py``.
        """
        super().play_mode()
        # small scene for playback; a single env when video recording is active (the scene
        # cameras render one env per view, and the deterministic play path frames env_0)
        self.scene.num_envs = 1 if self.record_video else min(self.scene.num_envs, 8)
        # freeze the friction randomization at the default (multiplier 1.0)
        self.events.randomize_ground_friction.params["friction_range"] = (1.0, 1.0)
        # single, deterministic start pose
        self.events.reset_start_pose.params["pose_weights"] = (1.0, 0.0, 0.0)
        self.events.reset_start_pose.params["joint_noise"] = 0.0
        self.events.reset_start_pose.params["xy_offset"] = 0.0
        self.events.reset_start_pose.params["yaw_range"] = (0.0, 0.0)
        # no curriculum: constant v_target at its final value
        self.curriculum.v_target = None
        self.curriculum.friction_scale = None
        self.events.apply_friction_scale = None
        self.v_target_start = self.v_target


@configclass
class CrawlPlayEnvCfg(CrawlEnvCfg):
    """Deterministic play/evaluation variant (kept for the ``-Play`` task id).

    In Isaac Lab 3.0 the canonical play path is :meth:`CrawlEnvCfg.play_mode` applied by the
    play entrypoint; this class keeps the ``Zombie-Crawl-SO100-Play-v0`` task id working and
    behaves identically (its ``play_mode`` is a no-op since it is already deterministic).
    """

    def __post_init__(self):
        # configure the deterministic behaviour before the base __post_init__ runs
        self.num_envs = 8
        self.use_curriculum = False
        self.actuator_preset = "servo"
        # with video recording the scene cameras render one env per view; a single env keeps
        # the deterministic play/evaluation video cheap (the play path frames env_0 anyway)
        if self.record_video or video_flags.sensor_video_enabled:
            self.num_envs = 1
        super().__post_init__()
        # freeze the friction randomization at the default (multiplier 1.0)
        self.events.randomize_ground_friction.params["friction_range"] = (1.0, 1.0)
        # single, deterministic start pose
        self.events.reset_start_pose.params["pose_weights"] = (1.0, 0.0, 0.0)
        self.events.reset_start_pose.params["joint_noise"] = 0.0
        self.events.reset_start_pose.params["xy_offset"] = 0.0
        self.events.reset_start_pose.params["yaw_range"] = (0.0, 0.0)
        # remove observation corruption
        self.observations.policy.enable_corruption = False

    def play_mode(self):
        # already deterministic; nothing to change
        return
