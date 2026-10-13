# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crawl RL environment with a single world-space training-video viewer.

Subclasses :class:`isaaclab.envs.ManagerBasedRLEnv` to force the render / Fabric
settings on when live training video is enabled.

Root cause of the static training video: the training video uses a single
world-space headless :class:`KitVisualizer` (one camera over the whole scene),
which creates no :class:`~isaaclab.sensors.Camera` sensor. A camera sensor is what
normally flips ``/isaaclab/render/rtx_sensors`` and
``/physics/fabricUpdateTransformations`` on (see
:meth:`isaaclab.sensors.camera.camera.Camera._initialize_impl`). Without them,
``sim.is_rendering`` stays ``False``, the env never calls ``sim.render()`` per step,
and PhysX does not continuously sync the Fabric transforms that RTX reads — so the
headless viewer's viewport is rendered once and then frozen (byte-identical frames).

Forcing both settings on makes ``sim.is_rendering`` ``True`` (it is read live every
step), so the env renders every ``render_interval`` steps with the current physics
state, and the world-space viewer produces fresh, non-static frames — while still
being a single camera (no per-env tiling, no OOM at large num_envs).
"""

from isaaclab.envs import ManagerBasedRLEnv

# Stream training/play video frames straight to ffmpeg instead of buffering gigabytes in RAM.
# manager_based_env instantiates the module-global ``VideoRecorder``, so swapping the symbol
# here (this module loads at gym.make time, after the app is up) makes every recorder use
# the streaming subclass. See streaming_recorder.py for details.
from . import streaming_recorder as _streaming_recorder
import isaaclab.envs.manager_based_env as _mb_env

_mb_env.VideoRecorder = _streaming_recorder.StreamingVideoRecorder


class CrawlRLEnv(ManagerBasedRLEnv):
    """Crawl env that keeps the headless training-video viewer rendering fresh frames."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(cfg, "record_training_video", False):
            # A Camera sensor sets these in Camera._initialize_impl; the world-space
            # headless viewer has no sensor, so set them here to keep the viewport
            # re-rendering each step (fresh, non-static video frames).
            self.sim.set_setting("/isaaclab/render/rtx_sensors", True)
            self.sim.set_setting("/physics/fabricUpdateTransformations", True)


class CrawlPlayRLEnv(ManagerBasedRLEnv):
    """Play/eval env: identical to the base env, but importing it here guarantees the
    streaming-video patch is applied for the deterministic play/eval recorders too."""
