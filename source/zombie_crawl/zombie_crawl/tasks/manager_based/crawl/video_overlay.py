"""3D command-vector overlay for training video recordings.

Drawn on the USD stage through Isaac Lab's own :class:`VisualizationMarkers` (a
``UsdGeom.PointInstancer``), so the arrows render inside the headless world-viewer render
product that the training video captures (this is the same marker path every Isaac Lab
sensor debug-vis uses). One slim cyan arrow floats over EVERY robot:

- cyan arrow: the COMMANDED crawl direction; the cone's TIP points where the robot must crawl.

Markers are procedural cones (no Nucleus asset download) tinted via a ``PreviewSurface``
material -- a bare prim has no material and renders un-coloured. All pxr / Isaac imports are
lazy so this module only loads once Kit is up.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class VectorArrowOverlay:
    """Commanded-direction (cyan) arrowheads via VisualizationMarkers."""

    def __init__(self, env, radius: float = 0.03, length: float = 0.28, lift: float = 0.12):
        from isaaclab.markers import VisualizationMarkersCfg
        from isaaclab.sim import ConeCfg, PreviewSurfaceCfg

        self._env = env
        self._length = length  # arrow length [m] (fixed -> uniform, clean look)
        self._lift = lift  # float this far above the base [m]

        self._cmd_cfg = VisualizationMarkersCfg(
            prim_path="/World/vector_overlay/cmd",
            markers={
                "arrow": ConeCfg(
                    radius=radius,
                    height=1.0,  # scaled per-instance by ``length``; cone apex sits at +Z
                    axis="Z",
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.85, 0.85), emissive_color=(0.0, 0.4, 0.4)
                    ),
                )
            },
        )
        self._cmd = None
        self._ids: torch.Tensor | None = None
        self._ready = False

    def setup(self) -> bool:
        from isaaclab.markers import VisualizationMarkers

        # tag every env: the PointInstancer draws them all in one instanced pass, and only the
        # ones inside the viewer frame are visible in the recorded clip.
        self._ids = self._env.scene.env_origins.new_tensor(range(self._env.num_envs), dtype=torch.long)
        self._cmd = VisualizationMarkers(self._cmd_cfg)
        self._ready = True
        logger.info("[VectorArrowOverlay] drawing %d command arrows", int(self._ids.numel()))
        return True

    @staticmethod
    def _quat_z_to_dir(angles: torch.Tensor) -> torch.Tensor:
        """Quaternion (xyzw) rotating local +Z onto the horizontal unit dir (cos, sin, 0)."""
        from isaaclab.utils.math import quat_from_matrix
        import torch.nn.functional as F

        dev = angles.device
        dirv = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1)
        # orthonormal frame with the third (local +Z) column = dir
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(dirv.shape[0], 3)
        e2 = F.normalize(torch.cross(dirv, ez, dim=-1), dim=-1)
        e1 = F.normalize(torch.cross(e2, dirv, dim=-1), dim=-1)
        mat = torch.stack([e1, e2, dirv], dim=-1)  # columns -> images of local axes
        return quat_from_matrix(mat)

    def update(self) -> None:
        if not self._ready:
            return
        env = self._env
        ids = self._ids
        robot = env.scene["robot"]
        pos = robot.data.root_pos_w.torch[ids]
        n = ids.shape[0]
        dev = pos.device
        ref = getattr(env, "zombie_ref_yaw", None)
        total = robot.data.root_pos_w.torch.shape[0]
        ref_i = ref[ids] if ref is not None and ref.shape[0] == total else torch.zeros(n, device=dev)

        # arrow: fixed length, cone apex (local +Z) aimed along the commanded heading.
        l = torch.full((n,), self._length, device=dev)
        q = self._quat_z_to_dir(ref_i)
        # center the cone half the arrow out from the base so the tip lands one length away
        t = pos + torch.stack(
            [0.5 * l * torch.cos(ref_i), 0.5 * l * torch.sin(ref_i), torch.full((n,), self._lift, device=dev)],
            dim=-1,
        )
        self._cmd.visualize(
            translations=t,
            orientations=q,
            scales=torch.stack([torch.ones_like(l), torch.ones_like(l), l], dim=-1),
        )

    def clear(self) -> None:
        if self._cmd is not None:
            self._cmd.set_visibility(False)
