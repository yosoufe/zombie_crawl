"""3D vector overlay for training video recordings.

Drawn on the USD stage through Isaac Lab's own :class:`VisualizationMarkers` (a
``UsdGeom.PointInstancer``), so the arrows render inside the headless world-viewer render
product that the training video captures (this is the same marker path every Isaac Lab
sensor debug-vis uses). One slim cyan arrow floats over EVERY robot:

- cyan arrow: the CLAW arrow (base -> wrist, the locomotion axis every directional reward
  is scored on); the cone's TIP points where the base must travel. The phase-1 cyan
  "commanded heading" arrow is gone with the removed heading command (P2-13) -- the claw
  arrow is the useful visual under a yaw-rate command (watch items W2/W3: does the base
  follow the arrow, or does only the arm swing?).
- red arrow(s): the CURRENT disturbance force on EVERY rigid body (base + every link),
  aimed along the last force sampled by ``mdp.disturb_all_bodies`` (read from
  ``env.extras["disturb_forces"]``); length grows with force magnitude, so a body being
  shoved hard shows a long arrow. Hidden while the curriculum has not turned the pushes on.

Markers are procedural cones (no Nucleus asset download) tinted via a ``PreviewSurface``
material -- a bare prim has no material and renders un-coloured. All pxr / Isaac imports are
lazy so this module only loads once Kit is up.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class VectorArrowOverlay:
    """Commanded-direction (cyan) + disturbance-force (red) arrowheads via VisualizationMarkers."""

    # force [N] that maps to the full red arrow length (disturb_all_bodies samples each axis
    # in +-0.4 at full curriculum scale -> |f| <= sqrt(3)*0.4 ~ 0.69 N)
    _FORCE_REF = 0.7
    _FORCE_EPS = 1e-3  # below this |f| the body is treated as undisturbed (arrow hidden)

    def __init__(
        self,
        env,
        radius: float = 0.02,
        length: float = 0.2,
        lift: float = 0.1,
        dist_radius: float = 0.012,
        dist_max_length: float = 0.12,
    ):
        from isaaclab.markers import VisualizationMarkersCfg
        from isaaclab.sim import ConeCfg, PreviewSurfaceCfg

        self._env = env
        self._length = length  # arrow length [m] (fixed -> uniform, clean look)
        self._lift = lift  # float this far above the base [m]
        self._dist_max_length = dist_max_length  # red arrow length at |f| = _FORCE_REF [m]

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
        self._dist_cfg = VisualizationMarkersCfg(
            prim_path="/World/vector_overlay/disturb",
            markers={
                "arrow": ConeCfg(
                    radius=dist_radius,
                    height=1.0,  # scaled per-instance by force magnitude; apex at +Z
                    axis="Z",
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.0, 0.0), emissive_color=(0.4, 0.0, 0.0)
                    ),
                )
            },
        )
        self._cmd = None
        self._dist = None
        self._ids: torch.Tensor | None = None
        self._ready = False

    def setup(self) -> bool:
        from isaaclab.markers import VisualizationMarkers

        # tag every env: the PointInstancer draws them all in one instanced pass, and only the
        # ones inside the viewer frame are visible in the recorded clip.
        self._ids = self._env.scene.env_origins.new_tensor(range(self._env.num_envs), dtype=torch.long)
        self._cmd = VisualizationMarkers(self._cmd_cfg)
        self._dist = VisualizationMarkers(self._dist_cfg)
        self._ready = True
        n_bodies = self._env.scene["robot"].num_bodies
        logger.info(
            "[VectorArrowOverlay] drawing %d command arrows + %d disturbance arrows (%d envs x %d bodies)",
            int(self._ids.numel()),
            int(self._ids.numel()) * n_bodies,
            int(self._ids.numel()),
            n_bodies,
        )
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

    @staticmethod
    def _quat_z_to_dir3d(dirv: torch.Tensor) -> torch.Tensor:
        """Quaternion (xyzw) rotating local +Z onto an arbitrary unit 3-D direction."""
        from isaaclab.utils.math import quat_from_matrix
        import torch.nn.functional as F

        # reference axis for building the in-plane basis; x-axis near dirv = +-Z keeps the
        # cross product non-degenerate
        up = torch.where(
            (dirv[:, 2].abs() > 0.999999).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0], device=dirv.device).expand_as(dirv),
            torch.tensor([0.0, 0.0, 1.0], device=dirv.device).expand_as(dirv),
        )
        e2 = F.normalize(torch.cross(up, dirv, dim=-1), dim=-1)
        e1 = torch.cross(e2, dirv, dim=-1)  # right-handed: e1 x e2 == dirv
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
        # claw arrow shared with the rewards/obs (mdp.get_claw_dir): the locomotion axis
        # the policy must drag its base along (the phase-1 commanded-heading arrow was
        # removed with the heading command -- P2-13)
        from .mdp.rewards import get_claw_dir

        cos_full, sin_full = get_claw_dir(env)
        total = robot.data.root_pos_w.torch.shape[0]
        if cos_full.shape[0] == total:
            ref_cos, ref_sin = cos_full[ids], sin_full[ids]
        else:
            ref_cos = torch.ones(n, device=dev)
            ref_sin = torch.zeros(n, device=dev)

        # arrow: fixed length, cone apex (local +Z) aimed along the commanded heading.
        l = torch.full((n,), self._length, device=dev)
        q = self._quat_z_to_dir(torch.atan2(ref_sin, ref_cos))
        # center the cone half the arrow out from the base so the tip lands one length away
        t = pos + torch.stack(
            [0.5 * l * ref_cos, 0.5 * l * ref_sin, torch.full((n,), self._lift, device=dev)],
            dim=-1,
        )
        self._cmd.visualize(
            translations=t,
            orientations=q,
            scales=torch.stack([torch.ones_like(l), torch.ones_like(l), l], dim=-1),
        )
        self._update_disturbance(robot, n)

    def _update_disturbance(self, robot, n: int) -> None:
        """One red arrow per rigid body, aimed along the currently applied disturbance force."""
        forces = self._env.extras.get("disturb_forces")
        m = n * robot.num_bodies
        dev = robot.data.root_pos_w.torch.device
        if forces is None or forces.shape[0] != n:
            # pushes never fired (e.g. deterministic eval) -> park every instance out of sight
            self._dist.visualize(
                translations=torch.full((m, 3), -1000.0, device=dev),
                orientations=torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(m, 4),
                scales=torch.full((m, 3), 1e-4, device=dev),
            )
            return
        f = forces.reshape(m, 3)
        mag = f.norm(dim=-1)
        dirv = f / mag.clamp(min=self._FORCE_EPS).unsqueeze(-1)
        # arrow length grows linearly with |f| up to dist_max_length at the worst-case
        # |f| actually in effect (published by disturb_all_bodies, tracks config ranges)
        ref = float(self._env.extras.get("disturb_force_ref", self._FORCE_REF))
        l = (mag.clamp(max=ref) / ref) * self._dist_max_length
        visible = mag > self._FORCE_EPS
        l = torch.where(visible, l, torch.zeros_like(l))
        q = self._quat_z_to_dir3d(dirv)
        # center the cone half an arrow out from the body COM so the tip leads the force
        t = robot.data.body_pos_w.torch.reshape(m, 3) + 0.5 * l.unsqueeze(-1) * dirv
        # invisible instances (zero force / pushes off): park far below the ground plane
        t = torch.where(visible.unsqueeze(-1), t, torch.full_like(t, -1000.0))
        self._dist.visualize(
            translations=t,
            orientations=q,
            scales=torch.stack([torch.ones_like(l), torch.ones_like(l), l.clamp(min=1e-4)], dim=-1),
        )

    def clear(self) -> None:
        if self._cmd is not None:
            self._cmd.set_visibility(False)
        if self._dist is not None:
            self._dist.set_visibility(False)
