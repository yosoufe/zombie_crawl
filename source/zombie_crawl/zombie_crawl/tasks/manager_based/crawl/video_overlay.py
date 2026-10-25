"""3D vector overlay for training/evaluation video recordings.

Drawn on the USD stage through Isaac Lab's own :class:`VisualizationMarkers` (a
``UsdGeom.PointInstancer``), so the overlay renders inside the headless world-viewer render
product that the training video captures (this is the same marker path every Isaac Lab
sensor debug-vis uses). Per robot:

- GREEN arc: the COMMANDED yaw rate ``yaw_rate_cmd`` (command term's current draw).
- MAGENTA arc: the ACTUAL filtered claw-heading rate ``r`` (``env.zombie_r_ema`` -- the
  same measurement the rewards/metrics score).
  Both arcs sit on the ground plane around the base, start at the claw axis, and sweep
  left/right with the sign of the rate (linear). A full circle equals
  ``arc_full_circle_rad_s`` (0.8 rad/s), so the 0.3 rad/s stage-2 command ceiling reads
  as a 3/8-circle (135 deg) and a 0.1 rad/s command as a 45 deg arc.
  Side-by-side green/magenta is the steering-lag visualization; with a zero command the
  green arc hides and magenta exposes any unwanted circling drift.
  Every arc is robot-bound -- no static scene markers are drawn (0.3 rad/s = 3/8 circle;
  a saturated full ring means |rate| >= 0.8 rad/s).
- red arrow(s): the CURRENT disturbance force on EVERY rigid body (base + every link),
  aimed along the last force sampled by ``mdp.disturb_all_bodies`` (read from
  ``env.extras["disturb_forces"]``); length grows with force magnitude, so a body being
  shoved hard shows a long arrow. Hidden while the curriculum has not turned the pushes on.

The old cyan claw arrow was removed on owner directive (its information is legible from
the robot pose itself; the arcs are the payload of these clips).

Markers are procedural cylinders/cones (no Nucleus asset download) tinted via a
``PreviewSurface`` material -- a bare prim has no material and renders un-coloured. All
pxr / Isaac imports are lazy so this module only loads once Kit is up.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class VectorArrowOverlay:
    """Commanded (green) + actual (magenta) yaw-rate ground arcs + disturbance (red) arrows."""

    # force [N] that maps to the full red arrow length (disturb_all_bodies samples each axis
    # in +-0.4 at full curriculum scale -> |f| <= sqrt(3)*0.4 ~ 0.69 N)
    _FORCE_REF = 0.7
    _FORCE_EPS = 1e-3  # below this |f| the body is treated as undisturbed (arrow hidden)
    _RATE_EPS = 1e-3  # below this |rad/s| the arc is hidden (parked out of sight)

    def __init__(
        self,
        env,
        dist_radius: float = 0.012,
        dist_max_length: float = 0.12,
        arc_radius: float = 0.22,
        arc_segments: int = 16,
        arc_full_circle_rad_s: float = 0.8,
        arc_lift: float = 0.03,
        arc_thickness: float = 0.014,
    ):
        from isaaclab.markers import VisualizationMarkersCfg
        from isaaclab.sim import ConeCfg, CylinderCfg, PreviewSurfaceCfg

        self._env = env
        self._dist_max_length = dist_max_length  # red arrow length at |f| = _FORCE_REF [m]
        self._arc_radius = arc_radius
        self._arc_n = int(arc_segments)
        self._arc_gain = 2.0 * torch.pi / float(arc_full_circle_rad_s)  # sweep [rad] per [rad/s]
        self._arc_lift = arc_lift
        self._arc_thickness = arc_thickness

        self._head_len = 9.0 * arc_thickness  # arrowhead length [m] at the sweep tip (env arcs ~13 cm)

        def _arc_cfg(path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
            return VisualizationMarkersCfg(
                prim_path=path,
                markers={
                    "seg": CylinderCfg(
                        radius=arc_thickness,
                        height=1.0,  # scaled per-instance to the segment arc length
                        axis="Z",
                        visual_material=PreviewSurfaceCfg(
                            diffuse_color=color, emissive_color=(0.4 * color[0], 0.4 * color[1], 0.4 * color[2])
                        ),
                    ),
                    "head": ConeCfg(
                        radius=2.2 * arc_thickness,
                        height=1.0,  # scaled per-instance to _head_len; apex leads the sweep
                        axis="Z",
                        visual_material=PreviewSurfaceCfg(
                            diffuse_color=color, emissive_color=(0.4 * color[0], 0.4 * color[1], 0.4 * color[2])
                        ),
                    ),
                },
            )

        self._arc_cmd_cfg = _arc_cfg("/World/vector_overlay/arc_cmd", (0.10, 0.95, 0.25))
        self._arc_act_cfg = _arc_cfg("/World/vector_overlay/arc_act", (0.95, 0.10, 0.95))
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
        self._arc_cmd = None
        self._arc_act = None
        self._dist = None
        self._ids: torch.Tensor | None = None
        self._ready = False

    def setup(self) -> bool:
        from isaaclab.markers import VisualizationMarkers

        # tag every env: the PointInstancer draws them all in one instanced pass, and only the
        # ones inside the viewer frame are visible in the recorded clip.
        self._ids = self._env.scene.env_origins.new_tensor(range(self._env.num_envs), dtype=torch.long)
        self._arc_cmd = VisualizationMarkers(self._arc_cmd_cfg)
        self._arc_act = VisualizationMarkers(self._arc_act_cfg)
        self._dist = VisualizationMarkers(self._dist_cfg)
        self._ready = True
        n_bodies = self._env.scene["robot"].num_bodies
        logger.info(
            "[VectorArrowOverlay] drawing cmd+act yaw arcs (%d segs each) for %d envs and %d disturbance arrows per env",
            self._arc_n,
            int(self._ids.numel()),
            n_bodies,
        )
        return True

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
            torch.tensor([0.0, 1.0, 0.0], device=dirv.device).expand_as(dirv),
        )
        e2 = F.normalize(torch.cross(up, dirv, dim=-1), dim=-1)
        e1 = torch.cross(e2, dirv, dim=-1)  # right-handed: e1 x e2 == dirv
        mat = torch.stack([e1, e2, dirv], dim=-1)  # columns -> images of local axes
        return quat_from_matrix(mat)

    def _arc_instances(self, values: torch.Tensor, centers: torch.Tensor, starts: torch.Tensor, radii: torch.Tensor):
        """Positions/orientations/scales for one arc per row (envs + legend rows).

        ``values`` [rad/s] per row, ``centers`` [M,3] arc centers (base positions),
        ``starts`` [M] start angle (claw heading), ``radii`` [M] arc radius per row.
        Returns flat [M*N] tensors; hidden rows (|value| < eps) are parked at -1000.
        """
        dev = values.device
        m = values.shape[0]
        n = self._arc_n
        sweep = torch.clamp(values * self._arc_gain, -2.0 * torch.pi, 2.0 * torch.pi)
        sgn = torch.sign(sweep)
        sweep_abs = sweep.abs()
        k = torch.arange(n, device=dev, dtype=values.dtype)
        # segment centre angles and the walk direction along the sweep
        frac = (k + 0.5).unsqueeze(0) / n  # [1,N]
        ang = starts.unsqueeze(-1) + sweep.unsqueeze(-1) * frac  # [M,N]
        cx, cy, cz = centers[:, 0:1], centers[:, 1:2], centers[:, 2:3]
        pos = torch.stack(
            [
                cx + radii.unsqueeze(-1) * torch.cos(ang),
                cy + radii.unsqueeze(-1) * torch.sin(ang),
                (cz + self._arc_lift).expand(m, n),
            ],
            dim=-1,
        ).reshape(m * n, 3)
        tang = torch.stack(
            [-sgn.unsqueeze(-1) * torch.sin(ang), sgn.unsqueeze(-1) * torch.cos(ang), torch.zeros_like(ang)],
            dim=-1,
        ).reshape(m * n, 3)
        # cylinder prim already carries the radius; only its (unit) height is scaled to the
        # segment arc length (slight overlap so the polyline reads as one continuous arc)
        seg_len = ((radii * sweep_abs / n).clamp(min=1e-4) * 1.25).unsqueeze(-1).expand(m, n).reshape(m, n, 1)
        scale = torch.cat(
            [
                torch.ones((m, n, 1), device=dev),
                torch.ones((m, n, 1), device=dev),
                seg_len,
            ],
            dim=-1,
        ).reshape(m * n, 3)
        # arrowhead cone at the sweep tip: centre just past the last segment, cone axis
        # along the walk direction (apex at +Z leads the sweep)
        # head length scales with the arc radius so the bigger legend arcs get proportionally
        # bigger (visible) arrowheads
        head_len = self._head_len * (radii / self._arc_radius)
        ang_end = starts + sweep * (1.0 - 0.5 / n) + (head_len * 0.5) / radii.clamp(min=1e-3)
        head_c = torch.stack(
            [
                cx.squeeze(-1) + radii * torch.cos(ang_end),
                cy.squeeze(-1) + radii * torch.sin(ang_end),
                (cz.squeeze(-1) + self._arc_lift),
            ],
            dim=-1,
        )
        head_t = torch.stack(
            [-sgn * torch.sin(ang_end), sgn * torch.cos(ang_end), torch.zeros_like(ang_end)], dim=-1
        )
        radius_scale = radii / self._arc_radius
        head_s = torch.stack([radius_scale, radius_scale, head_len], dim=-1)
        pos = torch.cat([pos.reshape(m, n, 3), head_c.unsqueeze(1)], dim=1).reshape(m * (n + 1), 3)
        tang = torch.cat([tang.reshape(m, n, 3), head_t.unsqueeze(1)], dim=1).reshape(m * (n + 1), 3)
        scale = torch.cat([scale.reshape(m, n, 3), head_s.unsqueeze(1)], dim=1).reshape(m * (n + 1), 3)
        indices = torch.cat([torch.zeros(m, n, device=dev), torch.ones(m, 1, device=dev)], dim=1).reshape(-1).long()

        visible = (values.abs() > self._RATE_EPS).repeat_interleave(n + 1)
        pos = torch.where(visible.unsqueeze(-1), pos, torch.full_like(pos, -1000.0))
        scale = torch.where(visible.unsqueeze(-1), scale, torch.full_like(scale, 1e-4))
        # zero-rate rows produce a null tangent (sgn=0); aim them +Z (parked anyway, but a
        # NaN quaternion would poison the instancer transform buffer)
        tang_ok = torch.where(
            tang.norm(dim=-1, keepdim=True) > 1e-6, tang, torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(tang)
        )
        quat = self._quat_z_to_dir3d(tang_ok)
        return pos, quat, scale, indices

    def update(self) -> None:
        if not self._ready:
            return
        env = self._env
        ids = self._ids
        robot = env.scene["robot"]
        pos = robot.data.root_pos_w.torch[ids]
        n = ids.shape[0]
        dev = pos.device
        # arc start angle = claw heading (mdp.get_claw_dir), shared with the rewards/obs
        from .mdp.rewards import get_claw_dir

        cos_full, sin_full = get_claw_dir(env)
        total = robot.data.root_pos_w.torch.shape[0]
        if cos_full.shape[0] == total:
            ref_cos, ref_sin = cos_full[ids], sin_full[ids]
        else:
            ref_cos = torch.ones(n, device=dev)
            ref_sin = torch.zeros(n, device=dev)
        start = torch.atan2(ref_sin, ref_cos)

        cmd_term = env.command_manager.get_term("yaw_rate_cmd")
        cmd = cmd_term.yaw_rate[ids] if cmd_term.yaw_rate.shape[0] == total else torch.zeros(n, device=dev)
        act = getattr(env, "zombie_r_ema", torch.zeros(n, device=dev))[ids]

        # legend: three reference arcs (cmd colours on the cmd marker, act on the act
        # marker) placed on the camera side of the grid, larger radius for legibility
        # no static legend: every drawn arc belongs to a robot (owner directive -- stray
        # fixed arcs at world positions read as bugs in the clips)
        centers = pos
        starts = start
        radii = torch.full((n,), self._arc_radius, device=dev)
        cmd_vals = cmd
        act_vals = act

        t, q, s, idx = self._arc_instances(cmd_vals, centers, starts, radii)
        self._arc_cmd.visualize(translations=t, orientations=q, scales=s, marker_indices=idx)
        t, q, s, idx = self._arc_instances(act_vals, centers, starts, radii)
        self._arc_act.visualize(translations=t, orientations=q, scales=s, marker_indices=idx)
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
        for marker in (self._arc_cmd, self._arc_act, self._dist):
            if marker is not None:
                marker.set_visibility(False)
