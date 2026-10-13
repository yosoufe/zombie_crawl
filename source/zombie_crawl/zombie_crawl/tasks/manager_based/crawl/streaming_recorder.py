"""Drop-in streaming replacement for Isaac Lab's VideoRecorder.

Isaac Lab's recorder accumulates every captured frame in a Python list of NumPy arrays and
encodes only when the clip closes (moviepy ``ImageSequenceClip``). At 3840x2160 RGBA that is
~33 MB per frame: a 2000-step (40 s) clip needs ~66 GB of host RAM and OOMs the machine, while
sub-sampling (``frame_stride``) keeps RAM low but makes the playback jittery/useless.

This subclass keeps the clip trigger/step-offset/keep-last-n logic of the base class but sends
each frame straight into an ``imageio_ffmpeg.write_frames`` generator -- an ffmpeg subprocess
fed over a rawvideo pipe -- so host memory stays constant at ~one frame while playback stays at
full capture rate.
"""

from __future__ import annotations

import logging

import numpy as np
from isaaclab.envs.utils.video_recorder import VideoRecorder

logger = logging.getLogger(__name__)


class StreamingVideoRecorder(VideoRecorder):
    """VideoRecorder that pipes frames to ffmpeg incrementally instead of buffering in RAM."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._writer = None
        self._clip_frame_count = 0
        self._first_frame = None
        self._overlay = None
        self._overlay_failed = False
        source = getattr(cfg, "source", "") or ""
        self._overlay_enabled = source.startswith("visualizer")

    def step(self) -> None:
        """Same trigger logic as the base class, but frames go to the pipe, not a list."""
        self._step_count += 1
        if self._step_count <= self.cfg.step_offset:
            return
        effective_step = self._step_count - self.cfg.step_offset
        if self._check_trigger(effective_step):
            if self._recording:
                self._close_clip()
            self._recording = True
            self._frames_step_count = 0
        if self._recording:
            self._frames_step_count += 1
            if self._frames_step_count % self.cfg.frame_stride == 0:
                self._update_overlay()
                frame = self._get_frame()
                if frame is not None:
                    self._send_frame(frame)
            if self._frames_step_count >= self.cfg.video_length:
                self._close_clip()

    def _update_overlay(self) -> None:
        """Draw the commanded-direction (cyan) arrow overlay on the stage."""
        if not self._overlay_enabled or self._overlay_failed:
            return
        try:
            if self._overlay is None:
                from .video_overlay import VectorArrowOverlay

                self._overlay = VectorArrowOverlay(self._env)
                if not self._overlay.setup():
                    self._overlay, self._overlay_failed = None, True
                    return
            self._overlay.update()
        except Exception:
            import traceback

            logger.warning("[StreamingVideoRecorder] vector overlay disabled. %s", traceback.format_exc())
            self._overlay, self._overlay_failed = None, True

    def close(self) -> None:
        if self._recording and self._writer is not None:
            self._close_clip()

    def _send_frame(self, frame: np.ndarray) -> None:
        try:
            if self._writer is None:
                self._open_writer(frame)
            self._writer.send(np.ascontiguousarray(frame))
            self._clip_frame_count += 1
        except Exception:
            if not self._frame_error_logged:
                import traceback

                logger.error(
                    "[StreamingVideoRecorder] frame write failed; recording stopped. %s",
                    traceback.format_exc(),
                )
                self._frame_error_logged = True
            self._abort_writer()
            self._recording = False

    def _resolve_fps(self) -> float:
        if self.cfg.fps:
            return float(self.cfg.fps)
        base_fps = self._env.metadata.get("render_fps") if hasattr(self._env, "metadata") else None
        if not base_fps:
            step_dt = getattr(self._env, "step_dt", None)
            base_fps = round(1.0 / step_dt) if step_dt else 30
        return float(max(1, round(base_fps / self.cfg.frame_stride)))

    def _open_writer(self, frame: np.ndarray) -> None:
        from imageio_ffmpeg import write_frames

        h, w = frame.shape[:2]
        pix_fmt_in = "rgba" if frame.shape[2] == 4 else "rgb24"
        path = self._clip_path(self._clip_index)
        import os

        os.makedirs(self._effective_output_dir(), exist_ok=True)
        self._writer = write_frames(
            path,
            size=(w, h),
            pix_fmt_in=pix_fmt_in,
            pix_fmt_out="yuv420p",
            fps=self._resolve_fps(),
            codec="libx264",
            # ultrafast keeps a 4K50 encoder ahead of the live pipe; monitor-quality is plenty
            output_params=["-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"],
            ffmpeg_log_level="error",
        )
        self._writer.send(None)  # seed the generator (spawns ffmpeg)
        self._first_frame = frame.copy()
        self._clip_frame_count = 0
        logger.info("[StreamingVideoRecorder] streaming clip %s (%dx%d @ %.1f fps)", path, w, h, self._resolve_fps())

    def _abort_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    def _close_clip(self) -> None:
        if self._writer is None or self._clip_frame_count == 0:
            self._recording = False
            self._abort_writer()
            return
        if self._first_frame is not None and float(np.mean(self._first_frame)) < 2.0:
            logger.warning("[StreamingVideoRecorder] clip looks mostly black (mean pixel < 2).")
        self._writer.close()  # flushes ffmpeg and finalizes the mp4
        logger.info("[StreamingVideoRecorder] Wrote %d frames (streamed) to clip %d", self._clip_frame_count, self._clip_index)
        self._abort_writer()
        self._recording = False
        self._clip_index += 1
        self._maybe_delete_old_clips()
