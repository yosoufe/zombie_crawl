#!/usr/bin/env python
"""Visualize the ESP32-S3-LCD-1.3 QMI8658 orientation in 3D.

Fusion (Madgwick AHRS) runs ON THE ESP32 (firmware: imu_stream.ino), which
streams orientation quaternions as CSV:  ms,qw,qx,qy,qz

Run:  /home/yousof/robotics/experimental/.venv/bin/python imu_visualizer.py [--port /dev/ttyACM0]
"""
import argparse
import queue
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import serial


def reader(port, q):
    with serial.Serial(port, 115200, timeout=1) as s:
        s.dtr = True
        s.rts = False  # release EN (auto-download circuit), don't hold board in reset
        while True:
            line = s.readline().decode(errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            try:
                q.put((int(parts[0]),
                       np.array([float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4])])))
            except ValueError:
                continue


def rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def board_vertices(w=49.0, h=24.0, t=8.0):
    """Waveshare board approx size (mm), z-up = screen normal."""
    v = np.array([[sx * w / 2, sy * h / 2, sz * t / 2]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 2, 6, 4],
             [1, 5, 7, 3], [0, 4, 5, 1], [2, 3, 7, 6]]
    return v, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()

    samples = queue.Queue(maxsize=200)
    threading.Thread(target=reader, args=(args.port, samples), daemon=True).start()

    v, faces = board_vertices()
    axes = np.eye(3) * 28  # 28 mm axis arrows

    fig = plt.figure("QMI8658 — 3D orientation (Madgwick on ESP32)", figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection([v[f] for f in faces], facecolor="#2a6df4",
                            edgecolor="k", alpha=0.85)
    ax.add_collection3d(poly)
    lines = Line3DCollection([[[0, 0, 0], [1, 0, 0]],
                              [[0, 0, 0], [0, 1, 0]],
                              [[0, 0, 0], [0, 0, 1]]], linewidths=3)
    ax.add_collection3d(lines)
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_zlim(-40, 40)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
    ax.set_box_aspect((1, 1, 1))
    rate = fig.text(0.02, 0.02, "", family="monospace")

    last_ms = [None]
    count = [0]

    def update(_frame):
        try:  # drain newest sample
            while True:
                ms, q = samples.get_nowait()
        except queue.Empty:
            return poly, lines

        if last_ms[0] is not None:
            count[0] += 1
        last_ms[0] = ms

        R = rotation_matrix(q)
        vr = v @ R.T
        poly.set_verts([vr[f] for f in faces])
        lines.set_segments([[[0, 0, 0], axes[i] @ R.T] for i in range(3)])
        lines.set_colors(["#d62828", "#2a9d3a", "#2a5bd6"])
        w_, x_, y_, z_ = q
        roll = np.degrees(np.arctan2(2*(w_*x_+y_*z_), 1-2*(x_*x_+y_*y_)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w_*y_-z_*x_), -1, 1)))
        yaw = np.degrees(np.arctan2(2*(w_*z_+x_*y_), 1-2*(y_*y_+z_*z_)))
        ax.set_title(f"roll {roll:6.1f}°  pitch {pitch:6.1f}°  yaw {yaw:6.1f}°")
        rate.set_text(f"t={ms}ms  n={count[0]}  q = [{w_:+.3f} {x_:+.3f} {y_:+.3f} {z_:+.3f}]")
        return poly, lines

    main.ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
