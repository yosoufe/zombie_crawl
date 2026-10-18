#!/usr/bin/env python3
"""Decisive IMU lag diagnostic.

You do ONE fast ~90 deg twist when told. The script logs raw gyro+accel+quat
and answers:
  1. Did the gyro saturate (peak near the configured DPS ceiling)?
  2. Does gyro-integrated angle match the accel-derived angle change? (scale err)
  3. How long does the estimate take to settle after you stop?

Run:  .venv/bin/python imu_diag.py
"""
import subprocess
import time

import numpy as np
import serial

PORT = "/dev/ttyACM0"
import sys
RANGE = float(sys.argv[1]) if len(sys.argv) > 1 else 512.0  # configured DPS ceiling


def q_angle(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return 2 * np.degrees(np.arccos(d))


def main():
    subprocess.run(["fuser", "-k", "8765/tcp"], capture_output=True)
    time.sleep(1)
    s = serial.Serial(PORT, 115200, timeout=1)
    s.dtr = True
    s.rts = False
    time.sleep(1.5)
    s.reset_input_buffer()
    s.write(b"RAW 1\n")
    time.sleep(0.3)
    s.reset_input_buffer()

    print("\nTest: hold board flat, screen up.")
    input("Press ENTER when the board lies flat and still... ")
    print("  BASELINE 3 s — keep still", flush=True)
    rows = []
    t_base = time.time() + 3
    while time.time() < t_base:
        ln = s.readline().decode(errors="ignore").strip()
        p = ln.split(",")
        if len(p) == 11 and p[0].isdigit():
            rows.append([time.time()] + [float(x) for x in p[1:]])

    input("Press ENTER, then IMMEDIATELY twist ~90 deg FAST (wrist snap),\n  and hold perfectly still until the script finishes... ")
    t_end = time.time() + 12
    while time.time() < t_end:
        ln = s.readline().decode(errors="ignore").strip()
        p = ln.split(",")
        if len(p) == 11 and p[0].isdigit():
            rows.append([time.time()] + [float(x) for x in p[1:]])
    s.write(b"RAW 0\n")
    s.close()

    a = np.array(rows)
    if len(a) < 100:
        sys.exit("too few samples")
    t = a[:, 0] - a[0, 0]
    acc, gyr, quat = a[:, 1:4], a[:, 4:7], a[:, 7:11]

    # hemisphere-fix quat series
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    speed = np.linalg.norm(gyr, axis=1)
    moving = speed > 40  # dps
    peak = np.abs(gyr).max()
    print("\n=== DIAGNOSTIC ===")
    print(f"samples: {len(a)}  duration: {t[-1]:.1f} s")
    print(f"peak |gyro| = {peak:.1f} dps  (ceiling {RANGE})"
          f"  -> {'SATURATED — raise range!' if peak > 0.9 * RANGE else 'no saturation'}")

    idx = np.where(moving)[0]
    if len(idx) < 3:
        print("no twist detected — rerun and twist harder/faster")
        return
    i0, i1 = idx[0], idx[-1]
    dur = t[i1] - t[i0]

    # dominant axis during the twist
    dom = int(np.argmax(np.abs(gyr[i0:i1 + 1]).max(axis=0)))
    dom_angle = np.cumsum(gyr[i0:i1 + 1, dom]) * np.gradient(t[i0:i1 + 1])
    gyro_dangle = abs(dom_angle[-1])
    print(f"twist: axis {'xyz'[dom]}, {dur:.2f} s, gyro-integrated change = {gyro_dangle:.1f} deg")

    # accel-derived attitude change: baseline vs hold
    pre = acc[max(0, i0 - 30):i0].mean(0)
    post = acc[i1 + 30:].mean(0)
    pre /= np.linalg.norm(pre)
    post /= np.linalg.norm(post)
    true_dangle = np.degrees(np.arccos(np.clip(np.dot(pre, post), -1, 1)))
    print(f"accel-derived change            = {true_dangle:.1f} deg")
    if true_dangle > 5:
        ratio = gyro_dangle / true_dangle
        flag = "" if 0.9 < ratio < 1.1 else "  <-- SCALE ERROR"
        print(f"gyro/true ratio                 = {ratio:.3f}{flag}")

    # settle: |dq/dt| after the twist until < 2 deg/s for 1 s
    dq = np.array([q_angle(quat[i - 1], quat[i]) for i in range(1, len(quat))])
    rate = np.r_[dq, dq[-1]] / np.maximum(np.gradient(t), 1e-6)
    settle = None
    for i in range(i1, len(rate) - 60):
        if np.all(rate[i:i + 60] < 2.0):
            settle = t[i] - t[i1]
            break
    total = q_angle(quat[i1], quat[-1])
    print(f"still-moving after stop         : {settle if settle is None else f'{settle:.2f} s'}"
          f"  (settle time after your stop)")
    print(f"quat still changed after stop   : {total:.2f} deg over the hold")

    ok = peak < 0.9 * RANGE and 0.9 < (gyro_dangle / max(true_dangle, 1e-9)) < 1.1
    print("\nVERDICT:", "sensor chain clean — lag is filter gain" if ok
          else "sensor data itself is deficient (see lines above)")


if __name__ == "__main__":
    import sys
    main()
