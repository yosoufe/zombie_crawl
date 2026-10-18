#!/usr/bin/env python
"""IMU 3D live view: ESP32 Madgwick quaternion -> WebSocket -> three.js.

Serves an HTML page at http://localhost:8765 and pushes frames on /ws.
Firmware: imu_stream.ino (streams ms,qw,qx,qy,qz at 100 Hz).

Run:  /home/yousof/robotics/experimental/.venv/bin/python imu_server.py [--port /dev/ttyACM0]
"""
import argparse
import asyncio
import json
import pathlib
import threading
import time

import serial
from aiohttp import web

HERE = pathlib.Path(__file__).parent
clients = set()
latest = {"ms": 0, "q": [1.0, 0.0, 0.0, 0.0], "acc": None, "t": 0.0}
ser = None


def serial_reader():
    global ser
    while True:
        try:
            with serial.Serial("/dev/ttyACM0", 115200, timeout=1) as ser:
                ser.dtr = True
                ser.rts = False
                while True:
                    line = ser.readline().decode(errors="replace").strip()
                    parts = line.split(",")
                    if len(parts) == 11 and parts[0].isdigit():
                        try:
                            vals = [float(x) for x in parts[1:]]
                            latest["ms"] = int(parts[0])
                            latest["acc"] = vals[0:3]
                            latest["q"] = vals[6:10]
                            latest["t"] = time.time()
                        except ValueError:
                            continue
                    elif len(parts) == 5 and parts[0].isdigit():
                        try:
                            latest["ms"] = int(parts[0])
                            latest["q"] = [float(x) for x in parts[1:]]
                            latest["t"] = time.time()
                        except ValueError:
                            continue
        except Exception as e:
            print(f"serial lost ({e}); reopening in 1 s", flush=True)
            time.sleep(1)


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    try:
        async for msg in ws:  # inbound: "BETA <v>" / "GYR <odr> <lpf>" -> firmware
            if msg.type == web.WSMsgType.TEXT:
                txt = msg.data.strip()
                if txt.upper().startswith(("BETA", "GYR")) and ser is not None:
                    ser.write((txt + "\n").encode())
                    await ws.send_str(json.dumps({"log": txt}))
    finally:
        clients.discard(ws)
    return ws


async def ws_broadcaster(app):
    loop = asyncio.get_running_loop()

    async def loop_task():
        while True:
            if clients:
                frame = json.dumps(latest)
                for ws in list(clients):
                    try:
                        await ws.send_str(frame)
                    except Exception:
                        clients.discard(ws)
            await asyncio.sleep(1 / 60)  # 60 fps push

    task = loop.create_task(loop_task())
    yield
    task.cancel()


async def index(request):
    return web.FileResponse(HERE / "index.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--http", type=int, default=8765)
    args = ap.parse_args()

    threading.Thread(target=serial_reader, daemon=True).start()

    app = web.Application()
    app.cleanup_ctx.append(ws_broadcaster)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    print(f"IMU server: http://localhost:{args.http}")
    web.run_app(app, host="127.0.0.1", port=args.http, print=None)


if __name__ == "__main__":
    main()
