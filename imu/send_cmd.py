#!/usr/bin/env python3
"""Send a firmware tuning command live:  send_cmd.py "GYR 2 4"  |  send_cmd.py "BETA 0.5" """
import asyncio
import sys
import websockets


async def send(cmd: str):
    async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
        await ws.send(cmd)
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            if '"log"' in msg:
                print("sent ->", msg)
                return

asyncio.run(send(sys.argv[1]))
