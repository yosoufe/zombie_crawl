#!/usr/bin/env python3
"""Tune Madgwick beta live:  set_beta.py 0.3   (no re-flash, no port conflict)."""
import asyncio
import sys
import websockets


async def send(value: str):
    async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
        await ws.send(f"BETA {value}")
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            if '"log"' in msg:
                print("board ack:", msg)
                return

asyncio.run(send(sys.argv[1]))
