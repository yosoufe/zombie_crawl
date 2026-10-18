# ESP32-S3-LCD-1.3 — IMU Setup

Board: [Waveshare ESP32-S3-LCD-1.3](https://www.waveshare.com/wiki/ESP32-S3-LCD-1.3)
- ESP32-S3R8 (16MB Flash, 8MB PSRAM), 1.3" 240x240 LCD (ST7789VW), TF card slot
- **IMU: QST QMI8658** (3-axis accel + 3-axis gyro, I2C SDA=47 SCL=48), driver = `SensorLib`
- USB: CH9102 serial bridge (UART0) → appears as `/dev/ttyACM0`; LCD backlight = GPIO20

## Known-good software matrix (IMPORTANT)

The IDE's auto-update broke the Waveshare demos; this is the verified working combo:

| Component | Version | Notes |
|---|---|---|
| ESP32 core | `esp32:esp32@2.0.12` | 3.3.12 crashes bundled TFT_eSPI in `begin()` |
| TFT_eSPI | bundled **2.5.0** + added `#define TFT_BL 20` / `#define TFT_BACKLIGHT_ON HIGH` to `User_Setup.h` | 2.5.43 crashes on core 2.0.12; BL was never enabled → black screen |
| SensorLib | **0.2.6** | 0.5.0's QMI8658 compat header crashes on core 2.0.12 |
| Patch | `04_Snake/WS_QMI8658.cpp` — dropped 4th `selfTest` arg from `configAccelerometer`/`configGyroscope` | 3-arg form compiles on both APIs |

Reinstall if the IDE auto-updates them:
```bash
arduino-cli core install esp32:esp32@2.0.12
arduino-cli lib install "SensorLib@0.2.6"
cp -r ~/robotics/experimental/imu/ESP32-S3-LCD-1.3-Demo/Arduino/libraries/TFT_eSPI ~/Arduino/libraries/
printf '\n#define TFT_BL   20\n#define TFT_BACKLIGHT_ON HIGH\n' >> ~/Arduino/libraries/TFT_eSPI/User_Setup.h
```

## Environment

| Tool | Location / Version |
|---|---|
| Arduino IDE 2.3.10 | `~/robotics/experimental/arduino-ide_2.3.10_Linux_64bit.AppImage` (menu: "Arduino IDE 2") |
| Arduino IDE 1.8.19 | `~/robotics/experimental/arduino-1.8.19/` (menu: "Arduino 1.8.19", `~/bin/arduino`; runs on system OpenJDK 25 with `-Dsun.java2d.uiScale=1.5`) |
| arduino-cli | `~/bin/arduino-cli` (v1.5.1) |
| Python venv | `/home/yousof/robotics/experimental/.venv` (pyserial, numpy, matplotlib, aiohttp, websockets — installed) |

IDE 1, IDE 2, and the CLI share `~/.arduino15` (cores) and `~/Arduino/libraries` (libraries).

## Files here

- `ESP32-S3-LCD-1.3-Demo.zip` / `ESP32-S3-LCD-1.3-Demo/` — official Waveshare demo package;
  IMU demos: `Arduino/examples/04_Snake` (tilt-controlled Snake), `03_SDIMG_Game2048`
- `imu_stream/imu_stream.ino` — firmware: QMI8658 (gyro **1024 DPS**/896.8 Hz/LPF3, accel 2 G) →
  **Madgwick AHRS on the ESP32** (measured dt, gyro auto-bias on boot, keep still ~1 s) →
  streams `ms,qw,qx,qy,qz` CSV on UART0; accepts live `BETA`/`GYR`/`RAW` commands on serial
- `imu_stream/mount_quat.h` — sensor→board mount quaternion (generated; currently ≈ identity —
  the chip frame matches the board model frame, confirmed by Waveshare FactoryProgram logic)
- `imu_server.py` + `index.html` — serial→WebSocket bridge + three.js browser 3D view (recommended)
- `imu_mount_calibrate.py` — 3-pose gravity calibration (Enter-paced): A flat screen-up,
  B screen-down, C held upright USB-top; does two-position accel-bias extraction, writes `mount_quat.h`
- `imu_diag.py` — lag diagnostic: one fast ~90° twist on Enter → reports gyro saturation,
  gyro-integrated vs accel-derived angle, post-stop settle (usage: `imu_diag.py [ceiling_dps]`)
- `send_cmd.py` / `set_beta.py` — forward live commands via the server's /ws
  (e.g. `send_cmd.py "GYR 3 3 6"`, `set_beta.py 0.3`)
- `imu_visualizer.py` — matplotlib fallback visualizer (animation flaky when backgrounded)

## Commands

### Flash firmware (stream or Snake demos)

```bash
FQBN="esp32:esp32:esp32s3:FlashSize=16M,PSRAM=opi,CDCOnBoot=default"
# Madgwick quaternion streamer
arduino-cli upload -b "$FQBN" -p /dev/ttyACM0 ~/robotics/experimental/imu/imu_stream/
# Waveshare Snake game (on-screen, no PC needed)
arduino-cli upload -b "$FQBN" -p /dev/ttyACM0 \
  ~/robotics/experimental/imu/ESP32-S3-LCD-1.3-Demo/Arduino/examples/04_Snake/
```

`CDCOnBoot=default` routes `Serial` to UART0 (the CH9102 bridge) → you can both flash and
read logs on `/dev/ttyACM0` with one cable.

### Live 3D visualization (browser — recommended)

```bash
cd ~/robotics/experimental/imu
# start server detached (survives terminal close)
setsid nohup /home/yousof/robotics/experimental/.venv/bin/python imu_server.py > server.log 2>&1 < /dev/null &
# verify
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/    # -> 200
# stop
fuser -k 8765/tcp    # NOT pkill -f imu_server.py — that matches your own shell too
```

Open **http://localhost:8765** — 60 fps three.js board follows your tilts;
HUD shows roll/pitch/yaw, quaternion, board uptime and fps (drag to orbit).

### Tuning & diagnostics (no reflash)

Live commands go to the firmware through the server (`send_cmd.py`), or paste into
`arduino-cli monitor`-style serial input. Port is exclusive — stop the server before
`imu_diag.py`/`imu_mount_calibrate.py` (both kill/restart the server themselves).
**`send_cmd.py`/`set_beta.py` are RAM-only**: a reboot or reflash reverts to the firmware
defaults. A tuning value is persistent ONLY once it's edited into the source
(`betaDef` in `MadgwickAHRS.cpp`, `configGyroscope` args in the sketch) and compiled+flashed.

```bash
V=/home/yousof/robotics/experimental/.venv/bin/python
$V ~/robotics/experimental/imu/send_cmd.py "BETA 0.3"     # Madgwick gain (default 0.1)
$V ~/robotics/experimental/imu/send_cmd.py "GYR 3 3 6"    # gyro odr lpf range (idx, see sketch comment)
$V ~/robotics/experimental/imu/send_cmd.py "RAW 1"        # 11-field raw frames (for the calibrate/diag scripts)
$V ~/robotics/experimental/imu/imu_diag.py 1024           # saturation / scale / settle test (one twist)
$V ~/robotics/experimental/imu/imu_mount_calibrate.py     # 3-pose mount solve -> mount_quat.h
```

Findings (2026-10): gyro at 256/512 DPS **clips** on fast ~90° snaps (peaks measured ~580 dps) →
multi-second accel-drag tail after rotation; fixed by `GYR_RANGE_1024DPS` (settle 23° → 2.5°).
Madgwick must use **measured dt** (fixed 10 ms assumed-rate under-integrated fast rotations).
Renderer world is **Z-up** (`camera.up.set(0,0,1)`), mount ≈ identity — chip axes match board model.

Boot sequence (~7 s): QMI **COD** on-demand calibration → 3 s gyro warm-up wait → gyro+accel
bias sampling (**board must be FLAT and still**; else the accel offset skews roll/pitch, and the
gyro transient bakes in yaw drift — a 48 dps boot bias caused constant 48°/s spin).
Serial markers: `# COD ok`, `# BIAS <gx gy gz dps>`, `# ABIAS <ax ay az g>`, `# READY`.
SensorLib's COD gains do **not** affect `getAccelerometer()` output — accel offsets are removed
in software (`bax/bay/baz`). At-rest quality now: roll/pitch σ≈0.02°, yaw ~0.1°/s (6-DOF, no compass).

### Matplotlib fallback

```bash
/home/yousof/robotics/experimental/.venv/bin/python ~/robotics/experimental/imu/imu_visualizer.py
```

### SensorLib QMI8658 examples (Arduino IDE 2)

With SensorLib 0.2.6: **File → Examples → SensorLib → Examples** → `QMI8658_GetDataExample`,
`QMI8658_MadgwickAHRS`, `QMI8658_ReadFromFifoExample`, `QMI8658_WakeOnMotion`, …

### Misc

```bash
arduino-cli board list                                  # check port
arduino-cli compile --clean -b "$FQBN" <sketch>         # build (use --clean after lib swaps!)
arduino-cli monitor -p /dev/ttyACM0 -c 115200           # serial monitor
```

Gotchas:
- `arduino-cli upload` has **no `--clean` flag** — run `compile --clean` first, else stale caches re-flash old binaries (check `ELF file SHA256` in boot logs).
- If the board sits in `waiting for download`, reset via:
  `~/.arduino15/packages/esp32/tools/esptool_py/5.3.1/esptool --chip esp32s3 -p /dev/ttyACM0 --after hard-reset chip_id`
- Crash backtrace decode: `xtensa-esp32s3-elf-addr2line -pfiaC -e <sketch>.elf <addrs>`
  (toolchain: `~/.arduino15/packages/esp32/tools/s3-gcc/2021r2-p5/bin/`).

# Fusion
madgwick for fusion — runs on the ESP32 (see `imu_stream/imu_stream.ino`); the PC only renders.
