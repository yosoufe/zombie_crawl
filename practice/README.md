# practice — constant velocity + yaw rate simulator

Lightweight pygame sandbox for practicing differential-drive kinematics:
the robot moves with a constant linear velocity `cmd_v` and yaw rate `cmd_w`
(`x += v·cos(yaw)·dt`, `y += v·sin(yaw)·dt`, `yaw += w·dt`).

## Requirements

- Python 3 with pygame 2 (installed in the conda base env):
  `python3 -c "import pygame; print(pygame.version.ver)"`
- If missing: `python3 -m pip install pygame`

## Run

```bash
python3 practice/robot_anim.py
```

The window is resizable. The camera follows the robot; each grid cell is 1 m × 1 m.
The orange trail is the robot's path.

## Setting commands

### Text boxes
- Click the `cmd_v` or `cmd_w` box (or press `Tab`), type a number, press `Enter` to apply.
- `Tab` switches between the two boxes, `Esc` leaves the box without applying.
- Clamped: `cmd_v` ±2.0 m/s, `cmd_w` ±3.0 rad/s.

### Keyboard shortcuts
| Key | Action |
| --- | --- |
| Up / Down | increase / decrease `cmd_v` by 0.1 m/s |
| Left / Right | increase / decrease `cmd_w` by 0.1 rad/s |
| Space | zero both commands |
| R | reset robot pose and trail |
| Esc | quit |

Keyboard shortcuts are ignored while a text box is focused.
