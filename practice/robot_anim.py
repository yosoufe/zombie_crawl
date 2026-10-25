import math
import sys

import pygame

W, H = 1400, 900
PPM = 90.0
DT = 1 / 60.0
V_STEP = 0.1
W_STEP = 0.1
V_LIMIT = 2.0
W_LIMIT = 3.0
BG = (24, 26, 33)
GRID = (40, 44, 54)
ROBOT = (80, 200, 120)
TRAIL = (255, 180, 60)
TEXT = (220, 220, 225)
FIELD_BG = (38, 42, 52)
FIELD_BORDER = (90, 96, 112)
FIELD_ACTIVE = (120, 200, 255)
TYPABLE = "0123456789.-+"


class Bot:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.cmd_v = 0.5
        self.cmd_w = 0.3
        self.trail = []

    def step(self):
        self.x += self.cmd_v * math.cos(self.yaw) * DT
        self.y += self.cmd_v * math.sin(self.yaw) * DT
        self.yaw += self.cmd_w * DT
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.trail.append((self.x, self.y))
        if len(self.trail) > 20000:
            self.trail = self.trail[-10000:]

    def reset(self):
        self.__init__()


def world_to_screen(bot, wx, wy):
    return (
        int(W / 2 + (wx - bot.x) * PPM),
        int(H / 2 - (wy - bot.y) * PPM),
    )


def draw_grid(screen, bot):
    left = bot.x - W / 2 / PPM
    right = bot.x + W / 2 / PPM
    bottom = bot.y - H / 2 / PPM
    top = bot.y + H / 2 / PPM
    for gx in range(int(math.floor(left)), int(math.ceil(right)) + 1):
        sx, _ = world_to_screen(bot, gx, 0)
        pygame.draw.line(screen, GRID, (sx, 0), (sx, H))
    for gy in range(int(math.floor(bottom)), int(math.ceil(top)) + 1):
        _, sy = world_to_screen(bot, 0, gy)
        pygame.draw.line(screen, GRID, (0, sy), (W, sy))


def draw_robot(screen, bot):
    cx, cy = W // 2, H // 2
    r = 20
    nose = (cx + r * math.cos(bot.yaw), cy - r * math.sin(bot.yaw))
    left = (
        cx + r * 0.7 * math.cos(bot.yaw + 2.5),
        cy - r * 0.7 * math.sin(bot.yaw + 2.5),
    )
    right = (
        cx + r * 0.7 * math.cos(bot.yaw - 2.5),
        cy - r * 0.7 * math.sin(bot.yaw - 2.5),
    )
    pygame.draw.polygon(screen, ROBOT, [nose, left, right])
    pygame.draw.circle(screen, ROBOT, (cx, cy), 6)


class TextBox:
    def __init__(self, label, rect, get, set, fmt):
        self.label = label
        self.rect = pygame.Rect(rect)
        self.get = get
        self.set = set
        self.fmt = fmt
        self.buf = ""
        self.box = pygame.Rect(rect[0] + 140, rect[1], 120, rect[3])

    def begin(self):
        self.buf = self.fmt(self.get())
        if self.buf.startswith("+"):
            self.buf = self.buf[1:]

    def handle_key(self, event):
        if event.key == pygame.K_RETURN:
            try:
                self.set(float(self.buf))
            except ValueError:
                pass
            return False
        if event.key == pygame.K_BACKSPACE:
            self.buf = self.buf[:-1]
        elif event.unicode in TYPABLE and len(self.buf) < 10:
            self.buf += event.unicode
        return True

    def draw(self, screen, font, focused, ticks):
        label = font.render(self.label, True, TEXT)
        screen.blit(label, (self.rect.x, self.rect.y + self.rect.height // 2 - label.get_height() // 2))
        tx = self.rect.x + label.get_width() + 8
        box = pygame.Rect(tx, self.rect.y, 120, self.rect.height)
        pygame.draw.rect(screen, FIELD_BG, box, border_radius=5)
        pygame.draw.rect(
            screen,
            FIELD_ACTIVE if focused else FIELD_BORDER,
            box,
            2,
            border_radius=5,
        )
        shown = self.buf if focused else self.fmt(self.get())
        text = font.render(shown, True, TEXT)
        screen.blit(text, (box.x + 8, box.y + box.height // 2 - text.get_height() // 2))
        if focused and ticks // 400 % 2 == 0:
            cx = box.x + 8 + font.size(shown)[0]
            pygame.draw.line(screen, TEXT, (cx + 2, box.y + 6), (cx + 2, box.bottom - 6), 2)
        self.box = box


def main():
    global W, H
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Constant velocity + yaw rate practice")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    bot = Bot()
    fields = {
        "v": TextBox(
            "cmd_v m/s ",
            (14, 150, 240, 36),
            lambda: bot.cmd_v,
            lambda val: setattr(bot, "cmd_v", max(-V_LIMIT, min(V_LIMIT, val))),
            lambda val: f"{val:+.2f}",
        ),
        "w": TextBox(
            "cmd_w rad/s ",
            (14, 196, 240, 36),
            lambda: bot.cmd_w,
            lambda val: setattr(bot, "cmd_w", max(-W_LIMIT, min(W_LIMIT, val))),
            lambda val: f"{val:+.2f}",
        ),
    }
    focus = None

    while True:
        ticks = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                focus = None
                for name, tb in fields.items():
                    if tb.box.collidepoint(event.pos):
                        focus = name
                        tb.begin()
            if event.type == pygame.KEYDOWN:
                if focus:
                    if event.key == pygame.K_ESCAPE:
                        focus = None
                    elif event.key == pygame.K_TAB:
                        focus = "w" if focus == "v" else "v"
                        fields[focus].begin()
                    elif not fields[focus].handle_key(event):
                        focus = None
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_TAB:
                    focus = "v"
                    fields["v"].begin()
                elif event.key == pygame.K_SPACE:
                    bot.cmd_v, bot.cmd_w = 0.0, 0.0
                elif event.key == pygame.K_r:
                    bot.reset()

        new_w, new_h = screen.get_size()
        if (new_w, new_h) != (W, H):
            W, H = new_w, new_h

        if focus is None:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                bot.cmd_v = min(V_LIMIT, bot.cmd_v + V_STEP)
            if keys[pygame.K_DOWN]:
                bot.cmd_v = max(-V_LIMIT, bot.cmd_v - V_STEP)
            if keys[pygame.K_LEFT]:
                bot.cmd_w = min(W_LIMIT, bot.cmd_w + W_STEP)
            if keys[pygame.K_RIGHT]:
                bot.cmd_w = max(-W_LIMIT, bot.cmd_w - W_STEP)

        bot.step()

        screen.fill(BG)
        draw_grid(screen, bot)
        pts = [world_to_screen(bot, wx, wy) for wx, wy in bot.trail[-5000:]]
        if len(pts) > 1:
            pygame.draw.lines(screen, TRAIL, False, pts, 3)
        draw_robot(screen, bot)

        hud = [
            f"pos    ({bot.x:+.2f}, {bot.y:+.2f}) m  yaw {math.degrees(bot.yaw):+.1f} deg",
            "Up/Down = velocity   Left/Right = yaw rate",
            "Click or Tab into a box, type value, Enter applies",
            "Space = zero commands   R = reset   Esc = quit",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, TEXT), (14, 12 + i * 32))
        for name, tb in fields.items():
            tb.draw(screen, font, focus == name, ticks)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
