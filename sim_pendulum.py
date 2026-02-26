"""
Simple Pendulum with Damping — Pygame Simulation
Equation: θ'' + (b/m)θ' + (g/L)sin(θ) = 0
"""

import sys
import math
import pygame
import numpy as np
from scipy.integrate import solve_ivp

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendulum Simulation")
clock = pygame.time.Clock()
FPS = 60

# ── Fonts & Colors ──────────────────────────────────────────────────
font_title = pygame.font.SysFont("DejaVu Sans", 22, bold=True)
font_label = pygame.font.SysFont("DejaVu Sans", 14)
font_value = pygame.font.SysFont("DejaVu Sans", 13, bold=True)
font_small = pygame.font.SysFont("DejaVu Sans", 12)
font_heading = pygame.font.SysFont("DejaVu Sans", 15, bold=True)

BG        = (24, 26, 33)
PANEL_BG  = (34, 37, 46)
WHITE     = (220, 225, 235)
GRAY      = (100, 105, 115)
BLUE      = (74, 144, 217)
RED       = (231, 76, 60)
GREEN     = (46, 204, 113)
YELLOW    = (241, 196, 15)
PURPLE    = (155, 89, 182)
DARK_GRAY = (55, 58, 68)
SLIDER_BG = (50, 54, 66)
SLIDER_FILL = (74, 144, 217)
KNOB      = (200, 210, 225)

G = 9.81


# ── UI widgets ──────────────────────────────────────────────────────
class Slider:
    def __init__(self, x, y, w, label, lo, hi, val, step=None, fmt=".1f"):
        self.rect = pygame.Rect(x, y, w, 20)
        self.label, self.lo, self.hi = label, lo, hi
        self.val = self.default = val
        self.step, self.fmt, self.dragging = step, fmt, False

    def draw(self, surf):
        surf.blit(font_label.render(self.label, True, WHITE), (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(surf, SLIDER_BG, self.rect, border_radius=4)
        frac = (self.val - self.lo) / (self.hi - self.lo)
        pygame.draw.rect(surf, SLIDER_FILL,
                         (self.rect.x, self.rect.y, int(frac * self.rect.w), self.rect.h),
                         border_radius=4)
        kx = self.rect.x + int(frac * self.rect.w)
        pygame.draw.circle(surf, KNOB, (kx, self.rect.centery), 10)
        pygame.draw.circle(surf, SLIDER_FILL, (kx, self.rect.centery), 10, 2)
        surf.blit(font_value.render(f"{self.val:{self.fmt}}", True, YELLOW),
                  (self.rect.right + 8, self.rect.y + 1))

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            frac = (self.val - self.lo) / (self.hi - self.lo)
            kx = self.rect.x + int(frac * self.rect.w)
            if math.hypot(ev.pos[0] - kx, ev.pos[1] - self.rect.centery) < 14 \
               or self.rect.collidepoint(ev.pos):
                self.dragging = True
                self._set(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set(ev.pos[0])

    def _set(self, mx):
        f = max(0, min(1, (mx - self.rect.x) / self.rect.w))
        v = self.lo + f * (self.hi - self.lo)
        if self.step: v = round(v / self.step) * self.step
        self.val = max(self.lo, min(self.hi, v))

    def reset(self):
        self.val = self.default


class Btn:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text

    def draw(self, surf):
        c = SLIDER_BG if self.rect.collidepoint(pygame.mouse.get_pos()) else DARK_GRAY
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        pygame.draw.rect(surf, GRAY, self.rect, 1, border_radius=6)
        t = font_label.render(self.text, True, WHITE)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, ev):
        return ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
               and self.rect.collidepoint(ev.pos)


# ── Graph helper ────────────────────────────────────────────────────
def draw_graph(surf, rect, tx, ty, color, xlabel, title):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surf, DARK_GRAY, rect, 1, border_radius=6)
    m = 45
    px, py, pw, ph = rect.x + m, rect.y + 28, rect.w - m - 15, rect.h - 50
    surf.blit(font_heading.render(title, True, WHITE), (rect.x + 10, rect.y + 6))
    if len(tx) < 2: return
    tmin, tmax = float(tx[0]), float(tx[-1])
    ymin, ymax = float(np.min(ty)), float(np.max(ty))
    if abs(ymax - ymin) < 1e-9: ymin -= 1; ymax += 1
    pad = (ymax - ymin) * 0.1; ymin -= pad; ymax += pad
    pygame.draw.line(surf, GRAY, (px, py + ph), (px + pw, py + ph), 1)
    pygame.draw.line(surf, GRAY, (px, py), (px, py + ph), 1)
    for i in range(6):
        f = i / 5; yy = py + ph - int(f * ph)
        pygame.draw.line(surf, DARK_GRAY, (px, yy), (px + pw, yy), 1)
        surf.blit(font_small.render(f"{ymin + f * (ymax - ymin):.1f}", True, GRAY), (px - 40, yy - 6))
    for i in range(6):
        f = i / 5; xx = px + int(f * pw)
        surf.blit(font_small.render(f"{tmin + f * (tmax - tmin):.1f}", True, GRAY), (xx - 10, py + ph + 3))
    surf.blit(font_small.render(xlabel, True, GRAY), (px + pw // 2 - 20, py + ph + 16))
    step = max(1, len(tx) // (pw // 2))
    pts = []
    for i in range(0, len(tx), step):
        fx = (float(tx[i]) - tmin) / (tmax - tmin)
        fy = (float(ty[i]) - ymin) / (ymax - ymin)
        pts.append((px + int(fx * pw), py + ph - int(fy * ph)))
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 2)
    return px, py, pw, ph, tmin, tmax, ymin, ymax


# ── Physics ─────────────────────────────────────────────────────────
T_SPAN = (0, 15)
T_EVAL = np.linspace(*T_SPAN, 1500)


def solve(L, m, b, theta0_deg):
    th0 = math.radians(theta0_deg)
    sol = solve_ivp(lambda t, y: [y[1], -(b / m) * y[1] - (G / L) * math.sin(y[0])],
                    T_SPAN, [th0, 0.0], t_eval=T_EVAL, method='RK45')
    return sol.t, sol.y[0], sol.y[1]


# ── Sliders ─────────────────────────────────────────────────────────
PY = 490
sl_L  = Slider(20, PY + 22, 260, "Length L (m)", 0.5, 3.0, 1.5, 0.1)
sl_m  = Slider(20, PY + 70, 260, "Mass m (kg)", 0.5, 10.0, 2.0, 0.1)
sl_b  = Slider(360, PY + 22, 260, "Damping b (Ns/m)", 0.0, 2.0, 0.3, 0.05, ".2f")
sl_th = Slider(360, PY + 70, 260, "Init. Angle θ₀ (°)", 5, 170, 120, 1, ".0f")
sliders = [sl_L, sl_m, sl_b, sl_th]
btn_reset = Btn(700, PY + 55, 100, 34, "Reset")

# ── State ───────────────────────────────────────────────────────────
t_arr, th_arr, om_arr = solve(1.5, 2.0, 0.3, 120)
prev = (1.5, 2.0, 0.3, 120)
sim_t = 0.0

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE: running = False
            if ev.key == pygame.K_r: sim_t = 0.0
        for s in sliders: s.handle(ev)
        if btn_reset.clicked(ev):
            for s in sliders: s.reset()
            sim_t = 0.0

    cur = (sl_L.val, sl_m.val, sl_b.val, sl_th.val)
    if cur != prev:
        t_arr, th_arr, om_arr = solve(*cur)
        prev = cur; sim_t = 0.0

    sim_t += dt
    if sim_t > T_SPAN[1]: sim_t = 0.0
    idx = max(0, min(int(sim_t / T_SPAN[1] * (len(t_arr) - 1)), len(t_arr) - 1))
    theta = float(th_arr[idx])

    screen.fill(BG)
    screen.blit(font_title.render("Simple Pendulum with Damping", True, WHITE), (WIDTH // 2 - 170, 10))

    # ── Pendulum visual ─────────────────────────────────────────────
    pivot = (180, 160)
    L_px = int(sl_L.val * 120)
    bob_x = pivot[0] + int(L_px * math.sin(theta))
    bob_y = pivot[1] + int(L_px * math.cos(theta))

    pygame.draw.rect(screen, GRAY, (pivot[0] - 30, pivot[1] - 4, 60, 8))
    pygame.draw.line(screen, WHITE, pivot, (bob_x, bob_y), 3)
    bob_r = int(12 + sl_m.val * 1.5)
    pygame.draw.circle(screen, RED, (bob_x, bob_y), bob_r)
    pygame.draw.circle(screen, WHITE, (bob_x, bob_y), bob_r, 2)
    pygame.draw.circle(screen, WHITE, pivot, 5)

    pygame.draw.line(screen, DARK_GRAY, pivot, (pivot[0], pivot[1] + L_px), 1)
    arc_r = 40
    start_a = -math.pi / 2 + min(0, theta)
    end_a = -math.pi / 2 + max(0, theta)
    if abs(theta) > 0.02:
        arc_pts = []
        for a in np.linspace(start_a, end_a, 30):
            arc_pts.append((pivot[0] + int(arc_r * math.cos(a)),
                            pivot[1] - int(arc_r * math.sin(a))))
        if len(arc_pts) > 1:
            pygame.draw.lines(screen, GREEN, False, arc_pts, 2)

    ang_txt = font_value.render(f"θ = {math.degrees(theta):.1f}°", True, YELLOW)
    screen.blit(ang_txt, (pivot[0] - ang_txt.get_width() // 2, pivot[1] + L_px + bob_r + 15))
    t_txt = font_small.render(f"t = {sim_t:.2f} s", True, GRAY)
    screen.blit(t_txt, (pivot[0] - t_txt.get_width() // 2, pivot[1] + L_px + bob_r + 35))

    # ── Graphs ──────────────────────────────────────────────────────
    g1 = pygame.Rect(370, 42, 430, 200)
    r1 = draw_graph(screen, g1, t_arr, np.degrees(th_arr), GREEN, "Time (s)", "Angle (°) vs Time")
    if r1:
        mx = r1[0] + int((sim_t - r1[4]) / (r1[5] - r1[4]) * r1[2])
        pygame.draw.line(screen, YELLOW, (mx, r1[1]), (mx, r1[1] + r1[3]), 1)

    g2 = pygame.Rect(370, 252, 430, 200)
    draw_graph(screen, g2, t_arr, om_arr, PURPLE, "Time (s)", "Angular Velocity (rad/s) vs Time")

    # ── Info ────────────────────────────────────────────────────────
    info = pygame.Rect(820, 42, 440, 420)
    pygame.draw.rect(screen, PANEL_BG, info, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, info, 1, border_radius=6)
    screen.blit(font_heading.render("Parameters", True, WHITE), (info.x + 15, info.y + 10))
    lines = [
        f"Length (L):       {sl_L.val:.1f} m",
        f"Mass (m):         {sl_m.val:.1f} kg",
        f"Damping (b):      {sl_b.val:.2f} Ns/m",
        f"Init. Angle (θ₀): {sl_th.val:.0f}°",
        "",
        f"Period (small θ): {2 * math.pi * math.sqrt(sl_L.val / G):.2f} s",
        f"Frequency:        {1 / (2 * math.pi * math.sqrt(sl_L.val / G)):.2f} Hz",
        "",
        f"Current θ:        {math.degrees(theta):.2f}°",
        f"Current ω:        {float(om_arr[idx]):.3f} rad/s",
    ]
    for i, l in enumerate(lines):
        screen.blit(font_small.render(l, True, WHITE if l else GRAY), (info.x + 15, info.y + 35 + i * 22))
    screen.blit(font_small.render("θ'' + (b/m)θ' + (g/L)sin(θ) = 0", True, GRAY),
                (info.x + 15, info.y + info.h - 30))

    # ── Controls bar ────────────────────────────────────────────────
    pygame.draw.rect(screen, PANEL_BG, (0, PY - 20, WIDTH, HEIGHT - PY + 20))
    pygame.draw.line(screen, DARK_GRAY, (0, PY - 20), (WIDTH, PY - 20), 1)
    for s in sliders: s.draw(screen)
    btn_reset.draw(screen)
    screen.blit(font_small.render("R: Restart  |  ESC: Quit", True, GRAY),
                (WIDTH // 2 - 80, HEIGHT - 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
