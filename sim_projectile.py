"""
Projectile Motion with Air Resistance — Pygame Simulation
Compares ideal trajectory vs trajectory with quadratic drag.
"""

import sys
import math
import pygame
import numpy as np
from scipy.integrate import solve_ivp

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projectile Motion Simulation")
clock = pygame.time.Clock()
FPS = 60

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
DARK_GRAY = (55, 58, 68)
SLIDER_BG = (50, 54, 66)
SLIDER_FILL = (74, 144, 217)
KNOB      = (200, 210, 225)

G = 9.81
RHO = 1.225
RADIUS = 0.05
A = math.pi * RADIUS**2


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
            if self.rect.collidepoint(ev.pos):
                self.dragging = True; self._set(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP: self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging: self._set(ev.pos[0])

    def _set(self, mx):
        f = max(0, min(1, (mx - self.rect.x) / self.rect.w))
        v = self.lo + f * (self.hi - self.lo)
        if self.step: v = round(v / self.step) * self.step
        self.val = max(self.lo, min(self.hi, v))

    def reset(self): self.val = self.default


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
        return ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos)


# ── Physics ─────────────────────────────────────────────────────────
def ideal(v0, ang_deg):
    a = math.radians(ang_deg)
    vx, vy = v0 * math.cos(a), v0 * math.sin(a)
    tf = 2 * vy / G
    t = np.linspace(0, tf, 500)
    x = vx * t; y = vy * t - 0.5 * G * t**2
    y = np.maximum(y, 0)
    return x, y, vy**2 / (2 * G), vx * tf, tf

def drag_ode(t, s, mass, Cd):
    x, y, vx, vy = s
    v = math.sqrt(vx**2 + vy**2)
    if v < 1e-12: return [vx, vy, 0, -G]
    Fd = 0.5 * Cd * RHO * A * v**2
    return [vx, vy, -Fd * vx / (v * mass), -G - Fd * vy / (v * mass)]

def hit_ground(t, s, mass, Cd): return s[1]
hit_ground.terminal = True; hit_ground.direction = -1

def with_drag(v0, ang_deg, Cd, mass):
    a = math.radians(ang_deg)
    sol = solve_ivp(drag_ode, [0, 200], [0, 0, v0 * math.cos(a), v0 * math.sin(a)],
                    events=hit_ground, args=(mass, Cd), max_step=0.05)
    x, y = sol.y[0], np.maximum(sol.y[1], 0)
    return x, y, float(y.max()), float(x[-1]), float(sol.t[-1])


# ── Controls ────────────────────────────────────────────────────────
PY = 490
sl_v0   = Slider(20, PY + 22, 260, "Velocity v₀ (m/s)", 5, 100, 40, 1, ".0f")
sl_ang  = Slider(20, PY + 70, 260, "Angle (°)", 5, 85, 45, 1, ".0f")
sl_cd   = Slider(350, PY + 22, 260, "Drag Coeff. Cd", 0.0, 1.0, 0.47, 0.01, ".2f")
sl_mass = Slider(350, PY + 70, 260, "Mass (kg)", 0.1, 10.0, 1.0, 0.1)
sliders = [sl_v0, sl_ang, sl_cd, sl_mass]
btn_reset = Btn(700, PY + 55, 100, 34, "Reset")

# ── State ───────────────────────────────────────────────────────────
prev = None
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

    cur = (sl_v0.val, sl_ang.val, sl_cd.val, sl_mass.val)
    if cur != prev:
        xi, yi, mhi, rngi, tfi = ideal(cur[0], cur[1])
        xd, yd, mhd, rngd, tfd = with_drag(*cur)
        prev = cur; sim_t = 0.0
        max_t = max(tfi, tfd)

    sim_t += dt
    if sim_t > max_t: sim_t = 0.0

    screen.fill(BG)
    screen.blit(font_title.render("Projectile Motion — Ideal vs Air Resistance", True, WHITE),
                (WIDTH // 2 - 240, 10))

    # ── Trajectory graph ────────────────────────────────────────────
    gr = pygame.Rect(40, 50, 750, 400)
    pygame.draw.rect(screen, PANEL_BG, gr, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, gr, 1, border_radius=6)

    mg = 55
    px, py2 = gr.x + mg, gr.y + 30
    pw, ph = gr.w - mg - 15, gr.h - 55

    x_max = max(float(xi.max()), float(xd.max())) * 1.1 + 1
    y_max = max(float(yi.max()), float(yd.max())) * 1.3 + 1

    pygame.draw.line(screen, GRAY, (px, py2 + ph), (px + pw, py2 + ph), 1)
    pygame.draw.line(screen, GRAY, (px, py2), (px, py2 + ph), 1)

    for i in range(6):
        f = i / 5
        yy = py2 + ph - int(f * ph)
        pygame.draw.line(screen, DARK_GRAY, (px, yy), (px + pw, yy), 1)
        screen.blit(font_small.render(f"{f * y_max:.0f}", True, GRAY), (px - 40, yy - 6))
        xx = px + int(f * pw)
        screen.blit(font_small.render(f"{f * x_max:.0f}", True, GRAY), (xx - 10, py2 + ph + 5))

    screen.blit(font_small.render("Distance (m)", True, GRAY), (px + pw // 2 - 30, py2 + ph + 18))
    screen.blit(font_heading.render("Trajectory", True, WHITE), (gr.x + 10, gr.y + 6))

    def to_screen(xarr, yarr):
        pts = []
        step = max(1, len(xarr) // (pw // 2))
        for j in range(0, len(xarr), step):
            sx = px + int(float(xarr[j]) / x_max * pw)
            sy = py2 + ph - int(float(yarr[j]) / y_max * ph)
            pts.append((sx, sy))
        return pts

    pts_i = to_screen(xi, yi)
    pts_d = to_screen(xd, yd)
    if len(pts_i) > 1: pygame.draw.lines(screen, BLUE, False, pts_i, 3)
    if len(pts_d) > 1: pygame.draw.lines(screen, RED, False, pts_d, 3)

    # Animated dot (ideal)
    frac_i = min(sim_t / tfi, 1.0) if tfi > 0 else 0
    idx_i = int(frac_i * (len(xi) - 1))
    if idx_i < len(xi):
        bx = px + int(float(xi[idx_i]) / x_max * pw)
        by = py2 + ph - int(float(yi[idx_i]) / y_max * ph)
        pygame.draw.circle(screen, BLUE, (bx, by), 6)

    # Animated dot (drag)
    frac_d = min(sim_t / tfd, 1.0) if tfd > 0 else 0
    idx_d = int(frac_d * (len(xd) - 1))
    if idx_d < len(xd):
        bx2 = px + int(float(xd[idx_d]) / x_max * pw)
        by2 = py2 + ph - int(float(yd[idx_d]) / y_max * ph)
        pygame.draw.circle(screen, RED, (bx2, by2), 6)

    # Legend
    pygame.draw.line(screen, BLUE, (gr.right - 200, gr.y + 14), (gr.right - 175, gr.y + 14), 3)
    screen.blit(font_small.render("Ideal (no drag)", True, BLUE), (gr.right - 170, gr.y + 8))
    pygame.draw.line(screen, RED, (gr.right - 200, gr.y + 30), (gr.right - 175, gr.y + 30), 3)
    screen.blit(font_small.render("With drag", True, RED), (gr.right - 170, gr.y + 24))

    # ── Info panel ──────────────────────────────────────────────────
    info = pygame.Rect(810, 50, 450, 400)
    pygame.draw.rect(screen, PANEL_BG, info, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, info, 1, border_radius=6)
    screen.blit(font_heading.render("Results", True, WHITE), (info.x + 15, info.y + 10))

    loss = (1 - rngd / rngi) * 100 if rngi > 0 else 0
    lines = [
        "─── Ideal (no drag) ───",
        f"  Range:       {rngi:.1f} m",
        f"  Max height:  {mhi:.1f} m",
        f"  Flight time: {tfi:.2f} s",
        "",
        "─── With drag ───",
        f"  Range:       {rngd:.1f} m",
        f"  Max height:  {mhd:.1f} m",
        f"  Flight time: {tfd:.2f} s",
        "",
        f"Range loss:    {loss:.1f}%",
        "",
        "─── Parameters ───",
        f"  v₀ = {sl_v0.val:.0f} m/s",
        f"  θ  = {sl_ang.val:.0f}°",
        f"  Cd = {sl_cd.val:.2f}",
        f"  m  = {sl_mass.val:.1f} kg",
    ]
    for i, l in enumerate(lines):
        c = YELLOW if "loss" in l.lower() else BLUE if "ideal" in l.lower() \
            else RED if "with drag" in l.lower() else WHITE if l.strip() else GRAY
        if l.startswith("───"): c = GRAY
        screen.blit(font_small.render(l, True, c), (info.x + 15, info.y + 35 + i * 21))

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
