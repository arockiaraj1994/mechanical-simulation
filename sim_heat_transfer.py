"""
1D Transient Heat Conduction — Pygame Simulation
Finite difference: dT/dt = alpha * d²T/dx²
"""

import sys
import math
import pygame
import numpy as np

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Heat Conduction Simulation")
clock = pygame.time.Clock()
FPS = 30

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

MATERIALS = {'Copper': 1.11e-4, 'Aluminum': 9.7e-5, 'Steel': 1.2e-5, 'Wood': 1.5e-7}
MAT_LIST = list(MATERIALS.keys())
T_INIT = 25.0
NX = 100


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
def solve_heat(alpha, T_left, T_right, rod_len, nx=NX):
    dx = rod_len / (nx - 1)
    dt = 0.4 * dx**2 / alpha
    x = np.linspace(0, rod_len, nx)
    total_time = rod_len**2 / (4 * alpha)
    n_steps = int(total_time / dt) + 1
    n_snaps = 200
    save_every = max(1, n_steps // n_snaps)
    T = np.full(nx, T_INIT)
    T[0], T[-1] = T_left, T_right
    snaps, times = [T.copy()], [0.0]
    r = alpha * dt / dx**2
    for step in range(1, n_steps + 1):
        Tn = T.copy()
        Tn[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
        Tn[0], Tn[-1] = T_left, T_right
        T = Tn
        if step % save_every == 0:
            snaps.append(T.copy()); times.append(step * dt)
    return x, np.array(snaps), np.array(times)


def temp_to_color(t_val, t_min, t_max):
    """Map temperature to blue-red gradient."""
    if t_max - t_min < 1e-9:
        f = 0.5
    else:
        f = max(0, min(1, (t_val - t_min) / (t_max - t_min)))
    r = int(50 + 205 * f)
    b = int(205 - 180 * f)
    g = int(50 + 100 * (1 - abs(f - 0.5) * 2))
    return (r, g, b)


# ── Controls ────────────────────────────────────────────────────────
PY = 490
sl_Tl   = Slider(20, PY + 22, 220, "T left (°C)", 0, 500, 400, 10, ".0f")
sl_Tr   = Slider(20, PY + 70, 220, "T right (°C)", 0, 500, 100, 10, ".0f")
sl_Ln   = Slider(310, PY + 22, 220, "Rod Length (m)", 0.2, 3.0, 1.0, 0.1)
sl_time = Slider(310, PY + 70, 220, "Time →", 0.0, 1.0, 0.0, 0.01, ".2f")
sliders = [sl_Tl, sl_Tr, sl_Ln, sl_time]

btn_mat = Btn(600, PY + 12, 120, 32, MAT_LIST[2])
btn_reset = Btn(600, PY + 56, 90, 32, "Reset")
mat_idx = 2

# ── Precompute ──────────────────────────────────────────────────────
x_arr, snaps, times = solve_heat(MATERIALS[MAT_LIST[mat_idx]], 400, 100, 1.0)
prev_phys = (MAT_LIST[mat_idx], 400.0, 100.0, 1.0)

running = True
while running:
    clock.tick(FPS)
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE: running = False
        for s in sliders: s.handle(ev)
        if btn_mat.clicked(ev):
            mat_idx = (mat_idx + 1) % len(MAT_LIST)
            btn_mat.text = MAT_LIST[mat_idx]
        if btn_reset.clicked(ev):
            for s in sliders: s.reset()
            mat_idx = 2; btn_mat.text = MAT_LIST[2]

    cur_phys = (MAT_LIST[mat_idx], sl_Tl.val, sl_Tr.val, sl_Ln.val)
    if cur_phys != prev_phys:
        alpha = MATERIALS[cur_phys[0]]
        x_arr, snaps, times = solve_heat(alpha, sl_Tl.val, sl_Tr.val, sl_Ln.val)
        prev_phys = cur_phys

    t_frac = sl_time.val
    snap_idx = int(t_frac * (len(snaps) - 1))
    snap_idx = max(0, min(snap_idx, len(snaps) - 1))
    T_cur = snaps[snap_idx]
    T_init_snap = snaps[0]
    T_ss = snaps[-1]
    elapsed = float(times[snap_idx]) if snap_idx < len(times) else float(times[-1])

    T_min = min(T_INIT, sl_Tl.val, sl_Tr.val) - 10
    T_max = max(sl_Tl.val, sl_Tr.val) + 10

    screen.fill(BG)
    mat_name = MAT_LIST[mat_idx]
    screen.blit(font_title.render(
        f"1D Heat Conduction — {mat_name} | Time: {elapsed:.2f} s", True, WHITE),
        (20, 10))

    # ── Color-mapped rod ────────────────────────────────────────────
    rod_rect = pygame.Rect(60, 55, 700, 50)
    pygame.draw.rect(screen, DARK_GRAY, rod_rect, border_radius=4)
    seg_w = rod_rect.w / len(T_cur)
    for i in range(len(T_cur)):
        c = temp_to_color(float(T_cur[i]), T_min, T_max)
        rx = rod_rect.x + int(i * seg_w)
        rw = max(1, int(seg_w) + 1)
        pygame.draw.rect(screen, c, (rx, rod_rect.y, rw, rod_rect.h))
    pygame.draw.rect(screen, GRAY, rod_rect, 2, border_radius=4)

    screen.blit(font_value.render(f"{sl_Tl.val:.0f}°C", True, RED),
                (rod_rect.x - 5, rod_rect.bottom + 5))
    screen.blit(font_value.render(f"{sl_Tr.val:.0f}°C", True, BLUE),
                (rod_rect.right - 35, rod_rect.bottom + 5))
    screen.blit(font_heading.render("Temperature Color Map", True, WHITE),
                (rod_rect.x + rod_rect.w // 2 - 80, rod_rect.y - 20))

    # Color scale bar
    csb_x, csb_y, csb_w, csb_h = rod_rect.right + 30, rod_rect.y, 20, rod_rect.h
    for row in range(csb_h):
        f = 1.0 - row / csb_h
        c = temp_to_color(T_min + f * (T_max - T_min), T_min, T_max)
        pygame.draw.line(screen, c, (csb_x, csb_y + row), (csb_x + csb_w, csb_y + row))
    pygame.draw.rect(screen, GRAY, (csb_x, csb_y, csb_w, csb_h), 1)
    screen.blit(font_small.render(f"{T_max:.0f}°C", True, WHITE), (csb_x + csb_w + 4, csb_y - 2))
    screen.blit(font_small.render(f"{T_min:.0f}°C", True, WHITE), (csb_x + csb_w + 4, csb_y + csb_h - 12))

    # ── Temperature profile graph ───────────────────────────────────
    gr = pygame.Rect(60, 140, 700, 300)
    pygame.draw.rect(screen, PANEL_BG, gr, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, gr, 1, border_radius=6)
    screen.blit(font_heading.render("Temperature Distribution", True, WHITE), (gr.x + 10, gr.y + 6))

    mg = 50; gpx, gpy = gr.x + mg, gr.y + 30
    gpw, gph = gr.w - mg - 15, gr.h - 55

    pygame.draw.line(screen, GRAY, (gpx, gpy + gph), (gpx + gpw, gpy + gph), 1)
    pygame.draw.line(screen, GRAY, (gpx, gpy), (gpx, gpy + gph), 1)

    for i in range(6):
        f = i / 5; yy = gpy + gph - int(f * gph)
        pygame.draw.line(screen, DARK_GRAY, (gpx, yy), (gpx + gpw, yy), 1)
        v = T_min + f * (T_max - T_min)
        screen.blit(font_small.render(f"{v:.0f}", True, GRAY), (gpx - 40, yy - 6))
    for i in range(6):
        f = i / 5; xx = gpx + int(f * gpw)
        v = f * sl_Ln.val
        screen.blit(font_small.render(f"{v:.2f}", True, GRAY), (xx - 12, gpy + gph + 4))
    screen.blit(font_small.render("Position (m)", True, GRAY), (gpx + gpw // 2 - 30, gpy + gph + 18))

    def plot_line(data, color, width=2):
        step = max(1, len(data) // (gpw // 2))
        pts = []
        for j in range(0, len(data), step):
            fx = j / (len(data) - 1)
            fy = (float(data[j]) - T_min) / (T_max - T_min)
            pts.append((gpx + int(fx * gpw), gpy + gph - int(fy * gph)))
        if len(pts) > 1:
            pygame.draw.lines(screen, color, False, pts, width)

    plot_line(T_init_snap, DARK_GRAY, 1)
    plot_line(T_ss, (80, 80, 140), 1)
    plot_line(T_cur, RED, 3)

    # Legend
    lx = gr.right - 180; ly = gr.y + 14
    pygame.draw.line(screen, RED, (lx, ly), (lx + 20, ly), 3)
    screen.blit(font_small.render("Current", True, RED), (lx + 25, ly - 6))
    pygame.draw.line(screen, DARK_GRAY, (lx, ly + 16), (lx + 20, ly + 16), 1)
    screen.blit(font_small.render("Initial", True, GRAY), (lx + 25, ly + 10))
    pygame.draw.line(screen, (80, 80, 140), (lx, ly + 32), (lx + 20, ly + 32), 1)
    screen.blit(font_small.render("Steady state", True, (80, 80, 140)), (lx + 25, ly + 26))

    # ── Info panel ──────────────────────────────────────────────────
    info = pygame.Rect(790, 140, 470, 300)
    pygame.draw.rect(screen, PANEL_BG, info, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, info, 1, border_radius=6)
    screen.blit(font_heading.render("Parameters", True, WHITE), (info.x + 15, info.y + 10))

    alpha_val = MATERIALS[mat_name]
    lines = [
        f"Material:       {mat_name}",
        f"α (diffusivity): {alpha_val:.2e} m²/s",
        f"Rod length:      {sl_Ln.val:.1f} m",
        f"T left:          {sl_Tl.val:.0f} °C",
        f"T right:         {sl_Tr.val:.0f} °C",
        f"Initial temp:    {T_INIT:.0f} °C",
        "",
        f"Time elapsed:    {elapsed:.3f} s",
        f"Time fraction:   {t_frac:.0%}",
        "",
        f"Min temp now:    {float(T_cur.min()):.1f} °C",
        f"Max temp now:    {float(T_cur.max()):.1f} °C",
    ]
    for i, l in enumerate(lines):
        screen.blit(font_small.render(l, True, WHITE if l else GRAY),
                    (info.x + 15, info.y + 35 + i * 21))

    # ── Controls bar ────────────────────────────────────────────────
    pygame.draw.rect(screen, PANEL_BG, (0, PY - 20, WIDTH, HEIGHT - PY + 20))
    pygame.draw.line(screen, DARK_GRAY, (0, PY - 20), (WIDTH, PY - 20), 1)
    for s in sliders: s.draw(screen)
    btn_mat.draw(screen)
    btn_reset.draw(screen)
    screen.blit(font_small.render("Click material button to cycle  |  Drag Time slider to scrub  |  ESC: Quit",
                                  True, GRAY), (WIDTH // 2 - 220, HEIGHT - 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
