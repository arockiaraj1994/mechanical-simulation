"""
Beam Bending / Deflection — Pygame Simulation
Euler-Bernoulli beam theory with point load.
Simply Supported and Cantilever beams.
"""

import sys
import math
import pygame
import numpy as np

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Beam Bending Simulation")
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
ORANGE    = (230, 126, 34)
DARK_GRAY = (55, 58, 68)
SLIDER_BG = (50, 54, 66)
SLIDER_FILL = (74, 144, 217)
KNOB      = (200, 210, 225)

MATERIALS = {'Steel': 200e9, 'Aluminum': 69e9, 'Wood': 12e9}
MAT_LIST = list(MATERIALS.keys())


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
    def __init__(self, x, y, w, h, text, toggle=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text, self.toggle, self.active = text, toggle, False

    def draw(self, surf):
        if self.toggle:
            c = GREEN if self.active else DARK_GRAY
        else:
            c = SLIDER_BG if self.rect.collidepoint(pygame.mouse.get_pos()) else DARK_GRAY
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        pygame.draw.rect(surf, GRAY, self.rect, 1, border_radius=6)
        surf.blit(font_label.render(self.text, True, WHITE),
                  font_label.render(self.text, True, WHITE).get_rect(center=self.rect.center))

    def clicked(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos):
            if self.toggle: self.active = not self.active
            return True
        return False


# ── Physics ─────────────────────────────────────────────────────────
N = 300

def calc_simply_supported(x, L, P, a, E, I):
    b = L - a; Ra = P * b / L
    defl = np.zeros_like(x); M = np.zeros_like(x); V = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= a:
            defl[i] = -(P * b * xi) / (6 * L * E * I) * (L**2 - b**2 - xi**2)
            M[i] = Ra * xi; V[i] = Ra
        else:
            defl[i] = -(P * a * (L - xi)) / (6 * L * E * I) * (2 * L * (L - xi) - a**2 - (L - xi)**2)
            M[i] = Ra * xi - P * (xi - a); V[i] = Ra - P
    return defl, M, V

def calc_cantilever(x, L, P, a, E, I):
    defl = np.zeros_like(x); M = np.zeros_like(x); V = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= a:
            defl[i] = -(P * xi**2) / (6 * E * I) * (3 * a - xi)
            M[i] = -P * (a - xi); V[i] = P
        else:
            defl[i] = -(P * a**2) / (6 * E * I) * (3 * xi - a)
    return defl, M, V

def compute(L, P, a_frac, mat_idx, I_val, cantilever):
    E = MATERIALS[MAT_LIST[mat_idx]]
    a = a_frac * L
    x = np.linspace(0, L, N)
    fn = calc_cantilever if cantilever else calc_simply_supported
    d, m, v = fn(x, L, P, a, E, I_val)
    return x, d, m, v, a


# ── Graph helper ────────────────────────────────────────────────────
def draw_graph(surf, rect, xd, yd, color, ylabel, title, fill=True):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surf, DARK_GRAY, rect, 1, border_radius=6)
    mg = 48; px, py = rect.x + mg, rect.y + 24
    pw, ph = rect.w - mg - 10, rect.h - 44
    surf.blit(font_heading.render(title, True, WHITE), (rect.x + 10, rect.y + 4))
    if len(xd) < 2: return
    xmin, xmax = float(xd[0]), float(xd[-1])
    ymin, ymax = float(np.min(yd)), float(np.max(yd))
    if abs(ymax - ymin) < 1e-12: ymin -= 1; ymax += 1
    pad = (ymax - ymin) * 0.12; ymin -= pad; ymax += pad
    zero_y = py + ph - int((0 - ymin) / (ymax - ymin) * ph)
    pygame.draw.line(surf, GRAY, (px, zero_y), (px + pw, zero_y), 1)
    pygame.draw.line(surf, GRAY, (px, py), (px, py + ph), 1)
    for i in range(5):
        f = i / 4; yy = py + ph - int(f * ph)
        pygame.draw.line(surf, DARK_GRAY, (px, yy), (px + pw, yy), 1)
        surf.blit(font_small.render(f"{ymin + f * (ymax - ymin):.2g}", True, GRAY), (px - 45, yy - 6))
    step = max(1, len(xd) // (pw // 2))
    pts = []
    for i in range(0, len(xd), step):
        fx = (float(xd[i]) - xmin) / (xmax - xmin)
        fy = (float(yd[i]) - ymin) / (ymax - ymin)
        pts.append((px + int(fx * pw), py + ph - int(fy * ph)))
    if len(pts) > 1:
        if fill:
            fill_pts = [(pts[0][0], zero_y)] + pts + [(pts[-1][0], zero_y)]
            pygame.draw.polygon(surf, (*color, 40) if len(color) == 3 else color, fill_pts)
        pygame.draw.lines(surf, color, False, pts, 2)
    lbl = font_small.render(ylabel, True, GRAY)
    surf.blit(lbl, (px - 45, py - 16))


# ── Controls ────────────────────────────────────────────────────────
PY = 490
sl_L  = Slider(20, PY + 22, 220, "Length L (m)", 1, 10, 5.0, 0.5)
sl_P  = Slider(20, PY + 70, 220, "Load P (N)", 100, 10000, 5000, 100, ".0f")
sl_a  = Slider(310, PY + 22, 220, "Load Position (frac)", 0.05, 0.95, 0.5, 0.05, ".2f")
sl_I  = Slider(310, PY + 70, 220, "I (×10⁻⁵ m⁴)", 0.1, 10, 1.0, 0.1)
sliders = [sl_L, sl_P, sl_a, sl_I]

btn_mat = Btn(600, PY + 12, 120, 32, MAT_LIST[0])
btn_type = Btn(600, PY + 56, 140, 32, "Simply Supported", toggle=True)
btn_reset = Btn(760, PY + 56, 90, 32, "Reset")
mat_idx = 0

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
        btn_type.clicked(ev)
        if btn_reset.clicked(ev):
            for s in sliders: s.reset()
            mat_idx = 0; btn_mat.text = MAT_LIST[0]; btn_type.active = False

    x, defl, M, V, a = compute(sl_L.val, sl_P.val, sl_a.val, mat_idx, sl_I.val * 1e-5, btn_type.active)
    defl_mm = defl * 1000
    M_kn = M / 1000
    V_kn = V / 1000

    screen.fill(BG)
    btype = "Cantilever" if btn_type.active else "Simply Supported"
    mat = MAT_LIST[mat_idx]
    E_gpa = MATERIALS[mat] / 1e9
    max_d = float(np.max(np.abs(defl_mm)))
    ttl = f"Beam Bending — {btype} | {mat} (E={E_gpa:.0f} GPa) | Max deflection: {max_d:.2f} mm"
    screen.blit(font_title.render(ttl, True, WHITE), (20, 10))

    # ── Beam visual ─────────────────────────────────────────────────
    beam_x, beam_y, beam_w = 60, 58, 500
    beam_h = 12
    pygame.draw.rect(screen, BLUE, (beam_x, beam_y, beam_w, beam_h), border_radius=2)

    load_px = beam_x + int(sl_a.val * beam_w)
    pygame.draw.line(screen, RED, (load_px, beam_y - 30), (load_px, beam_y), 3)
    pts = [(load_px - 6, beam_y - 8), (load_px + 6, beam_y - 8), (load_px, beam_y)]
    pygame.draw.polygon(screen, RED, pts)
    screen.blit(font_small.render(f"P={sl_P.val:.0f}N", True, RED), (load_px - 25, beam_y - 45))

    if btn_type.active:
        pygame.draw.rect(screen, GRAY, (beam_x - 10, beam_y - 5, 10, beam_h + 20))
        for yy in range(beam_y - 5, beam_y + beam_h + 15, 6):
            pygame.draw.line(screen, DARK_GRAY, (beam_x - 10, yy), (beam_x - 16, yy + 4), 1)
    else:
        tri_h = 18
        pygame.draw.polygon(screen, GRAY,
                            [(beam_x, beam_y + beam_h), (beam_x - 8, beam_y + beam_h + tri_h),
                             (beam_x + 8, beam_y + beam_h + tri_h)])
        pygame.draw.polygon(screen, GRAY,
                            [(beam_x + beam_w, beam_y + beam_h),
                             (beam_x + beam_w - 8, beam_y + beam_h + tri_h),
                             (beam_x + beam_w + 8, beam_y + beam_h + tri_h)])

    scale = beam_w / sl_L.val
    defl_scale = 80 / (max_d + 0.001)
    defl_pts = []
    step = max(1, N // beam_w)
    for i in range(0, N, step):
        px_i = beam_x + int(float(x[i]) * scale)
        py_i = beam_y + beam_h + 5 - int(float(defl_mm[i]) * defl_scale)
        defl_pts.append((px_i, py_i))
    if len(defl_pts) > 1:
        pygame.draw.lines(screen, YELLOW, False, defl_pts, 2)

    # ── 3 diagrams ──────────────────────────────────────────────────
    gw = 380; gh = 130
    draw_graph(screen, pygame.Rect(600, 42, gw, gh), x, defl_mm, BLUE, "mm", "Deflection")
    draw_graph(screen, pygame.Rect(600, 180, gw, gh), x, M_kn, ORANGE, "kN·m", "Bending Moment")
    draw_graph(screen, pygame.Rect(600, 318, gw, gh), x, V_kn, GREEN, "kN", "Shear Force")

    # ── Info box ────────────────────────────────────────────────────
    info = pygame.Rect(60, 100, 500, 360)
    pygame.draw.rect(screen, PANEL_BG, info, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, info, 1, border_radius=6)
    screen.blit(font_heading.render("Parameters", True, WHITE), (info.x + 15, info.y + 10))
    lines = [
        f"Beam type:         {btype}",
        f"Material:          {mat} (E = {E_gpa:.0f} GPa)",
        f"Length (L):        {sl_L.val:.1f} m",
        f"Load (P):          {sl_P.val:.0f} N",
        f"Load position:     {sl_a.val * sl_L.val:.2f} m  ({sl_a.val:.0%} of L)",
        f"Moment of I:       {sl_I.val:.1f} × 10⁻⁵ m⁴",
        "",
        f"Max deflection:    {max_d:.3f} mm",
        f"Max moment:        {float(np.max(np.abs(M_kn))):.2f} kN·m",
        f"Max shear:         {float(np.max(np.abs(V_kn))):.2f} kN",
    ]
    for i, l in enumerate(lines):
        screen.blit(font_small.render(l, True, WHITE if l else GRAY), (info.x + 15, info.y + 35 + i * 22))

    # ── Controls bar ────────────────────────────────────────────────
    pygame.draw.rect(screen, PANEL_BG, (0, PY - 20, WIDTH, HEIGHT - PY + 20))
    pygame.draw.line(screen, DARK_GRAY, (0, PY - 20), (WIDTH, PY - 20), 1)
    for s in sliders: s.draw(screen)
    btn_mat.draw(screen)
    btn_type.text = "Cantilever" if btn_type.active else "Simply Supp."
    btn_type.draw(screen)
    btn_reset.draw(screen)
    screen.blit(font_small.render("Click material button to cycle  |  ESC: Quit", True, GRAY),
                (WIDTH // 2 - 150, HEIGHT - 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
