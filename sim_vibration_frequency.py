"""
Vibration & Frequency Analysis — 5 Topics in One Pygame App

Keys 1-5 switch topics live. Sliders adjust parameters in real time.

1 = Free Vibration         (undamped sine wave)
2 = Damped Vibration       (exponentially decaying sine)
3 = Forced Vibration       (steady-state response to external force)
4 = Resonance              (amplitude spike when forcing freq = natural freq)
5 = FFT of Vibration       (time-domain signal → frequency spectrum)
"""

import sys
import math
import pygame
import numpy as np

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vibration & Frequency Analysis")
clock = pygame.time.Clock()
FPS = 60

# ── Fonts & Colors ──────────────────────────────────────────────────
font_title = pygame.font.SysFont("DejaVu Sans", 22, bold=True)
font_label = pygame.font.SysFont("DejaVu Sans", 14)
font_value = pygame.font.SysFont("DejaVu Sans", 13, bold=True)
font_small = pygame.font.SysFont("DejaVu Sans", 12)
font_heading = pygame.font.SysFont("DejaVu Sans", 15, bold=True)
font_tab = pygame.font.SysFont("DejaVu Sans", 13, bold=True)
font_physics = pygame.font.SysFont("DejaVu Sans", 12)

BG          = (24, 26, 33)
PANEL_BG    = (34, 37, 46)
WHITE       = (220, 225, 235)
GRAY        = (100, 105, 115)
BLUE        = (74, 144, 217)
RED         = (231, 76, 60)
GREEN       = (46, 204, 113)
YELLOW      = (241, 196, 15)
ORANGE      = (230, 126, 34)
PURPLE      = (155, 89, 182)
DARK_GRAY   = (55, 58, 68)
SLIDER_BG   = (50, 54, 66)
SLIDER_FILL = (74, 144, 217)
KNOB        = (200, 210, 225)
TAB_ACTIVE  = (74, 144, 217)
TAB_INACTIVE = (44, 47, 56)


# ── UI Widgets ──────────────────────────────────────────────────────
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
        self.rect = pygame.Rect(x, y, w, h); self.text = text

    def draw(self, surf):
        c = SLIDER_BG if self.rect.collidepoint(pygame.mouse.get_pos()) else DARK_GRAY
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        pygame.draw.rect(surf, GRAY, self.rect, 1, border_radius=6)
        t = font_label.render(self.text, True, WHITE)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, ev):
        return ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos)


# ── Graph drawing ───────────────────────────────────────────────────
def draw_graph(surf, rect, xd, yd, color, xlabel, ylabel, title,
               extra_lines=None, marker_idx=None, fill=False):
    """Generic graph renderer. extra_lines = [(xd, yd, color, width, label), ...]"""
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surf, DARK_GRAY, rect, 1, border_radius=6)
    mg = 55
    px, py, pw, ph = rect.x + mg, rect.y + 28, rect.w - mg - 15, rect.h - 55
    surf.blit(font_heading.render(title, True, WHITE), (rect.x + 10, rect.y + 6))
    if len(xd) < 2: return

    xmin, xmax = float(np.min(xd)), float(np.max(xd))
    ymin, ymax = float(np.min(yd)), float(np.max(yd))
    if extra_lines:
        for exd, eyd, *_ in extra_lines:
            ymin = min(ymin, float(np.min(eyd)))
            ymax = max(ymax, float(np.max(eyd)))
    if abs(ymax - ymin) < 1e-9: ymin -= 1; ymax += 1
    pad = (ymax - ymin) * 0.1; ymin -= pad; ymax += pad
    if abs(xmax - xmin) < 1e-9: xmax = xmin + 1

    # Axes and grid
    pygame.draw.line(surf, GRAY, (px, py + ph), (px + pw, py + ph), 1)
    pygame.draw.line(surf, GRAY, (px, py), (px, py + ph), 1)
    for i in range(6):
        f = i / 5
        yy = py + ph - int(f * ph)
        pygame.draw.line(surf, DARK_GRAY, (px, yy), (px + pw, yy), 1)
        surf.blit(font_small.render(f"{ymin + f * (ymax - ymin):.2g}", True, GRAY), (px - 50, yy - 6))
        xx = px + int(f * pw)
        surf.blit(font_small.render(f"{xmin + f * (xmax - xmin):.2g}", True, GRAY), (xx - 12, py + ph + 4))
    surf.blit(font_small.render(xlabel, True, GRAY), (px + pw // 2 - 25, py + ph + 18))

    def to_pts(xarr, yarr):
        step = max(1, len(xarr) // (pw // 2))
        pts = []
        for j in range(0, len(xarr), step):
            fx = (float(xarr[j]) - xmin) / (xmax - xmin)
            fy = (float(yarr[j]) - ymin) / (ymax - ymin)
            pts.append((px + int(fx * pw), py + ph - int(fy * ph)))
        return pts

    if extra_lines:
        for exd, eyd, ec, ew, *rest in extra_lines:
            epts = to_pts(exd, eyd)
            if len(epts) > 1:
                pygame.draw.lines(surf, ec, False, epts, ew)

    pts = to_pts(xd, yd)
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 2)

    if marker_idx is not None and 0 <= marker_idx < len(xd):
        fx = (float(xd[marker_idx]) - xmin) / (xmax - xmin)
        fy = (float(yd[marker_idx]) - ymin) / (ymax - ymin)
        mx_p = px + int(fx * pw)
        my_p = py + ph - int(fy * ph)
        pygame.draw.circle(surf, YELLOW, (mx_p, my_p), 6)


def draw_bar_chart(surf, rect, freqs, mags, color, xlabel, ylabel, title,
                   annotations=None):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surf, DARK_GRAY, rect, 1, border_radius=6)
    mg = 55
    px, py, pw, ph = rect.x + mg, rect.y + 28, rect.w - mg - 15, rect.h - 55
    surf.blit(font_heading.render(title, True, WHITE), (rect.x + 10, rect.y + 6))
    if len(freqs) < 2: return

    xmin, xmax = float(freqs[0]), float(freqs[-1])
    ymax = float(np.max(mags)) * 1.2 + 0.01
    ymin = 0

    pygame.draw.line(surf, GRAY, (px, py + ph), (px + pw, py + ph), 1)
    pygame.draw.line(surf, GRAY, (px, py), (px, py + ph), 1)
    for i in range(6):
        f = i / 5
        yy = py + ph - int(f * ph)
        pygame.draw.line(surf, DARK_GRAY, (px, yy), (px + pw, yy), 1)
        surf.blit(font_small.render(f"{f * ymax:.2f}", True, GRAY), (px - 45, yy - 6))
        xx = px + int(f * pw)
        surf.blit(font_small.render(f"{xmin + f * (xmax - xmin):.0f}", True, GRAY), (xx - 10, py + ph + 4))
    surf.blit(font_small.render(xlabel, True, GRAY), (px + pw // 2 - 25, py + ph + 18))

    bar_w = max(1, pw // len(freqs))
    for j in range(len(freqs)):
        fx = (float(freqs[j]) - xmin) / (xmax - xmin) if xmax > xmin else 0
        fy = float(mags[j]) / ymax if ymax > 0 else 0
        bx = px + int(fx * pw)
        bh = int(fy * ph)
        if bh > 0:
            pygame.draw.rect(surf, color, (bx, py + ph - bh, max(bar_w, 2), bh))

    if annotations:
        for freq_val, label in annotations:
            fx = (freq_val - xmin) / (xmax - xmin) if xmax > xmin else 0
            idx = int(np.argmin(np.abs(freqs - freq_val)))
            fy = float(mags[idx]) / ymax if ymax > 0 else 0
            ax_p = px + int(fx * pw)
            ay_p = py + ph - int(fy * ph)
            surf.blit(font_value.render(label, True, WHITE), (ax_p + 5, ay_p - 18))
            pygame.draw.circle(surf, YELLOW, (ax_p, ay_p), 4)


# ── Sliders ─────────────────────────────────────────────────────────
PY = 510
sl_m  = Slider(20,  PY + 22, 220, "Mass m (kg)", 0.5, 10.0, 1.0, 0.1)
sl_k  = Slider(20,  PY + 70, 220, "Stiffness k (N/m)", 5, 200, 40.0, 1, ".0f")
sl_c  = Slider(310, PY + 22, 220, "Damping c (Ns/m)", 0.0, 20.0, 2.0, 0.1)
sl_x0 = Slider(310, PY + 70, 220, "Init. Disp. x₀ (m)", 0.1, 3.0, 1.0, 0.1)
sl_F0 = Slider(600, PY + 22, 220, "Force F₀ (N)", 0.5, 20.0, 5.0, 0.5)
sliders = [sl_m, sl_k, sl_c, sl_x0, sl_F0]
btn_reset = Btn(880, PY + 60, 90, 32, "Reset")

# ── Topic tabs ──────────────────────────────────────────────────────
TAB_NAMES = [
    "1: Free Vibration",
    "2: Damped",
    "3: Forced",
    "4: Resonance",
    "5: FFT",
]
TAB_W = 150
TAB_H = 32
TAB_Y = 8
topic = 0
sim_t = 0.0

# ── Main loop ───────────────────────────────────────────────────────
running = True
prev_params = None

while running:
    dt = clock.tick(FPS) / 1000.0

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE: running = False
            if ev.key == pygame.K_r: sim_t = 0.0
            if ev.key in (pygame.K_1, pygame.K_KP1): topic = 0; sim_t = 0.0
            if ev.key in (pygame.K_2, pygame.K_KP2): topic = 1; sim_t = 0.0
            if ev.key in (pygame.K_3, pygame.K_KP3): topic = 2; sim_t = 0.0
            if ev.key in (pygame.K_4, pygame.K_KP4): topic = 3; sim_t = 0.0
            if ev.key in (pygame.K_5, pygame.K_KP5): topic = 4; sim_t = 0.0
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i in range(5):
                tx = 20 + i * (TAB_W + 6)
                r = pygame.Rect(tx, TAB_Y, TAB_W, TAB_H)
                if r.collidepoint(ev.pos):
                    topic = i; sim_t = 0.0
        for s in sliders: s.handle(ev)
        if btn_reset.clicked(ev):
            for s in sliders: s.reset()
            sim_t = 0.0

    # ── Derived physics ─────────────────────────────────────────────
    m_v = sl_m.val; k_v = sl_k.val; c_v = sl_c.val
    x0_v = sl_x0.val; F0_v = sl_F0.val
    wn = math.sqrt(k_v / m_v)
    fn = wn / (2 * math.pi)
    zeta = c_v / (2 * math.sqrt(k_v * m_v))
    wd = wn * math.sqrt(1 - zeta**2) if zeta < 1 else 0.0

    screen.fill(BG)

    # ── Tabs ────────────────────────────────────────────────────────
    for i, name in enumerate(TAB_NAMES):
        tx = 20 + i * (TAB_W + 6)
        r = pygame.Rect(tx, TAB_Y, TAB_W, TAB_H)
        c = TAB_ACTIVE if i == topic else TAB_INACTIVE
        pygame.draw.rect(screen, c, r, border_radius=6)
        pygame.draw.rect(screen, GRAY if i != topic else BLUE, r, 1, border_radius=6)
        screen.blit(font_tab.render(name, True, WHITE), (tx + 10, TAB_Y + 8))

    # ── Common info bar ─────────────────────────────────────────────
    info_y = 46
    info_str = (f"ωn = {wn:.2f} rad/s ({fn:.2f} Hz)   |   "
                f"ζ = {zeta:.3f}   |   "
                f"{'Underdamped' if zeta < 1 else 'Critically' if abs(zeta-1)<0.01 else 'Overdamped'}")
    if zeta < 1:
        info_str += f"   |   ωd = {wd:.2f} rad/s"
    screen.blit(font_value.render(info_str, True, GRAY), (20, info_y))

    graph_rect = pygame.Rect(20, 68, 780, 400)
    info_rect = pygame.Rect(820, 68, 440, 400)

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 1 — Free Vibration
    # ═════════════════════════════════════════════════════════════════
    if topic == 0:
        # x(t) = x0 * cos(ωn * t) — pure cosine, no energy loss
        T_END = 6.0
        t = np.linspace(0, T_END, 800)
        x = x0_v * np.cos(wn * t)

        sim_t += dt
        if sim_t > T_END: sim_t = 0.0
        idx = int(sim_t / T_END * (len(t) - 1))
        idx = max(0, min(idx, len(t) - 1))

        draw_graph(screen, graph_rect, t[:idx+1], x[:idx+1], BLUE,
                   "Time (s)", "x (m)",
                   f"Free Vibration (undamped) — x(t) = x₀ cos(ωn t)",
                   marker_idx=idx)

        # Info
        pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)
        screen.blit(font_heading.render("Free Vibration", True, GREEN), (info_rect.x + 15, info_rect.y + 10))
        lines = [
            "",
            "Physics:",
            "  A mass on a spring with NO friction",
            "  oscillates forever at the natural",
            "  frequency.",
            "",
            "Equation:",
            "  mx'' + kx = 0",
            "  x(t) = x₀ cos(ωn t)",
            "",
            f"  ωn = sqrt(k/m) = {wn:.2f} rad/s",
            f"  Period T = {2*math.pi/wn:.3f} s",
            f"  Frequency = {fn:.2f} Hz",
            "",
            f"  Current x = {float(x[idx]):.4f} m",
            f"  Time = {sim_t:.2f} s",
        ]
        for i, l in enumerate(lines):
            c = GREEN if "Physics" in l else ORANGE if "Equation" in l else WHITE
            screen.blit(font_physics.render(l, True, c), (info_rect.x + 15, info_rect.y + 30 + i * 21))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 2 — Damped Vibration
    # ═════════════════════════════════════════════════════════════════
    elif topic == 1:
        # x(t) = x0 * e^(-ζωn t) * cos(ωd t) for underdamped
        T_END = 8.0
        t = np.linspace(0, T_END, 1000)

        if zeta < 1:
            x = x0_v * np.exp(-zeta * wn * t) * np.cos(wd * t)
            dtype = "Underdamped"
        elif abs(zeta - 1) < 0.05:
            x = x0_v * (1 + wn * t) * np.exp(-wn * t)
            dtype = "Critically Damped"
        else:
            s1 = -zeta * wn + wn * math.sqrt(zeta**2 - 1)
            s2 = -zeta * wn - wn * math.sqrt(zeta**2 - 1)
            A_ = x0_v * s2 / (s2 - s1); B_ = -x0_v * s1 / (s2 - s1)
            x = A_ * np.exp(s1 * t) + B_ * np.exp(s2 * t)
            dtype = "Overdamped"

        envelope = x0_v * np.exp(-zeta * wn * t)

        sim_t += dt
        if sim_t > T_END: sim_t = 0.0
        idx = int(sim_t / T_END * (len(t) - 1))
        idx = max(0, min(idx, len(t) - 1))

        draw_graph(screen, graph_rect, t[:idx+1], x[:idx+1], GREEN,
                   "Time (s)", "x (m)",
                   f"Damped Vibration — {dtype} (ζ = {zeta:.3f})",
                   extra_lines=[(t, envelope, GRAY, 1), (t, -envelope, GRAY, 1)],
                   marker_idx=idx)

        pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)
        screen.blit(font_heading.render("Damped Vibration", True, GREEN), (info_rect.x + 15, info_rect.y + 10))
        lines = [
            "",
            "Physics:",
            "  Energy is lost to friction. The",
            "  amplitude decays exponentially.",
            "",
            "Equation:",
            "  mx'' + cx' + kx = 0",
            f"  ζ = c/(2√(km)) = {zeta:.3f}",
            "",
            "  ζ < 1 → Underdamped (oscillates)",
            "  ζ = 1 → Critically damped",
            "  ζ > 1 → Overdamped (no oscillation)",
            "",
            f"  System is: {dtype}",
            f"  Current x = {float(x[idx]):.4f} m",
        ]
        for i, l in enumerate(lines):
            c = GREEN if "Physics" in l else ORANGE if "Equation" in l \
                else YELLOW if "System is" in l else WHITE
            screen.blit(font_physics.render(l, True, c), (info_rect.x + 15, info_rect.y + 30 + i * 21))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 3 — Forced Vibration
    # ═════════════════════════════════════════════════════════════════
    elif topic == 2:
        # Transient + steady-state response to F(t) = F0 sin(ω_f t)
        w_f = wn * 0.6
        r_ratio = w_f / wn
        X_ss = (F0_v / k_v) / math.sqrt((1 - r_ratio**2)**2 + (2 * zeta * r_ratio)**2)
        phi = math.atan2(2 * zeta * r_ratio, 1 - r_ratio**2)

        T_END = 10.0
        t = np.linspace(0, T_END, 1200)
        if zeta < 1:
            x_tr = x0_v * np.exp(-zeta * wn * t) * np.cos(wd * t)
        else:
            x_tr = x0_v * np.exp(-wn * t)
        x_ss = X_ss * np.sin(w_f * t - phi)
        x_total = x_tr + x_ss

        sim_t += dt
        if sim_t > T_END: sim_t = 0.0
        idx = int(sim_t / T_END * (len(t) - 1))
        idx = max(0, min(idx, len(t) - 1))

        draw_graph(screen, graph_rect, t[:idx+1], x_total[:idx+1], ORANGE,
                   "Time (s)", "x (m)",
                   f"Forced Vibration — ω_force = {w_f:.2f}, ωn = {wn:.2f}",
                   extra_lines=[(t, x_ss, GRAY, 1)],
                   marker_idx=idx)

        pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)
        screen.blit(font_heading.render("Forced Vibration", True, GREEN), (info_rect.x + 15, info_rect.y + 10))
        lines = [
            "",
            "Physics:",
            "  An external force F₀ sin(ωt) drives",
            "  the system. Response = transient +",
            "  steady-state. Transient decays away.",
            "",
            "Equation:",
            "  mx'' + cx' + kx = F₀ sin(ωt)",
            "",
            f"  Forcing freq: {w_f:.2f} rad/s",
            f"  Freq ratio r: {r_ratio:.2f}",
            f"  Steady-state amp X: {X_ss:.4f} m",
            f"  Phase lag φ: {math.degrees(phi):.1f}°",
            "",
            "  Gray line = steady-state only",
            f"  Current x = {float(x_total[idx]):.4f} m",
        ]
        for i, l in enumerate(lines):
            c = GREEN if "Physics" in l else ORANGE if "Equation" in l else WHITE
            screen.blit(font_physics.render(l, True, c), (info_rect.x + 15, info_rect.y + 30 + i * 21))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 4 — Resonance
    # ═════════════════════════════════════════════════════════════════
    elif topic == 3:
        # Amplitude ratio vs frequency ratio for multiple damping values
        r_vals = np.linspace(0.01, 3.0, 400)
        zeta_list = [0.05, 0.1, 0.2, 0.5, 1.0]
        colors_res = [RED, ORANGE, YELLOW, GREEN, BLUE]

        pygame.draw.rect(screen, PANEL_BG, graph_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, graph_rect, 1, border_radius=6)
        screen.blit(font_heading.render("Resonance — Amplitude vs Forcing Frequency", True, WHITE),
                    (graph_rect.x + 10, graph_rect.y + 6))

        mg = 55
        gpx, gpy = graph_rect.x + mg, graph_rect.y + 28
        gpw, gph = graph_rect.w - mg - 15, graph_rect.h - 55
        y_max_plot = 12.0

        pygame.draw.line(screen, GRAY, (gpx, gpy + gph), (gpx + gpw, gpy + gph), 1)
        pygame.draw.line(screen, GRAY, (gpx, gpy), (gpx, gpy + gph), 1)
        for i in range(7):
            f = i / 6; yy = gpy + gph - int(f * gph)
            pygame.draw.line(screen, DARK_GRAY, (gpx, yy), (gpx + gpw, yy), 1)
            screen.blit(font_small.render(f"{f * y_max_plot:.0f}", True, GRAY), (gpx - 30, yy - 6))
        for i in range(7):
            f = i / 6; xx = gpx + int(f * gpw)
            screen.blit(font_small.render(f"{f * 3:.1f}", True, GRAY), (xx - 10, gpy + gph + 4))
        screen.blit(font_small.render("Freq ratio r = ω/ωn", True, GRAY),
                    (gpx + gpw // 2 - 50, gpy + gph + 18))

        # Resonance line
        r1_x = gpx + int(1.0 / 3.0 * gpw)
        pygame.draw.line(screen, GRAY, (r1_x, gpy), (r1_x, gpy + gph), 1)
        screen.blit(font_small.render("r=1", True, GRAY), (r1_x + 3, gpy + 2))

        # Current damping marker
        X_cur = 1.0 / (2 * max(zeta, 0.001))
        cur_y = gpy + gph - int(min(X_cur, y_max_plot) / y_max_plot * gph)
        pygame.draw.circle(screen, YELLOW, (r1_x, cur_y), 7)
        screen.blit(font_value.render(f"Your ζ={zeta:.3f}", True, YELLOW), (r1_x + 12, cur_y - 8))

        for z_i, col in zip(zeta_list, colors_res):
            X_ratio = 1.0 / np.sqrt((1 - r_vals**2)**2 + (2 * z_i * r_vals)**2)
            step = max(1, len(r_vals) // (gpw // 2))
            pts = []
            for j in range(0, len(r_vals), step):
                fx = float(r_vals[j]) / 3.0
                fy = min(float(X_ratio[j]), y_max_plot) / y_max_plot
                pts.append((gpx + int(fx * gpw), gpy + gph - int(fy * gph)))
            if len(pts) > 1:
                pygame.draw.lines(screen, col, False, pts, 2)

        # Legend
        lx = graph_rect.right - 150
        for i, (z_i, col) in enumerate(zip(zeta_list, colors_res)):
            ly = graph_rect.y + 20 + i * 18
            pygame.draw.line(screen, col, (lx, ly + 6), (lx + 20, ly + 6), 2)
            screen.blit(font_small.render(f"ζ = {z_i}", True, col), (lx + 25, ly))

        pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)
        screen.blit(font_heading.render("Resonance", True, GREEN), (info_rect.x + 15, info_rect.y + 10))
        peak_amp = 1.0 / (2 * max(zeta, 0.001))
        lines = [
            "",
            "Physics:",
            "  When forcing frequency equals",
            "  natural frequency (ω = ωn, r = 1),",
            "  the system absorbs max energy.",
            "  Amplitude becomes very large!",
            "",
            "  Only DAMPING limits the peak.",
            "  At resonance: X = 1/(2ζ)",
            "",
            f"  Your ζ = {zeta:.3f}",
            f"  Peak amplitude ratio = {peak_amp:.1f}",
            "",
            "  Low damping = DANGEROUS!",
            "  This breaks bridges & machines.",
        ]
        for i, l in enumerate(lines):
            c = GREEN if "Physics" in l else RED if "DANGEROUS" in l else WHITE
            screen.blit(font_physics.render(l, True, c), (info_rect.x + 15, info_rect.y + 30 + i * 21))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 5 — FFT
    # ═════════════════════════════════════════════════════════════════
    elif topic == 4:
        fs = 500; T_sig = 2.0
        t_sig = np.arange(0, T_sig, 1 / fs)
        f1, f2, f3 = fn, 25.0, 60.0
        signal = (1.0 * np.sin(2 * np.pi * f1 * t_sig)
                  + 0.5 * np.sin(2 * np.pi * f2 * t_sig)
                  + 0.3 * np.sin(2 * np.pi * f3 * t_sig)
                  + 0.15 * np.random.randn(len(t_sig)))
        N_s = len(t_sig)
        fft_mag = 2.0 / N_s * np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(N_s, 1 / fs)
        mask = freqs <= 100
        freqs_m, mag_m = freqs[mask], fft_mag[mask]

        # Time domain (top half of graph area)
        g_top = pygame.Rect(graph_rect.x, graph_rect.y, graph_rect.w, graph_rect.h // 2 - 5)
        draw_graph(screen, g_top, t_sig, signal, BLUE, "Time (s)", "Amp",
                   "Time Domain — Vibration Signal (3 freqs + noise)")

        # Frequency domain (bottom half)
        g_bot = pygame.Rect(graph_rect.x, graph_rect.y + graph_rect.h // 2 + 5,
                            graph_rect.w, graph_rect.h // 2 - 5)
        anns = [(f1, f"f1={f1:.1f}Hz"), (f2, f"f2={f2:.0f}Hz"), (f3, f"f3={f3:.0f}Hz")]
        draw_bar_chart(screen, g_bot, freqs_m, mag_m, RED, "Frequency (Hz)", "Mag",
                       "Frequency Spectrum (FFT)", annotations=anns)

        pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
        pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)
        screen.blit(font_heading.render("FFT Analysis", True, GREEN), (info_rect.x + 15, info_rect.y + 10))
        lines = [
            "",
            "Physics:",
            "  Real vibration signals contain",
            "  multiple frequencies mixed together.",
            "  FFT decomposes the signal into its",
            "  individual frequency components.",
            "",
            "Signal contains:",
            f"  f1 = {f1:.1f} Hz (natural freq)",
            f"  f2 = {f2:.0f} Hz",
            f"  f3 = {f3:.0f} Hz",
            "  + random noise",
            "",
            "  Engineers use FFT to find which",
            "  part of a machine is vibrating.",
        ]
        for i, l in enumerate(lines):
            c = GREEN if "Physics" in l else ORANGE if "Signal" in l else WHITE
            screen.blit(font_physics.render(l, True, c), (info_rect.x + 15, info_rect.y + 30 + i * 21))

    # ── Controls bar ────────────────────────────────────────────────
    pygame.draw.rect(screen, PANEL_BG, (0, PY - 20, WIDTH, HEIGHT - PY + 20))
    pygame.draw.line(screen, DARK_GRAY, (0, PY - 20), (WIDTH, PY - 20), 1)
    for s in sliders: s.draw(screen)
    btn_reset.draw(screen)
    screen.blit(font_small.render("Keys 1-5: Switch topic  |  R: Restart  |  ESC: Quit", True, GRAY),
                (WIDTH // 2 - 160, HEIGHT - 20))

    pygame.display.flip()

pygame.quit()
sys.exit()
