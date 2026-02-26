"""
Spring-Mass Damper System (Quarter-Car Suspension Model)
Interactive real-time simulation using Pygame.
Equation: mx'' + cx' + kx = F(t)
"""

import sys
import math
import pygame
import numpy as np
from scipy.integrate import solve_ivp

# ── Pygame init ─────────────────────────────────────────────────────
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spring-Mass Damper — Suspension Simulation")
clock = pygame.time.Clock()
FPS = 60

# ── Fonts ───────────────────────────────────────────────────────────
font_title = pygame.font.SysFont("DejaVu Sans", 22, bold=True)
font_label = pygame.font.SysFont("DejaVu Sans", 14)
font_value = pygame.font.SysFont("DejaVu Sans", 13, bold=True)
font_small = pygame.font.SysFont("DejaVu Sans", 12)
font_heading = pygame.font.SysFont("DejaVu Sans", 15, bold=True)

# ── Colors ──────────────────────────────────────────────────────────
BG = (24, 26, 33)
PANEL_BG = (34, 37, 46)
WHITE = (220, 225, 235)
GRAY = (100, 105, 115)
BLUE = (74, 144, 217)
RED = (231, 76, 60)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
DARK_GRAY = (55, 58, 68)
SLIDER_BG = (50, 54, 66)
SLIDER_FILL = (74, 144, 217)
KNOB_COLOR = (200, 210, 225)
SPRING_COLOR = (100, 180, 255)
DAMPER_COLOR = (160, 165, 175)
GROUND_COLOR = (130, 135, 145)
MASS_COLOR = (74, 144, 217)


# ── Slider class ────────────────────────────────────────────────────
class Slider:
    def __init__(self, x, y, w, label, min_val, max_val, default, step=None, fmt=".1f"):
        self.rect = pygame.Rect(x, y, w, 20)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.val = default
        self.default = default
        self.step = step
        self.fmt = fmt
        self.dragging = False

    def draw(self, surface):
        lbl = font_label.render(self.label, True, WHITE)
        surface.blit(lbl, (self.rect.x, self.rect.y - 18))

        pygame.draw.rect(surface, SLIDER_BG, self.rect, border_radius=4)

        frac = (self.val - self.min_val) / (self.max_val - self.min_val)
        fill_w = int(frac * self.rect.w)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pygame.draw.rect(surface, SLIDER_FILL, fill_rect, border_radius=4)

        knob_x = self.rect.x + fill_w
        knob_y = self.rect.centery
        pygame.draw.circle(surface, KNOB_COLOR, (knob_x, knob_y), 10)
        pygame.draw.circle(surface, SLIDER_FILL, (knob_x, knob_y), 10, 2)

        val_str = f"{self.val:{self.fmt}}"
        val_surf = font_value.render(val_str, True, YELLOW)
        surface.blit(val_surf, (self.rect.right + 8, self.rect.y + 1))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            knob_frac = (self.val - self.min_val) / (self.max_val - self.min_val)
            knob_x = self.rect.x + int(knob_frac * self.rect.w)
            knob_y = self.rect.centery
            if math.hypot(event.pos[0] - knob_x, event.pos[1] - knob_y) < 14:
                self.dragging = True
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(event.pos[0])

    def _update_from_mouse(self, mx):
        frac = max(0.0, min(1.0, (mx - self.rect.x) / self.rect.w))
        raw = self.min_val + frac * (self.max_val - self.min_val)
        if self.step:
            raw = round(raw / self.step) * self.step
        self.val = max(self.min_val, min(self.max_val, raw))

    def reset(self):
        self.val = self.default


# ── Button class ────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, text, color=DARK_GRAY, hover_color=SLIDER_BG):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        c = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(surface, c, self.rect, border_radius=6)
        pygame.draw.rect(surface, GRAY, self.rect, 1, border_radius=6)
        txt = font_label.render(self.text, True, WHITE)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


class ToggleButton:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.active = False

    def draw(self, surface):
        c = GREEN if self.active else DARK_GRAY
        pygame.draw.rect(surface, c, self.rect, border_radius=6)
        pygame.draw.rect(surface, GRAY, self.rect, 1, border_radius=6)
        txt = font_label.render(self.text, True, WHITE)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos)):
            self.active = not self.active


# ── Physics ─────────────────────────────────────────────────────────
T_SPAN = (0, 10)
T_EVAL = np.linspace(*T_SPAN, 1000)


def spring_mass_ode(t, y, m, k, c, bump):
    x, v = y
    F = 0.0
    if bump:
        F = 50 * np.exp(-((t - 1.0) ** 2) / 0.05)
    return [v, (F - c * v - k * x) / m]


def solve(m, k, c, x0, bump):
    sol = solve_ivp(
        spring_mass_ode, T_SPAN, [x0, 0.0],
        t_eval=T_EVAL, args=(m, k, c, bump), method='RK45'
    )
    return sol.t, sol.y[0], sol.y[1]


# ── Drawing helpers ─────────────────────────────────────────────────
def draw_spring_visual(surface, cx, ground_y, mass_y, mass_h):
    """Draw spring zig-zag, damper, mass block, and ground."""
    spring_top = mass_y + mass_h // 2
    spring_bot = ground_y

    for hx in range(-80, 81, 20):
        pygame.draw.line(surface, GROUND_COLOR,
                         (cx + hx, ground_y), (cx + hx - 6, ground_y + 10), 2)
    pygame.draw.line(surface, GROUND_COLOR, (cx - 90, ground_y), (cx + 90, ground_y), 3)

    n_coils = 10
    seg_h = (spring_bot - spring_top) / (n_coils * 2 + 1)
    coil_w = 22
    pts = [(cx - 30, spring_top)]
    for i in range(1, n_coils * 2 + 1):
        py = spring_top + i * seg_h
        px = cx - 30 + (coil_w if i % 2 == 1 else -coil_w)
        pts.append((px, py))
    pts.append((cx - 30, spring_bot))
    if len(pts) > 1:
        pygame.draw.lines(surface, SPRING_COLOR, False, pts, 3)

    damper_x = cx + 30
    piston_mid = (spring_top + spring_bot) // 2
    dw = 12
    dh = 30
    pygame.draw.line(surface, DAMPER_COLOR, (damper_x, spring_bot), (damper_x, piston_mid + dh), 2)
    pygame.draw.line(surface, DAMPER_COLOR, (damper_x, spring_top), (damper_x, piston_mid - 5), 2)
    pygame.draw.rect(surface, DAMPER_COLOR,
                     (damper_x - dw, piston_mid - dh, dw * 2, dh * 2), 2)
    pygame.draw.line(surface, DAMPER_COLOR, (damper_x, spring_top),
                     (cx - 30, spring_top), 2)  # top bar to spring
    pygame.draw.line(surface, DAMPER_COLOR, (damper_x, spring_bot),
                     (cx - 30, spring_bot), 2)  # bottom bar to spring

    mass_rect = pygame.Rect(cx - 40, mass_y - mass_h // 2, 80, mass_h)
    pygame.draw.rect(surface, MASS_COLOR, mass_rect, border_radius=8)
    pygame.draw.rect(surface, WHITE, mass_rect, 2, border_radius=8)
    m_txt = font_label.render("m", True, WHITE)
    surface.blit(m_txt, m_txt.get_rect(center=mass_rect.center))

    lbl_s = font_small.render("k", True, SPRING_COLOR)
    surface.blit(lbl_s, (cx - 60, (spring_top + spring_bot) // 2 - 6))
    lbl_c = font_small.render("c", True, DAMPER_COLOR)
    surface.blit(lbl_c, (cx + 48, (spring_top + spring_bot) // 2 - 6))


def draw_graph(surface, rect, t_data, y_data, color, xlabel, ylabel, title):
    """Draw a simple line graph inside a rect region."""
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, DARK_GRAY, rect, 1, border_radius=6)

    margin = 45
    plot_x = rect.x + margin
    plot_y = rect.y + 28
    plot_w = rect.w - margin - 15
    plot_h = rect.h - 50

    title_surf = font_heading.render(title, True, WHITE)
    surface.blit(title_surf, (rect.x + 10, rect.y + 6))

    if len(t_data) < 2 or len(y_data) < 2:
        return

    t_min, t_max = float(t_data[0]), float(t_data[-1])
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    y_pad = (y_max - y_min) * 0.1
    y_min -= y_pad
    y_max += y_pad

    pygame.draw.line(surface, GRAY, (plot_x, plot_y + plot_h),
                     (plot_x + plot_w, plot_y + plot_h), 1)
    pygame.draw.line(surface, GRAY, (plot_x, plot_y),
                     (plot_x, plot_y + plot_h), 1)

    n_yticks = 5
    for i in range(n_yticks + 1):
        frac = i / n_yticks
        yy = plot_y + plot_h - int(frac * plot_h)
        val = y_min + frac * (y_max - y_min)
        pygame.draw.line(surface, DARK_GRAY, (plot_x, yy), (plot_x + plot_w, yy), 1)
        lbl = font_small.render(f"{val:.2f}", True, GRAY)
        surface.blit(lbl, (plot_x - 40, yy - 6))

    n_xticks = 5
    for i in range(n_xticks + 1):
        frac = i / n_xticks
        xx = plot_x + int(frac * plot_w)
        val = t_min + frac * (t_max - t_min)
        lbl = font_small.render(f"{val:.1f}", True, GRAY)
        surface.blit(lbl, (xx - 10, plot_y + plot_h + 3))

    xl = font_small.render(xlabel, True, GRAY)
    surface.blit(xl, (plot_x + plot_w // 2 - xl.get_width() // 2, plot_y + plot_h + 16))

    step = max(1, len(t_data) // (plot_w // 2))
    points = []
    for i in range(0, len(t_data), step):
        fx = (float(t_data[i]) - t_min) / (t_max - t_min)
        fy = (float(y_data[i]) - y_min) / (y_max - y_min)
        px = plot_x + int(fx * plot_w)
        py = plot_y + plot_h - int(fy * plot_h)
        points.append((px, py))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)


# ── UI elements ─────────────────────────────────────────────────────
PANEL_X = 20
PANEL_Y = 480

sl_m = Slider(PANEL_X, PANEL_Y + 22, 260, "Mass  m (kg)", 1, 50, 10.0, step=0.5)
sl_k = Slider(PANEL_X, PANEL_Y + 70, 260, "Spring  k (N/m)", 10, 500, 200.0, step=5, fmt=".0f")
sl_c = Slider(PANEL_X + 340, PANEL_Y + 22, 260, "Damping  c (Ns/m)", 0, 50, 10.0, step=0.5)
sl_x0 = Slider(PANEL_X + 340, PANEL_Y + 70, 260, "Initial Disp.  x₀ (m)", -0.5, 0.5, 0.3, step=0.01, fmt=".2f")
sliders = [sl_m, sl_k, sl_c, sl_x0]

btn_reset = Button(PANEL_X + 700, PANEL_Y + 55, 100, 34, "Reset")
btn_bump = ToggleButton(PANEL_X + 700, PANEL_Y + 10, 130, 34, "Road Bump")

# ── Simulation state ────────────────────────────────────────────────
sim_time = 0.0
t_data, x_data, v_data = solve(10.0, 200.0, 10.0, 0.3, False)
prev_params = (10.0, 200.0, 10.0, 0.3, False)
paused = False

# ── Main loop ───────────────────────────────────────────────────────
running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                paused = not paused
            if event.key == pygame.K_r:
                sim_time = 0.0
        for s in sliders:
            s.handle_event(event)
        btn_bump.handle_event(event)
        if btn_reset.clicked(event):
            for s in sliders:
                s.reset()
            btn_bump.active = False
            sim_time = 0.0

    cur_params = (sl_m.val, sl_k.val, sl_c.val, sl_x0.val, btn_bump.active)
    if cur_params != prev_params:
        t_data, x_data, v_data = solve(*cur_params)
        prev_params = cur_params
        sim_time = 0.0

    if not paused:
        sim_time += dt * 1.0
    if sim_time > T_SPAN[1]:
        sim_time = 0.0

    idx = int(sim_time / T_SPAN[1] * (len(t_data) - 1))
    idx = max(0, min(idx, len(t_data) - 1))
    current_x = float(x_data[idx])

    # ── Draw ────────────────────────────────────────────────────────
    screen.fill(BG)

    title = font_title.render("Spring-Mass Damper System — Quarter-Car Suspension", True, WHITE)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

    # Spring-mass visual (left)
    vis_cx = 160
    vis_ground_y = 400
    vis_mass_h = 60
    scale = 250
    vis_mass_y = int(200 - current_x * scale)
    draw_spring_visual(screen, vis_cx, vis_ground_y, vis_mass_y, vis_mass_h)

    disp_txt = font_value.render(f"x = {current_x:.3f} m", True, YELLOW)
    screen.blit(disp_txt, (vis_cx - disp_txt.get_width() // 2, 130))

    time_txt = font_small.render(f"t = {sim_time:.2f} s", True, GRAY)
    screen.blit(time_txt, (vis_cx - time_txt.get_width() // 2, 148))

    # Displacement vs Time graph (top-right)
    graph1_rect = pygame.Rect(310, 42, 460, 210)
    draw_graph(screen, graph1_rect, t_data, x_data, BLUE,
               "Time (s)", "x (m)", "Displacement vs Time")

    time_frac = sim_time / T_SPAN[1]
    marker_x = graph1_rect.x + 45 + int(time_frac * (graph1_rect.w - 60))
    pygame.draw.line(screen, YELLOW, (marker_x, graph1_rect.y + 28),
                     (marker_x, graph1_rect.y + graph1_rect.h - 22), 1)

    # Phase portrait (bottom-right)
    graph2_rect = pygame.Rect(310, 262, 460, 210)
    draw_graph(screen, graph2_rect, x_data, v_data, RED,
               "Displacement (m)", "v (m/s)", "Phase Portrait (x vs v)")

    if idx > 0:
        g2_margin = 45
        g2_px = graph2_rect.x + g2_margin
        g2_py = graph2_rect.y + 28
        g2_pw = graph2_rect.w - g2_margin - 15
        g2_ph = graph2_rect.h - 50
        x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
        v_min, v_max = float(np.min(v_data)), float(np.max(v_data))
        if abs(x_max - x_min) < 1e-9:
            x_min -= 0.5; x_max += 0.5
        if abs(v_max - v_min) < 1e-9:
            v_min -= 0.5; v_max += 0.5
        xp = (x_max - x_min) * 0.1; vp = (v_max - v_min) * 0.1
        x_min -= xp; x_max += xp; v_min -= vp; v_max += vp
        fx = (float(x_data[idx]) - x_min) / (x_max - x_min)
        fy = (float(v_data[idx]) - v_min) / (v_max - v_min)
        dot_x = g2_px + int(fx * g2_pw)
        dot_y = g2_py + g2_ph - int(fy * g2_ph)
        pygame.draw.circle(screen, YELLOW, (dot_x, dot_y), 5)

    # Info panel
    info_rect = pygame.Rect(790, 42, 470, 430)
    pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=6)
    pygame.draw.rect(screen, DARK_GRAY, info_rect, 1, border_radius=6)

    info_title = font_heading.render("Parameters", True, WHITE)
    screen.blit(info_title, (info_rect.x + 15, info_rect.y + 10))

    params_text = [
        f"Mass (m):          {sl_m.val:.1f} kg",
        f"Spring (k):        {sl_k.val:.0f} N/m",
        f"Damping (c):       {sl_c.val:.1f} Ns/m",
        f"Init. Disp (x₀):   {sl_x0.val:.2f} m",
        f"Road Bump:         {'ON' if btn_bump.active else 'OFF'}",
        "",
        f"Natural freq:      {math.sqrt(sl_k.val / sl_m.val) / (2 * math.pi):.2f} Hz",
        f"ωn:                {math.sqrt(sl_k.val / sl_m.val):.2f} rad/s",
        f"Damping ratio ζ:   {sl_c.val / (2 * math.sqrt(sl_k.val * sl_m.val)):.3f}",
        "",
        f"Current x:         {current_x:.4f} m",
        f"Current v:         {float(v_data[idx]):.4f} m/s",
    ]
    for i, line in enumerate(params_text):
        color = WHITE if line else GRAY
        txt = font_small.render(line, True, color)
        screen.blit(txt, (info_rect.x + 15, info_rect.y + 35 + i * 22))

    damping_ratio = sl_c.val / (2 * math.sqrt(sl_k.val * sl_m.val))
    if damping_ratio < 1:
        sys_type = "Underdamped"
        type_color = GREEN
    elif abs(damping_ratio - 1) < 0.01:
        sys_type = "Critically Damped"
        type_color = YELLOW
    else:
        sys_type = "Overdamped"
        type_color = RED
    sys_surf = font_heading.render(f"System: {sys_type}", True, type_color)
    screen.blit(sys_surf, (info_rect.x + 15, info_rect.y + 35 + 13 * 22))

    eq_txt = font_small.render("mx'' + cx' + kx = F(t)", True, GRAY)
    screen.blit(eq_txt, (info_rect.x + 15, info_rect.y + info_rect.h - 30))

    # Controls bar
    controls_bg = pygame.Rect(0, PANEL_Y - 20, WIDTH, HEIGHT - PANEL_Y + 20)
    pygame.draw.rect(screen, PANEL_BG, controls_bg)
    pygame.draw.line(screen, DARK_GRAY, (0, PANEL_Y - 20), (WIDTH, PANEL_Y - 20), 1)

    for s in sliders:
        s.draw(screen)
    btn_reset.draw(screen)
    btn_bump.draw(screen)

    hint = font_small.render("SPACE: Pause  |  R: Restart  |  ESC: Quit", True, GRAY)
    screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
