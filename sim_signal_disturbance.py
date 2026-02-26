"""
Signal Disturbance Visualization — 5 Topics, No Input Required

Click tabs or press keys 1-5 to switch topics. Everything auto-runs.

1 = Clean vs Noisy Signal
2 = Signal Filtering (moving average)
3 = Beat Frequency
4 = FFT Analysis
5 = Aliasing Effect
"""

import sys
import math
import numpy as np
import pygame

pygame.init()
W, H = 900, 560
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Signal Disturbance Visualization")
clock = pygame.time.Clock()
FPS = 60

# ── Fonts & Colors ──────────────────────────────────────────────────
font_title = pygame.font.SysFont("DejaVu Sans", 16, bold=True)
font_label = pygame.font.SysFont("DejaVu Sans", 13, bold=True)
font_small = pygame.font.SysFont("DejaVu Sans", 11)
font_tab   = pygame.font.SysFont("DejaVu Sans", 12, bold=True)
font_info  = pygame.font.SysFont("DejaVu Sans", 11)

BG       = (18, 20, 28)
PANEL    = (28, 31, 40)
WHITE    = (210, 215, 225)
GRAY     = (90, 95, 105)
DARK     = (40, 44, 55)
CLEAN    = (74, 144, 217)
NOISY    = (231, 76, 60)
FILTERED = (46, 204, 113)
FFT_BAR  = (230, 126, 34)
BEAT_COL = (155, 89, 182)
ALIAS_COL = (241, 196, 15)
TAB_ON   = (74, 144, 217)
TAB_OFF  = (40, 44, 55)

# ── Fixed internal parameters ───────────────────────────────────────
FREQ1       = 5.0
FREQ2       = 5.5
FREQ_HIGH   = 40.0
NOISE       = 0.35
SAMPLE_RATE = 12.0
FILT_W      = 15
BUF         = 800

# ── Buffers ─────────────────────────────────────────────────────────
buf_clean    = np.zeros(BUF)
buf_noisy    = np.zeros(BUF)
buf_filtered = np.zeros(BUF)
buf_beat     = np.zeros(BUF)
buf_fft      = np.zeros(BUF)
t_acc        = 0.0
dt           = 1.0 / FPS
topic        = 0
paused       = False

# ── Tabs ────────────────────────────────────────────────────────────
TAB_NAMES = ["1: Clean vs Noisy", "2: Filtering", "3: Beat Freq", "4: FFT", "5: Aliasing"]
TAB_W, TAB_H, TAB_Y = 148, 28, 6


# ── Drawing helpers ─────────────────────────────────────────────────
def draw_wave(surf, rect, data, color, label):
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, DARK, rect, 1, border_radius=4)
    cy = rect.centery
    pygame.draw.line(surf, DARK, (rect.x + 5, cy), (rect.right - 5, cy), 1)
    surf.blit(font_label.render(label, True, color), (rect.x + 10, rect.y + 4))
    if len(data) < 2:
        return
    uw = rect.w - 20
    n = min(len(data), uw)
    hh = (rect.h - 30) // 2
    dm = max(abs(np.max(data[-n:])), abs(np.min(data[-n:])), 0.01)
    pts = []
    for i in range(n):
        px = rect.x + 10 + int(i / n * uw)
        py = cy - int(float(data[-(n - i)]) / dm * hh)
        py = max(rect.y + 18, min(rect.bottom - 8, py))
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 2)


def draw_fft_bars(surf, rect, freqs, mags, label, max_freq=50):
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, DARK, rect, 1, border_radius=4)
    surf.blit(font_label.render(label, True, FFT_BAR), (rect.x + 10, rect.y + 4))
    mask = freqs <= max_freq
    fs, ms = freqs[mask], mags[mask]
    if len(fs) < 2:
        return
    mm = float(np.max(ms)) * 1.2 + 0.001
    uw, uh = rect.w - 30, rect.h - 42
    by = rect.bottom - 14
    pygame.draw.line(surf, GRAY, (rect.x + 15, by), (rect.right - 15, by), 1)
    for fv in range(0, int(max_freq) + 1, 10):
        fx = rect.x + 15 + int(fv / max_freq * uw)
        surf.blit(font_small.render(f"{fv}", True, GRAY), (fx - 6, by + 1))
        pygame.draw.line(surf, DARK, (fx, rect.y + 22), (fx, by), 1)
    surf.blit(font_small.render("Hz", True, GRAY), (rect.right - 25, by + 1))
    bw = max(2, uw // len(fs))
    for i in range(len(fs)):
        fx = rect.x + 15 + int(float(fs[i]) / max_freq * uw)
        bh = int(float(ms[i]) / mm * uh)
        if bh > 1:
            pygame.draw.rect(surf, FFT_BAR, (fx, by - bh, max(bw, 2), bh))
    peak_idx = np.argsort(ms)[-3:]
    for pi in peak_idx:
        if ms[pi] > mm * 0.08:
            fx = rect.x + 15 + int(float(fs[pi]) / max_freq * uw)
            bh = int(float(ms[pi]) / mm * uh)
            surf.blit(font_small.render(f"{fs[pi]:.1f}Hz", True, WHITE), (fx - 12, by - bh - 14))


def draw_info(surf, rect, lines):
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, DARK, rect, 1, border_radius=4)
    surf.blit(font_label.render("How it works", True, FILTERED), (rect.x + 10, rect.y + 6))
    for i, (txt, col) in enumerate(lines):
        surf.blit(font_info.render(txt, True, col), (rect.x + 10, rect.y + 26 + i * 17))


def reset_bufs():
    global t_acc
    buf_clean[:] = 0; buf_noisy[:] = 0; buf_filtered[:] = 0
    buf_beat[:] = 0; buf_fft[:] = 0; t_acc = 0.0


# ── Main loop ───────────────────────────────────────────────────────
running = True
while running:
    clock.tick(FPS)

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE: running = False
            if ev.key == pygame.K_SPACE: paused = not paused
            if ev.key == pygame.K_r: reset_bufs()
            for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]):
                if ev.key == k: topic = i; reset_bufs()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i in range(5):
                r = pygame.Rect(10 + i * (TAB_W + 6), TAB_Y, TAB_W, TAB_H)
                if r.collidepoint(ev.pos):
                    topic = i; reset_bufs()

    # ── Generate samples ────────────────────────────────────────────
    if not paused:
        t_acc += dt
        sc = math.sin(2 * math.pi * FREQ1 * t_acc)
        sn = sc + NOISE * np.random.randn()
        buf_clean[:-1] = buf_clean[1:]; buf_clean[-1] = sc
        buf_noisy[:-1] = buf_noisy[1:]; buf_noisy[-1] = sn
        buf_filtered[:-1] = buf_filtered[1:]; buf_filtered[-1] = np.mean(buf_noisy[-FILT_W:])
        buf_beat[:-1] = buf_beat[1:]
        buf_beat[-1] = math.sin(2 * math.pi * FREQ1 * t_acc) + math.sin(2 * math.pi * FREQ2 * t_acc)
        buf_fft[:-1] = buf_fft[1:]; buf_fft[-1] = sn

    screen.fill(BG)

    # ── Tabs ────────────────────────────────────────────────────────
    for i, name in enumerate(TAB_NAMES):
        r = pygame.Rect(10 + i * (TAB_W + 6), TAB_Y, TAB_W, TAB_H)
        c = TAB_ON if i == topic else TAB_OFF
        pygame.draw.rect(screen, c, r, border_radius=5)
        pygame.draw.rect(screen, CLEAN if i == topic else GRAY, r, 1, border_radius=5)
        screen.blit(font_tab.render(name, True, WHITE), (r.x + 8, r.y + 6))

    GY = 42
    LW, LH = 580, 0  # graph left area width; height set per topic
    IW = W - LW - 30  # info panel width

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 1 — Clean vs Noisy
    # ═════════════════════════════════════════════════════════════════
    if topic == 0:
        r1 = pygame.Rect(10, GY, LW, 220)
        r2 = pygame.Rect(10, GY + 228, LW, 220)
        draw_wave(screen, r1, buf_clean, CLEAN, "Clean Signal  (5 Hz pure sine)")
        draw_wave(screen, r2, buf_noisy, NOISY, "Noisy Signal  (same + random noise)")

        draw_info(screen, pygame.Rect(LW + 20, GY, IW, 448), [
            ("In real systems, signals are never", WHITE),
            ("perfectly clean. Noise comes from:", WHITE),
            ("", GRAY),
            ("  - Electrical interference", WHITE),
            ("  - Sensor imperfections", WHITE),
            ("  - Mechanical vibrations", WHITE),
            ("  - Temperature fluctuations", WHITE),
            ("", GRAY),
            ("What you see:", FILTERED),
            ("  Top: Pure sine wave at 5 Hz", CLEAN),
            ("  Bottom: Same wave + Gaussian", NOISY),
            ("  noise (level = 0.35)", NOISY),
            ("", GRAY),
            ("The noise is random — each sample", WHITE),
            ("gets a different disturbance.", WHITE),
            ("", GRAY),
            ("Signal-to-Noise Ratio (SNR) tells", WHITE),
            ("how much signal vs noise you have.", WHITE),
            ("Higher SNR = cleaner signal.", WHITE),
        ])

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 2 — Signal Filtering
    # ═════════════════════════════════════════════════════════════════
    elif topic == 1:
        r1 = pygame.Rect(10, GY, LW, 220)
        r2 = pygame.Rect(10, GY + 228, LW, 220)
        draw_wave(screen, r1, buf_noisy, NOISY, "Noisy Input")
        draw_wave(screen, r2, buf_filtered, FILTERED, "Filtered Output  (moving average)")

        draw_info(screen, pygame.Rect(LW + 20, GY, IW, 448), [
            ("A moving-average filter smooths", WHITE),
            ("the signal by averaging the last", WHITE),
            ("N samples at each point.", WHITE),
            ("", GRAY),
            ("  y[n] = mean(x[n-14] ... x[n])", FILTERED),
            ("  Window size = 15 samples", FILTERED),
            ("", GRAY),
            ("What you see:", FILTERED),
            ("  Top: Noisy signal (red)", NOISY),
            ("  Bottom: Smoothed signal (green)", FILTERED),
            ("", GRAY),
            ("Trade-off:", ALIAS_COL),
            ("  Larger window = smoother output", WHITE),
            ("  but more time delay (lag).", WHITE),
            ("", GRAY),
            ("Used in: sensor data processing,", WHITE),
            ("audio noise reduction, stock price", WHITE),
            ("smoothing, vibration analysis.", WHITE),
        ])

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 3 — Beat Frequency
    # ═════════════════════════════════════════════════════════════════
    elif topic == 2:
        f_beat = abs(FREQ1 - FREQ2)
        t_now = np.linspace(t_acc - BUF * dt, t_acc, BUF)
        w1 = np.sin(2 * np.pi * FREQ1 * t_now)
        w2 = np.sin(2 * np.pi * FREQ2 * t_now)

        r1 = pygame.Rect(10, GY, LW, 140)
        r2 = pygame.Rect(10, GY + 148, LW, 140)
        r3 = pygame.Rect(10, GY + 296, LW, 152)
        draw_wave(screen, r1, w1, CLEAN, f"Wave 1:  {FREQ1} Hz")
        draw_wave(screen, r2, w2, BEAT_COL, f"Wave 2:  {FREQ2} Hz")
        draw_wave(screen, r3, buf_beat, ALIAS_COL,
                  f"Combined = Beat  ({f_beat:.1f} Hz envelope)")

        draw_info(screen, pygame.Rect(LW + 20, GY, IW, 448), [
            ("When two waves with CLOSE", WHITE),
            ("frequencies are added, the result", WHITE),
            ("oscillates with a slow envelope.", WHITE),
            ("This is called BEATING.", WHITE),
            ("", GRAY),
            (f"  f1 = {FREQ1} Hz", CLEAN),
            (f"  f2 = {FREQ2} Hz", BEAT_COL),
            (f"  Beat freq = |f1-f2| = {f_beat} Hz", ALIAS_COL),
            ("", GRAY),
            ("The amplitude rises and falls at", WHITE),
            ("the beat frequency.", WHITE),
            ("", GRAY),
            ("Used in:", FILTERED),
            ("  - Tuning musical instruments", WHITE),
            ("  - Vibration diagnostics", WHITE),
            ("  - Radio signal processing", WHITE),
            ("  - Detecting small freq shifts", WHITE),
        ])

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 4 — FFT Analysis
    # ═════════════════════════════════════════════════════════════════
    elif topic == 3:
        r1 = pygame.Rect(10, GY, LW, 200)
        r2 = pygame.Rect(10, GY + 208, LW, 240)
        draw_wave(screen, r1, buf_fft, NOISY, "Live Signal  (time domain)")

        fft_v = np.fft.rfft(buf_fft)
        fft_m = 2.0 / BUF * np.abs(fft_v)
        fft_f = np.fft.rfftfreq(BUF, dt)
        draw_fft_bars(screen, r2, fft_f, fft_m, "Frequency Spectrum  (FFT)", max_freq=50)

        draw_info(screen, pygame.Rect(LW + 20, GY, IW, 448), [
            ("FFT (Fast Fourier Transform)", WHITE),
            ("converts a time signal into its", WHITE),
            ("frequency components.", WHITE),
            ("", GRAY),
            ("What you see:", FILTERED),
            ("  Top: Noisy signal scrolling", NOISY),
            ("  Bottom: FFT bar chart", FFT_BAR),
            ("", GRAY),
            ("The tall bar shows the main", WHITE),
            (f"frequency at {FREQ1} Hz — the same", WHITE),
            ("sine wave hidden in the noise.", WHITE),
            ("", GRAY),
            ("Engineers use FFT to:", FILTERED),
            ("  - Find vibration sources", WHITE),
            ("  - Diagnose machine faults", WHITE),
            ("  - Analyze audio/speech", WHITE),
            ("  - Detect structural damage", WHITE),
        ])

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 5 — Aliasing
    # ═════════════════════════════════════════════════════════════════
    elif topic == 4:
        nyquist = SAMPLE_RATE / 2
        aliased = FREQ_HIGH > nyquist
        alias_f = abs(FREQ_HIGH - round(FREQ_HIGH / SAMPLE_RATE) * SAMPLE_RATE)

        t_cont = np.linspace(t_acc - 2.0, t_acc, BUF)
        y_cont = np.sin(2 * np.pi * FREQ_HIGH * t_cont)

        t_start = max(0, t_acc - 2.0)
        ns = max(int(2.0 * SAMPLE_RATE), 2)
        t_samp = np.linspace(t_start, t_acc, ns)
        y_samp = np.sin(2 * np.pi * FREQ_HIGH * t_samp)
        y_recon = np.interp(t_cont, t_samp, y_samp)

        r1 = pygame.Rect(10, GY, LW, 210)
        r2 = pygame.Rect(10, GY + 218, LW, 210)
        draw_wave(screen, r1, y_cont, CLEAN, f"Original: {FREQ_HIGH:.0f} Hz (continuous)")
        draw_wave(screen, r2, y_recon, NOISY, f"Sampled at {SAMPLE_RATE:.0f} Hz → appears {alias_f:.1f} Hz")

        # Sample dots on top panel
        cy1 = r1.centery; hh1 = (r1.h - 30) // 2
        for i in range(len(t_samp)):
            frac = (float(t_samp[i]) - float(t_cont[0])) / (float(t_cont[-1]) - float(t_cont[0]))
            if 0 <= frac <= 1:
                px = r1.x + 10 + int(frac * (r1.w - 20))
                py = cy1 - int(float(y_samp[i]) / 1.0 * hh1)
                py = max(r1.y + 18, min(r1.bottom - 8, py))
                pygame.draw.circle(screen, ALIAS_COL, (px, py), 4)
                pygame.draw.circle(screen, WHITE, (px, py), 4, 1)

        status_c = NOISY if aliased else FILTERED
        status_t = "ALIASED!" if aliased else "OK"
        screen.blit(font_label.render(f"Status: {status_t}", True, status_c), (10, GY + 436))

        draw_info(screen, pygame.Rect(LW + 20, GY, IW, 448), [
            ("Aliasing happens when the sample", WHITE),
            ("rate is too LOW for the signal.", WHITE),
            ("", GRAY),
            ("Nyquist theorem:", ALIAS_COL),
            ("  Sample rate must be > 2x the", WHITE),
            ("  highest frequency in the signal.", WHITE),
            ("", GRAY),
            (f"  Signal: {FREQ_HIGH:.0f} Hz", CLEAN),
            (f"  Nyquist needs: {2*FREQ_HIGH:.0f} Hz", CLEAN),
            (f"  Actual sample rate: {SAMPLE_RATE:.0f} Hz", NOISY),
            (f"  Result: appears as {alias_f:.1f} Hz", NOISY),
            ("", GRAY),
            ("The yellow dots are the samples.", ALIAS_COL),
            ("Connecting them reconstructs a", WHITE),
            ("WRONG lower-frequency wave.", WHITE),
            ("", GRAY),
            ("Example: 44.1 kHz audio captures", WHITE),
            ("up to ~22 kHz (human hearing).", WHITE),
        ])

    # ── Footer ──────────────────────────────────────────────────────
    ft = f"SPACE: {'Play' if paused else 'Pause'}  |  R: Reset  |  1-5: Switch topic  |  ESC: Quit"
    screen.blit(font_small.render(ft, True, GRAY), (10, H - 18))
    screen.blit(font_small.render(f"t = {t_acc:.1f}s", True, GRAY), (W - 60, H - 18))

    pygame.display.flip()

pygame.quit()
sys.exit()
