"""
Signal Disturbance Visualization — 5 Topics in One Pygame App

Set TOPIC = 1..5 to pick which demo to run.

1 = Clean vs Noisy Signal   (pure sine + same wave with random noise)
2 = Signal Filtering         (noisy signal → filtered smooth signal)
3 = Beat Frequency           (two close frequencies combined → beating)
4 = FFT Analysis             (live signal + frequency spectrum bars)
5 = Aliasing Effect          (high-freq wave vs under-sampled distorted wave)

Works in Google Colab:
    !pip install pygame
    Then run this file. Uses headless display via os.environ trick.

Also works locally with a normal display.
"""

# ── Headless display for Google Colab ───────────────────────────────
import os
import sys

_in_colab = 'google.colab' in sys.modules if 'google.colab' in sys.modules else False
try:
    from google.colab import output as _colab_out
    _in_colab = True
except ImportError:
    pass

if _in_colab:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

# ═══════════════════════════════════════════════════════════════════
#  CHOOSE TOPIC (1 to 5)
# ═══════════════════════════════════════════════════════════════════
TOPIC = 1

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════
FREQ_MAIN     = 5.0     # primary signal frequency (Hz)
FREQ_SECOND   = 5.5     # second frequency for beat (Hz) — topic 3
FREQ_HIGH     = 40.0    # high-frequency signal for aliasing — topic 5
NOISE_LEVEL   = 0.35    # noise amplitude (0 = none, 1 = heavy) — topics 1, 2, 4
SAMPLE_RATE   = 12.0    # under-sampling rate (Hz) for aliasing — topic 5
FILTER_WINDOW = 15      # moving-average filter width (samples) — topic 2

# ═══════════════════════════════════════════════════════════════════
import math
import numpy as np
import pygame

pygame.init()

W, H = 800, 500
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"Signal Disturbance — Topic {TOPIC}")
clock = pygame.time.Clock()
FPS = 60

# ── Fonts & Colors ──────────────────────────────────────────────────
font_title = pygame.font.SysFont("DejaVu Sans", 18, bold=True)
font_label = pygame.font.SysFont("DejaVu Sans", 14, bold=True)
font_small = pygame.font.SysFont("DejaVu Sans", 12)
font_btn   = pygame.font.SysFont("DejaVu Sans", 13, bold=True)

BG       = (18, 20, 28)
PANEL    = (28, 31, 40)
WHITE    = (210, 215, 225)
GRAY     = (90, 95, 105)
DARK     = (40, 44, 55)
CLEAN    = (74, 144, 217)    # blue
NOISY    = (231, 76, 60)     # red
FILTERED = (46, 204, 113)    # green
FFT_BAR  = (230, 126, 34)   # orange
BEAT_COL = (155, 89, 182)   # purple
ALIAS_COL= (241, 196, 15)   # yellow
BTN_BG   = (50, 54, 66)
BTN_HOVER= (70, 75, 88)


# ── Play/Pause button ──────────────────────────────────────────────
btn_rect = pygame.Rect(W - 100, H - 38, 85, 28)
paused = False


def draw_button(surf):
    mouse = pygame.mouse.get_pos()
    c = BTN_HOVER if btn_rect.collidepoint(mouse) else BTN_BG
    pygame.draw.rect(surf, c, btn_rect, border_radius=5)
    pygame.draw.rect(surf, GRAY, btn_rect, 1, border_radius=5)
    text = "▶ Play" if paused else "❚❚ Pause"
    t = font_btn.render(text, True, WHITE)
    surf.blit(t, t.get_rect(center=btn_rect.center))


# ── Drawing helpers ─────────────────────────────────────────────────
def draw_waveform(surf, rect, data, color, label, lw=2):
    """Draw a scrolling waveform inside rect with label."""
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, DARK, rect, 1, border_radius=4)

    # Centre axis
    cy = rect.centery
    pygame.draw.line(surf, DARK, (rect.x + 5, cy), (rect.right - 5, cy), 1)

    # Label
    surf.blit(font_label.render(label, True, color), (rect.x + 10, rect.y + 4))

    if len(data) < 2:
        return

    usable_w = rect.w - 20
    n = min(len(data), usable_w)
    half_h = (rect.h - 30) // 2
    d_max = max(abs(np.max(data[-n:])), abs(np.min(data[-n:])), 0.01)

    pts = []
    for i in range(n):
        px = rect.x + 10 + int(i / n * usable_w)
        val = float(data[-(n - i)])
        py = cy - int(val / d_max * half_h)
        py = max(rect.y + 18, min(rect.bottom - 8, py))
        pts.append((px, py))

    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, lw)


def draw_fft_bars(surf, rect, freqs, mags, label, max_freq=80):
    """Draw frequency spectrum as vertical bars."""
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, DARK, rect, 1, border_radius=4)
    surf.blit(font_label.render(label, True, FFT_BAR), (rect.x + 10, rect.y + 4))

    mask = freqs <= max_freq
    f_show = freqs[mask]
    m_show = mags[mask]
    if len(f_show) < 2:
        return

    m_max = float(np.max(m_show)) * 1.2 + 0.001
    usable_w = rect.w - 30
    usable_h = rect.h - 40
    base_y = rect.bottom - 12

    # Axis
    pygame.draw.line(surf, GRAY, (rect.x + 15, base_y), (rect.right - 15, base_y), 1)

    # Frequency labels
    for fv in range(0, int(max_freq) + 1, 10):
        fx = rect.x + 15 + int(fv / max_freq * usable_w)
        surf.blit(font_small.render(f"{fv}", True, GRAY), (fx - 6, base_y + 1))
        pygame.draw.line(surf, DARK, (fx, rect.y + 22), (fx, base_y), 1)
    surf.blit(font_small.render("Hz", True, GRAY), (rect.right - 25, base_y + 1))

    bar_w = max(2, usable_w // len(f_show))
    for i in range(len(f_show)):
        fx = rect.x + 15 + int(float(f_show[i]) / max_freq * usable_w)
        bh = int(float(m_show[i]) / m_max * usable_h)
        if bh > 1:
            pygame.draw.rect(surf, FFT_BAR, (fx, base_y - bh, max(bar_w, 2), bh))

    # Peak labels
    if len(m_show) > 5:
        peak_indices = np.argsort(m_show)[-3:]
        for pi in peak_indices:
            if m_show[pi] > m_max * 0.1:
                fx = rect.x + 15 + int(float(f_show[pi]) / max_freq * usable_w)
                bh = int(float(m_show[pi]) / m_max * usable_h)
                surf.blit(font_small.render(f"{f_show[pi]:.1f}Hz", True, WHITE),
                          (fx - 12, base_y - bh - 14))


def draw_sampled_points(surf, rect, t_samp, y_samp, color, cy, half_h, d_max, t_range):
    """Draw sampling dots on top of a waveform rect."""
    for i in range(len(t_samp)):
        frac = (float(t_samp[i]) - t_range[0]) / (t_range[1] - t_range[0])
        if 0 <= frac <= 1:
            px = rect.x + 10 + int(frac * (rect.w - 20))
            py = cy - int(float(y_samp[i]) / d_max * half_h)
            py = max(rect.y + 18, min(rect.bottom - 8, py))
            pygame.draw.circle(surf, color, (px, py), 5)
            pygame.draw.circle(surf, WHITE, (px, py), 5, 1)


# ── Signal buffers ──────────────────────────────────────────────────
BUF_LEN = 800
t_acc = 0.0
dt_sig = 1.0 / FPS

buf_clean   = np.zeros(BUF_LEN)
buf_noisy   = np.zeros(BUF_LEN)
buf_filtered = np.zeros(BUF_LEN)
buf_beat    = np.zeros(BUF_LEN)
buf_fft_sig = np.zeros(BUF_LEN)

# ── Title map ───────────────────────────────────────────────────────
TITLES = {
    1: "Clean vs Noisy Signal",
    2: "Signal Filtering (Moving Average)",
    3: "Beat Frequency",
    4: "FFT Analysis",
    5: "Aliasing Effect",
}

# ── Main loop ───────────────────────────────────────────────────────
running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            if ev.key == pygame.K_SPACE:
                paused = not paused
            if ev.key == pygame.K_r:
                t_acc = 0.0
                buf_clean[:] = 0; buf_noisy[:] = 0
                buf_filtered[:] = 0; buf_beat[:] = 0; buf_fft_sig[:] = 0
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if btn_rect.collidepoint(ev.pos):
                paused = not paused

    if not paused:
        t_acc += dt_sig

        # Generate one new sample for each buffer
        s_clean = math.sin(2 * math.pi * FREQ_MAIN * t_acc)
        s_noise = s_clean + NOISE_LEVEL * np.random.randn()

        buf_clean = np.roll(buf_clean, -1)
        buf_clean[-1] = s_clean
        buf_noisy = np.roll(buf_noisy, -1)
        buf_noisy[-1] = s_noise
        buf_filtered = np.roll(buf_filtered, -1)
        buf_filtered[-1] = np.mean(buf_noisy[-FILTER_WINDOW:])
        buf_beat = np.roll(buf_beat, -1)
        buf_beat[-1] = math.sin(2 * math.pi * FREQ_MAIN * t_acc) + \
                        math.sin(2 * math.pi * FREQ_SECOND * t_acc)
        buf_fft_sig = np.roll(buf_fft_sig, -1)
        buf_fft_sig[-1] = s_noise

    screen.fill(BG)

    # Title
    title = TITLES.get(TOPIC, "Unknown Topic")
    screen.blit(font_title.render(f"Topic {TOPIC}: {title}", True, WHITE), (15, 8))

    # Parameter info
    param_str = f"Freq: {FREQ_MAIN:.1f}Hz  |  Noise: {NOISE_LEVEL:.2f}  |  t = {t_acc:.1f}s"
    screen.blit(font_small.render(param_str, True, GRAY), (15, 32))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 1 — Clean vs Noisy
    # ═════════════════════════════════════════════════════════════════
    if TOPIC == 1:
        # Pure sine wave on top, same wave corrupted by random noise below.
        # In real engineering, noise comes from electrical interference,
        # mechanical vibrations, or sensor imperfections.
        r1 = pygame.Rect(15, 55, W - 30, 195)
        r2 = pygame.Rect(15, 260, W - 30, 195)
        draw_waveform(screen, r1, buf_clean, CLEAN, "Clean Signal  (pure sine)")
        draw_waveform(screen, r2, buf_noisy, NOISY,
                      f"Noisy Signal  (noise = {NOISE_LEVEL:.2f})")

        screen.blit(font_small.render(
            "Top: Original sine wave  |  Bottom: Same wave + random Gaussian noise",
            True, GRAY), (15, 462))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 2 — Signal Filtering
    # ═════════════════════════════════════════════════════════════════
    elif TOPIC == 2:
        # A moving-average filter smooths the noisy signal by averaging
        # the last N samples. This removes high-frequency noise but
        # introduces a small time delay (lag).
        r1 = pygame.Rect(15, 55, W - 30, 195)
        r2 = pygame.Rect(15, 260, W - 30, 195)
        draw_waveform(screen, r1, buf_noisy, NOISY,
                      f"Noisy Input  (noise = {NOISE_LEVEL:.2f})")
        draw_waveform(screen, r2, buf_filtered, FILTERED,
                      f"Filtered Output  (window = {FILTER_WINDOW} samples)")

        screen.blit(font_small.render(
            f"Moving average filter: y[n] = mean of last {FILTER_WINDOW} samples  |  "
            "Removes noise, adds slight lag",
            True, GRAY), (15, 462))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 3 — Beat Frequency
    # ═════════════════════════════════════════════════════════════════
    elif TOPIC == 3:
        # When two waves with CLOSE frequencies are added together,
        # the result oscillates with a slow "envelope" called the beat.
        # Beat frequency = |f1 - f2|. Musicians use this to tune instruments.
        f_beat = abs(FREQ_MAIN - FREQ_SECOND)

        r1 = pygame.Rect(15, 55, W - 30, 130)
        r2 = pygame.Rect(15, 195, W - 30, 130)
        r3 = pygame.Rect(15, 335, W - 30, 120)

        # Individual waves
        buf_w1 = np.sin(2 * np.pi * FREQ_MAIN * (t_acc + np.arange(BUF_LEN) * dt_sig / FPS * FPS - BUF_LEN * dt_sig))
        buf_w2 = np.sin(2 * np.pi * FREQ_SECOND * (t_acc + np.arange(BUF_LEN) * dt_sig / FPS * FPS - BUF_LEN * dt_sig))

        draw_waveform(screen, r1, buf_w1[-BUF_LEN:], CLEAN,
                      f"Wave 1: {FREQ_MAIN:.1f} Hz")
        draw_waveform(screen, r2, buf_w2[-BUF_LEN:], BEAT_COL,
                      f"Wave 2: {FREQ_SECOND:.1f} Hz")
        draw_waveform(screen, r3, buf_beat, ALIAS_COL,
                      f"Combined = Beat  |  Beat freq = |{FREQ_MAIN}-{FREQ_SECOND}| = {f_beat:.1f} Hz")

        screen.blit(font_small.render(
            f"Beat frequency = |f1 - f2| = {f_beat:.1f} Hz  |  "
            "Used in instrument tuning and vibration diagnostics",
            True, GRAY), (15, 462))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 4 — FFT Analysis
    # ═════════════════════════════════════════════════════════════════
    elif TOPIC == 4:
        # FFT decomposes a time-domain signal into frequency components.
        # The bars show which frequencies are present and how strong they are.
        r1 = pygame.Rect(15, 55, W - 30, 180)
        r2 = pygame.Rect(15, 245, W - 30, 210)

        draw_waveform(screen, r1, buf_fft_sig, NOISY, "Live Signal (time domain)")

        # Compute FFT on the buffer
        fft_vals = np.fft.rfft(buf_fft_sig)
        fft_mag = 2.0 / BUF_LEN * np.abs(fft_vals)
        fft_freqs = np.fft.rfftfreq(BUF_LEN, dt_sig)

        draw_fft_bars(screen, r2, fft_freqs, fft_mag,
                      "Frequency Spectrum (FFT)", max_freq=50)

        screen.blit(font_small.render(
            "Top: Noisy signal in time  |  Bottom: FFT reveals the hidden frequency peaks",
            True, GRAY), (15, 462))

    # ═════════════════════════════════════════════════════════════════
    #  TOPIC 5 — Aliasing
    # ═════════════════════════════════════════════════════════════════
    elif TOPIC == 5:
        # Aliasing occurs when sampling rate < 2 × signal frequency (Nyquist).
        # The under-sampled signal appears to be a DIFFERENT (lower) frequency.
        # This is why digital audio uses 44.1 kHz — to capture up to ~22 kHz.
        nyquist = SAMPLE_RATE / 2
        is_aliased = FREQ_HIGH > nyquist
        alias_freq = abs(FREQ_HIGH - round(FREQ_HIGH / SAMPLE_RATE) * SAMPLE_RATE)

        r1 = pygame.Rect(15, 55, W - 30, 180)
        r2 = pygame.Rect(15, 245, W - 30, 180)

        # Continuous high-freq signal
        t_cont = np.linspace(t_acc - 2.0, t_acc, BUF_LEN)
        y_cont = np.sin(2 * np.pi * FREQ_HIGH * t_cont)
        draw_waveform(screen, r1, y_cont, CLEAN,
                      f"Original Signal: {FREQ_HIGH:.0f} Hz (continuous)")

        # Sampled points
        t_start = max(0, t_acc - 2.0)
        n_samp = int(2.0 * SAMPLE_RATE)
        t_samp = np.linspace(t_start, t_acc, max(n_samp, 2))
        y_samp = np.sin(2 * np.pi * FREQ_HIGH * t_samp)

        # Reconstructed signal (what the sampled data looks like)
        y_recon = np.interp(t_cont, t_samp, y_samp)
        draw_waveform(screen, r2, y_recon, NOISY,
                      f"Sampled at {SAMPLE_RATE:.0f} Hz → appears as {alias_freq:.1f} Hz")

        # Draw sample dots on top panel
        cy1 = r1.centery
        half_h1 = (r1.h - 30) // 2
        d_max1 = 1.0
        draw_sampled_points(screen, r1, t_samp, y_samp, ALIAS_COL,
                            cy1, half_h1, d_max1, (float(t_cont[0]), float(t_cont[-1])))

        status = "ALIASED!" if is_aliased else "OK (Nyquist satisfied)"
        status_col = NOISY if is_aliased else FILTERED
        screen.blit(font_small.render(
            f"Nyquist rate = 2×f = {2*FREQ_HIGH:.0f} Hz  |  "
            f"Sample rate = {SAMPLE_RATE:.0f} Hz  |  Status: {status}",
            True, status_col), (15, 435))
        screen.blit(font_small.render(
            "Aliasing: under-sampling makes a high-freq signal look like a different low-freq signal",
            True, GRAY), (15, 462))

    else:
        screen.blit(font_title.render(
            f"Invalid TOPIC = {TOPIC}. Set to 1, 2, 3, 4, or 5.", True, NOISY), (30, 200))

    # ── Play/Pause + footer ─────────────────────────────────────────
    draw_button(screen)
    screen.blit(font_small.render("SPACE: Play/Pause  |  R: Reset  |  ESC: Quit",
                                  True, GRAY), (15, H - 32))

    pygame.display.flip()

    # Colab: save a single frame as image (non-interactive)
    if _in_colab and t_acc > 3.0:
        pygame.image.save(screen, "/tmp/signal_disturbance.png")
        print(f"Saved frame to /tmp/signal_disturbance.png (Topic {TOPIC})")
        running = False

pygame.quit()
if not _in_colab:
    sys.exit()
