"""
Vibration & Frequency Analysis — 5 Topics in One File (Google Colab)

Set TOPIC = 1..5 to pick which demo to run.

1 = Free Vibration         (undamped sine wave)
2 = Damped Vibration       (exponentially decaying sine)
3 = Forced Vibration       (steady-state response to external force)
4 = Resonance              (amplitude vs forcing frequency — spike at ωn)
5 = FFT of Vibration       (time-domain signal → frequency spectrum)

Works in Google Colab: animated topics render via HTML(anim.to_jshtml()).
Also works locally (auto-detects environment).
"""

# Auto-select a working matplotlib backend when running locally
_in_notebook = False
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None and 'IPKernelApp' in _ip.config:
        _in_notebook = True
except Exception:
    pass

if not _in_notebook:
    import matplotlib
    matplotlib.use('webagg')

# ═══════════════════════════════════════════════════════════════════
#  CHOOSE TOPIC HERE (1 to 5)
# ═══════════════════════════════════════════════════════════════════
TOPIC = 1

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS — tweak these to explore the physics
# ═══════════════════════════════════════════════════════════════════
m = 1.0          # mass (kg)
k = 40.0         # spring stiffness (N/m)
c = 2.0          # damping coefficient (Ns/m) — used in topics 2-5
F0 = 5.0         # amplitude of external force (N) — used in topics 3-4
x0 = 1.0         # initial displacement (m)

# Derived quantities
import numpy as np
wn = np.sqrt(k / m)                       # natural frequency (rad/s)
fn = wn / (2 * np.pi)                     # natural frequency (Hz)
zeta = c / (2 * np.sqrt(k * m))           # damping ratio (dimensionless)
wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0  # damped natural freq

import matplotlib.pyplot as plt
import matplotlib.animation as animation

print(f"Natural frequency  ωn = {wn:.2f} rad/s  ({fn:.2f} Hz)")
print(f"Damping ratio       ζ = {zeta:.3f}")
if zeta < 1:
    print(f"Damped frequency   ωd = {wd:.2f} rad/s")
print(f"Running Topic {TOPIC}\n")

# ═══════════════════════════════════════════════════════════════════
#  TOPIC 1 — Free Vibration (no damping)
# ═══════════════════════════════════════════════════════════════════
if TOPIC == 1:
    # Physics: A mass on a spring with no friction oscillates forever.
    # x(t) = x0 * cos(ωn * t)
    # The motion is a pure cosine at the natural frequency ωn = √(k/m).

    t = np.linspace(0, 4, 1000)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 4)
    ax.set_ylim(-x0 * 1.4, x0 * 1.4)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement x (m)", fontsize=12)
    ax.set_title(f"Free Vibration (undamped) — ωn = {wn:.2f} rad/s", fontsize=14)
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], "#3498DB", lw=2)
    dot, = ax.plot([], [], "o", color="#E74C3C", ms=8)

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def animate(i):
        ti = t[:i + 1]
        xi = x0 * np.cos(wn * ti)
        line.set_data(ti, xi)
        dot.set_data([ti[-1]], [xi[-1]])
        return line, dot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    try:
        from IPython.display import HTML
        plt.close()
        HTML(anim.to_jshtml())
    except ImportError:
        plt.show()


# ═══════════════════════════════════════════════════════════════════
#  TOPIC 2 — Damped Vibration
# ═══════════════════════════════════════════════════════════════════
elif TOPIC == 2:
    # Physics: Real systems lose energy to friction/air resistance.
    # The amplitude decays exponentially: x(t) = x0 * e^(-ζωn t) * cos(ωd t)
    # ζ < 1  →  underdamped (oscillates while decaying)
    # ζ = 1  →  critically damped (fastest return, no oscillation)
    # ζ > 1  →  overdamped (slow, no oscillation)

    t = np.linspace(0, 6, 1200)

    if zeta < 1:
        x = x0 * np.exp(-zeta * wn * t) * np.cos(wd * t)
        label = f"Underdamped (ζ = {zeta:.3f})"
    elif abs(zeta - 1) < 0.01:
        x = x0 * (1 + wn * t) * np.exp(-wn * t)
        label = "Critically damped (ζ ≈ 1)"
    else:
        s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
        s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
        A = x0 * s2 / (s2 - s1)
        B = -x0 * s1 / (s2 - s1)
        x = A * np.exp(s1 * t) + B * np.exp(s2 * t)
        label = f"Overdamped (ζ = {zeta:.3f})"

    envelope = x0 * np.exp(-zeta * wn * t)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 6)
    ax.set_ylim(-x0 * 1.4, x0 * 1.4)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement x (m)", fontsize=12)
    ax.set_title(f"Damped Vibration — {label}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.plot(t, envelope, "--", color="#95A5A6", lw=1, label="Decay envelope")
    ax.plot(t, -envelope, "--", color="#95A5A6", lw=1)
    line, = ax.plot([], [], "#2ECC71", lw=2, label=label)
    dot, = ax.plot([], [], "o", color="#E74C3C", ms=8)
    ax.legend(fontsize=10)

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def animate(i):
        line.set_data(t[:i + 1], x[:i + 1])
        dot.set_data([t[i]], [x[i]])
        return line, dot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    try:
        from IPython.display import HTML
        plt.close()
        HTML(anim.to_jshtml())
    except ImportError:
        plt.show()


# ═══════════════════════════════════════════════════════════════════
#  TOPIC 3 — Forced Vibration
# ═══════════════════════════════════════════════════════════════════
elif TOPIC == 3:
    # Physics: An external harmonic force F(t) = F0*sin(ω*t) drives the system.
    # After the transient dies out, the mass oscillates at the FORCING frequency ω
    # (not the natural frequency). The steady-state amplitude depends on how
    # close ω is to ωn.
    #
    # Full response = transient (decays) + steady-state (persists)
    # x_ss(t) = X * sin(ω*t - φ)
    # where X = F0/k / √[(1-r²)² + (2ζr)²],  r = ω/ωn

    w_force = wn * 0.6       # forcing frequency (chosen away from resonance)
    r = w_force / wn
    X_ss = (F0 / k) / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
    phi = np.arctan2(2 * zeta * r, 1 - r**2)

    t = np.linspace(0, 8, 1600)

    # Transient part (decays away)
    if zeta < 1:
        x_transient = x0 * np.exp(-zeta * wn * t) * np.cos(wd * t)
    else:
        x_transient = x0 * np.exp(-wn * t)
    # Steady-state part
    x_steady = X_ss * np.sin(w_force * t - phi)
    x_total = x_transient + x_steady

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 8)
    ymax = max(abs(x_total)) * 1.3
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement x (m)", fontsize=12)
    ax.set_title(f"Forced Vibration — ω_force = {w_force:.2f} rad/s, "
                 f"ωn = {wn:.2f} rad/s", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax.plot(t, x_steady, "--", color="#95A5A6", lw=1, alpha=0.6, label="Steady-state only")
    line, = ax.plot([], [], "#E67E22", lw=2, label="Total response")
    dot, = ax.plot([], [], "o", color="#E74C3C", ms=8)
    ax.legend(fontsize=10)

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def animate(i):
        line.set_data(t[:i + 1], x_total[:i + 1])
        dot.set_data([t[i]], [x_total[i]])
        return line, dot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    try:
        from IPython.display import HTML
        plt.close()
        HTML(anim.to_jshtml())
    except ImportError:
        plt.show()


# ═══════════════════════════════════════════════════════════════════
#  TOPIC 4 — Resonance
# ═══════════════════════════════════════════════════════════════════
elif TOPIC == 4:
    # Physics: When the forcing frequency ω equals the natural frequency ωn,
    # the system absorbs maximum energy → amplitude becomes very large.
    # This is RESONANCE — the most dangerous condition in mechanical design.
    #
    # Amplitude ratio X/(F0/k) = 1 / √[(1-r²)² + (2ζr)²]
    # At r = 1 (resonance), amplitude = 1/(2ζ) — only damping limits it.

    r_vals = np.linspace(0.01, 3.0, 500)   # frequency ratio ω/ωn
    zeta_list = [0.05, 0.1, 0.2, 0.5, 1.0]
    colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for z, col in zip(zeta_list, colors):
        X_ratio = 1.0 / np.sqrt((1 - r_vals**2)**2 + (2 * z * r_vals)**2)
        ax.plot(r_vals, X_ratio, color=col, lw=2, label=f"ζ = {z}")

    ax.axvline(x=1.0, color="#95A5A6", ls="--", lw=1, alpha=0.7)
    ax.annotate("Resonance\n(ω = ωn)", xy=(1.0, 0), xytext=(1.3, 8),
                fontsize=11, color="#95A5A6",
                arrowprops=dict(arrowstyle="->", color="#95A5A6"))

    ax.set_xlabel("Frequency Ratio  r = ω / ωn", fontsize=12)
    ax.set_ylabel("Amplitude Ratio  X / (F₀/k)", fontsize=12)
    ax.set_title("Resonance — Amplitude vs Forcing Frequency", fontsize=14)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=11, title="Damping ratio", title_fontsize=11)
    ax.grid(True, alpha=0.3)

    # This is a static plot — no animation needed
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  TOPIC 5 — FFT of Vibration Signal
# ═══════════════════════════════════════════════════════════════════
elif TOPIC == 5:
    # Physics: Real vibration signals are messy — they contain multiple
    # frequency components. The FFT (Fast Fourier Transform) decomposes
    # a time-domain signal into its constituent frequencies.
    # Engineers use this to identify which frequencies are present
    # (e.g., to find which component in a machine is vibrating).

    fs = 500                # sampling rate (Hz)
    T = 2.0                 # signal duration (s)
    t = np.arange(0, T, 1 / fs)

    # Build a signal with 3 known frequencies + noise
    f1, f2, f3 = fn, 25.0, 60.0   # Hz (fn is the natural frequency)
    signal = (1.0 * np.sin(2 * np.pi * f1 * t)
              + 0.5 * np.sin(2 * np.pi * f2 * t)
              + 0.3 * np.sin(2 * np.pi * f3 * t)
              + 0.2 * np.random.randn(len(t)))

    # Compute FFT
    N = len(t)
    fft_vals = np.fft.rfft(signal)
    fft_mag = 2.0 / N * np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, 1 / fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Time domain
    ax1.plot(t, signal, "#3498DB", lw=0.8)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.set_title("Time Domain — Vibration Signal (3 frequencies + noise)", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Frequency domain (bar chart style)
    mask = freqs <= 100  # show up to 100 Hz
    ax2.bar(freqs[mask], fft_mag[mask], width=0.5, color="#E74C3C", alpha=0.85)
    for fi, label in [(f1, f"f₁={f1:.1f}Hz"), (f2, f"f₂={f2:.0f}Hz"), (f3, f"f₃={f3:.0f}Hz")]:
        ax2.annotate(label, xy=(fi, fft_mag[np.argmin(np.abs(freqs - fi))]),
                     xytext=(fi + 3, fft_mag[np.argmin(np.abs(freqs - fi))] + 0.05),
                     fontsize=10, color="#2C3E50",
                     arrowprops=dict(arrowstyle="->", color="#2C3E50"))

    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("Magnitude", fontsize=12)
    ax2.set_title("Frequency Spectrum (FFT) — Peaks reveal hidden frequencies", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print(f"Invalid TOPIC = {TOPIC}. Set TOPIC to 1, 2, 3, 4, or 5.")
