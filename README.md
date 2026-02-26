# Mechanical Engineering Interactive Simulations

Interactive Python simulations built with **Pygame** for a Mechanical Engineering college student conference. Each simulation has real-time animation, draggable sliders, and a live parameter panel.

## Simulations

| # | Simulation | File | What it shows |
|---|-----------|------|---------------|
| 1 | **Spring-Mass Damper** | `sim_spring_mass.py` | Animated spring + damper, displacement vs time, phase portrait |
| 2 | **Pendulum** | `sim_pendulum.py` | Swinging pendulum, angle vs time, angular velocity |
| 3 | **Beam Bending** | `sim_beam_bending.py` | Beam deflection, bending moment, shear force diagrams |
| 4 | **Projectile Motion** | `sim_projectile.py` | Trajectory with and without air drag, animated projectiles |
| 5 | **Heat Conduction** | `sim_heat_transfer.py` | Color-mapped rod, temperature profile over time |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python sim_spring_mass.py
python sim_pendulum.py
python sim_beam_bending.py
python sim_projectile.py
python sim_heat_transfer.py
```

## Controls

- **Drag sliders** at the bottom to change parameters (plots update instantly)
- **SPACE** — Pause / resume animation (spring-mass, pendulum, projectile)
- **R** — Restart animation
- **ESC** — Quit
- **Click buttons** to toggle options (road bump, beam type, material)

## Dependencies

- Python 3.10+
- Pygame
- NumPy
- SciPy
