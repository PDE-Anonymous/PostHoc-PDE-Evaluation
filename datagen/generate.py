#!/usr/bin/env python3
import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import make_integrator as _shared_make_integrator


def two_soliton(x, t, L, c0=1.0, c1=0.5, x00=None, x01=None):
    if x00 is None:
        x00 = L / 2 - 10
    if x01 is None:
        x01 = L / 2 - 5
    Xg, Tg = np.meshgrid(x, t)
    xi0 = np.sqrt(c0) / 2 * (Xg - c0 * Tg - x00)
    xi1 = np.sqrt(c1) / 2 * (Xg - c1 * Tg - x01)
    num = c0 * np.cosh(xi1) ** 2 + c1 * np.sinh(xi0) ** 2
    den = ((np.sqrt(c0) - np.sqrt(c1)) * np.cosh(xi0 + xi1) + (np.sqrt(c0) + np.sqrt(c1)) * np.cosh(xi0 - xi1))
    return 2 * (c0 - c1) * num / den ** 2


def two_soliton_ic(x, L):
    return two_soliton(x, np.array([0.0]), L)[0]


PDE_CONFIG = {"burgers": {"length": 2 * np.pi, "N": 512, "dt": 0.0025, "T": 2.0, "linear": {2: 0.05}, "nonlinear": -1.0, "dealias": False, "ic": lambda x, L: np.cos(x), "rhs": lambda U, U_x, U_xx, nu: -U * U_x + nu * U_xx, "nu": 0.05}, 
              "kdv": {"length": 60.0, "N": 512, "dt": 0.1, "T": 20.0, "linear": {3: -1.0}, "nonlinear": -6.0, "dealias": True, "ic": two_soliton_ic, "rhs": lambda U, U_x, U_xx, U_xxx, U_xxxx: -6 * U * U_x - U_xxx}, 
              "ks": {"length": 22.0, "N": 1024, "dt": 50.0 / 999, "T": 50.0, "linear": {2: -1.0, 4: -1.0}, "nonlinear": -1.0, "dealias": True, "ic": lambda x, L: np.cos(2 * np.pi * x / L) + 0.5 * np.cos(4 * np.pi * x / L + 0.3), "rhs": lambda U, U_x, U_xx, U_xxx, U_xxxx, dealias=None: -U * U_x - U_xx - U_xxxx}, 
              "advection_diffusion": {"length": 2 * np.pi, "N": 512, "dt": 0.0025, "T": 2.5, "linear": {1: -1.0, 2: 0.1}, "nonlinear": 0.0, "dealias": False, "ic": lambda x, L: np.exp(-((((x - L / 4) + L / 2) % L - L / 2) / 0.4) ** 2), "rhs": lambda U, U_x, U_xx: -U_x + 0.1 * U_xx}}


EXPLODE_FACTOR = 1e4
CHECK_EVERY = 100
MAX_DTFAC = 8


def _make_step(pde, k, dt, x):
    config = PDE_CONFIG[pde]
    nonlinear = config.get("nonlinear", 0.0)
    quad = [(0, 1, nonlinear)] if nonlinear != 0.0 else []  # (0, 1) = u * u_x
    return _shared_make_integrator(0.0, config["linear"], quad, k, dt, x_coord=x)


def simulate(pde, x, T, dt):
    config = PDE_CONFIG[pde]
    length = config["length"]
    k = 2 * np.pi * np.fft.rfftfreq(len(x), d=length / len(x))
    u0 = config["ic"](x, length)
    u0_max = float(np.max(np.abs(u0)))
    dt = float(dt)

    for attempt in range(MAX_DTFAC):  # halve dt and retry if the run blows up
        n_time = max(2, round(T / dt) + 1)
        step = _make_step(pde, k, dt, x)
        values = np.zeros((len(x), n_time))
        values[:, 0] = u = u0.copy()
        t_sim = 0.0
        exploded = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n_time - 1):
                u = step(u, t_sim)
                t_sim += dt
                values[:, i + 1] = u
                if i % CHECK_EVERY == 0 and (not np.all(np.isfinite(u)) or np.max(np.abs(u)) > EXPLODE_FACTOR * max(u0_max, 1.0)):
                    exploded = True
                    break
        if not exploded:
            if attempt > 0:
                print(f"  (stable dt={dt:.6g} after {attempt + 1} attempt(s))")
            return values, np.arange(n_time) * dt
        print(f"  dt={dt:.6g} unstable, retrying with dt/2 ...")
        dt /= 2

    raise RuntimeError(f"Simulation of {pde} exploded after {MAX_DTFAC} dt reductions (final dt={dt:.6g}). Try a shorter T or finer dx.")


def plot_data(path, pde, x, t, values):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    T_grid, X_grid = np.meshgrid(t, x)
    im = ax.pcolormesh(T_grid, X_grid, values, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_ylabel("space")
    ax.set_xlabel("time")
    fig.colorbar(im, ax=ax, label="u(x,t)")
    ax.set_title(f"{pde.upper()} data")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate 1D PDE data")
    parser.add_argument("--pde", choices=sorted(PDE_CONFIG), default="ks")
    parser.add_argument("--dx", type=float, default=0.1, help="Spatial resolution")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--T", type=float, default=None, help="Final time")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = PDE_CONFIG[args.pde]
    length = config["length"]
    N = config["N"]
    dt = args.dt if args.dt is not None else config["dt"]
    T = args.T if args.T is not None else config["T"]
    dx = args.dx if args.dx is not None else length / N

    n_space = max(4, round(length / dx))
    if n_space % 2 != 0:
        n_space += 1
    x = np.linspace(0, length, n_space, endpoint=False)
    values, t = simulate(args.pde, x, T, dt)
    data = {"x": x, "t": t, "u": values, "pde": np.array(args.pde), "disposition": np.array("uniform")}

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.pde}.npz")
    np.savez(output, **data)
    plot_path = os.path.splitext(output)[0] + ".png"
    plot_data(plot_path, args.pde, x, t, values)
    print(f"Saved {args.pde} data to {output}")
    print(f"Saved data plot to {plot_path}")
    print(f"  points: {values.size}")
    print(f"  dx={dx}, dt={t[1]-t[0]}, T={t[-1]:g}")


if __name__ == "__main__":
    main()
