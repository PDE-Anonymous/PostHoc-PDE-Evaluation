# sec:generalization -- out-of-distribution generalization
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from errors import rollout
from utils import make_integrator, simulate

NEEDS_SIMULATION = True


def _ic1(x, L):
    # default IC (utils.py's simulate_ks)
    return np.cos(2 * np.pi * x / L) + 0.5 * np.cos(4 * np.pi * x / L + 0.3)


def _ic2(x, L):
    # two sine modes at an odd phase, so KS can't trap the solution on a symmetric sub-manifold
    return np.sin(2 * np.pi * x / L) + 0.3 * np.sin(6 * np.pi * x / L + 0.5)


def _sol2(x, L, c0=1.0, c1=0.5, x00=None, x01=None):
    # two-soliton profile of KdV at t=0, same convention as datagen/generate.py
    if x00 is None:
        x00 = L / 2 - 10
    if x01 is None:
        x01 = L / 2 - 5
    s0 = np.sqrt(c0) / 2 * (x - x00)
    s1 = np.sqrt(c1) / 2 * (x - x01)
    num = c0 * np.cosh(s1) ** 2 + c1 * np.sinh(s0) ** 2
    den = ((np.sqrt(c0) - np.sqrt(c1)) * np.cosh(s0 + s1) + (np.sqrt(c0) + np.sqrt(c1)) * np.cosh(s0 - s1))
    return 2 * (c0 - c1) * num / den ** 2


# ic=1 is the training IC, ic=2 a physically-motivated OOD IC (falls back to the generic modes below)
def _burgers_ic1(x, L):
    return np.cos(x)


def _burgers_ic2(x, L):
    # two-shock train -- steep fronts that blow up the discovered PDEs' rollout error around T~40
    return (-1.5 * np.tanh((x - L / 3) / 0.2) + 1.5 * np.tanh((x - 2 * L / 3) / 0.2))


def _kdv_ic1(x, L):
    return _sol2(x, L)  # training IC: analytic two-soliton at t=0


def _kdv_ic2(x, L):
    # not a real KdV soliton, so it disperses under the true PDE instead of translating cleanly -- sits close to pdefind's instability cliff without actually blowing up
    return 0.45 / np.cosh((x - L / 2) / 3.0) ** 2


def _ad_ic1(x, L):
    return np.exp(-((((x - L / 4) + L / 2) % L - L / 2) / 0.4) ** 2)


def _ad_ic2(x, L):
    # same bump as training, amplitude 6 -- makes both candidates blow up near t=5.6; deepmod looks worse at the short T=2.5 training time, pdefind worse near the T~5.6 blowup
    return 6.0 * np.exp(-((((x - L / 4) + L / 2) % L - L / 2) / 0.4) ** 2)


PDE_IC = {"burgers": {1: _burgers_ic1, 2: _burgers_ic2}, "kdv":     {1: _kdv_ic1, 2: _kdv_ic2}, "ks":      {1: _ic1, 2: _ic2}, "advection_diffusion": {1: _ad_ic1, 2: _ad_ic2}}

# Generic KS-flavoured modes (fallback for PDEs without a dedicated entry).
GENERIC_IC = {1: _ic1, 2: _ic2}


def ic_func(pde_name, ic_choice):
    if pde_name in PDE_IC and ic_choice in PDE_IC[pde_name]:
        return PDE_IC[pde_name][ic_choice]
    return GENERIC_IC[ic_choice]


def _simulate_ic(data, ic_choice):
    # resimulates from the chosen IC, independent of whatever U_ref/U_disc the caller already has
    if ic_choice not in GENERIC_IC and not (data.get("pde_name") in PDE_IC and ic_choice in PDE_IC[data.get("pde_name")]):
        raise ValueError(f"Unknown IC choice {ic_choice!r}; must be 1 or 2.")

    x, kk, dt = data["x"], data["kk"], data["dt"]
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    u0 = ic_func(data.get("pde_name"), ic_choice)(x, L)
    nt = data["U_ref"].shape[1]

    ref_int = make_integrator(data["b_ref"], data["lin_ref"], data["quad_ref"], kk, dt, x_coord=x, x_lin=data.get("xlin_ref", {}), t_lin=data.get("tlin_ref", {}), coord=data.get("coord_ref", {}))
    U_ref_ic = simulate(ref_int, u0, nt, dt=dt)

    disc_int = make_integrator(data["b_disc"], data["lin_disc"], data["quad_disc"], kk, dt, x_coord=x, x_lin=data.get("xlin_disc", {}), t_lin=data.get("tlin_disc", {}), coord=data.get("coord_disc", {}))
    U_disc_ic = simulate(disc_int, u0, nt, dt=dt)

    return u0, U_ref_ic, U_disc_ic


def IC_nMAE(data):
    # data["ic"] picks the IC to resimulate from (default 1 = training IC, not actually OOD)
    ic_choice = data.get("ic", 1)
    _, U_ref_ic, U_disc_ic = _simulate_ic(data, ic_choice)
    num = float(np.sum(np.abs(U_disc_ic - U_ref_ic)))
    den = float(np.sum(np.abs(U_ref_ic)))
    return num / den if den > 0 else float("nan")


def rollout_IC(data):
    # same as rollout(), but starting from a chosen IC instead of the training one
    ic_choice = data.get("ic", 1)
    _, U_ref_ic, U_disc_ic = _simulate_ic(data, ic_choice)
    d = dict(data)
    d["U_ref"], d["U_disc"] = U_ref_ic, U_disc_ic
    return rollout(d)


def conv_t_series(b, lin, quad, x, kk, u0, T, dt0, n_refine, xlin=None, tlin=None, coord=None):
    # simulates at successively halved dt; a shrinking diffs sequence means convergence, plateauing/growing flags numerical instability
    dts = [dt0 / (2 ** i) for i in range(n_refine + 1)]
    finals = []
    for dt_i in dts:
        nt_i = max(2, round(T / dt_i) + 1)
        integrator = make_integrator(b, lin, quad, kk, dt_i, x_coord=x, x_lin=xlin or {}, t_lin=tlin or {}, coord=coord or {})
        finals.append(simulate(integrator, u0, nt_i, dt=dt_i)[:, -1])

    diffs = []
    for i in range(len(finals) - 1):
        num = float(np.sum(np.abs(finals[i + 1] - finals[i])))
        den = float(np.sum(np.abs(finals[i + 1])))
        diffs.append(num / den if den > 0 else float("nan"))
    return dts, diffs


def conv_x_series(b, lin, quad, x0, dt, T, ic_choice, n_refine, pde_name=None, xlin=None, tlin=None, coord=None):
    # same idea as conv_t_series but doubling N -- dyadic grids so the coarser points are an exact subset of the finer ones (no interpolation needed)
    L = float(x0[-1] - x0[0] + (x0[1] - x0[0]))
    Ns = [len(x0) * (2 ** i) for i in range(n_refine + 1)]
    finals = []
    for N_i in Ns:
        x_i = np.linspace(0, L, N_i, endpoint=False)
        kk_i = np.fft.rfftfreq(N_i, d=L / N_i) * 2 * np.pi
        u0_i = ic_func(pde_name, ic_choice)(x_i, L)
        nt_i = max(2, round(T / dt) + 1)
        integrator = make_integrator(b, lin, quad, kk_i, dt, x_coord=x_i, x_lin=xlin or {}, t_lin=tlin or {}, coord=coord or {})
        finals.append(simulate(integrator, u0_i, nt_i, dt=dt)[:, -1])

    diffs = []
    for i in range(len(finals) - 1):
        fine_sub = finals[i + 1][::2]  # exact point match, since Ns are dyadic
        num = float(np.sum(np.abs(fine_sub - finals[i])))
        den = float(np.sum(np.abs(fine_sub)))
        diffs.append(num / den if den > 0 else float("nan"))
    return Ns, diffs


def Conv_t(data):
    # self-consistency check of the discovered PDE alone, no GT involved
    b, lin, quad = data["b_disc"], data["lin_disc"], data["quad_disc"]
    x, kk = data["x"], data["kk"]
    u0 = data["U_ref"][:, 0]
    T = float(data["t"][-1])
    dt0 = float(data["dt"])
    n_refine = int(data.get("n_refine", 6))

    dts, diffs = conv_t_series(b, lin, quad, x, kk, u0, T, dt0, n_refine, data.get("xlin_disc"), data.get("tlin_disc"), data.get("coord_disc"))
    return {f"dt={dts[i]:.2g}->dt={dts[i + 1]:.2g}": diffs[i] for i in range(len(diffs))}


def Conv_x(data):
    # same idea as Conv_t, refining the mesh instead
    b, lin, quad = data["b_disc"], data["lin_disc"], data["quad_disc"]
    x0 = data["x"]
    dt = float(data["dt"])
    T = float(data["t"][-1])
    ic_choice = data.get("ic", 1)
    n_refine = int(data.get("n_refine", 6))

    Ns, diffs = conv_x_series(b, lin, quad, x0, dt, T, ic_choice, n_refine, data.get("pde_name"), data.get("xlin_disc"), data.get("tlin_disc"), data.get("coord_disc"))
    return {f"N={Ns[i]}->N={Ns[i + 1]}": diffs[i] for i in range(len(diffs))}


METRICS = {"IC_nMAE":     IC_nMAE, "rollout_IC":  rollout_IC, "Conv_t":      Conv_t, "Conv_x":      Conv_x}
