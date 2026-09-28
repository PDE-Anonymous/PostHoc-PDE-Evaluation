import json
import os
import re

import numpy as np
import sympy as sp

DIR = os.path.dirname(os.path.abspath(__file__))
PDE_ROOT = os.path.join(DIR, "pde_files")

ORDER_NAME = {0: "u", 1: "u_x", 2: "u_xx", 3: "u_xxx", 4: "u_xxxx"}


# lets configs write ICs as plain text formulas instead of Python code
def _eval_expr(expr_str, namespace):
    names = list(namespace)
    expr = sp.sympify(expr_str)
    fn = sp.lambdify(names, expr, modules=["numpy"])
    return fn(*(namespace[n] for n in names))


# params (scalars) and vars (arrays) can reference x/L and anything defined before them
def _config_ic(config, x, L):
    namespace = {"x": x, "L": L}
    for name, val in (config.get("ic_params") or {}).items():
        namespace[name] = val if isinstance(val, (int, float)) else _eval_expr(str(val), namespace)
    for name, expr_str in (config.get("ic_vars") or {}).items():
        namespace[name] = _eval_expr(expr_str, namespace)
    return _eval_expr(config["ic"], namespace)


# a config only holds the domain/IC -- the PDE itself always comes from --fileGT
def load_reference_config(path):
    with open(path) as f:
        raw = json.load(f)
    config = dict(raw)
    config.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return config


# just builds the domain + IC, no PDE integration (that happens elsewhere from --fileGT)
def simulate_from_config(config, T=None):
    N, L, dt = config["N"], config["L"], config["dt"]
    T_eff = T if T is not None else config["T"]
    default_nt = config.get("default_nt", round(config["T"] / dt) + 1)
    nt = default_nt if T is None else round(T_eff / dt) + 1
    t = np.arange(nt) * dt
    x = np.linspace(0, L, N, endpoint=False)
    u0 = _config_ic(config, x, L)
    print(f"Loaded {config.get('name', 'custom')} domain/IC from config (N={N}, nt={nt})")
    return x, t, {"u": u0[:, None]}


# PDE parsing
def deriv_order(name):
    return name.count("x")


# 'u' -> 0, 'u_xxx' -> 3, None if not a u-family term
def factor_order(name):
    if name == "u":
        return 0
    if name.startswith("u_") and name[2:] and all(c == "x" for c in name[2:]):
        return len(name) - 2
    return None


_SUPERSCRIPT = {"⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9}


def _expand_factor(name):
    # 'u²' -> ['u', 'u'], 'u_xx⁴' -> 4x ['u_xx'], etc.
    m = re.match(r"^(.*?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", name)
    if not m:
        return [name]
    base, exp_str = m.groups()
    if not base:
        return [name]
    n = 0
    for ch in exp_str:
        n = n * 10 + _SUPERSCRIPT[ch]
    if n <= 0:
        return [name]
    return [base] * n


def _parse_factors(key):
    # 'u²·u_xx' -> ['u', 'u', 'u_xx']
    factors = []
    for part in key.split("·"):
        factors.extend(_expand_factor(part.strip()))
    return factors


def parse_pde_file(path):
    # returns (b, linear, quad, x_lin, t_lin, coord) -- coord is always {}, pure coordinate terms like "t"/"x"/"t·x" get dropped with a warning
    def _drop(key):
        print(f"  note: dropping pure-coordinate term '{key}:' (not part of the dictionary)")

    b = 0.0; linear = {}; quad = []; x_lin = {}; t_lin = {}; coord = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            key, val_str = s.split(":", 1)
            key = key.strip()
            if key in ("pde", "base", "source"):
                continue
            try:
                val = float(val_str.strip())
            except ValueError:
                continue
            if key in ("b", "bias"):
                b = val
                continue
            factors = _parse_factors(key)
            if len(factors) == 1:
                p = factors[0]
                if p in ("t", "T", "x", "X"):
                    _drop(key)
                    continue
                o = factor_order(p)
                if o is not None:
                    linear[o] = linear.get(o, 0.0) + val
            elif len(factors) == 2:
                a, c = factors
                oa, oc = factor_order(a), factor_order(c)
                if oa is not None and oc is not None:
                    quad.append((oa, oc, val))
                elif a in ("x", "X") and oc is not None:
                    x_lin[oc] = x_lin.get(oc, 0.0) + val
                elif c in ("x", "X") and oa is not None:
                    x_lin[oa] = x_lin.get(oa, 0.0) + val
                elif a in ("t", "T") and oc is not None:
                    t_lin[oc] = t_lin.get(oc, 0.0) + val
                elif c in ("t", "T") and oa is not None:
                    t_lin[oa] = t_lin.get(oa, 0.0) + val
                else:
                    _drop(key)  # coordinate-only pair (t·t / x·x / t·x), no u factor
            else:
                # 3+ factors: keep pure u-family products (u·u·u, u²·u_xx, ...), skip mixed t/x ones
                ords = [factor_order(p) for p in factors]
                if all(o is not None for o in ords):
                    quad.append((*ords, val))
    return b, linear, quad, x_lin, t_lin, coord


# Extracts the PDE system name (e.g. 'ks') from a file's containing directory
def pde_from_dir(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def pde_label(b, linear, quad, sep=",  ", x_lin=None, t_lin=None, coord=None):
    parts = []
    if abs(b) > 1e-12:
        parts.append(f"b={b:.3g}")
    for o in sorted(linear):
        parts.append(f"{ORDER_NAME.get(o, 'u_'+'x'*o)}={linear[o]:.3g}")
    for *ords, val in quad:
        name = "·".join(ORDER_NAME.get(o, 'u_' + 'x' * o) for o in ords)
        parts.append(f'{name}={val:.3g}')
    for o in sorted(x_lin or {}):
        nu = ORDER_NAME.get(o, 'u_' + 'x'*o)
        parts.append(f'x·{nu}={x_lin[o]:.3g}')
    for o in sorted(t_lin or {}):
        nu = ORDER_NAME.get(o, 'u_' + 'x'*o)
        parts.append(f't·{nu}={t_lin[o]:.3g}')
    return sep.join(parts)


# ETD2-RK integrator (Cox & Matthews 2002)
def make_integrator(b, linear, quad, kk, dt, x_coord=None, x_lin=None, t_lin=None, coord=None, dealias=None):
    # ETD2-RK (not ETD1): needed for long-horizon stability on PDEs like KdV. dealias=None auto-enables 2/3 dealiasing whenever quad/x_lin/t_lin forcing terms are present, matching datagen/generate.py.
    L_hat  = sum(c * (1j * kk) ** o for o, c in linear.items()).astype(complex) if linear else np.zeros_like(kk, dtype=complex)
    eL     = np.exp(L_hat * dt)
    eL2    = np.exp(L_hat * dt / 2)
    safe_L = np.where(np.abs(L_hat) < 1e-14, 1.0, L_hat)
    safe2  = np.where(np.abs(L_hat ** 2) < 1e-14, 1.0, L_hat ** 2)
    c1     = np.where(np.abs(L_hat) < 1e-14, dt,     (eL  - 1.0) / safe_L)
    ch     = np.where(np.abs(L_hat) < 1e-14, dt / 2, (eL2 - 1.0) / safe_L)
    av     = np.where(np.abs(L_hat) < 1e-14, dt / 2, (eL - 1.0 - L_hat * dt) / (safe2 * dt))
    N      = len(kk) * 2 - 2
    x_lin  = x_lin or {}
    t_lin  = t_lin or {}
    coord  = coord or {}
    xr     = np.asarray(x_coord, dtype=float) if x_coord is not None else np.zeros(N)
    if dealias is None:
        dealias = bool(quad) or bool(x_lin) or bool(t_lin) or bool(coord)
    mask   = (np.abs(kk) < (2 / 3) * kk.max()).astype(float) if dealias else np.ones(len(kk))
    cache  = {}

    def d(u, o):
        if o not in cache:
            uh = np.fft.rfft(u)
            cache[o] = u if o == 0 else np.fft.irfft((1j * kk) ** o * uh, n=N)
        return cache[o]

    def nl_hat(u, t_sim):
        cache.clear()
        nl = np.full(N, b, dtype=float)
        for *ords, val in quad:
            prod = d(u, ords[0])
            for o in ords[1:]:
                prod = prod * d(u, o)
            nl += val * prod
        for o, c in x_lin.items():
            nl += c * xr * d(u, o)
        for o, c in t_lin.items():
            nl += c * t_sim * d(u, o)
        return mask * np.fft.rfft(nl)

    def step(u, t_sim):
        uh = np.fft.rfft(u)
        N1 = nl_hat(u, t_sim)
        ua = np.fft.irfft(eL2 * uh + ch * N1, n=N)
        N2 = nl_hat(ua, t_sim + dt / 2)
        return np.fft.irfft(eL * uh + c1 * N1 + av * (N2 - N1), n=N)

    return step


def simulate(step_fn, u0, nt, nt_inner=1, dt=None, t0=0.0):
    # nt_inner inner steps per output frame; t_sim threaded through so time-dependent terms stay correct
    N = len(u0)
    U = np.zeros((N, nt))
    u = u0.copy(); U[:, 0] = u
    t_sim = t0
    for n in range(nt - 1):
        for _ in range(nt_inner):
            u = step_fn(u, t_sim)
            if dt is not None:
                t_sim += dt
        U[:, n + 1] = u
    return U


# coefficient vector
def coeff_vector(linear, quad):
    return np.array([linear[o] for o in sorted(linear)] + [p[-1] for p in quad])


# RHS and term weights
def get_derivs(U, kk):
    N, nt = U.shape
    cache = {0: U}

    def get(o):
        if o not in cache:
            cache[o] = np.array([np.fft.irfft((1j*kk)**o * np.fft.rfft(U[:, n]), n=N) for n in range(nt)]).T
        return cache[o]

    return get


def eval_rhs(b, linear, quad, U, kk):
    # u-family terms only; see eval_rhs_full for x·/t· products too
    return eval_rhs_full(b, linear, quad, U, kk)


def eval_rhs_full(b, linear, quad, U, kk, x_lin=None, t_lin=None, coord=None, x=None, t=None):
    # coord kept only for signature compat with callers, always ignored
    get = get_derivs(U, kk)
    rhs = np.full(U.shape, b, dtype=float)
    for o, c in linear.items():
        rhs += c * get(o)
    for *ords, val in quad:
        prod = get(ords[0])
        for o in ords[1:]:
            prod = prod * get(o)
        rhs += val * prod
    if x is not None:
        xc = np.broadcast_to(np.asarray(x, dtype=float)[:, None], U.shape)
        for o, c in (x_lin or {}).items():
            rhs += c * xc * get(o)
    if t is not None:
        tg = np.broadcast_to(np.asarray(t, dtype=float)[None, :], U.shape)
        for o, c in (t_lin or {}).items():
            rhs += c * tg * get(o)
    return rhs


def term_weights(linear, quad, U_ref, kk):
    get = get_derivs(U_ref, kk)
    w_lin  = {o: float(np.mean(np.abs(get(o)))) for o in linear}

    def w_product(ords):
        prod = get(ords[0])
        for o in ords[1:]:
            prod = prod * get(o)
        return float(np.mean(np.abs(prod)))

    w_quad = [(*ords, w_product(ords)) for *ords, _ in quad]
    return w_lin, w_quad
