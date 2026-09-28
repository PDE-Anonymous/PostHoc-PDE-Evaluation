import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared import pde_terms_from_data


def _trajectory_fn(keys, kk, dt, N, x_coord, u0, t0, nt, nt_inner):
    # JAX version of utils.make_integrator's ETD2-RK step, differentiable w.r.t. vals (term coeffs)
    kk = jnp.asarray(kk)
    x_coord = jnp.asarray(x_coord) if x_coord is not None else None
    u_orders = tuple(sorted({f - 2 for key in keys for f in key if f >= 2}))
    lin_idxs = [(i, key[0] - 2) for i, key in enumerate(keys) if len(key) == 1 and key[0] >= 2]

    def derivs(u):
        uh = jnp.fft.rfft(u)
        return {o: (u if o == 0 else jnp.fft.irfft((1j * kk) ** o * uh, n=N)) for o in u_orders}

    def run(vals):
        L_hat = (sum(vals[i] * (1j * kk) ** o for i, o in lin_idxs).astype(complex) if lin_idxs else jnp.zeros_like(kk, dtype=complex))
        eL = jnp.exp(L_hat * dt); eL2 = jnp.exp(L_hat * dt / 2)
        safe_L = jnp.where(jnp.abs(L_hat) < 1e-14, 1.0, L_hat)
        safe2 = jnp.where(jnp.abs(L_hat ** 2) < 1e-14, 1.0, L_hat ** 2)
        c1 = jnp.where(jnp.abs(L_hat) < 1e-14, dt, (eL - 1.0) / safe_L)
        ch = jnp.where(jnp.abs(L_hat) < 1e-14, dt / 2, (eL2 - 1.0) / safe_L)
        av = jnp.where(jnp.abs(L_hat) < 1e-14, dt / 2, (eL - 1.0 - L_hat * dt) / (safe2 * dt))
        mask = ((jnp.abs(kk) < (2 / 3) * kk.max()).astype(jnp.float64) if any(len(key) >= 2 for key in keys) else jnp.ones_like(kk))

        def nl_of(u, t_sim):
            nl = jnp.full((N,), vals[0] if len(keys) and len(keys[0]) == 0 else 0.0, dtype=jnp.float64) if (keys and len(keys[0]) == 0) else jnp.full((N,), 0.0, dtype=jnp.float64)
            d = derivs(u)
            for i, key in enumerate(keys):
                if len(key) == 1 and key[0] >= 2:
                    continue  # linear u_o handled exactly by exp_L
                v = vals[i]
                if len(key) == 0:  # b (only if index 0 not already used)
                    continue
                if len(key) == 1:
                    if key[0] == 0:  # t
                        nl = nl + v * t_sim
                    else:  # x
                        nl = nl + v * x_coord
                else:
                    prod = None
                    for f in key:
                        fv = t_sim if f == 0 else (x_coord if f == 1 else d[f - 2])
                        prod = fv if prod is None else prod * fv
                    nl = nl + v * prod
            return mask * jnp.fft.rfft(nl)

        def step(u, t_sim):
            uh = jnp.fft.rfft(u)
            N1 = nl_of(u, t_sim)
            ua = jnp.fft.irfft(eL2 * uh + ch * N1, n=N)
            N2 = nl_of(ua, t_sim + dt / 2)
            return jnp.fft.irfft(eL * uh + c1 * N1 + av * (N2 - N1), n=N)

        def inner(carry, _):
            u, t_sim = carry
            u_new = step(u, t_sim)
            return (u_new, t_sim + dt), None

        def outer(carry, _):
            (u_end, t_end), _ = jax.lax.scan(inner, carry, None, length=nt_inner)
            return (u_end, t_end), u_end

        _, U_rest = jax.lax.scan(outer, (u0, t0), None, length=nt - 1)
        return jnp.concatenate([u0[None, :], U_rest], axis=0)  # (nt, N)

    return run


def sensitivity_weights(data):
    # returns (keys, w): term keys and their trajectory-jacobian sensitivity weight
    ref  = pde_terms_from_data(data, "ref")
    disc = pde_terms_from_data(data, "disc")
    keys = sorted(set(ref) | set(disc), key=lambda k: (len(k), k))
    vals = jnp.array([ref.get(k, 0.0) for k in keys], dtype=jnp.float64)

    U_ref = data["U_ref"]
    N, nt = U_ref.shape
    dt = float(data["dt"])
    nt_inner = max(1, round(float(data["t"][1] - data["t"][0]) / dt))
    u0 = jnp.asarray(U_ref[:, 0], dtype=jnp.float64)
    t0 = float(data["t"][0])

    run = _trajectory_fn(keys, data["kk"], dt, N, data.get("x"), u0, t0, nt, nt_inner)
    jac = jax.jacfwd(run)(vals)  # (nt, N, n_params)
    w = np.array(jnp.sum(jac ** 2, axis=(0, 1)))

    return keys, w
