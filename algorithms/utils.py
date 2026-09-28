import os
import warnings

import numpy as np


def pde_name_from_npz(path):
    if not path.lower().endswith(".npz"):
        return ""
    with np.load(path) as arch:
        raw = arch["pde"] if "pde" in arch else None
    if raw is None:
        return ""
    return raw.decode().strip().lower() if isinstance(raw, bytes) else str(raw).strip().lower()


def load_data(path, dx=None, dt=None):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".npz":
        with np.load(path) as archive:
            if "u" not in archive:
                raise ValueError(f"{path!r} must contain an array named 'u'")
            u = archive["u"]
            x = archive["x"] if "x" in archive else None
            t = archive["t"] if "t" in archive else None
            u_t = archive["u_t"] if "u_t" in archive else None
    elif extension == ".npy":
        raw = np.load(path, allow_pickle=True)
        raw = raw.item() if raw.ndim == 0 else raw
        if isinstance(raw, dict) and "u" in raw:
            u = raw["u"]
            x_raw, t_raw = raw.get("x"), raw.get("t")
            u_t = raw.get("u_t")
            x = x_raw[:, 0] if x_raw is not None and x_raw.ndim == 2 else x_raw
            t = t_raw[0, :] if t_raw is not None and t_raw.ndim == 2 else t_raw
        else:
            u, x, t, u_t = np.asarray(raw, dtype=float), None, None, None
    else:
        u, x, t, u_t = np.loadtxt(path, dtype=float), None, None, None

    if u.ndim != 2:
        raise ValueError(f"u must be a 2D array with shape (space, time), got {u.shape}")
    if x is None:
        x = np.arange(u.shape[0], dtype=float) * (1.0 if dx is None else dx)
    if t is None:
        t = np.arange(u.shape[1], dtype=float) * (1.0 if dt is None else dt)
    if len(x) != u.shape[0] or len(t) != u.shape[1]:
        raise ValueError("Coordinate lengths must match u.shape=(n_space, n_time)")
    if len(x) < 3 or len(t) < 3:
        raise ValueError("At least three spatial and temporal samples are required")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        raise ValueError("u, x and t must contain only finite values")
    return {"u": u, "x": x, "t": t, "u_t": u_t}


def _spectral_derivative(u, k, order, axis=0):
    k = k[:, None] if axis == 0 else k[None, :]
    return np.fft.irfft((1j * k) ** order * np.fft.rfft(u, axis=axis), n=u.shape[axis], axis=axis)


def _finite_derivative(u, x, order, axis=0):
    result = u
    for _ in range(order):
        result = np.gradient(result, x, axis=axis, edge_order=2)
    return result


def _spline_derivative(u, coord, order, axis=0, periodic=False):
    from scipy.interpolate import make_interp_spline

    degree = max(3, order + 1)
    n = u.shape[axis]
    if n <= degree:
        degree = n - 1
    if periodic:
        spacing = coord[1] - coord[0]
        coord_ext = np.concatenate([coord, [coord[0] + (coord[-1] - coord[0]) + spacing]])  # wrap one extra point around
    else:
        coord_ext = coord
    n_other = u.shape[1 - axis]

    def one_curve(y):
        y_ext = np.concatenate([y, y[:1]]) if periodic else y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = make_interp_spline(coord_ext, y_ext, k=degree, bc_type="periodic" if periodic else None)
        return spline.derivative(order)(coord)

    result = np.empty_like(u)
    for i in range(n_other):
        if axis == 0:
            result[:, i] = one_curve(u[:, i])
        else:
            result[i, :] = one_curve(u[i, :])
    return result


def spatial_derivative(u, x, order, deriv="spectral"):
    if order == 0:
        return u
    if deriv == "spectral":
        spacing = x[1] - x[0]
        k = 2 * np.pi * np.fft.rfftfreq(u.shape[0], d=spacing)
        return _spectral_derivative(u, k, order)
    if deriv == "spline":
        return _spline_derivative(u, x, order, axis=0, periodic=True)
    if deriv == "fd":
        return _finite_derivative(u, x, order, axis=0)
    raise ValueError(f"unknown derivative method {deriv!r}")


def temporal_derivative(u, t, deriv="fd"):
    if deriv == "fd":
        return _finite_derivative(u, t, 1, axis=1)
    if deriv == "spline":
        return _spline_derivative(u, t, 1, axis=1, periodic=False)
    if deriv == "spectral":
        spacing = t[1] - t[0]
        k = 2 * np.pi * np.fft.rfftfreq(u.shape[1], d=spacing)
        return _spectral_derivative(u, k, 1, axis=1)
    raise ValueError(f"unknown derivative method {deriv!r}")


def save_pde(path, coefficients):
    with open(path, "w", encoding="utf-8") as handle:
        for term, coefficient in coefficients.items():
            handle.write(f"{term}: {coefficient:.12g}\n")
