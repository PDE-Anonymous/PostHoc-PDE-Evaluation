import argparse, itertools, os, re, sys, warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path: sys.path.remove(_SCRIPT_DIR)
import numpy as np
import pysindy as ps
sys.path.insert(0, _SCRIPT_DIR)  # re-add last so our own utils.py wins over any other "utils" already on the path
from utils import load_data, save_pde, spatial_derivative, temporal_derivative, pde_name_from_npz


def _factor(u, x, t, token, deriv):
    if token == "u": return u
    if token == "x": return np.broadcast_to(x[:, None], u.shape)
    if token == "t": return np.broadcast_to(t[None, :], u.shape)
    if re.fullmatch(r"u_x+", token): return spatial_derivative(u, x, token.count("x"), deriv)
    raise ValueError(f"unknown dictionary factor {token!r}")


def _coord_only(term):
    return all(token in ("t", "x") for token in term.split())


# mirrors deepmod.py's own defaults so both algorithms train on the same grid
_DEFAULT_STRIDE = {"burgers": (1, 6), "kdv": (7, 25), "ks": (4, 100), "advection_diffusion": (2, 5)}
_DEFAULT_THRESHOLD = {"burgers": 0.0015, "kdv": 0.05, "ks": 0.1, "advection_diffusion": 0.005}
_DEFAULT_DERIV = {"burgers": "fd", "kdv": "spline", "ks": "fd", "advection_diffusion": "spline"}
_DEFAULT_RIDGE = {"kdv": 0.0001, "advection_diffusion": 0.001}


def _default_terms():
    base = ["u"] + ["u_" + "x" * k for k in range(1, 5)]
    coords = ["t", "x"]
    terms = (coords + base) + [" ".join(pair) for pair in itertools.combinations_with_replacement(coords + base, 2)]
    return [term for term in terms if not _coord_only(term)]


def build_dictionary(u, x, t, terms=None, deriv="spectral"):
    if terms is None: terms = _default_terms()
    columns, labels = [np.ones_like(u)], ["b"]
    for term in terms:
        column = np.ones(u.shape)
        for token in term.split(): column = column * _factor(u, x, t, token, deriv)
        columns.append(column)
        labels.append("·".join(term.split()))
    theta = np.column_stack([column.ravel() for column in columns])
    return theta, labels


def discover(data, terms=None, threshold=None, ridge=None, trim=0, deriv=None, t_deriv="fd", max_iter=20, stride_x=None, stride_t=None, pde_name=""):
    if trim < 0 or 2 * trim >= data["u"].shape[1]:
        raise ValueError("trim must leave at least one time slice")
    if stride_x is None or stride_t is None:
        stride_x, stride_t = _DEFAULT_STRIDE.get(pde_name, (1, 1))
    stride_x, stride_t = max(1, int(stride_x)), max(1, int(stride_t))
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD.get(pde_name, 0.05)
    if deriv is None:
        deriv = _DEFAULT_DERIV.get(pde_name, "spectral")
    if ridge is None:
        ridge = _DEFAULT_RIDGE.get(pde_name, 1e-8)

    u, x, t, u_t = data["u"], data["x"], data["t"], data["u_t"]
    if stride_x != 1 or stride_t != 1:
        ix, it = np.arange(0, u.shape[0], stride_x), np.arange(0, u.shape[1], stride_t)
        x, t, u = x[ix], t[it], u[np.ix_(ix, it)]
        if u_t is not None:
            u_t = u_t[np.ix_(ix, it)]
    if u_t is None:
        u_t = temporal_derivative(u, t, t_deriv)
    if u_t.shape != u.shape:
        raise ValueError("u_t must have the same shape as u")
    theta, labels = build_dictionary(u, x, t, terms=terms, deriv=deriv)
    print(f"Dictionary ({len(labels)} terms): {labels}")
    print(f"Uniform-grid subsampling: stride_x={stride_x}, stride_t={stride_t}")
    n_time = u.shape[1]
    target = u_t[:, trim : n_time - trim].ravel()
    theta = theta.reshape(u.shape[0], u.shape[1], -1)[:, trim : n_time - trim].reshape(-1, theta.shape[1])
    optimizer = ps.STLSQ(threshold=threshold, alpha=ridge, max_iter=max_iter, normalize_columns=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficients = optimizer.fit(theta, target).coef_.ravel()
    return {label: float(value) for label, value in zip(labels, coefficients) if abs(value) > 1e-12}


def main():
    parser = argparse.ArgumentParser(description="Discover a 1D PDE with pdefind STLSQ on a custom dictionary")
    parser.add_argument("--data", required=True, help="Input .npz, .npy, CSV, or whitespace text file")
    parser.add_argument("--output", default=None, help="Output PDE file (default: algorithms/discovered_pde_<data-name>_pdefind.txt)")
    parser.add_argument("--dx", type=float, default=None, help="Spatial spacing for files without x")
    parser.add_argument("--dt", type=float, default=None, help="Temporal spacing for files without t")
    parser.add_argument("--terms", default=None, help="Comma-separated custom dictionary, each term space-separated factors in {u, u_x..., x, t} (e.g. 'u, u_x, u_x u_x, x u_xx, t u_x')")
    parser.add_argument("--threshold", type=float, default=None, help="Sparsity threshold on raw-scale coefficients (default: per-PDE, 0.05 unless overridden)")
    parser.add_argument("--ridge", type=float, default=None, help="L2 regularization strength for STLSQ (default: per-PDE, 1e-8 unless overridden)")
    parser.add_argument("--max_iter", type=int, default=20, help="Max STLSQ threshold-refit iterations")
    parser.add_argument("--trim", type=int, default=0, help="Number of initial/final time slices to exclude")
    parser.add_argument("--stride-x", type=int, default=None, help="Keep every Nth spatial point on the uniform grid (default: per-PDE, 1 = full grid)")
    parser.add_argument("--stride-t", type=int, default=None, help="Keep every Nth time point on the uniform grid (default: per-PDE, 1 = full grid)")
    parser.add_argument("--deriv", type=str, choices=("spectral", "fd", "spline"), default=None, help="Spatial derivative estimator (default: per-PDE, spectral unless overridden)")
    parser.add_argument("--t-deriv", type=str, choices=("fd", "spectral", "spline"), default="fd", help="Temporal derivative estimator for u_t (default: fd)")
    args = parser.parse_args()
    data = load_data(args.data, dx=args.dx, dt=args.dt)
    pde_name = pde_name_from_npz(args.data)
    print(f"Loaded data from {args.data}" + (f"  (pde={pde_name!r})" if pde_name else ""))
    terms = [term.strip() for term in args.terms.split(",") if term.strip()] if args.terms else None
    coefficients = discover(data, terms=terms, threshold=args.threshold, ridge=args.ridge, trim=args.trim, deriv=args.deriv, t_deriv=args.t_deriv, max_iter=args.max_iter, stride_x=args.stride_x, stride_t=args.stride_t, pde_name=pde_name)
    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.data))[0]
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"discovered_pde_{stem}_pdefind.txt")
    save_pde(output, coefficients)
    print(f"Discovered PDE written to {output}")
    for term, coefficient in coefficients.items(): print(f"  {term}: {coefficient:.6g}")


if __name__ == "__main__":
    main()
