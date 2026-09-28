import argparse, io, itertools, os, sys
from contextlib import redirect_stdout

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path: sys.path.remove(_SCRIPT_DIR)
import numpy as np
import torch
from torch.autograd import grad
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.model.func_approx import NN
from deepymod.model.constraint import LeastSquares
from deepymod.model.deepmod import Library
from deepymod.model.sparse_estimators import Threshold
from deepymod.training.sparsity_scheduler import Periodic, TrainTestPeriodic
from deepymod.training.training import train
sys.path.insert(0, _SCRIPT_DIR)  # re-add last so our own utils.py wins over any other "utils" already on the path
from utils import load_data, save_pde, pde_name_from_npz


class FullLibrary1D(Library):
    # mirrors pdefind.py's dictionary: u, u_x, ..., u_xxxx and every pairwise product
    def __init__(self, diff_order=4, exclude_coords=False):
        super().__init__()
        self.diff_order = diff_order
        base = ["u"] + ["u_" + "x" * k for k in range(1, diff_order + 1)]
        coords = ["t", "x"]
        tokens = base if exclude_coords else (coords + base)
        terms = tokens + [" ".join(pair) for pair in itertools.combinations_with_replacement(tokens, 2)]
        self.terms = [term for term in terms if any(token not in ("t", "x") for token in term.split())]
        self.labels = ["b"] + ["·".join(term.split()) for term in self.terms]

    def _spatial_derivatives(self, prediction, data):
        current = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]
        derivs = {"u_x": current}
        for order in range(2, self.diff_order + 1):
            current = grad(current, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]
            derivs["u_" + "x" * order] = current
        return derivs

    def library(self, input):
        prediction, data = input
        u_t = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 0:1]
        features = {"t": data[:, 0:1], "x": data[:, 1:2], "u": prediction}
        features.update(self._spatial_derivatives(prediction, data))
        columns = [torch.ones_like(prediction)]
        for term in self.terms:
            tokens = term.split()
            column = features[tokens[0]]
            for token in tokens[1:]:
                column = column * features[token]
            columns.append(column)
        theta = torch.cat(columns, dim=1)
        return [u_t], [theta]


# per-PDE tuning, falls back to the generic defaults in discover() below
_SCHEDULER_PATIENCE = {"burgers": 200, "kdv":     10, "ks":      200}

_DEFAULT_THRESHOLD = {"burgers": 0.02, "kdv": 0.04, "ks": 0.5, "advection_diffusion": 0.02}

_DEFAULT_HIDDEN = {"kdv": (40, 40, 40, 40)}

_DEFAULT_STRIDE = {"burgers": (1, 6), "kdv": (7, 25), "ks": (4, 100), "advection_diffusion": (2, 5)}

_SEED = 42


def _subsample_uniform(u, x, t, stride_x=1, stride_t=1):
    stride_x = max(1, int(stride_x))
    stride_t = max(1, int(stride_t))
    return u[::stride_x, ::stride_t], x[::stride_x], t[::stride_t]


def _make_load_function(u, x, t):
    # Dataset calls this itself with its own kwargs, so u/x/t have to come in via closure
    def load(**kwargs):
        Xt, Yx = np.meshgrid(t, x)
        coords = np.column_stack([Xt.ravel(), Yx.ravel()]).astype(np.float32)
        values = u.ravel()[:, None].astype(np.float32)
        return torch.tensor(coords), torch.tensor(values)
    return load


def _test_mse(model, test_loader):
    with torch.no_grad():
        total, count = 0.0, 0
        for data_test, target_test in test_loader:
            prediction = model.func_approx(data_test)[0]
            total += torch.mean((prediction - target_test) ** 2).item() * len(data_test)
            count += len(data_test)
    return total / max(count, 1)


def discover(data, diff_order=4, threshold=None, max_iterations=100000, hidden=None, scheduler_patience=None, pde_name="", scheduler="periodic", exclude_coords=False, stride_x=None, stride_t=None):
    if stride_x is None or stride_t is None:
        stride_x, stride_t = _DEFAULT_STRIDE.get(pde_name, (1, 1))
    u, x, t = data["u"], data["x"], data["t"]
    if stride_x > 1 or stride_t > 1:
        u, x, t = _subsample_uniform(u, x, t, stride_x, stride_t)
    x_mean = np.mean(x)
    x = x - x_mean  # centering helps the Lasso conditioning

    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    library_obj = FullLibrary1D(diff_order=diff_order, exclude_coords=exclude_coords)
    labels = library_obj.labels

    if scheduler_patience is None:
        scheduler_patience = _SCHEDULER_PATIENCE.get(pde_name, 200)

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD.get(pde_name, 0.1)

    if hidden is None:
        hidden = _DEFAULT_HIDDEN.get(pde_name, (30, 30, 30, 30))

    print(f"Data: u={u.shape}, x={x.shape}, t={t.shape}")
    print(f"Library: pdefind ({len(labels)} terms) -> {labels}")
    print(f"Scheduler patience: {scheduler_patience}")
    print(f"Uniform-grid subsampling: stride_x={stride_x}, stride_t={stride_t}")

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepymod_runs")
    os.makedirs(run_dir, exist_ok=True)

    load_function = _make_load_function(u, x, t)

    with redirect_stdout(io.StringIO()):
        dataset = Dataset(load_function=load_function, preprocess_kwargs={"random_state": _SEED, "normalize_coords": False, "normalize_data": False}, shuffle=True, device=None)
        train_loader, test_loader = get_train_test_loader(dataset, train_test_split=0.8)

    net = NN(2, list(hidden), 1)

    print(f"Building model (network=MLP, library=pdefind ({len(labels)} terms), threshold={threshold})")
    sparse = Threshold(threshold=threshold)
    model = DeepMoD(net, library_obj, sparse, LeastSquares())

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

    if scheduler == "periodic":
        sparsity_scheduler = Periodic(periodicity=50, initial_iteration=1000)
    else:
        sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=scheduler_patience, delta=1e-5)

    log_dir = os.path.join(run_dir, f"run_{os.getpid()}_{_SEED}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training up to {max_iterations} iterations...")
    with redirect_stdout(io.StringIO()):
        train(model, train_loader, test_loader, optimizer, sparsity_scheduler, log_dir=log_dir, max_iterations=max_iterations, patience=200, delta=1e-3)
    print("Training done.")

    mse = _test_mse(model, test_loader)
    print(f"Validation MSE: {mse:.3e}")

    coeffs = model.constraint_coeffs(sparse=True, scaled=False)[0].detach().cpu().numpy().ravel()
    return {label: float(v) for label, v in zip(labels, coeffs) if abs(v) > 1e-12}


def main():
    parser = argparse.ArgumentParser(description="Discover a 1D PDE with DeepMoD")
    parser.add_argument("--data", required=True, help="Input .npz from datagen/")
    parser.add_argument("--output", default=None, help="Output PDE file (default: algorithms/discovered_pde_<data-name>_deepmod.txt)")
    parser.add_argument("--diff-order", type=int, default=4, help="Library derivative order (default: 4, matching pdefind)")
    parser.add_argument("--max-iterations", type=int, default=100000, help="Max training iterations (default: 100000)")
    parser.add_argument("--threshold", type=float, default=None, help="Sparsity threshold for the Threshold estimator (default: per-PDE default, 0.1 unless overridden)")
    parser.add_argument("--scheduler", choices=("periodic", "train-test"), default="periodic", help="Stock DeepMoD sparsity scheduler (default: periodic, applies mask every 50 iters after iter 1000; train-test only prunes after val loss plateaus)")
    parser.add_argument("--exclude-coords", action="store_true", help="Drop t/x coordinate columns from the library (keeps only u and its derivatives and their products)")
    parser.add_argument("--hidden", default=None, help="Comma-separated hidden layer sizes (default: per-PDE default, see _DEFAULT_HIDDEN, falling back to 30,30,30,30)")
    parser.add_argument("--scheduler-patience", type=int, default=None, help="Sparsity scheduler patience (default: auto from npz pde name, 200 otherwise)")
    parser.add_argument("--stride-x", type=int, default=None, help="Uniform subsampling stride along x (keep every stride-th column; default: per-PDE, 1 = all). When >1, the whole subsampled uniform grid is used")
    parser.add_argument("--stride-t", type=int, default=None, help="Uniform subsampling stride along t (keep every stride-th row; default: per-PDE, 1 = all). When >1, the whole subsampled uniform grid is used")
    args = parser.parse_args()

    data = load_data(args.data)
    pde_name = pde_name_from_npz(args.data)
    print(f"Loaded data from {args.data}" + (f"  (pde={pde_name!r})" if pde_name else ""))
    hidden = [int(size) for size in args.hidden.split(",") if size.strip()] if args.hidden else None

    coefficients = discover(data, diff_order=args.diff_order, threshold=args.threshold, max_iterations=args.max_iterations, hidden=hidden, scheduler_patience=args.scheduler_patience, pde_name=pde_name, scheduler=args.scheduler, exclude_coords=args.exclude_coords, stride_x=args.stride_x, stride_t=args.stride_t)

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.data))[0]
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"discovered_pde_{stem}_deepmod.txt")
    save_pde(output, coefficients)
    print(f"Discovered PDE written to {output}")
    for term, coefficient in coefficients.items():
        print(f"  {term}: {coefficient:.6g}")


if __name__ == "__main__":
    main()
