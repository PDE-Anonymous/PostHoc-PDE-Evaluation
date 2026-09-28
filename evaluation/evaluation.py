import argparse
import glob
import importlib.util
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import PDE_ROOT, parse_pde_file, pde_from_dir, pde_label, make_integrator, simulate, load_reference_config, simulate_from_config

parser = argparse.ArgumentParser()
parser.add_argument("--file",      required=True, nargs="+", help="Candidate PDE file(s) (any filename), or a single system name (e.g. ks) to evaluate every *.txt file found under pde_files/<system>/ besides --fileGT")
parser.add_argument("--fileGT",    required=True, help="Ground-truth PDE file (any filename); its containing directory identifies the reference system to simulate")
parser.add_argument("--config",    required=True, help="Reference-system config JSON giving the domain and initial condition to simulate on (see configs/*.json) -- the PDE itself always comes from --fileGT")
parser.add_argument("--metric",    required=True, nargs="+")
parser.add_argument("--dt_factor", type=float, default=1.0, help="dt is divided by this factor (finer timestep for larger values, e.g. 100 = 100 inner substeps per output frame)")
parser.add_argument("--T_max",     type=float, default=[None], nargs="+", help="One or more simulation end times; repeats the full run for each")
parser.add_argument("--k_max",     type=float, default=None, help="Max spatial wavenumber for fMSE (default: 1%%-of-peak-energy auto cutoff)")
parser.add_argument("--theta_size", type=int, default=31, help="Size of the full candidate dictionary Theta, required by Sterms (default: 31)")
parser.add_argument("--baseline_file", default=None, help="PDE file used as the baseline model for Score (default: null model u_t=0)")
parser.add_argument("--ic",        type=int, default=None, help="OOD initial condition choice for the IC_* / rollout_IC metrics: 1 (default/training IC), 2 (generic OOD sines, or a system-specific override)")
parser.add_argument("--n_refine",  type=int, default=None, help="Number of successive halvings/doublings for Conv_t/Conv_x (default: 6)")
# sparsity.py / tradeoff.py hyperparameters -- None means "use that metric's own default"
parser.add_argument("--w1",             type=float, default=None, help="ExpTreeV2 node-count weight (default: 1.0)")
parser.add_argument("--w2",             type=float, default=None, help="ExpTreeV2 depth weight (default: 1.0)")
parser.add_argument("--w3",             type=float, default=None, help="ExpTreeV2 operator-weight weight (default: 1.0)")
parser.add_argument("--c0",             type=float, default=None, help="Reward1 sparsity/accuracy trade-off (default: 0.2)")
parser.add_argument("--xi1",            type=float, default=None, help="Reward2 sparsity weight (default: 0.01)")
parser.add_argument("--xi2",            type=float, default=None, help="Reward2 tree-depth weight (default: 0.0001)")
parser.add_argument("--rank",           type=float, default=None, help="MDL_Fey candidate rank N(alpha_hat) (default: 1, no penalty)")
parser.add_argument("--mdl_lambda",     type=float, default=None, help="MDL_Fey trade-off lambda (default: sqrt(n data points))")
parser.add_argument("--eps_d",          type=float, default=None, help="MDL_Fey normalization constant (default: 1e-15)")
parser.add_argument("--c_max",          type=float, default=None, help="MDL_Sym max constant magnitude (default: 10.0)")
parser.add_argument("--eps_enc",        type=float, default=None, help="MDL_Sym constant encoding precision (default: 1e-6)")
parser.add_argument("--mdl_sym_lambda", type=float, default=None, help="MDL_Sym trade-off lambda (default: 0.1)")
args = parser.parse_args()

# build metric registry from all .py files in this folder
HERE = os.path.dirname(__file__)
registry = {}  # name -> (fn, needs_simulation)
by_file  = {}  # module stem (e.g. "coef", "errors") -> [metric names]
for fname in sorted(os.listdir(HERE)):
    if not fname.endswith(".py") or fname in ("evaluation.py", "utils.py"):
        continue
    stem = fname[:-3]
    path = os.path.join(HERE, fname)
    spec = importlib.util.spec_from_file_location(stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "METRICS"):
        continue
    needs_sim = getattr(mod, "NEEDS_SIMULATION", True)
    for name, fn in mod.METRICS.items():
        registry[name] = (fn, needs_sim)
        by_file.setdefault(stem, []).append(name)

metric_source = {m: stem for stem, ms in by_file.items() for m in ms}

grouped = None  # (stem, metric) pairs; set only when the user selected a whole file / "all"
if args.metric == ["all"]:
    grouped = [(stem, m) for stem in sorted(by_file) for m in sorted(by_file[stem])]
    args.metric = [m for _, m in grouped]
else:
    expanded = []
    for m in args.metric:
        expanded.extend(sorted(by_file[m]) if m in by_file else [m])
    args.metric = expanded

unknown = [m for m in args.metric if m not in registry]
if unknown:
    print(f"Unknown metric(s): {unknown}. Available:\n  " + "\n  ".join(sorted(registry)) + "\n  Or a file name to select all its metrics:\n  " + "\n  ".join(sorted(by_file)))
    sys.exit(1)

needs_simulation = any(registry[m][1] for m in args.metric)

# resolve files
ref_path = args.fileGT
assert os.path.isabs(ref_path) or os.path.exists(ref_path), f"Ground-truth file not found: {ref_path}"
pde_name = pde_from_dir(ref_path)
pde_dir  = os.path.join(PDE_ROOT, pde_name)

if len(args.file) == 1 and not (os.path.isabs(args.file[0]) or os.path.exists(args.file[0])) and os.path.isdir(os.path.join(PDE_ROOT, args.file[0])):
    batch_dir     = os.path.join(PDE_ROOT, args.file[0])
    compare_files = sorted(f for f in glob.glob(os.path.join(batch_dir, "*.txt")) if os.path.abspath(f) != os.path.abspath(ref_path))
    assert compare_files, f"No candidate files found in {batch_dir} (besides --fileGT)"
else:

    # Resolves a bare filename to its full path, falling back to the GT's directory
    def _resolve(path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        candidate = os.path.join(pde_dir, path)
        assert os.path.exists(candidate), f"Candidate file '{path}' not found (looked for it as given, and under {pde_dir})"
        return candidate
    compare_files = [_resolve(p) for p in args.file]

config = load_reference_config(args.config)

for T_max in args.T_max:
    if len(args.T_max) > 1:
        print(f"\n{'=' * 20} T_max={T_max} {'=' * 20}")

    # domain
    x, t, fields = simulate_from_config(config, T=T_max)
    u0       = fields["u"][:, 0].copy()
    N        = len(x)
    nt       = len(t)
    nt_inner = max(1, round(args.dt_factor))
    dt       = float(t[1] - t[0]) / nt_inner
    L        = float(x[-1] - x[0] + (x[1] - x[0]))
    kk       = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    # reference simulation
    print(f"Simulating reference: {ref_path}")
    b_r, lin_r, quad_r, xlin_r, tlin_r, coord_r = parse_pde_file(ref_path)
    U_ref = simulate(make_integrator(b_r, lin_r, quad_r, kk, dt, x_coord=x, x_lin=xlin_r, t_lin=tlin_r, coord=coord_r), u0, nt, nt_inner, dt=dt, t0=0.0)
    mean_abs_ref = float(np.mean(np.abs(U_ref)))

    baseline = None
    if args.baseline_file:
        bl_path = args.baseline_file
        if not (os.path.isabs(bl_path) or os.path.exists(bl_path)):
            bl_path = os.path.join(pde_dir, bl_path)
            assert os.path.exists(bl_path), f"Baseline file '{args.baseline_file}' not found (looked for it as given, and under {pde_dir})"
        print(f"Simulating baseline (Score): {bl_path}")
        b_bl, lin_bl, quad_bl, xlin_bl, tlin_bl, coord_bl = parse_pde_file(bl_path)
        U_bl = simulate(make_integrator(b_bl, lin_bl, quad_bl, kk, dt, x_coord=x, x_lin=xlin_bl, t_lin=tlin_bl, coord=coord_bl), u0, nt, nt_inner, dt=dt, t0=0.0)
        baseline = {"b_base": b_bl, "lin_base": lin_bl, "quad_base": quad_bl, "xlin_base": xlin_bl, "tlin_base": tlin_bl, "coord_base": coord_bl, "U_base": U_bl}

    # evaluate
    print(f"\nMetrics: {', '.join(args.metric)}\n" + "-" * 40)
    for fpath in compare_files:
        b, lin, quad, xlin, tlin, coord = parse_pde_file(fpath)
        print(f"\nEvaluating: {os.path.basename(fpath)}")
        print(f"  PDE: u_t = {pde_label(b, lin, quad, x_lin=xlin, t_lin=tlin, coord=coord)}")

        U_disc = simulate(make_integrator(b, lin, quad, kk, dt, x_coord=x, x_lin=xlin, t_lin=tlin, coord=coord), u0, nt, nt_inner, dt=dt, t0=0.0) if needs_simulation else None

        data = {"U_ref": U_ref, "U_disc": U_disc, "x": x, "t": t, "kk": kk, "dt": dt, "mean_abs_ref": mean_abs_ref, "k_max": args.k_max, "theta_size": args.theta_size, "b_ref":  b_r,  "lin_ref":  lin_r,  "quad_ref":  quad_r, "xlin_ref": xlin_r, "tlin_ref": tlin_r, "coord_ref": coord_r, "b_disc": b,    "lin_disc": lin,    "quad_disc": quad, "xlin_disc": xlin, "tlin_disc": tlin, "coord_disc": coord, "pde_name": pde_name}
        if baseline is not None:
            data.update(baseline)
        for key in ("w1", "w2", "w3", "c0", "xi1", "xi2", "rank", "mdl_lambda", "eps_d", "c_max", "eps_enc", "mdl_sym_lambda", "ic", "n_refine"):
            val = getattr(args, key)
            if val is not None:
                data[key] = val

        prev_stem = None
        for m in args.metric:
            stem = metric_source.get(m)
            if grouped is not None and stem != prev_stem:
                prev_stem = stem
                print(f"\n  [{stem}.py]")
            compute, _ = registry[m]
            try:
                result = compute(data)
            except ValueError:
                # required argument (e.g. Sterms's theta_size, Score's baseline_file) missing
                print(f"  {m}: argument missing")
                continue
            if isinstance(result, dict):
                print(f"  {m}:")
                for k, v in result.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {m}: {result}")
