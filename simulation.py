import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from utils import (PDE_ROOT, parse_pde_file, pde_from_dir, pde_label, make_integrator, simulate, load_reference_config, simulate_from_config)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))
from generalization import ic_func, conv_t_series, conv_x_series

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, nargs='+', help="Candidate PDE file(s) (any filename), or a single system name (e.g. ks) to plot every *.txt file found under pde_files/<system>/ besides --fileGT")
parser.add_argument("--fileGT", required=True, help="Ground-truth PDE file (any filename); its containing directory identifies the reference system to simulate")
parser.add_argument("--config", required=True, help="Reference-system config JSON giving the domain and initial condition to simulate on (see configs/*.json) -- the PDE itself always comes from --fileGT")
parser.add_argument("--dt_factor", type=float, default=1.0, help="dt is divided by this factor (finer timestep for larger values, e.g. 100 = 100 inner substeps per output frame)")
parser.add_argument("--vmax_diff", type=float, default=None)
parser.add_argument("--T_max", type=float, default=None, help="Override the config's default simulation end time")
parser.add_argument("--rollout",   action="store_true", help="Also save a rollout-error plot in rollout_error/<pde>/")
parser.add_argument("--rollout_on_sim", action="store_true", help="Overlay rollout error curve on the simulation plots")
parser.add_argument("--plot_rollout_error", action="store_true", help="Overlay nMAE(t) curve on the error heatmap")
parser.add_argument("--fourier",   action="store_true", help="Also save a Fourier-space error heatmap in fourier_error/<pde>/")
parser.add_argument("--k_max",        type=float, default=None, help="Max spatial wavenumber shown in the Fourier heatmap")
parser.add_argument("--fourier_log",  action="store_true", help="Log colorscale for the Fourier heatmap")
parser.add_argument("--fourier_norm", action="store_true", help="Normalize each wavenumber row independently in the Fourier heatmap")
parser.add_argument("--rollout_t", type=float, default=None, help="Print rollout error at this specific time (e.g. 300)")
parser.add_argument("--cmap_err", type=str, default="Reds", help="Colormap for error heatmaps (default: Reds)")
parser.add_argument("--clean", action="store_true", help="Remove all axis ticks, labels, titles, and colorbars from plots")
parser.add_argument("--ic", type=int, default=1, help="OOD initial condition: 1 (default/training IC), 2 (generic OOD sines, or a system-specific override), 3 (generic OOD cosines)")
parser.add_argument("--eq_title", action="store_true", help="Show the PDE name and its discovered equation as the plot title")
parser.add_argument("--conv", action="store_true", help="Also save Conv_t/Conv_x numerical-convergence plots (candidates only)")
parser.add_argument("--n_refine", type=int, default=6, help="Number of successive halvings/doublings for --conv (default: 6)")
args = parser.parse_args()

def candidate_label(path):
    suffix = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    return "GT" if suffix.lower() == "true" else suffix


def candidate_name(path):
    return os.path.splitext(os.path.basename(path))[0]

ref_path = args.fileGT
assert os.path.isabs(ref_path) or os.path.exists(ref_path), f"Ground-truth file not found: {ref_path}"
pde_name = pde_from_dir(ref_path)
pde_dir  = os.path.join(PDE_ROOT, pde_name)

if len(args.file) == 1 and not (os.path.isabs(args.file[0]) or os.path.exists(args.file[0])) and os.path.isdir(os.path.join(PDE_ROOT, args.file[0])):
    batch_dir     = os.path.join(PDE_ROOT, args.file[0])
    compare_files = sorted(f for f in glob.glob(os.path.join(batch_dir, "*.txt")) if os.path.abspath(f) != os.path.abspath(ref_path))
    assert compare_files, f"No candidate files found in {batch_dir} (besides --fileGT)"
    prefix = args.file[0]
else:
    def _resolve(path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        candidate = os.path.join(pde_dir, path)
        assert os.path.exists(candidate), f"Candidate file '{path}' not found (looked for it as given, and under {pde_dir})"
        return candidate

    compare_files = [_resolve(p) for p in args.file]
    prefix = candidate_name(compare_files[0])

# tag all outputs (sims/errors/rollout/fourier/convergence) with the IC used
if args.ic != 1:
    prefix = f"{prefix}_ic{args.ic}"

# domain and IC
print(f"Loading {pde_name} domain / IC...")
x, t, fields = simulate_from_config(load_reference_config(args.config), T=args.T_max)
u0 = fields["u"][:, 0].copy()
N = len(x)
nt = len(t)
dt_orig = float(t[1] - t[0])
L_dom = float(x[-1] - x[0] + (x[1] - x[0]))
kk = np.fft.rfftfreq(N, d=L_dom / N) * 2 * np.pi

if args.ic != 1:
    u0 = ic_func(pde_name, args.ic)(x, L_dom)
    print(f"Using OOD initial condition {args.ic}")

nt_inner = max(1, round(args.dt_factor))
dt = dt_orig / nt_inner

# reference
print(f"Simulating reference: {ref_path}")
b_r, lin_r, quad_r, x_lin_r, t_lin_r, coord_r = parse_pde_file(ref_path)
ref_lbl = pde_label(b_r, lin_r, quad_r, x_lin=x_lin_r, t_lin=t_lin_r, coord=coord_r)
U_ref = simulate(make_integrator(b_r, lin_r, quad_r, kk, dt, x_coord=x, x_lin=x_lin_r, t_lin=t_lin_r, coord=coord_r), u0, nt, nt_inner, dt=dt)
vr = float(np.nanmax(np.abs(U_ref)))
mean_abs_ref = float(np.nanmean(np.abs(U_ref)))
if not (np.isfinite(mean_abs_ref) and mean_abs_ref > 0):
    mean_abs_ref = 1.0

# simulate each file
results = []
for fpath in compare_files:
    print(f"Simulating: {fpath}")
    b, lin, quad, x_lin, t_lin, coord = parse_pde_file(fpath)
    U_disc = simulate(make_integrator(b, lin, quad, kk, dt, x_coord=x, x_lin=x_lin, t_lin=t_lin, coord=coord), u0, nt, nt_inner, dt=dt)
    diff = U_disc - U_ref
    dx = float(x[1] - x[0])
    e_t = np.sum(diff**2, axis=0) * dx  # Riemann-sum approx. of ||.||_{L^2(Omega)}^2
    eps_rollout_t = np.cumsum(e_t) / np.arange(1, len(e_t) + 1)
    results.append((U_disc, diff, pde_label(b, lin, quad, x_lin=x_lin, t_lin=t_lin, coord=coord), eps_rollout_t))

if args.vmax_diff is not None:
    vd_global = args.vmax_diff
else:
    dmax = [float(np.max(np.abs(r[1])[np.isfinite(r[1])])) for r in results if np.any(np.isfinite(r[1]))]
    vd_global = (max(dmax) / mean_abs_ref) if dmax else 1.0
_scale = "explicit" if args.vmax_diff is not None else "max"
print(f"Unified error scale: 0 – {vd_global:.4e} ({_scale})")
rollout_fin = [float(np.max(r[3][np.isfinite(r[3])])) for r in results if np.any(np.isfinite(r[3]))]
rollout_global_max = max(rollout_fin) if rollout_fin else 1.0


def rollout_disp(eps):
    return np.minimum(eps, rollout_global_max)


def err_im_args(arr):
    return np.clip(arr, 0.0, vd_global), dict(vmin=0.0, vmax=vd_global)

if args.fourier:
    # k_cut only depends on the (shared) reference spectrum / --k_max, so compute once
    F_ref = np.fft.rfft(U_ref, axis=0)
    if args.k_max is not None:
        k_cut = int(np.searchsorted(kk, args.k_max)) + 1
    else:
        mean_energy = np.abs(F_ref).mean(axis=1)
        k_cut = int(np.where(mean_energy > mean_energy.max() * 1e-2)[0].max()) + 1
    k_cut   = min(k_cut, len(kk))
    kk_plot = kk[1:k_cut]
    err_plots = [np.abs(np.fft.rfft(r[0], axis=0) - F_ref)[1:k_cut] for r in results]
    print(f"Fourier metric/heatmap restricted to k in [{kk_plot[0]:.3g}, {kk_plot[-1]:.3g}] ({len(kk_plot)} modes)")
    if not args.fourier_norm:
        emax = [float(np.max(e[np.isfinite(e)])) for e in err_plots if np.any(np.isfinite(e))]
        vmax_fourier_global = max(emax) if emax else 1.0
        print(f"Unified Fourier-error scale: 0 – {vmax_fourier_global:.4e}")

if args.rollout_t is not None:
    idx = int(np.argmin(np.abs(t - args.rollout_t)))
    print(f"\nRollout error at t={t[idx]:.2f}:")
    for fpath, (_, _, _, eps_rollout_t) in zip(compare_files, results):
        print(f"  {os.path.basename(fpath)}: {eps_rollout_t[idx]:.4e}")

# plots
col_w = 5.0
col_h = 4.0


def _latex_var(v):
    # u_xx -> u_{xx}, u_xxxx -> u_{xxxx}, u_x -> u_{x}, u -> u
    return re.sub(r'_(x+)', r'_{\1}', v.strip())


def _latex_term(name):
    # "name_i·name_j" renders as juxtaposition (u u_x), no explicit multiplication dot
    return "".join(_latex_var(p) for p in name.split("·"))


def eq_title_str(tex, lbl):
    # renders as proper LaTeX (-1.0uu_x-1.0u_{xx}-1.0u_{xxxx}), wrapping two terms per line past the first 3
    terms = [t.strip() for t in lbl.split(",")]

    parts = []
    for i, term in enumerate(terms):
        name, val_str = term.split("=")
        name, val = name.strip(), float(val_str)
        sign = "-" if val < 0 else ("+" if i > 0 else "")
        parts.append(f"{sign}{abs(val):g}{_latex_term(name)}")

    lines = ["".join(parts[i:i + 3]) for i in range(0, len(parts), 3)]
    return f"{tex}\n$u_t=" + lines[0] + "$" + "".join(f"\n${ln}$" for ln in lines[1:])


def eq_title_nlines(lbl):
    # total title lines: 1 for the PDE-name line + ceil(n_terms/3) equation lines
    n_terms = len(lbl.split(","))
    return 1 + -(-n_terms // 3)


def eq_title_extra_h(lbl):
    # extra height per title line beyond the first, so a multi-line title doesn't shrink the plot area
    return 0.35 * (eq_title_nlines(lbl) - 1) if args.eq_title else 0.0


def _label_or_title(ax, tex, lbl):
    if args.eq_title:
        ax.set_title(eq_title_str(tex, lbl), fontsize=13)
    else:
        ax.text(0.02, 0.98, tex, transform=ax.transAxes, fontsize=16, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


def _colorbar(fig, im, ax, fmt=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(fmt)
    return cb


# GT + all candidates, side by side in one figure, sharing the same color scale (vr)
n_panels = 1 + len(compare_files)
sbs_extra_h = max([eq_title_extra_h(ref_lbl)] + [eq_title_extra_h(r[2]) for r in results]) if args.eq_title else 0.0
fig_sbs, axes_sbs = plt.subplots(1, n_panels, figsize=(col_w * n_panels, col_h + sbs_extra_h), layout="constrained", sharey=True)
im_sbs = axes_sbs[0].pcolormesh(t, x, U_ref, shading="auto", cmap="RdBu_r", vmin=-vr, vmax=vr)
_label_or_title(axes_sbs[0], r"$PDE_{\mathcal{GT}}$", ref_lbl)
axes_sbs[0].set_ylabel("space", fontsize=13)
axes_sbs[0].set_xlabel("time", fontsize=13)
for ax, fpath, (U_disc, _, lbl, _) in zip(axes_sbs[1:], compare_files, results):
    ax.pcolormesh(t, x, U_disc, shading="auto", cmap="RdBu_r", vmin=-vr, vmax=vr)
    suffix = candidate_label(fpath)
    sbs_tex = rf"$PDE_{{\mathcal{{{suffix}}}}}$"
    _label_or_title(ax, sbs_tex, lbl)
    ax.set_xlabel("time", fontsize=13)
fig_sbs.colorbar(im_sbs, ax=axes_sbs, fraction=0.023, pad=0.02)
out_sbs = os.path.join("simulations", pde_name, f"side_by_side_{prefix}.png")
os.makedirs(os.path.dirname(out_sbs), exist_ok=True)
fig_sbs.savefig(out_sbs, dpi=150)
plt.close(fig_sbs)
print(f"Side-by-side plot saved → {out_sbs}")

for i, (fpath, (U_disc, diff, lbl, eps_rollout_t)) in enumerate(zip(compare_files, results)):
    stem = candidate_name(fpath)
    if args.ic != 1:
        stem = f"{stem}_ic{args.ic}"
    suffix = candidate_label(fpath)
    pde_tex = rf"$PDE_{{\mathcal{{{suffix}}}}}$"
    extra_h = eq_title_extra_h(lbl)
    # error heatmap colorbar: plain decimals under 10, scientific notation above
    err_sfmt = ticker.ScalarFormatter(useMathText=True)
    err_sfmt.set_powerlimits((0, 1))
    err_data, err_kw = err_im_args(np.abs(diff) / mean_abs_ref)
    mae = float(np.mean(np.abs(diff))) / mean_abs_ref

    # solution
    fig_sol, ax_sol = plt.subplots(figsize=(col_w + (0.7 if args.rollout_on_sim else 0), col_h + extra_h), layout="constrained")
    if args.rollout_on_sim:
        ax_roll_sim = ax_sol.twinx()
        ax_roll_sim.plot(t, rollout_disp(eps_rollout_t), color="black", lw=2.5, linestyle="--", alpha=0.9)
        ax_roll_sim.set_ylabel(r"$\varepsilon_{\mathrm{rollout}}(t)$", fontsize=11, color="black")
        ax_roll_sim.tick_params(axis='y', labelsize=10, labelcolor="black", colors="black")
        ax_roll_sim.set_ylim(0., rollout_global_max)
    im_sol = ax_sol.pcolormesh(t, x, U_disc, shading="auto", cmap="RdBu_r", vmin=-vr, vmax=vr)
    ax_sol.set_aspect("auto")
    if args.clean:
        ax_sol.set_axis_off()
    else:
        ax_sol.set_ylabel("space", fontsize=13)
        ax_sol.set_xlabel("time", fontsize=13)
        ax_sol.tick_params(labelsize=11)
        _label_or_title(ax_sol, pde_tex, lbl)
        cb_ax = [ax_sol, ax_roll_sim] if args.rollout_on_sim else ax_sol
        _colorbar(fig_sol, im_sol, cb_ax)
    out_sol = os.path.join("simulations", pde_name, f"simulation_{stem}.png")
    os.makedirs(os.path.dirname(out_sol), exist_ok=True)
    fig_sol.savefig(out_sol, dpi=150)
    plt.close(fig_sol)
    print(f"Plot saved → {out_sol}")

    # error
    fig_err, ax_err = plt.subplots(figsize=(col_w + (1.5 if args.plot_rollout_error else 0), col_h + extra_h), layout="constrained")
    if args.plot_rollout_error:
        ax_nmae = ax_err.twinx()
        ax_nmae.plot(t, rollout_disp(eps_rollout_t), color="tab:orange", lw=2.5, linestyle="--", alpha=0.9)
        ax_nmae.set_ylabel(r"$\varepsilon_{\mathrm{rollout}}(t)$", fontsize=11, color="tab:orange")
        ax_nmae.tick_params(axis="y", labelsize=10, labelcolor="tab:orange", colors="tab:orange")
        ax_nmae.set_ylim(0., rollout_global_max)
    im_err = ax_err.pcolormesh(t, x, err_data, shading="auto", cmap=args.cmap_err, **err_kw)
    ax_err.set_aspect("auto")
    if args.clean:
        ax_err.set_axis_off()
        if args.plot_rollout_error:
            ax_nmae.set_axis_off()
    else:
        ax_err.set_ylabel("space", fontsize=13)
        ax_err.set_xlabel("time", fontsize=13)
        ax_err.tick_params(labelsize=11)
        ax_err.text(0.98, 0.97, f"nMAE={mae:.2e}", transform=ax_err.transAxes, fontsize=11, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        _label_or_title(ax_err, pde_tex, lbl)
        cb_ax = [ax_err, ax_nmae] if args.plot_rollout_error else ax_err
        _colorbar(fig_err, im_err, cb_ax, err_sfmt)
    out_err = os.path.join("errors", pde_name, f"error_{stem}.png")
    os.makedirs(os.path.dirname(out_err), exist_ok=True)
    fig_err.savefig(out_err, dpi=150)
    plt.close(fig_err)
    print(f"Plot saved → {out_err}")

    # rollout-error plot
    if args.rollout:
        fig_roll, ax_mesh = plt.subplots(figsize=(col_w + 0.7, col_h + extra_h), layout="constrained")
        ax_roll = ax_mesh.twinx()

        im_roll = ax_mesh.pcolormesh(t, x, err_data, shading="auto", cmap=args.cmap_err, **err_kw)
        ax_roll.plot(t, rollout_disp(eps_rollout_t), color="black", lw=1.5, linestyle="--", alpha=0.9)
        ax_roll.set_ylim(0, rollout_global_max)
        if args.clean:
            ax_mesh.set_axis_off()
            ax_roll.set_axis_off()
        else:
            ax_mesh.set_ylabel("space", fontsize=13)
            ax_mesh.set_xlabel("time", fontsize=13)
            ax_mesh.tick_params(labelsize=11)
            ax_mesh.text(0.98, 0.97, f"nMAE={mae:.2e}", transform=ax_mesh.transAxes, fontsize=11, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            _label_or_title(ax_mesh, pde_tex, lbl)
            ax_roll.set_ylabel(r"$\varepsilon_{\mathrm{rollout}}(t)$", fontsize=11, color="black")
            ax_roll.tick_params(axis='y', labelsize=10, labelcolor="black", colors="black")
            _colorbar(fig_roll, im_roll, [ax_mesh, ax_roll], err_sfmt)

        out_roll = os.path.join("rollout_error", pde_name, f"rollout_{stem}.png")
        os.makedirs(os.path.dirname(out_roll), exist_ok=True)
        fig_roll.savefig(out_roll, dpi=150)
        plt.close(fig_roll)
        print(f"Rollout plot saved → {out_roll}")

    # Fourier-space heatmap
    if args.fourier:
        # precomputed, already |F(û)(k,t)-F(u)(k,t)|, trimmed to [1:k_cut] (excludes k=0)
        err_plot_raw = err_plots[i]
        err_plot     = err_plot_raw

        if args.fourier_norm:
            row_max  = err_plot.max(axis=1, keepdims=True)
            err_plot = err_plot_raw / np.where(row_max > 0, row_max, 1)

        fig_f, ax_f = plt.subplots(figsize=(col_w, col_h + extra_h), layout="constrained")
        if args.fourier_log:
            vmax_f = float(err_plot.max()) if args.fourier_norm else vmax_fourier_global
            norm_f = LogNorm(vmin=max(vmax_f * 1e-4, 1e-10), vmax=vmax_f)
            im_f = ax_f.pcolormesh(t, kk_plot, err_plot + 1e-10, shading="auto", cmap=args.cmap_err, norm=norm_f)
        else:
            vmax_f = None if args.fourier_norm else vmax_fourier_global
            im_f = ax_f.pcolormesh(t, kk_plot, err_plot, shading="auto", cmap=args.cmap_err, vmin=0, vmax=vmax_f)

        # fMSE over the same [1, k_cut) band as the heatmap, not the full Nyquist range
        n_modes  = err_plot_raw.shape[0]
        fmse_t   = np.sum(err_plot_raw ** 2, axis=0) / n_modes
        fmse_val = float(fmse_t.mean())

        if args.clean:
            ax_f.set_axis_off()
        else:
            ax_f.set_xlabel("time", fontsize=13)
            ax_f.set_ylabel("spatial wavenumber $k$", fontsize=13)
            ax_f.tick_params(labelsize=11)
            ax_f.text(0.98, 0.97, f"fMSE={fmse_val:.2e}", transform=ax_f.transAxes, fontsize=11, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            _label_or_title(ax_f, pde_tex, lbl)
            _colorbar(fig_f, im_f, ax_f)

        out_f = os.path.join("fourier_error", pde_name, f"fourier_{stem}.png")
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        fig_f.savefig(out_f, dpi=150)
        plt.close(fig_f)
        print(f"Fourier plot saved → {out_f}")

# Conv_t / Conv_x numerical-convergence plots (candidates only)
if args.conv:
    conv_entries = []
    for fpath in compare_files:
        b, lin, quad, x_lin, t_lin, coord = parse_pde_file(fpath)
        suffix = candidate_label(fpath)
        conv_entries.append((rf"$PDE_{{\mathcal{{{suffix}}}}}$", b, lin, quad, x_lin, t_lin, coord))

    T_final = float(t[-1])

    fig_ct, ax_ct = plt.subplots(figsize=(col_w, col_h), layout="constrained")
    for tex, b, lin, quad, x_lin, t_lin, coord in conv_entries:
        dts, diffs = conv_t_series(b, lin, quad, x, kk, u0, T_final, dt, args.n_refine, x_lin, t_lin, coord)
        ax_ct.loglog(dts[1:], diffs, "o-", label=tex)
    ax_ct.set_xlabel(r"$\delta_t$", fontsize=13)
    ax_ct.set_ylabel(r"nMAE$(u_{\delta_t}, u_{\delta_t/2})$", fontsize=13)
    ax_ct.legend(fontsize=9)
    ax_ct.tick_params(labelsize=11)
    out_ct = os.path.join("convergence", pde_name, f"conv_t_{prefix}.png")
    os.makedirs(os.path.dirname(out_ct), exist_ok=True)
    fig_ct.savefig(out_ct, dpi=150)
    plt.close(fig_ct)
    print(f"Conv_t plot saved → {out_ct}")

    fig_cx, ax_cx = plt.subplots(figsize=(col_w, col_h), layout="constrained")
    for tex, b, lin, quad, x_lin, t_lin, coord in conv_entries:
        Ns, diffs = conv_x_series(b, lin, quad, x, dt, T_final, args.ic, args.n_refine, pde_name, x_lin, t_lin, coord)
        ax_cx.loglog(Ns[1:], diffs, "o-", label=tex)
    ax_cx.set_xlabel(r"$N$", fontsize=13)
    ax_cx.set_ylabel(r"nMAE$(u_N, u_{2N})$", fontsize=13)
    ax_cx.legend(fontsize=9)
    ax_cx.tick_params(labelsize=11)
    out_cx = os.path.join("convergence", pde_name, f"conv_x_{prefix}.png")
    os.makedirs(os.path.dirname(out_cx), exist_ok=True)
    fig_cx.savefig(out_cx, dpi=150)
    plt.close(fig_cx)
    print(f"Conv_x plot saved → {out_cx}")
