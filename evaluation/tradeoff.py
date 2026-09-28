# sec:tradeoff-sparsity-accuracy -- PIC (eq:PIC) not implemented, needs a PINN surrogate + moving-horizon regression
import os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (align, support, pde_terms_from_data, expr_from_data, expr_tree, tree_depth, tree_size)
from utils import eval_rhs_full, make_integrator, simulate

NEEDS_SIMULATION = False  # all residual terms are evaluated pointwise on U_ref, like n_utMSE


def _get(data):
    return align(pde_terms_from_data(data, "ref"), pde_terms_from_data(data, "disc"))


def _base_tree(data):
    return expr_tree(data["b_base"], data["lin_base"], data["quad_base"], data.get("xlin_base"), data.get("tlin_base"), data.get("coord_base"))


def _residual_ut(data):
    # same idea as errors.py's n_utMSE: ut_true/ut_hat both evaluated on U_ref
    def rhs(p):
        return eval_rhs_full(data[f"b_{p}"], data[f"lin_{p}"], data[f"quad_{p}"], data["U_ref"], data["kk"], x_lin=data.get(f"xlin_{p}"), t_lin=data.get(f"tlin_{p}"), coord=data.get(f"coord_{p}"), x=data["x"], t=data["t"])
    return rhs("ref"), rhs("disc")


def _nMAE(U_disc, U_ref):
    num = float(np.sum(np.abs(U_disc - U_ref)))
    den = float(np.sum(np.abs(U_ref)))
    return num / den if den > 0 else float("nan")


def Score(data):
    # compares the candidate against its predecessor on a complexity Pareto front, so it needs a baseline PDE -- no principled default, must come from --baseline_file
    if "lin_base" not in data:
        raise ValueError("Score requires a baseline PDE (data['lin_base']/['quad_base']/['b_base'], " "e.g. via evaluation.py's --baseline_file) -- it has no meaningful default.")

    U_ref = data["U_ref"]
    u0 = U_ref[:, 0]

    b, lin, quad = data["b_disc"], data["lin_disc"], data["quad_disc"]
    U_disc = data.get("U_disc")
    if U_disc is None:
        dt = float(data["dt"])
        integrator = make_integrator(b, lin, quad, data["kk"], dt, x_coord=data.get("x"), x_lin=data.get("xlin_disc", {}), t_lin=data.get("tlin_disc", {}), coord=data.get("coord_disc", {}))
        U_disc = simulate(integrator, u0, U_ref.shape[1], dt=dt)

    b_base, lin_base, quad_base = data["b_base"], data["lin_base"], data["quad_base"]
    U_base = data.get("U_base")
    if U_base is None:
        dt = float(data["dt"])
        integrator = make_integrator(b_base, lin_base, quad_base, data["kk"], dt, x_coord=data.get("x"), x_lin=data.get("xlin_base", {}), t_lin=data.get("tlin_base", {}), coord=data.get("coord_base", {}))
        U_base = simulate(integrator, u0, U_ref.shape[1], dt=dt)

    nmae_disc = _nMAE(U_disc, U_ref)
    nmae_base = _nMAE(U_base, U_ref)

    C_disc = tree_size(expr_from_data(data, "disc"))
    C_base = tree_size(_base_tree(data))
    dC = C_disc - C_base
    if dC == 0 or nmae_disc <= 0 or nmae_base <= 0:
        return float("nan")

    return float(-(np.log(nmae_disc) - np.log(nmae_base)) / dC)


def Reward1(data):
    # eq:reward1. c0 default per the paper's parameter table; override via data["c0"].
    c0 = data.get("c0", 0.2)
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    if s_size == 0:
        return float("-inf")
    sparsity = 1 - c0 * np.log10(s_size)

    ut_true, ut_hat = _residual_ut(data)
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    tss = float(np.sum((ut_true - ut_true.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return float(sparsity * r2)


def Reward2(data):
    # eq:reward2. xi1, xi2 defaults per the paper's parameter table; override via data["xi1"]/data["xi2"].
    xi1 = data.get("xi1", 0.01)
    xi2 = data.get("xi2", 0.0001)
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    depth = tree_depth(expr_from_data(data, "disc"))

    ut_true, ut_hat = _residual_ut(data)
    err = float(np.sum((ut_true - ut_hat) ** 2))
    return float((1 - xi1 * s_size - xi2 * depth) / (1 + err))


def AICc(data):
    # eq:cAIC, using the RSS-based Gaussian log-likelihood on u_t residuals.
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    return float(n * np.log(rss / n) + 2 * s_size)


def BIC(data):
    # same log-likelihood as AICc, just a different complexity penalty: log(n)*|S| vs 2*|S|
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    return float(np.log(n) * s_size + n * np.log(rss / n))


def MDL_Fey(data):
    # rank (N) isn't derivable from a single (ref, disc) pair, defaults to 1 -- override via data["rank"]
    rank = data.get("rank", 1)
    eps_d = data.get("eps_d", 1e-15)

    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    lam = data.get("mdl_lambda", np.sqrt(n))
    eps = float(np.mean((ut_true - ut_hat) ** 2))
    return float(np.log2(rank) + lam * np.log2(max(1.0, eps / eps_d)))


def _description_length(expr, data, theta_size):
    # |O_v| (SymLang paper) is each operator node's own arity, not a fixed alphabet size -- sympy's Add/Mul are n-ary so this is just len(node.args); leaves cost 0, no per-symbol term
    c_max = data.get("c_max", 10.0)
    eps = data.get("eps_enc", 1e-6)

    bits = 0.0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Add, sp.Mul)):
            bits += np.log2(len(node.args))
    bits += theta_size * np.log2(c_max / eps)
    return bits


def MDL_Sym(data):
    # -log p(T|e) + lambda*len(e); likelihood term is half of AICc/BIC's (-log p vs -2log p)
    lam = data.get("mdl_sym_lambda", 0.1)
    _, c_disc, _ = _get(data)
    theta_size = data.get("theta_size") or len(c_disc)

    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    neg_log_p = 0.5 * n * np.log(rss / n)

    expr = expr_from_data(data, "disc")
    length = _description_length(expr, data, theta_size)

    return float(neg_log_p + lam * length)


METRICS = {"Score":    Score, "MDL_Sym":  MDL_Sym, "Reward1": Reward1, "Reward2": Reward2, "AICc":    AICc, "BIC":     BIC, "MDL_Fey": MDL_Fey}
