# sec:errors -- prediction errors
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import eval_rhs_full

NEEDS_SIMULATION = True


def MSE(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    return float(np.mean((U_disc - U_ref) ** 2))


def nMSE(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    num = float(np.sum((U_disc - U_ref) ** 2))
    den = float(np.sum(U_ref ** 2))
    return num / den if den > 0 else float("nan")


def nMSE_bar(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    num = float(np.sum((U_disc - U_ref) ** 2))
    u_bar = np.mean(U_ref, axis=0)
    den = float(np.sum((U_ref - u_bar) ** 2))
    return num / den if den > 0 else float("nan")


def nMAE(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    num = float(np.sum(np.abs(U_disc - U_ref)))
    den = float(np.sum(np.abs(U_ref)))
    return num / den if den > 0 else float("nan")


def n_utMSE(data):
    def rhs(p):
        return eval_rhs_full(data[f"b_{p}"], data[f"lin_{p}"], data[f"quad_{p}"], data["U_ref"], data["kk"], x_lin=data.get(f"xlin_{p}"), t_lin=data.get(f"tlin_{p}"), coord=data.get(f"coord_{p}"), x=data["x"], t=data["t"])
    ut_true = rhs("ref")
    ut_hat  = rhs("disc")
    num = float(np.sum((ut_hat - ut_true) ** 2))
    den = float(np.sum(ut_true ** 2))
    return num / den if den > 0 else float("nan")


def fMSE(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    kk = data["kk"]
    F_ref  = np.fft.rfft(U_ref,  axis=0)
    F_disc = np.fft.rfft(U_disc, axis=0)
    diff2  = np.abs(F_disc - F_ref) ** 2  # (n_k, nt)

    k_max = data.get("k_max")
    if k_max is not None:
        k_cut = int(np.searchsorted(kk, k_max)) + 1
    else:
        mean_energy = np.abs(F_ref).mean(axis=1)
        k_cut = int(np.where(mean_energy > mean_energy.max() * 1e-2)[0].max()) + 1
    k_cut = min(k_cut, len(kk))

    kmin, kmax = 1, k_cut - 1
    return float(np.mean(diff2[kmin:kmax + 1, :]))


def rollout(data):
    U_ref, U_disc = data["U_ref"], data["U_disc"]
    dx = float(data["x"][1] - data["x"][0])
    l2_sq_per_t = np.sum((U_disc - U_ref) ** 2, axis=0) * dx
    return float(np.mean(l2_sq_per_t))


METRICS = {"MSE":      MSE, "nMSE":      nMSE, "nMSE_bar":  nMSE_bar, "nMAE":     nMAE, "n_utMSE":  n_utMSE, "fMSE":     fMSE, "rollout":  rollout}
