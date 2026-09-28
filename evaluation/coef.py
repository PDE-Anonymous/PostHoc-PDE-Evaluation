# sec:coef -- recovery of the PDE coefficients
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared import align, pde_terms_from_data, support

NEEDS_SIMULATION = False


def _get(data):
    return align(pde_terms_from_data(data, "ref"), pde_terms_from_data(data, "disc"))


def nL2coef(data):
    c_ref, c_disc, _ = _get(data)
    d = float(np.sum((c_ref - c_disc) ** 2))
    n = float(np.sum(c_ref ** 2))
    return d / n if n > 0 else float("nan")


def nL1coef(data):
    c_ref, c_disc, _ = _get(data)
    return float(np.sum(np.abs(c_ref - c_disc))) / float(np.sum(np.abs(c_ref)))


def maxerror(data):
    c_ref, c_disc, _ = _get(data)
    s = support(c_ref)
    return float(np.max(np.abs((c_ref - c_disc)[s]) / np.abs(c_ref[s]))) if np.any(s) else float("nan")


def _DCG(v):
    sorted_v = np.sort(np.abs(v))[::-1]
    denom = np.log2(np.arange(2, len(sorted_v) + 2))
    return float(np.sum(sorted_v / denom))


def NDCG(data):
    c_ref, c_disc, _ = _get(data)
    dcg  = _DCG(c_disc)
    idcg = _DCG(c_ref)
    return dcg / idcg if idcg > 0 else 1.0


def wL2coef(data):
    from sensitivity import sensitivity_weights  # lazy: only wL2coef needs jax
    c_ref, c_disc, _ = _get(data)
    keys, w = sensitivity_weights(data)
    diff = c_ref - c_disc
    return float(np.sum(w * diff ** 2) / np.sum(w)) if w.sum() > 0 else float("nan")


def Tanimoto(data):
    c_ref, c_disc, _ = _get(data)
    dot  = float(c_ref @ c_disc)
    norm = float(c_ref @ c_ref) + float(c_disc @ c_disc) - dot
    return dot / norm if norm != 0 else 1.0


METRICS = {"nL2coef":        nL2coef, "nL1coef":        nL1coef, "maxerror":       maxerror, "NDCG":           NDCG, "wL2coef":        wL2coef, "Tanimoto":       Tanimoto}
