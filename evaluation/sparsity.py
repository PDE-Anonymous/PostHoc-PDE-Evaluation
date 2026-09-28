# sec:sparsity -- parsimony of the discovered PDE
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared import align, support, expr_from_data, tree_size, tree_depth, operator_weight, pde_terms_from_data

NEEDS_SIMULATION = False


def _get(data):
    return align(pde_terms_from_data(data, "ref"), pde_terms_from_data(data, "disc"))


def Sterms(data):
    # counts #nonzero, not #zero -- align()'s union drops terms that are zero in both PDEs, so #zero on that truncated set would undercount relative to the true |Theta|
    _, c_disc, _ = _get(data)
    n_nonzero = int(np.sum(support(c_disc)))
    if n_nonzero == 0:
        return float("inf")
    theta_size = data.get("theta_size")
    if theta_size is None:
        raise ValueError("Sterms requires data['theta_size'] (size of the full " "candidate dictionary Theta) -- it cannot be inferred " "from a single ref/disc comparison.")
    return n_nonzero / theta_size


def ExpTree(data):
    # eq:ExpTree
    expr = expr_from_data(data, "disc")
    return float(tree_size(expr))


def ExpTreeV2(data):
    # weighted complexity: w1*NbNodes + w2*Depth + w3*OperatorWeight, weights default to 1.0
    w1 = data.get("w1", 1.0)
    w2 = data.get("w2", 1.0)
    w3 = data.get("w3", 1.0)
    expr = expr_from_data(data, "disc")
    return float(w1 * tree_size(expr) + w2 * tree_depth(expr) + w3 * operator_weight(expr))


METRICS = {"Sterms":    Sterms, "ExpTree":   ExpTree, "ExpTreeV2": ExpTreeV2}
