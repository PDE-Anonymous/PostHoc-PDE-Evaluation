# sec:terms -- term-level classification metrics for PDE discovery
import os, sys
from functools import lru_cache
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(__file__))
from shared import align, pde_terms_from_data, support, expr_from_data, tree_size

NEEDS_SIMULATION = False


def _get(data):
    return align(pde_terms_from_data(data, "ref"), pde_terms_from_data(data, "disc"))


def _confusion(c_ref, c_disc):
    s_ref = support(c_ref)
    s_disc = support(c_disc)
    tp = int(np.sum(s_ref & s_disc))
    fp = int(np.sum(~s_ref & s_disc))
    fn = int(np.sum(s_ref & ~s_disc))
    return tp, fp, fn


def TPR(data):
    c_ref, c_disc, _ = _get(data)
    tp, fp, fn = _confusion(c_ref, c_disc)
    denom = tp + fn + fp
    return tp / denom if denom > 0 else 1.0


def Precision(data):
    c_ref, c_disc, _ = _get(data)
    tp, fp, fn = _confusion(c_ref, c_disc)
    denom = tp + fp
    return tp / denom if denom > 0 else 1.0


def Recall(data):
    c_ref, c_disc, _ = _get(data)
    tp, fp, fn = _confusion(c_ref, c_disc)
    denom = tp + fn
    return tp / denom if denom > 0 else 1.0


# Tuples (unlike sympy exprs) are hashable, so this is cacheable by _tree_edit_distance below
def _to_tuple_tree(expr):
    if isinstance(expr, sp.Add):
        label = "Add"
    elif isinstance(expr, sp.Mul):
        label = "Mul"
    elif isinstance(expr, sp.Pow):
        label = "Pow"
    else:
        label = str(expr) if not expr.args else type(expr).__name__
    return (label, tuple(_to_tuple_tree(a) for a in expr.args))


def _node_size(node):
    _, children = node
    return 1 + sum(_node_size(c) for c in children)


@lru_cache(maxsize=None)
def _tree_edit_distance(t1, t2):
    return _forest_edit_distance((t1,), (t2,))


# Ordered forest edit distance (Zhang-Shasha recurrence, unrestricted keyroots)
@lru_cache(maxsize=None)
def _forest_edit_distance(f1, f2):
    if not f1 and not f2:
        return 0
    if not f1:
        return sum(_node_size(n) for n in f2)
    if not f2:
        return sum(_node_size(n) for n in f1)
    last1, last2 = f1[-1], f2[-1]
    rest1, rest2 = f1[:-1], f2[:-1]
    delete_cost = _forest_edit_distance(rest1, f2) + _node_size(last1)
    insert_cost = _forest_edit_distance(f1, rest2) + _node_size(last2)
    rename_cost = 0 if last1[0] == last2[0] else 1
    match_cost = _forest_edit_distance(last1[1], last2[1]) + _forest_edit_distance(rest1, rest2) + rename_cost
    return min(delete_cost, insert_cost, match_cost)


# Normalized tree edit distance between the GT and candidate expression trees
def TED(data):
    ref_expr = expr_from_data(data, "ref")
    disc_expr = expr_from_data(data, "disc")
    dist = _tree_edit_distance(_to_tuple_tree(ref_expr), _to_tuple_tree(disc_expr))
    denom = tree_size(ref_expr) + tree_size(disc_expr)
    return float(dist) / denom if denom > 0 else 0.0


# Symbolic equivalence check: candidate and GT differ by an additive or multiplicative constant
def CAS(data):
    ref_expr = expr_from_data(data, "ref")
    disc_expr = expr_from_data(data, "disc")
    diff = sp.simplify(disc_expr - ref_expr)
    if not diff.free_symbols:
        return f"1.0 (c0={float(diff):.4g})"
    if ref_expr != 0:
        ratio = sp.simplify(disc_expr / ref_expr)
        if not ratio.free_symbols and ratio != 0:
            return f"1.0 (c1={float(ratio):.4g})"
    return 0.0


METRICS = {"TPR":      TPR, "Precision": Precision, "Recall":   Recall, "TED":      TED, "CAS":      CAS}
