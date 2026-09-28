import os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import ORDER_NAME

THRESH = 1e-10  # threshold below which a coefficient is considered zero

# term-key encoding: b -> (), u_o -> (2+o,), x·u_o -> (1, 2+o), t·u_o -> (0, 2+o)
FACTOR_NAMES = {0: "t", 1: "x", 2: "u", 3: "u_x", 4: "u_xx", 5: "u_xxx", 6: "u_xxxx"}
FACTOR_ORDER = {name: fid for fid, name in FACTOR_NAMES.items()}
COORD_IDS = {}  # empty on purpose: pure coordinate terms (t, x, t·t, ...) aren't in the dictionary


def _term_key(factors):
    return tuple(sorted(factors))


def pde_terms(b, linear, quad, x_lin=None, t_lin=None, coord=None, thresh=THRESH):
    terms = {}

    def add(factors, v):
        if abs(v) > thresh:
            key = _term_key(factors)
            terms[key] = terms.get(key, 0.0) + v

    add((), b)
    for o, c in linear.items():
        add((2 + o,), c)
    for *ords, v in quad:
        add(tuple(2 + o for o in ords), v)
    for o, c in (x_lin or {}).items():
        add((1, 2 + o), c)
    for o, c in (t_lin or {}).items():
        add((0, 2 + o), c)
    for key, c in (coord or {}).items():
        ids = COORD_IDS.get(key)
        if ids is not None:
            add(ids, c)
    return terms


def pde_terms_from_data(data, which):
    p = "ref" if which == "ref" else "disc"
    return pde_terms(data[f"b_{p}"], data[f"lin_{p}"], data[f"quad_{p}"], data.get(f"xlin_{p}"), data.get(f"tlin_{p}"), data.get(f"coord_{p}"))


def term_label(key):
    if not key:
        return "b"
    return "·".join(FACTOR_NAMES[f] for f in key)


def align(terms_ref, terms_disc):
    # union of both PDEs' term keys, ordered like pdefind's own dictionary columns
    keys = sorted(set(terms_ref) | set(terms_disc), key=lambda k: (len(k), k))
    c_ref  = np.array([terms_ref.get(k, 0.0)  for k in keys])
    c_disc = np.array([terms_disc.get(k, 0.0) for k in keys])
    labels = [term_label(k) for k in keys]
    return c_ref, c_disc, labels


def support(c, thresh=THRESH):
    return np.abs(c) > thresh


def symbol(o):
    return sp.Symbol(ORDER_NAME.get(o, 'u_' + 'x' * o))


def expr_tree(b, linear, quad, x_lin=None, t_lin=None, coord=None):
    # each derivative order is its own independent symbol (u, u_x, u_xx, ...)
    X_SYM, T_SYM = sp.Symbol("x"), sp.Symbol("t")
    terms = []
    if abs(b) > 1e-12:
        terms.append(sp.Float(b))
    for o, c in linear.items():
        if abs(c) > 1e-12:
            terms.append(c * symbol(o))
    for *ords, v in quad:
        if abs(v) > 1e-12:
            prod = symbol(ords[0])
            for o in ords[1:]:
                prod = prod * symbol(o)
            terms.append(v * prod)
    for o, c in (x_lin or {}).items():
        if abs(c) > 1e-12:
            terms.append(c * X_SYM * symbol(o))
    for o, c in (t_lin or {}).items():
        if abs(c) > 1e-12:
            terms.append(c * T_SYM * symbol(o))
    return sp.Add(*terms) if terms else sp.Integer(0)


def expr_from_data(data, which):
    p = "ref" if which == "ref" else "disc"
    return expr_tree(data[f"b_{p}"], data[f"lin_{p}"], data[f"quad_{p}"], data.get(f"xlin_{p}"), data.get(f"tlin_{p}"), data.get(f"coord_{p}"))


def tree_size(expr):
    return len(list(sp.preorder_traversal(expr)))


def tree_depth(expr):
    if not expr.args:
        return 1
    return 1 + max(tree_depth(a) for a in expr.args)


def operator_weight(expr):
    # sum of arities of Add/Mul nodes; leaves (Symbol, Float, Integer) contribute 0
    w = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Add, sp.Mul)):
            w += len(node.args)
    return w
