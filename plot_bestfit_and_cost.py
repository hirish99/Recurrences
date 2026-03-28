"""
Generate the bestfit and cost figures for the paper (Figures 6, 9).

  - error-bestfit-laplace.pgf (Figure 6): Relative error of a single
    large-|x1| recurrence step for Laplace 2D (n=9) as a function of
    |x1|/xbar, with a linear least-squares fit in log-log space.

  - cost-combined.pgf (Figure 9): Flop count comparison for computing a
    line-Taylor expansion under three configurations (no rotation, rotation
    without recurrence, rotation with recurrence).

Output: scripts/output/{error-bestfit-laplace,cost-combined}.pdf

Usage:
    conda activate inteq
    python scripts/plot_bestfit_and_cost.py [bestfit|cost|all]
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Load recurrence modules
# ---------------------------------------------------------------------------

import sumpy.recurrence as _recurrence

_make_sympy_vec = _recurrence._make_sympy_vec
get_large_x1_recurrence = _recurrence.get_large_x1_recurrence

from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

NDIM = 2
var = _make_sympy_vec("x", NDIM)


# =========================================================================
# error-bestfit-laplace — Single recurrence step error vs |x1|/xbar
# =========================================================================

def generate_bestfit():
    """
    For Laplace 2D, derivative order n=9:
    At each grid point (x1, x2), compute derivative n using the large-|x1|
    recurrence for ONE step (all prior derivatives from sympy). Compare to
    the sympy ground truth. Plot the relative error vs |x1|/xbar where
    xbar = x2 (the off-axis coordinate).
    """
    print("Generating bestfit figure...")

    # PDE and Green's function
    w = make_identity_diff_op(NDIM)
    pde = laplacian(w)
    g = (-1 / (2 * sp.pi)) * sp.log(sp.sqrt(var[0]**2 + var[1]**2))
    deriv_order = 9

    # Get recurrence
    n_initial, recur_order, recurrence = get_large_x1_recurrence(pde)
    s = sp.Function("s")
    n = sp.symbols("n")

    # Build the recurrence step evaluator for derivative n=deriv_order
    # It needs s(n-1), s(n-2), ..., s(n-recur_order), x0, x1
    args = []
    for j in range(recur_order, 0, -1):
        args.append(s(deriv_order - j))
    args.extend(var)
    recur_fn = sp.lambdify(args, recurrence.subs(n, deriv_order),
                           modules=["scipy", "numpy"])

    # Build sympy evaluators for derivatives 0..deriv_order
    deriv_fns = []
    for i in range(deriv_order + 1):
        expr = sp.diff(g, var[0], i)
        deriv_fns.append(sp.lambdify([var[0], var[1]], expr,
                                      modules=["scipy", "numpy"]))

    # Sample points: sweep |x1|/xbar from 10^-9 to 10^-1
    # For each ratio, pick several random (x1, x2) pairs
    ratios = np.logspace(-9, -1, 50)
    x2_base = 1.0  # fix x2 = 1

    errors = []
    ratio_vals = []

    for ratio in ratios:
        x1_val = ratio * x2_base
        x2_val = x2_base

        # Compute prior derivatives from sympy (exact)
        prior = []
        for j in range(recur_order, 0, -1):
            idx = deriv_order - j
            with np.errstate(all="ignore"):
                prior.append(deriv_fns[idx](x1_val, x2_val))

        # One recurrence step
        with np.errstate(all="ignore"):
            recur_val = recur_fn(*prior, x1_val, x2_val)

        # Ground truth
        with np.errstate(all="ignore"):
            true_val = deriv_fns[deriv_order](x1_val, x2_val)

        if true_val != 0 and np.isfinite(recur_val) and np.isfinite(true_val):
            rel_err = abs((recur_val - true_val) / true_val)
            if rel_err > 0:
                errors.append(rel_err)
                ratio_vals.append(ratio)

    ratio_vals = np.array(ratio_vals)
    errors = np.array(errors)

    # Linear least squares fit in log-log space
    log_r = np.log10(ratio_vals)
    log_e = np.log10(errors)
    slope, intercept = np.polyfit(log_r, log_e, 1)

    fit_line = 10**(slope * log_r + intercept)

    print(f"  Fit slope: {slope:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.4))
    ax.scatter(ratio_vals, errors, s=10, label="Relative Error", zorder=3)
    ax.plot(ratio_vals, fit_line, "r-",
            label=f"Linear Least Squares Fit Slope: {slope:.4f}", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Parameter $|x_1|/\overline{x}$", fontsize=10)
    ax.set_ylabel("Relative Rounding Error (eq. 44)")
    ax.set_title(
        r"Relative Error in Single Recurrence Step, Laplace 2D, $n=9$",
        fontsize=12,
    )
    ax.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "output", "error-bestfit-laplace.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    print(f"  Saved: {out}")
    return fig


# =========================================================================
# cost-combined — Flop count: recurrence vs no recurrence
# =========================================================================

def _count_flops(expr):
    """
    Count the number of floating-point operations (additions and
    multiplications) in a sympy expression after CSE.

    This counts every Add and Mul node in the expression tree,
    which approximates the flop count.
    """
    # Apply CSE to reduce the expression
    replacements, reduced = sp.cse(expr)

    count = 0
    all_exprs = [repl_expr for _, repl_expr in replacements] + list(reduced)

    for e in all_exprs:
        for node in sp.preorder_traversal(e):
            if isinstance(node, sp.Add):
                count += len(node.args) - 1  # n-ary add = n-1 additions
            elif isinstance(node, sp.Mul):
                count += len(node.args) - 1  # n-ary mul = n-1 multiplications
            elif isinstance(node, sp.Pow):
                exp = node.args[1]
                if isinstance(exp, sp.Integer) and exp > 0:
                    count += int(exp) - 1  # x^n = n-1 multiplications
                else:
                    count += 1  # general power = 1 op
    return count


def generate_cost():
    """
    For Helmholtz 2D, compute the flop count for a line-Taylor expansion
    at orders 1, 3, 5, 7, 9, 11, 13, 15, comparing:
      - "Recurrence": use the large-|x1| recurrence to get derivatives
      - "No Recurrence": differentiate the Green's function directly with sympy

    The flop count is the number of arithmetic operations in the CSE'd
    expression for all derivatives up to the given order.
    """
    print("Generating cost-combined figure...")

    w = make_identity_diff_op(NDIM)
    pde = laplacian(w) + w

    # Green's function for Helmholtz 2D
    r = sp.sqrt(var[0]**2 + var[1]**2)
    g = sp.Rational(1, 4) * sp.I * sp.hankel1(0, r)

    # Helmholtz Hankel function differentiation is expensive at high orders.
    # Orders up to ~9 are feasible; beyond that sympy takes minutes per order.
    orders = [1, 3, 5, 7, 9]

    # Get recurrence
    n_initial, recur_order, recurrence = get_large_x1_recurrence(pde)
    s = sp.Function("s")
    n = sp.symbols("n")

    flops_recurrence = []
    flops_no_recurrence = []

    # Compute derivatives incrementally (diff once per order, not from scratch)
    derivs = [g]
    prev_deriv = g
    max_order = max(orders)

    print("  Computing symbolic derivatives incrementally...")
    for i in range(1, max_order + 1):
        import time as _time
        t0 = _time.time()
        prev_deriv = sp.diff(prev_deriv, var[0])
        derivs.append(prev_deriv)
        print(f"    d^{i}/dx0^{i}: {_time.time()-t0:.1f}s")

    for p in orders:
        print(f"  Order p={p}")

        # --- No recurrence: total flops for derivatives 0..p ---
        no_rec_total = 0
        for i in range(p + 1):
            no_rec_total += _count_flops(derivs[i])
        flops_no_recurrence.append(no_rec_total)

        # --- Recurrence: base cases from sympy, rest from recurrence ---
        rec_total = 0
        for i in range(p + 1):
            if i < n_initial:
                rec_total += _count_flops(derivs[i])
            else:
                rec_total += _count_flops(recurrence.subs(n, i))
        flops_recurrence.append(rec_total)

        print(f"    Recurrence: {rec_total}, No recurrence: {no_rec_total}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 5.3))
    ax.plot(orders, flops_recurrence, "o-", label="Recurrence")
    ax.plot(orders, flops_no_recurrence, "s-", label="No Recurrence")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Line-Taylor Order", fontsize=10)
    ax.set_ylabel("Flop Count", fontsize=10)
    ax.set_title("Helmholtz 2D Line-Taylor Flop Comparison", fontsize=12)
    ax.set_xticks(orders)
    ax.set_xticklabels([str(o) for o in orders])
    ax.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "output", "cost-combined.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = ["all"]

    if "all" in keys:
        keys = ["bestfit", "cost"]

    for key in keys:
        if key == "bestfit":
            generate_bestfit()
        elif key == "cost":
            generate_cost()
        else:
            print(f"Unknown: {key}. Choose from bestfit, cost, all")
            sys.exit(1)
