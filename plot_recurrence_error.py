"""
Generate the 3-panel error heatmap figures for the paper (Figures 3--5):
  - error-heatmap-laplace-2d.pgf
  - error-heatmap-helmholtz-2d.pgf
  - error-heatmap-biharmonic-2d.pgf

Each figure shows the relative error of computing the n-th derivative of
the Green's function via recurrence vs sympy, on a log-spaced 2D grid of
(x1, x2) points. The three panels are:
  1. Large-|x1| recurrence only
  2. Small-|x1| recurrence (Taylor expansion with p_offaxis terms)
  3. Blended large/small-|x1| (switching at threshold xi)

Output: scripts/output/error-heatmap-{laplace,helmholtz,biharmonic}-2d.pdf

Usage:
    conda activate inteq
    python scripts/plot_recurrence_error.py [laplace|helmholtz|biharmonic|all]
"""

from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Load recurrence modules from the parametric_recurrence branch
# ---------------------------------------------------------------------------

import sumpy.recurrence as _recurrence

_make_sympy_vec = _recurrence._make_sympy_vec
get_large_x1_recurrence = _recurrence.get_large_x1_recurrence
get_small_x1_expansion = _recurrence.get_small_x1_expansion

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
    LinearPDESystemOperator,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Grid resolution for the heatmap (number of points along each axis)
GRID_N = 60

# Grid range: log10 of min and max for x1 and x2
GRID_LOG_MIN = -8
GRID_LOG_MAX = 0

# Colorbar range
CBAR_VMIN = 1e-16
CBAR_VMAX = 1e1

# Small-|x1| Taylor expansion order
P_OFFAXIS = 12

# Blending threshold: use small-|x1| when |x2|/|x1| > XI
XI = 10

# ---------------------------------------------------------------------------
# PDE definitions
# ---------------------------------------------------------------------------

NDIM = 2
var = _make_sympy_vec("x", NDIM)


def _make_laplace_2d():
    """Laplace PDE in 2D: nabla^2 u = 0."""
    w = make_identity_diff_op(NDIM)
    pde = laplacian(w)
    # Green's function: G = -1/(2*pi) * log(r)
    g = (-1 / (2 * sp.pi)) * sp.log(sp.sqrt(var[0]**2 + var[1]**2))
    return pde, g, "Laplace 2D", 9


def _make_helmholtz_2d():
    """Helmholtz PDE in 2D: (nabla^2 + 1) u = 0."""
    w = make_identity_diff_op(NDIM)
    pde = laplacian(w) + w
    # Green's function: G = i/4 * H_0^(1)(r)
    # For derivative comparison we use the sympy besselj/bessely representation
    r = sp.sqrt(var[0]**2 + var[1]**2)
    g = sp.Rational(1, 4) * sp.I * sp.hankel1(0, r)
    return pde, g, "Helmholtz 2D", 8


def _make_biharmonic_2d():
    """Biharmonic PDE in 2D: nabla^4 u = 0."""
    w = make_identity_diff_op(NDIM)
    pde = laplacian(laplacian(w))
    # Green's function: G = 1/(8*pi) * r^2 * log(r)
    r = sp.sqrt(var[0]**2 + var[1]**2)
    g = sp.Rational(1, 8) / sp.pi * r**2 * sp.log(r)
    return pde, g, "Biharmonic 2D", 8


PDE_REGISTRY = {
    "laplace": (_make_laplace_2d, "error-heatmap-laplace-2d"),
    "helmholtz": (_make_helmholtz_2d, "error-heatmap-helmholtz-2d"),
    "biharmonic": (_make_biharmonic_2d, "error-heatmap-biharmonic-2d"),
}


# ---------------------------------------------------------------------------
# Core computation: evaluate n-th derivative via recurrence on a grid
# ---------------------------------------------------------------------------


def _build_recurrence_evaluator(pde, g, deriv_order):
    """
    Build lambdified functions for evaluating the n-th x1-derivative of g
    using the large-|x1| recurrence.

    Returns:
        eval_fns: list of callables, one per derivative order 0..deriv_order.
                  Each takes (prev_values..., x0, x1) and returns the derivative.
        n_initial: number of initial derivatives computed from sympy (base cases).
        recur_order: number of prior derivatives needed by the recurrence.
    """
    n_initial, recur_order, recurrence = get_large_x1_recurrence(pde)

    s = sp.Function("s")
    n = sp.symbols("n")

    eval_fns = []
    for i in range(deriv_order + 1):
        # Build argument list: s(i-order), ..., s(i-1), x0, x1
        args = []
        for j in range(recur_order, 0, -1):
            args.append(s(i - j))
        args.extend(var)

        if i < n_initial:
            # Base case: compute from the Green's function directly
            expr = sp.diff(g, var[0], i)
        else:
            # Recurrence step
            expr = recurrence.subs(n, i)

        eval_fns.append(sp.lambdify(args, expr, modules=["scipy", "numpy"]))

    return eval_fns, n_initial, recur_order


def _build_small_x1_evaluator(pde, g, deriv_order, taylor_order=P_OFFAXIS):
    """
    Build lambdified functions for evaluating the n-th x1-derivative using
    the small-|x1| expansion (Taylor expansion around x1=0).

    Returns:
        expansion_fn: callable(prev_s_values..., x0, x1, n_val) -> derivative
        s_n_initial: number of initial values for the small-|x1| recurrence
        s_order: recurrence order for the small-|x1| recurrence
        s_eval_fns: list of callables for the small-|x1| recurrence values
    """
    # Get the small-|x1| recurrence (for s(n) at x1=0)
    s_n_initial, s_order, s_recurrence = _recurrence.get_small_x1_recurrence(pde)

    # The effective start order for the recurrence must account for the
    # large-|x1| recurrence order as well (matching recurrence_qbx.py).
    _, large_recur_order, _ = get_large_x1_recurrence(pde)
    effective_start = max(s_n_initial, large_recur_order)

    s = sp.Function("s")  # noqa: F841 (used in lambdify arg construction)
    n = sp.symbols("n")

    # Build evaluators for the small-|x1| recurrence sequence
    s_eval_fns = []
    for i in range(deriv_order + taylor_order + 1):
        args = []
        for j in range(s_order, 0, -1):
            args.append(s(i - j))
        args.extend(var)

        if i < effective_start:
            # Base case: d^i/dx1^i G evaluated at x1=0
            expr = sp.diff(g, var[0], i)
            expr = expr.subs(var[0], 0)
        else:
            expr = s_recurrence.subs(n, i)

        s_eval_fns.append(sp.lambdify(args, expr, modules=["scipy", "numpy"]))

    # Get the Taylor expansion formula
    expansion, exp_s_initial, exp_s_order = get_small_x1_expansion(
        pde, taylor_order=taylor_order
    )

    # The expansion contains s(n), s(n-1), etc. as symbolic function calls.
    # Replace them with dummy symbols so we can lambdify properly.
    # IMPORTANT: arg order must match recurrence_qbx.py convention:
    #   [s(n-exp_s_order), s(n-exp_s_order+1), ..., s(n-1), s(n), x0, x1]
    s_dummies = []
    expansion_subst = expansion
    for j in range(exp_s_order, -1, -1):
        dummy = sp.Symbol(f"_s_nm{j}")
        expansion_subst = expansion_subst.subs(s(n - j), dummy)
        s_dummies.append(dummy)

    exp_args = s_dummies + list(var) + [n]
    expansion_fn = sp.lambdify(exp_args, expansion_subst,
                               modules=["scipy", "numpy"])

    return expansion_fn, effective_start, s_order, s_eval_fns, exp_s_order


def evaluate_derivative_large_x1(eval_fns, recur_order, x1_vals, x2_vals,
                                  deriv_order):
    """
    Evaluate the deriv_order-th derivative on a grid using only the
    large-|x1| recurrence.

    Returns: 2D array of derivative values, shape (len(x2_vals), len(x1_vals))
    """
    nx1, nx2 = len(x1_vals), len(x2_vals)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Storage for the last `recur_order` derivative values at each grid point
    storage = [np.zeros_like(X1)] * recur_order

    result = None
    for i in range(deriv_order + 1):
        fn = eval_fns[i]
        with np.errstate(all="ignore"):
            s_new = fn(*storage, X1, X2)
        if not isinstance(s_new, np.ndarray):
            s_new = s_new * np.ones_like(X1)
        # Replace nan/inf with 0 (happens at x1=0 singularity)
        s_new = np.where(np.isfinite(s_new), s_new, 0.0)

        storage.pop(0)
        storage.append(s_new)

        if i == deriv_order:
            result = s_new

    return result


def evaluate_derivative_small_x1(expansion_fn, s_eval_fns, s_order,
                                  effective_start, exp_s_order,
                                  x1_vals, x2_vals, deriv_order,
                                  g_expr=None):
    """
    Evaluate the deriv_order-th derivative on a grid using the
    small-|x1| expansion, following the same approach as recurrence_qbx.py.

    For orders < effective_start: use sympy derivatives directly.
    For orders >= effective_start: use the Taylor expansion from the
    small-|x1| recurrence values.
    """
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Step 1: compute the small-|x1| recurrence sequence s(0)..s(N) at x1=0
    storage_taylor_order = max(s_order, exp_s_order + 1)
    N = deriv_order + P_OFFAXIS + 1
    s_storage = [np.zeros_like(X2[:, 0])] * storage_taylor_order
    s_all = []  # indexed by order i

    for i in range(N):
        fn = s_eval_fns[i]
        with np.errstate(all="ignore"):
            s_new = fn(*s_storage[-s_order:], np.zeros_like(X2[:, 0]), X2[:, 0])
        if not isinstance(s_new, np.ndarray):
            s_new = s_new * np.ones_like(X2[:, 0])
        s_new = np.where(np.isfinite(s_new), s_new, 0.0)

        s_storage.pop(0)
        s_storage.append(s_new)
        s_all.append(s_new)

    # Step 2: evaluate the derivative at order deriv_order
    # For base cases (i < effective_start), use direct sympy derivative
    if deriv_order < effective_start and g_expr is not None:
        deriv_expr = sp.diff(g_expr, var[0], deriv_order)
        deriv_fn = sp.lambdify([var[0], var[1]], deriv_expr,
                               modules=["scipy", "numpy"])
        with np.errstate(all="ignore"):
            result = deriv_fn(X1, X2)
        if not isinstance(result, np.ndarray):
            result = result * np.ones_like(X1)
    else:
        # Use the Taylor expansion: needs s values from storage
        # Gather s values in order [s(n-exp_s_order), ..., s(n-1), s(n)]
        s_args = []
        for j in range(exp_s_order, -1, -1):
            idx = deriv_order - j
            if 0 <= idx < len(s_all):
                s_args.append(s_all[idx][:, None] * np.ones_like(X1))
            else:
                s_args.append(np.zeros_like(X1))

        with np.errstate(all="ignore"):
            result = expansion_fn(*s_args, X1, X2, deriv_order)
        if not isinstance(result, np.ndarray):
            result = result * np.ones_like(X1)

    result = np.where(np.isfinite(result), result, 0.0)
    return result


def evaluate_derivative_sympy(g, x1_vals, x2_vals, deriv_order):
    """
    Evaluate the deriv_order-th derivative using sympy (ground truth).
    """
    deriv_expr = sp.diff(g, var[0], deriv_order)
    # Use scipy for special functions (hankel1, besselj, etc.)
    deriv_fn = sp.lambdify([var[0], var[1]], deriv_expr,
                           modules=["scipy", "numpy"])

    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    with np.errstate(all="ignore"):
        result = deriv_fn(X1, X2)
    if not isinstance(result, np.ndarray):
        result = result * np.ones_like(X1)
    return result


def compute_relative_error(recur_vals, sympy_vals):
    """Compute |(recur - sympy) / recur|, clipped to colorbar range."""
    with np.errstate(all="ignore"):
        err = np.abs((recur_vals - sympy_vals) / recur_vals)
    err = np.where(np.isfinite(err), err, CBAR_VMAX)
    err = np.clip(err, CBAR_VMIN, CBAR_VMAX)
    return err


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_heatmap_figure(err_large, err_small, err_blend,
                         x1_vals, x2_vals, pde_name, deriv_order):
    """
    Generate the 3-panel error heatmap.

    Returns the matplotlib Figure.
    """
    panels = [
        ("Large-$|x_1|$ Recurrence", err_large),
        (f"Small-$|x_1|$ Expansion ($p_{{offaxis}}$={P_OFFAXIS})", err_small),
        (rf"Hybrid Approach ($\xi$={XI})", err_blend),
    ]

    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.2), sharey=True)
    norm = mcolors.LogNorm(vmin=CBAR_VMIN, vmax=CBAR_VMAX)

    for ax, (title, err) in zip(axes, panels):
        pcm = ax.pcolormesh(X1, X2, err, norm=norm, cmap="RdBu_r",
                            shading="auto", rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$x_1$-coordinate", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel("$x_2$-coordinate", fontsize=7)

    fig.colorbar(pcm, ax=axes, orientation="horizontal",
                 fraction=0.04, pad=0.18, aspect=50)

    fig.suptitle(
        rf"{pde_name}: {deriv_order}th Order Derivative Evaluation Error"
        r" $(u_{recur}-u_{sympy})/u_{recur}$",
        fontsize=8,
    )

    return fig


# ---------------------------------------------------------------------------
# Main driver for one PDE
# ---------------------------------------------------------------------------


def compute_heatmap_data(pde_key):
    """Compute error arrays for a given PDE.

    Returns:
        (err_large, err_small, err_blend, x1_vals, x2_vals, pde_name, deriv_order)
    """
    make_pde, output_name = PDE_REGISTRY[pde_key]
    pde, g, pde_name, deriv_order = make_pde()

    print(f"\n{'='*60}")
    print(f"Computing: {pde_name}, derivative order {deriv_order}")
    print(f"{'='*60}")

    x1_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)
    x2_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)

    # --- Ground truth (sympy) ---
    print("  Computing ground truth (sympy)...")
    t0 = time.time()
    sympy_vals = evaluate_derivative_sympy(g, x1_vals, x2_vals, deriv_order)
    print(f"    done ({time.time()-t0:.1f}s)")

    # --- Large-|x1| recurrence ---
    print("  Building large-|x1| evaluator...")
    t0 = time.time()
    eval_fns, n_initial, recur_order = _build_recurrence_evaluator(
        pde, g, deriv_order
    )
    print(f"  Computing large-|x1| derivatives...")
    large_vals = evaluate_derivative_large_x1(
        eval_fns, recur_order, x1_vals, x2_vals, deriv_order
    )
    err_large = compute_relative_error(large_vals, sympy_vals)
    print(f"    done ({time.time()-t0:.1f}s)")

    # --- Small-|x1| expansion ---
    print("  Building small-|x1| evaluator...")
    t0 = time.time()
    try:
        expansion_fn, effective_start, s_order, s_eval_fns, exp_s_order = \
            _build_small_x1_evaluator(pde, g, deriv_order)
        print(f"  Computing small-|x1| derivatives...")
        small_vals = evaluate_derivative_small_x1(
            expansion_fn, s_eval_fns, s_order, effective_start, exp_s_order,
            x1_vals, x2_vals, deriv_order, g_expr=g
        )
        err_small = compute_relative_error(small_vals, sympy_vals)
    except Exception as e:
        print(f"    WARNING: small-|x1| failed: {e}")
        import traceback; traceback.print_exc()
        small_vals = np.zeros((len(x2_vals), len(x1_vals)))
        err_small = CBAR_VMAX * np.ones((len(x2_vals), len(x1_vals)))
    print(f"    done ({time.time()-t0:.1f}s)")

    # --- Blended ---
    print("  Computing blended derivatives...")
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    use_small = np.abs(X2) > XI * np.abs(X1)
    blend_vals = np.where(use_small, small_vals, large_vals)
    err_blend = compute_relative_error(blend_vals, sympy_vals)

    return err_large, err_small, err_blend, x1_vals, x2_vals, pde_name, deriv_order


def generate_figure(pde_key):
    """Compute data and generate the error heatmap figure for a given PDE."""
    _, output_name = PDE_REGISTRY[pde_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output", f"{output_name}.pdf")

    err_large, err_small, err_blend, x1_vals, x2_vals, pde_name, deriv_order = \
        compute_heatmap_data(pde_key)

    print("  Generating figure...")
    fig = make_heatmap_figure(err_large, err_small, err_blend,
                               x1_vals, x2_vals, pde_name, deriv_order)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate error heatmap figures.")
    parser.add_argument("pdes", nargs="*", default=["all"],
                        choices=["all"] + list(PDE_REGISTRY.keys()),
                        help="Which PDEs to generate (default: all)")
    parser.add_argument("--grid", type=int, default=GRID_N,
                        help=f"Grid resolution (default: {GRID_N})")
    parser.add_argument("--p-offaxis", type=int, default=P_OFFAXIS,
                        help=f"Small-|x1| Taylor order (default: {P_OFFAXIS})")
    parser.add_argument("--xi", type=int, default=XI,
                        help=f"Blending threshold (default: {XI})")
    args = parser.parse_args()

    GRID_N = args.grid
    P_OFFAXIS = args.p_offaxis
    XI = args.xi

    keys = args.pdes
    if "all" in keys:
        keys = list(PDE_REGISTRY.keys())

    for key in keys:
        if key not in PDE_REGISTRY:
            print(f"Unknown PDE: {key}. Choose from {list(PDE_REGISTRY.keys())}")
            sys.exit(1)
        generate_figure(key)
