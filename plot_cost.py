"""
Generate flop count comparison figures for Line-Taylor expansions.

Three configurations compared:
  1. No rotation, no recurrence: Line-Taylor along a general direction v,
     requiring the full chain rule (mixed partial derivatives).
  2. Rotation, no recurrence: Line-Taylor along x1-axis after rotation,
     only 1D derivatives needed, computed from sympy.
  3. Rotation + recurrence: Same as (2), but derivatives come from the
     recurrence formula.

For each configuration, the full line-Taylor sum is built symbolically,
CSE is applied to the entire expression, and arithmetic operations are
counted in the CSE output.

Data is cached to scripts/output/cost_data_{kernel}.json so plots can be
regenerated without recomputing.

Output: scripts/output/cost-{kernel}.pdf, scripts/output/cost-combined.pdf

Usage:
    conda activate paper2025
    python scripts/plot_cost.py [laplace|helmholtz|laplace3d|helmholtz3d|all]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

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

# Symbols for the line-Taylor expansion
t_sym = sp.Symbol("t")
radius_sym = sp.Symbol("radius")


# ---------------------------------------------------------------------------
# Flop counting
# ---------------------------------------------------------------------------

def count_flops_in_cse(cse_result):
    """
    Count arithmetic operations in a CSE result.

    Args:
        cse_result: tuple (replacements, reduced_exprs) from sp.cse()

    Returns:
        Total number of floating-point operations (add, mul, pow).
    """
    replacements, reduced = cse_result
    count = 0
    all_exprs = [e for _, e in replacements] + list(reduced)

    for expr in all_exprs:
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Add):
                count += len(node.args) - 1
            elif isinstance(node, sp.Mul):
                count += len(node.args) - 1
            elif isinstance(node, sp.Pow):
                exp = node.args[1]
                if isinstance(exp, sp.Integer) and exp > 0:
                    # x^n computed as n-1 multiplications
                    count += int(exp) - 1
                else:
                    count += 1
    return count


def count_flops(expr):
    """CSE an expression and count flops."""
    return count_flops_in_cse(sp.cse(expr))


# ---------------------------------------------------------------------------
# Line-Taylor sum builders
# ---------------------------------------------------------------------------

def build_line_taylor_no_rotation(g_expr, p, ndim, var):
    """
    Case 1: No rotation, no recurrence.

    Build the line-Taylor expansion of G along a general direction v:
        sum_{i=0}^{p} [d^i/dt^i G(|x + t*v|)]_{t=0} * radius^i / i!

    This forces sympy to compute the chain rule, producing expressions
    with mixed partial derivatives.

    Returns the symbolic sum.
    """
    v_syms = sp.symbols(f"v0:{ndim}")

    # G evaluated along the line: G(sqrt(sum_j (x_j + t*v_j)^2))
    r_of_t = sp.sqrt(sum(
        (var[j] + t_sym * v_syms[j])**2 for j in range(ndim)
    ))

    # Substitute r into the Green's function expression
    r_orig = sp.sqrt(sum(var[j]**2 for j in range(ndim)))
    g_of_t = g_expr.subs(r_orig, r_of_t)

    # If direct substitution didn't work (e.g. for log(sqrt(...))),
    # do a more explicit substitution
    if g_of_t == g_expr:
        g_of_t = g_expr
        for j in range(ndim):
            g_of_t = g_of_t.subs(var[j], var[j] + t_sym * v_syms[j])

    total = sp.Integer(0)
    current_deriv = g_of_t

    for i in range(p + 1):
        if i > 0:
            current_deriv = sp.diff(current_deriv, t_sym)

        term = current_deriv.subs(t_sym, 0) * radius_sym**i / math.factorial(i)
        total += term

        print(f"      no-rot order {i}/{p} done")

    return total


def build_line_taylor_rotation_no_recurrence(g_expr, p, var,
                                              derivs_cache=None):
    """
    Case 2: Rotation, no recurrence.

    After rotation, the line is along x0-axis, so:
        sum_{i=0}^{p} [d^i/dx0^i G] * radius^i / i!

    Derivatives computed from sympy differentiation.

    Args:
        derivs_cache: optional list of precomputed derivatives [d^0 G, d^1 G, ...]

    Returns the symbolic sum and the list of derivatives.
    """
    total = sp.Integer(0)

    if derivs_cache is None:
        derivs_cache = []
        prev = g_expr
        for i in range(p + 1):
            if i == 0:
                derivs_cache.append(g_expr)
            else:
                prev = sp.diff(prev, var[0])
                derivs_cache.append(prev)
            print(f"      rot/no-rec diff order {i}/{p} done")
    else:
        # Extend cache if needed
        while len(derivs_cache) <= p:
            i = len(derivs_cache)
            prev = sp.diff(derivs_cache[-1], var[0])
            derivs_cache.append(prev)
            print(f"      rot/no-rec diff order {i}/{p} done")

    for i in range(p + 1):
        total += derivs_cache[i] * radius_sym**i / math.factorial(i)

    return total, derivs_cache


def build_line_taylor_rotation_recurrence(g_expr, pde, p, var,
                                           derivs_cache=None):
    """
    Case 3: Rotation + recurrence.

    Base-case derivatives from sympy, remaining from the recurrence formula.
    The recurrence references prior terms as s(n-1), s(n-2), etc. — we
    substitute those with the actual expressions to get a self-contained
    symbolic sum suitable for CSE.

    Returns the symbolic sum.
    """
    n_initial, recur_order, recurrence = get_large_x1_recurrence(pde)
    s = sp.Function("s")
    n = sp.symbols("n")

    # Build the list of derivative expressions
    deriv_exprs = []

    if derivs_cache is None:
        derivs_cache = []
        prev = g_expr
        for i in range(min(n_initial, p + 1)):
            if i == 0:
                derivs_cache.append(g_expr)
            else:
                prev = sp.diff(prev, var[0])
                derivs_cache.append(prev)
    else:
        while len(derivs_cache) < n_initial and len(derivs_cache) <= p:
            if len(derivs_cache) == 0:
                derivs_cache.append(g_expr)
            else:
                derivs_cache.append(sp.diff(derivs_cache[-1], var[0]))

    # Base cases: from sympy
    for i in range(min(n_initial, p + 1)):
        deriv_exprs.append(derivs_cache[i])

    # Recurrence cases: substitute s(n-j) with actual prior expressions
    for i in range(n_initial, p + 1):
        expr = recurrence.subs(n, i)
        # Replace s(i-j) references with actual derivative expressions
        for j in range(recur_order, 0, -1):
            idx = i - j
            if 0 <= idx < len(deriv_exprs):
                expr = expr.subs(s(idx), deriv_exprs[idx])
            else:
                expr = expr.subs(s(idx), sp.Integer(0))
        deriv_exprs.append(expr)
        print(f"      rot+rec order {i}/{p} done")

    total = sp.Integer(0)
    for i in range(p + 1):
        total += deriv_exprs[i] * radius_sym**i / math.factorial(i)

    return total


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def _make_laplace_2d():
    ndim = 2
    var = _make_sympy_vec("x", ndim)
    w = make_identity_diff_op(ndim)
    pde = laplacian(w)
    g = (-1 / (2 * sp.pi)) * sp.log(sp.sqrt(var[0]**2 + var[1]**2))
    return pde, g, "Laplace 2D", ndim, var


def _make_helmholtz_2d():
    ndim = 2
    var = _make_sympy_vec("x", ndim)
    w = make_identity_diff_op(ndim)
    pde = laplacian(w) + w
    r = sp.sqrt(var[0]**2 + var[1]**2)
    g = sp.Rational(1, 4) * sp.I * sp.hankel1(0, r)
    return pde, g, "Helmholtz 2D", ndim, var


def _make_laplace_3d():
    ndim = 3
    var = _make_sympy_vec("x", ndim)
    w = make_identity_diff_op(ndim)
    pde = laplacian(w)
    r = sp.sqrt(var[0]**2 + var[1]**2 + var[2]**2)
    g = sp.Rational(-1, 1) / (4 * sp.pi * r)
    return pde, g, "Laplace 3D", ndim, var


def _make_helmholtz_3d():
    ndim = 3
    var = _make_sympy_vec("x", ndim)
    w = make_identity_diff_op(ndim)
    pde = laplacian(w) + w  # k=1
    r = sp.sqrt(var[0]**2 + var[1]**2 + var[2]**2)
    g = sp.exp(sp.I * r) / (4 * sp.pi * r)
    return pde, g, "Helmholtz 3D", ndim, var


KERNELS = {
    "laplace": (_make_laplace_2d, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
    "helmholtz": (_make_helmholtz_2d, [1, 3, 5, 7, 8]),
    "laplace3d": (_make_laplace_3d, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
    "helmholtz3d": (_make_helmholtz_3d, [1, 3, 5, 7, 9, 11]),
}


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_flop_data(kernel_key):
    """Compute flop counts for all three cases and all orders."""
    make_kernel, orders = KERNELS[kernel_key]
    pde, g, name, ndim, var = make_kernel()

    print(f"\n{'='*60}")
    print(f"Computing flop data for {name}")
    print(f"Orders: {orders}")
    print(f"{'='*60}")

    results = {
        "kernel": kernel_key,
        "name": name,
        "orders": orders,
        "flops_no_rot_no_rec": [],
        "flops_rot_no_rec": [],
        "flops_rot_rec": [],
    }

    derivs_cache = None  # shared across orders for case 2

    for p in orders:
        print(f"\n--- Order p={p} ---")

        # Case 2: rotation, no recurrence (compute first to build derivs_cache)
        print("    Case 2: rotation, no recurrence")
        t0 = time.time()
        lt2, derivs_cache = build_line_taylor_rotation_no_recurrence(
            g, p, var, derivs_cache
        )
        flops2 = count_flops(lt2)
        dt2 = time.time() - t0
        results["flops_rot_no_rec"].append(flops2)
        print(f"      -> {flops2} flops ({dt2:.1f}s)")

        # Case 3: rotation + recurrence
        print("    Case 3: rotation + recurrence")
        t0 = time.time()
        lt3 = build_line_taylor_rotation_recurrence(g, pde, p, var,
                                                     derivs_cache)
        flops3 = count_flops(lt3)
        dt3 = time.time() - t0
        results["flops_rot_rec"].append(flops3)
        print(f"      -> {flops3} flops ({dt3:.1f}s)")

        # Case 1: no rotation, no recurrence
        print("    Case 1: no rotation, no recurrence")
        t0 = time.time()
        lt1 = build_line_taylor_no_rotation(g, p, ndim, var)
        flops1 = count_flops(lt1)
        dt1 = time.time() - t0
        results["flops_no_rot_no_rec"].append(flops1)
        print(f"      -> {flops1} flops ({dt1:.1f}s)")

    return results


def save_data(results, output_dir):
    """Save computed flop data to JSON for later reuse."""
    path = os.path.join(output_dir, f"cost_data_{results['kernel']}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Data saved: {path}")


def load_data(kernel_key, output_dir):
    """Load cached flop data if available."""
    path = os.path.join(output_dir, f"cost_data_{kernel_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_cost_figure(results):
    """Generate a single-kernel flop count comparison figure.

    Returns the matplotlib Figure.
    """
    orders = results["orders"]
    name = results["name"]

    fig, ax = plt.subplots(1, 1, figsize=(3.35, 3.49))

    ax.plot(orders, results["flops_no_rot_no_rec"], "d-",
            label="No Rotation, No Recurrence", color="gray")
    ax.plot(orders, results["flops_rot_no_rec"], "s-",
            label="Rotation, No Recurrence", color="tab:orange")
    ax.plot(orders, results["flops_rot_rec"], "o-",
            label="Rotation, Recurrence", color="tab:blue")

    ax.set_yscale("log")
    ax.set_xlabel("Line-Taylor Order")
    ax.set_ylabel("Flop Count")
    ax.set_title(f"{name} Line-Taylor Flop Comparison")
    ax.set_xticks(orders)
    ax.set_xticklabels([str(o) for o in orders])
    ax.set_xlim(orders[0] - 0.5, orders[-1] + 0.5)
    ax.legend()

    return fig


def _make_combined_panel(ax, subset_results, ymax=1e4):
    """Plot a set of kernels on a single axes."""
    kernel_colors = {
        "laplace": "tab:blue",
        "helmholtz": "tab:red",
        "laplace3d": "tab:green",
        "helmholtz3d": "tab:brown",
    }
    kernel_labels = {
        "laplace": "Laplace",
        "helmholtz": "Helmholtz",
        "laplace3d": "Laplace",
        "helmholtz3d": "Helmholtz",
    }
    cases = [
        ("flops_no_rot_no_rec", "No Rot, No Rec", "-.", "d"),
        ("flops_rot_no_rec", "Rot, No Rec", "--", "s"),
        ("flops_rot_rec", "Rot + Rec", "-", "o"),
    ]

    for key, results in subset_results.items():
        orders = results["orders"]
        color = kernel_colors.get(key, "black")
        klabel = kernel_labels.get(key, key)
        for data_key, case_label, ls, marker in cases:
            flops = results[data_key]
            n = len(flops)
            ord_arr = np.array(orders[:n], dtype=float)
            flop_arr = np.array(flops, dtype=float)

            if len(ord_arr) >= 2 and np.all(flop_arr > 0):
                coeffs = np.polyfit(np.log(ord_arr), np.log(flop_arr), 1)
                slope_str = f", slope={coeffs[0]:.1f}"
            else:
                slope_str = ""

            ax.plot(ord_arr, flop_arr,
                    linestyle=ls, marker=marker, color=color,
                    label=f"{klabel}: {case_label}{slope_str}",
                    markersize=5)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Line-Taylor Order")
    ax.set_ylabel("Flop Count")
    ax.set_ylim(top=ymax)

    all_orders = sorted(set(
        o for r in subset_results.values() for o in r["orders"]
    ))
    ax.set_xticks(all_orders)
    ax.set_xticklabels([str(o) for o in all_orders])
    ax.set_xlim(all_orders[0] * 0.8, all_orders[-1] * 1.1)

    ax.legend(fontsize=6, ncol=2, handlelength=3, loc="upper left")


def make_combined_cost_figure_2d(all_results):
    """Generate the 2D cost figure (Laplace 2D + Helmholtz 2D)."""
    subset = {k: v for k, v in all_results.items() if "3d" not in k}
    if not subset:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    _make_combined_panel(ax, subset)
    plt.grid()
    ax.set_title("Line-Taylor Flop Comparison (2D)")
    return fig


def make_combined_cost_figure_3d(all_results):
    """Generate the 3D cost figure (Laplace 3D + Helmholtz 3D)."""
    subset = {k: v for k, v in all_results.items() if "3d" in k}
    if not subset:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    _make_combined_panel(ax, subset)
    plt.grid()
    ax.set_title("Line-Taylor Flop Comparison (3D)")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = ["all"]

    if "all" in keys:
        keys = list(KERNELS.keys())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for key in keys:
        if key not in KERNELS:
            print(f"Unknown kernel: {key}. Choose from {list(KERNELS.keys())}")
            sys.exit(1)

        # Check for cached data
        cached = load_data(key, output_dir)
        if cached:
            print(f"Using cached data for {key}")
            results = cached
        else:
            results = compute_flop_data(key)
            save_data(results, output_dir)

        all_results[key] = results

        # Individual figure
        fig = make_cost_figure(results)
        out_path = os.path.join(output_dir, f"cost-{key}.pdf")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved: {out_path}")
        plt.close(fig)

    # Combined 2D and 3D figures
    if len(all_results) > 1:
        for suffix, make_fn in [("2d", make_combined_cost_figure_2d),
                                ("3d", make_combined_cost_figure_3d)]:
            fig = make_fn(all_results)
            if fig is not None:
                out_path = os.path.join(output_dir, f"cost-combined-{suffix}.pdf")
                fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
                print(f"Saved: {out_path}")
                plt.close(fig)
