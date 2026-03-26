"""
Generate the appendix asymptotic behavior figures for the paper (Figures 11--12):
  - asymptotic-c5.pgf: Asymptotic behavior of odd-order (c=5) derivatives
  - asymptotic-d6.pgf: Asymptotic behavior of even-order (d=6) derivatives

Each figure is a 3-panel heatmap (Laplace 2D, Helmholtz 2D, Biharmonic 2D)
showing the ratio of the derivative magnitude to its predicted asymptotic
bound, validating the assumptions in the paper.

Output: scripts/output/{asymptotic-c5,asymptotic-d6}.pdf

Usage:
    conda activate inteq
    python scripts/plot_appendix.py [c5|d6|all]
"""

from __future__ import annotations

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
# Load recurrence modules
# ---------------------------------------------------------------------------

import sumpy.recurrence as _recurrence

_make_sympy_vec = _recurrence._make_sympy_vec
from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

NDIM = 2
var = _make_sympy_vec("x", NDIM)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_N = 60
GRID_LOG_MIN = -8
GRID_LOG_MAX = 0
CBAR_VMIN = 1e-2
CBAR_VMAX = 1e3

# ---------------------------------------------------------------------------
# PDE / Green's function definitions
# ---------------------------------------------------------------------------


def _make_greens_fns():
    """Return dict of (name, Green's function expression) for each PDE."""
    r = sp.sqrt(var[0]**2 + var[1]**2)
    return {
        "Laplace 2D": (-1 / (2 * sp.pi)) * sp.log(r),
        "Helmholtz 2D": sp.Rational(1, 4) * sp.I * sp.hankel1(0, r),
        "Biharmonic 2D": sp.Rational(1, 8) / sp.pi * r**2 * sp.log(r),
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_max_deriv_over_interval(g, deriv_order, x1_vals, x2_vals):
    """
    For each (x1, x2) grid point, approximate:
        max_{0 <= xi <= x1} |d^c/dx1^c G(xi, x2)|
    by evaluating the derivative at a set of xi samples in [0, x1].

    Returns a 2D array of shape (len(x2_vals), len(x1_vals)).
    """
    deriv_expr = sp.diff(g, var[0], deriv_order)
    deriv_fn = sp.lambdify([var[0], var[1]], deriv_expr,
                           modules=["scipy", "numpy"])

    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    result = np.zeros_like(X1)

    # Sample xi at several points in [0, x1] for each grid point
    n_samples = 20
    for k in range(n_samples + 1):
        frac = k / n_samples
        XI = frac * X1
        with np.errstate(all="ignore"):
            vals = np.abs(deriv_fn(XI, X2))
        if not isinstance(vals, np.ndarray):
            vals = vals * np.ones_like(X1)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        result = np.maximum(result, vals)

    return result


def _compute_ratio_c5(pde_name, max_deriv, x1_vals, x2_vals):
    """
    Compute the ratio for c=5 (odd derivative):
      Laplace/Helmholtz: |max deriv| / (|x1| / xbar^(c+1))
      Biharmonic:        |max deriv| / (|x1|^3 / xbar^(c+1))
    where xbar = x2, c = 5.
    """
    c = 5
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    xbar = X2  # off-axis coordinate

    if "Biharmonic" in pde_name:
        bound = np.abs(X1)**3 / xbar**(c + 1)
    else:
        bound = np.abs(X1) / xbar**(c + 1)

    with np.errstate(all="ignore"):
        ratio = max_deriv / bound
    ratio = np.where(np.isfinite(ratio), ratio, CBAR_VMAX)
    ratio = np.clip(ratio, CBAR_VMIN, CBAR_VMAX)
    # Mask out region where |x1|/xbar >= 1 (bound only holds when |x1| < xbar)
    ratio = np.where(np.abs(X1) < xbar, ratio, np.nan)
    return ratio


def _compute_ratio_d6(pde_name, max_deriv, x1_vals, x2_vals):
    """
    Compute the ratio for d=6 (even derivative):
      Laplace/Helmholtz: |max deriv| / (1 / xbar^d)
      Biharmonic:        |max deriv| / (|x1|^2 / xbar^d)
    where xbar = x2, d = 6.
    """
    d = 6
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    xbar = X2

    if "Biharmonic" in pde_name:
        bound = np.abs(X1)**2 / xbar**d
    else:
        bound = 1.0 / xbar**d

    with np.errstate(all="ignore"):
        ratio = max_deriv / bound
    ratio = np.where(np.isfinite(ratio), ratio, CBAR_VMAX)
    ratio = np.clip(ratio, CBAR_VMIN, CBAR_VMAX)
    # Mask out region where |x1|/xbar >= 1 (bound only holds when |x1| < xbar)
    ratio = np.where(np.abs(X1) < xbar, ratio, np.nan)
    return ratio


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _make_appendix_figure(ratios, pde_names, eqref_labels, title, output_path):
    """
    Generate a 3-panel heatmap for the appendix figure.
    """
    x1_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)
    x2_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.2), sharey=True)
    norm = mcolors.LogNorm(vmin=CBAR_VMIN, vmax=CBAR_VMAX)

    for ax, ratio, name, eqref in zip(axes, ratios, pde_names, eqref_labels):
        pcm = ax.pcolormesh(X1, X2, ratio, norm=norm, cmap="RdBu_r",
                            shading="auto", rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$x_1$-coordinate", fontsize=7)
        ax.set_title(f"{name} ({eqref})", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel("$x_2$-coordinate", fontsize=7)

    cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal",
                         fraction=0.04, pad=0.18, aspect=50)

    fig.suptitle(title, fontsize=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"  Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def generate_mu5():
    """Generate asymptotic-mu5 — odd derivative (mu=5) asymptotic behavior."""
    print("Generating mu5 figure (mu=5, odd derivatives)...")
    c = 5
    greens = _make_greens_fns()
    x1_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)
    x2_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)

    ratios = []
    pde_names = ["Laplace 2D", "Helmholtz 2D", "Biharmonic 2D"]
    eqref_labels = [
        "odd bound",
        "odd bound",
        "odd bound (biharm)",
    ]

    for name in pde_names:
        print(f"  Computing {name}...")
        t0 = time.time()
        max_deriv = _compute_max_deriv_over_interval(
            greens[name], c, x1_vals, x2_vals
        )
        ratio = _compute_ratio_c5(name, max_deriv, x1_vals, x2_vals)
        ratios.append(ratio)
        print(f"    done ({time.time()-t0:.1f}s)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "output", "asymptotic-mu5.pdf")
    return _make_appendix_figure(
        ratios, pde_names, eqref_labels,
        "Asymptotic Behavior of Derivatives ($\\mu=5$)", out
    )


def generate_nu6():
    """Generate asymptotic-nu6 — even derivative (d=6) asymptotic behavior."""
    print("Generating nu6 figure (d=6, even derivatives)...")
    d = 6
    greens = _make_greens_fns()
    x1_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)
    x2_vals = np.logspace(GRID_LOG_MIN, GRID_LOG_MAX, GRID_N)

    ratios = []
    pde_names = ["Laplace 2D", "Helmholtz 2D", "Biharmonic 2D"]
    eqref_labels = [
        "even bound",
        "even bound",
        "even bound (biharm)",
    ]

    for name in pde_names:
        print(f"  Computing {name}...")
        t0 = time.time()
        max_deriv = _compute_max_deriv_over_interval(
            greens[name], d, x1_vals, x2_vals
        )
        ratio = _compute_ratio_d6(name, max_deriv, x1_vals, x2_vals)
        ratios.append(ratio)
        print(f"    done ({time.time()-t0:.1f}s)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "output", "asymptotic-nu6.pdf")
    return _make_appendix_figure(
        ratios, pde_names, eqref_labels,
        "Asymptotic Behavior of Derivatives ($\\nu=6$)", out
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate appendix asymptotic figures.")
    parser.add_argument("figures", nargs="*", default=["all"],
                        choices=["all", "c5", "d6"],
                        help="Which figures to generate (default: all)")
    parser.add_argument("--grid", type=int, default=GRID_N,
                        help=f"Grid resolution (default: {GRID_N})")
    args = parser.parse_args()

    GRID_N = args.grid
    keys = args.figures
    if "all" in keys:
        keys = ["mu5", "nu6"]

    for key in keys:
        if key == "mu5":
            generate_mu5()
        elif key == "nu6":
            generate_nu6()
        else:
            print(f"Unknown: {key}.")
            sys.exit(1)
