"""
Generate the QBX convergence figure (qbx.pgf) for the paper.

This script computes the relative error of evaluating the single layer potential
for the Laplace PDE in 2D on an ellipse, comparing:
  - u_qbxrec: QBX evaluation using the recurrence-based derivative computation
  - u_qbx:    Standard QBX evaluation (sumpy's built-in layer potential)

Both are compared against an analytically known true solution for an oscillating
source density on an ellipse.

The figure plots relative L_inf error vs mesh resolution h for QBX orders
p = 5, 7, 9, 11.

Output: scripts/output/qbx-convergence.pgf

Usage:
    conda activate inteq
    python scripts/plot_qbx.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Load recurrence modules from the parametric_recurrence branch.
# ---------------------------------------------------------------------------

import sumpy.recurrence as _recurrence
import sumpy.recurrence_qbx as _recurrence_qbx

_make_sympy_vec = _recurrence._make_sympy_vec
recurrence_qbx_lp = _recurrence_qbx.recurrence_qbx_lp

# Standard sumpy imports (from the installed main-branch sumpy)
from sumpy.array_context import _acf
from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import LaplaceKernel
from sumpy.qbx import LayerPotential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# QBX orders to sweep over
ORDERS = [5, 7, 9, 11]

# Number of panels on the ellipse (each panel has GL_ORDER nodes)
# h_panel = 2*pi / n_panels
RESOLUTIONS = [60, 200, 360]

# Ellipse semi-major axis (eccentricity = 1/a)
ELLIPSE_A = 2

# Mode number for the oscillating source density cos(mode_nr * t)
MODE_NR = 10

# Ellipse arc length (precomputed for a=2), used to convert n_p -> h
ELLIPSE_ARC_LENGTH = 9.68845

# Colors for each QBX order (one per entry in ORDERS)
COLORS = ["b", "g", "r", "c"]

# Small-|x1| expansion order used internally by recurrence_qbx_lp
P_OFFAXIS = 12

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


GL_ORDER = 16  # Gauss-Legendre order per panel


def create_ellipse(n_panels, a=ELLIPSE_A, mode_nr=MODE_NR, quad_convg_rate=10,
                   gl_order=GL_ORDER):
    """
    Create a composite Gauss-Legendre discretization of an ellipse
    (a*cos(t), sin(t)) with QBX expansion centers and an oscillating density.

    The interval [0, 2*pi] is split into n_panels equal panels, each
    discretized with gl_order Gauss-Legendre nodes.

    The QBX expansion radius is set to (h_panel / 4) * quad_convg_rate,
    where h_panel = 2*pi / n_panels is the panel size in parameter space.

    Returns:
        sources:  (2, n_total) array of source points on the ellipse
        centers:  (2, n_total) array of QBX expansion centers (offset inward)
        normals:  (2, n_total) array of outward unit normals
        density:  (n_total,) array, density = cos(mode_nr * t) / |gamma'(t)|
        jacobs:   (n_total,) array of Jacobian |gamma'(t)|
        weights:  (n_total,) array of composite GL quadrature weights
        radius:   float, QBX expansion radius
    """
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(gl_order)

    h_panel = 2 * np.pi / n_panels
    all_t = []
    all_w = []
    for k in range(n_panels):
        t_lo = k * h_panel
        # Map GL nodes from [-1, 1] to [t_lo, t_lo + h_panel]
        t_k = t_lo + (gl_nodes + 1) * h_panel / 2
        w_k = gl_weights * h_panel / 2
        all_t.append(t_k)
        all_w.append(w_k)

    t = np.concatenate(all_t)
    quad_weights = np.concatenate(all_w)

    phi = sp.symbols("phi")
    jacob_sym = sp.sqrt(a**2 * sp.sin(phi)**2 + sp.cos(phi)**2)
    jacob_fn = sp.lambdify(phi, jacob_sym)
    inv_jacob_fn = sp.lambdify(phi, 1 / jacob_sym)

    jacobs = jacob_fn(t)

    # QBX expansion radius based on panel size
    h_min = h_panel * np.min(jacobs)
    radius = (h_min / 4) * quad_convg_rate

    # Ellipse points
    sources = np.array([a * np.cos(t), np.sin(t)])

    # Outward unit normal
    normals = np.array([np.cos(t), a * np.sin(t)])
    normals = normals / np.linalg.norm(normals, axis=0)

    # Expansion centers: offset inward along the normal
    centers = sources - normals * radius

    # Density normalized by Jacobian (so quadrature weight = quad_weight * jacob)
    density = np.cos(mode_nr * t) * inv_jacob_fn(t)

    return sources, centers, normals, density, jacobs, quad_weights, radius


# ---------------------------------------------------------------------------
# True solution (analytic)
# ---------------------------------------------------------------------------


def laplace_2d_true_solution(density, jacobs, a=ELLIPSE_A, n=MODE_NR):
    """
    Analytic single layer potential for the Laplace equation on an ellipse
    with density cos(n*t).

    Uses the known eigenvalue mu_n = 1/(2n) * (1 + ((1-r)/(1+r))^n)
    for the single layer operator on an ellipse with eccentricity r = 1/a.
    """
    r = 1 / a
    mu_n = 1 / (2 * n) * (1 + ((1 - r) / (1 + r))**n)
    return mu_n * jacobs * density


# ---------------------------------------------------------------------------
# Standard QBX evaluation (sumpy built-in, no recurrence)
# ---------------------------------------------------------------------------

actx = _acf()
lknl2d = LaplaceKernel(2)


def qbx_lp_standard(sources, targets, centers, radius, strengths, order):
    """
    Evaluate the layer potential using sumpy's built-in QBX (no recurrence).

    This uses the LineTaylorLocalExpansion from the installed sumpy, which
    computes derivatives symbolically rather than via recurrence.
    """
    lpot = LayerPotential(
        expansion=LineTaylorLocalExpansion(lknl2d, order),
        target_kernels=(lknl2d,),
        source_kernels=(lknl2d,),
    )
    expansion_radii = actx.from_numpy(radius * np.ones(sources.shape[1]))
    result = lpot(
        actx,
        targets=actx.from_numpy(targets),
        sources=actx.from_numpy(sources),
        centers=actx.from_numpy(centers),
        strengths=(actx.from_numpy(strengths),),
        expansion_radii=expansion_radii,
    )
    return actx.to_numpy(result[0])


# ---------------------------------------------------------------------------
# Sweep: compute errors for all (order, resolution) combinations
# ---------------------------------------------------------------------------


def compute_errors(orders, resolutions):
    """
    For each (order, resolution) pair, compute the relative L_inf error of:
      - recurrence-based QBX  (err_rec)
      - standard QBX          (err_std)
    against the analytic true solution.

    Returns:
        err_rec:  list of lists, err_rec[i][j] = error for orders[i], resolutions[j]
        err_std:  same shape, for standard QBX
    """
    # Set up the Laplace 2D PDE and Green's function for the recurrence
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    greens_fn = (-1 / (2 * np.pi)) * sp.log(
        sp.sqrt((var[0] - var_t[0])**2 + (var[1] - var_t[1])**2)
    )

    err_rec = []
    err_std = []

    for p in orders:
        err_rec_p = []
        err_std_p = []
        for n_p in resolutions:
            print(f"  Order p={p}, resolution n_p={n_p}")
            sources, centers, normals, density, jacobs, weights, radius = \
                create_ellipse(n_p)
            strengths = jacobs * density * weights

            # Recurrence-based QBX
            rec_res = recurrence_qbx_lp(
                sources, centers, normals, strengths, radius,
                laplace2d, greens_fn, 2, p,
            )

            # Standard QBX
            std_res = qbx_lp_standard(
                sources, sources, centers, radius, strengths, p,
            )

            # True solution
            true_sol = laplace_2d_true_solution(density, jacobs)
            true_max = np.max(np.abs(true_sol))

            err_rec_p.append(np.max(np.abs(rec_res - true_sol)) / true_max)
            err_std_p.append(np.max(np.abs(std_res - true_sol)) / true_max)

        err_rec.append(err_rec_p)
        err_std.append(err_std_p)

    return err_rec, err_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_qbx_figure(err_rec, err_std, orders, resolutions):
    """
    Generate the QBX convergence scatter plot.

    x-axis: mesh spacing h = arc_length / n_p
    y-axis: relative L_inf error (log scale)

    Returns the matplotlib Figure.
    """
    h_values = 2 * np.pi / np.array(resolutions)  # panel size in parameter space

    fig, ax = plt.subplots(1, 1, figsize=(3.83, 3.83))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for i, p in enumerate(orders):
        ax.scatter(
            h_values, err_rec[i],
            marker="+", c=COLORS[i], s=50,
            label=rf"$u = u_{{qbxrec}}$ ($p_{{QBX}}$={p})",
        )
        ax.scatter(
            h_values, err_std[i],
            marker="x", c=COLORS[i], s=50,
            label=rf"$u = u_{{qbx}}$ ($p_{{QBX}}$={p})",
        )

    ax.set_xlabel("Mesh Resolution ($h$)")
    ax.set_ylabel(r"$E(u) = \|u - u_{\mathrm{true}}\|_\infty / \|u_{\mathrm{true}}\|_\infty$")
    ax.set_title(r"$(u-u_{true})/u_{true}$")
    ax.set_xticks(h_values)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        "Laplace 2D: Ellipse SLP Boundary Evaluation Error"
        rf" ($\xi={10}$, $p_{{small}}={P_OFFAXIS}$)",
    )

    return fig


# ---------------------------------------------------------------------------
# Data save/load
# ---------------------------------------------------------------------------

import json


def save_qbx_data(err_rec, err_std, orders, resolutions, path):
    data = {
        "orders": orders,
        "resolutions": resolutions,
        "err_rec": err_rec,
        "err_std": err_std,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved data: {path}")


def load_qbx_data(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate QBX convergence figure.")
    parser.add_argument("--orders", type=int, nargs="+", default=ORDERS,
                        help=f"QBX orders (default: {ORDERS})")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS,
                        help=f"Mesh resolutions (default: {RESOLUTIONS})")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation even if cached data exists")
    args = parser.parse_args()

    ORDERS = args.orders
    RESOLUTIONS = args.resolutions

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qbx-convergence.pdf")
    data_path = os.path.join(output_dir, "qbx_data.json")

    cached = None if args.recompute else load_qbx_data(data_path)
    if cached and cached["orders"] == ORDERS and cached["resolutions"] == RESOLUTIONS:
        print("Using cached QBX data")
        err_rec = cached["err_rec"]
        err_std = cached["err_std"]
    else:
        print("Computing QBX convergence data...")
        print(f"  Orders: {ORDERS}")
        print(f"  Resolutions: {RESOLUTIONS}")
        err_rec, err_std = compute_errors(ORDERS, RESOLUTIONS)
        save_qbx_data(err_rec, err_std, ORDERS, RESOLUTIONS, data_path)

    print("Generating figure...")
    fig = make_qbx_figure(err_rec, err_std, ORDERS, RESOLUTIONS)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {output_path}")
    plt.close(fig)
