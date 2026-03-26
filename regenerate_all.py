"""
Regenerate all paper figures into output/.

This is a wrapper that imports each plotting script and runs the figure
generation.

Usage:
    conda activate recurrences
    python regenerate_all.py
"""

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
})

import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def patch_savefig(fig, name):
    """Save figure as PDF into output/."""
    pdf_path = os.path.join(OUTPUT_DIR, name + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    print(f"  Saved PDF: {pdf_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    what = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in what:
        what = ["qbx", "heatmaps", "bestfit", "cost", "appendix"]

    # --- QBX convergence ---
    if "qbx" in what:
        print("\n=== QBX convergence ===")
        import plot_qbx
        err_rec, err_std = plot_qbx.compute_errors(
            plot_qbx.ORDERS, plot_qbx.RESOLUTIONS
        )
        fig = plot_qbx.make_qbx_figure(err_rec, err_std,
                                        plot_qbx.ORDERS, plot_qbx.RESOLUTIONS)
        patch_savefig(fig, "qbx-convergence")
        plt.close(fig)

    # --- Error heatmaps ---
    if "heatmaps" in what:
        import plot_recurrence_error as pre
        for key in ["laplace", "helmholtz", "biharmonic"]:
            print(f"\n=== Error heatmap: {key} ===")
            _, output_name = pre.PDE_REGISTRY[key]
            err_large, err_small, err_blend, x1v, x2v, pde_name, deriv_order = \
                pre.compute_heatmap_data(key)
            fig = pre.make_heatmap_figure(err_large, err_small, err_blend,
                                          x1v, x2v, pde_name, deriv_order)
            patch_savefig(fig, output_name)
            plt.close(fig)

    # --- Bestfit ---
    if "bestfit" in what:
        print("\n=== Bestfit ===")
        import plot_bestfit_and_cost as pbc
        fig = pbc.generate_bestfit()
        patch_savefig(fig, "error-bestfit-laplace")
        plt.close(fig)

    # --- Cost ---
    if "cost" in what:
        print("\n=== Cost figures ===")
        import plot_cost
        all_results = {}
        for key in plot_cost.KERNELS:
            cached = plot_cost.load_data(key, OUTPUT_DIR)
            if cached:
                results = cached
            else:
                results = plot_cost.compute_flop_data(key)
                plot_cost.save_data(results, OUTPUT_DIR)
            all_results[key] = results
            fig = plot_cost.make_cost_figure(results)
            patch_savefig(fig, f"cost-{key}")
            plt.close(fig)

        # Combined figure (used by Paper.tex)
        if len(all_results) > 1:

            for suffix, make_fn in [("2d", plot_cost.make_combined_cost_figure_2d),
                                    ("3d", plot_cost.make_combined_cost_figure_3d)]:
                fig = make_fn(all_results)
                patch_savefig(fig, f"cost-combined-{suffix}")
                plt.close(fig)

    # --- Appendix ---
    if "appendix" in what:
        print("\n=== Appendix figures ===")
        import plot_appendix
        for key, gen_fn, name in [
            ("c5", plot_appendix.generate_mu5, "asymptotic-mu5"),
            ("d6", plot_appendix.generate_nu6, "asymptotic-nu6"),
        ]:
            fig = gen_fn()
            patch_savefig(fig, name)
            plt.close(fig)

    print("\n=== Done ===")
