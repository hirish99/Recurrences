# Recurrences

Reproduction scripts for the paper:

> **Fast Evaluation of Derivatives of Green's Functions Using Recurrences**
> Hirish Chandrasekaran, Andreas Kloeckner

The recurrence library code lives in
[`sumpy.recurrence`](https://github.com/hirish99/sumpy/blob/parametric_recurrence/sumpy/recurrence.py)
and
[`sumpy.recurrence_qbx`](https://github.com/hirish99/sumpy/blob/parametric_recurrence/sumpy/recurrence_qbx.py)
on the `parametric_recurrence` branch of [hirish99/sumpy](https://github.com/hirish99/sumpy/tree/parametric_recurrence).

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/hirish99/Recurrences.git
cd Recurrences

# 2. Clone the recurrence branch of sumpy
git clone --single-branch --branch parametric_recurrence \
    https://github.com/hirish99/sumpy.git sumpy-parametric

# 3. Create conda environment and install dependencies
conda create -n recurrences python=3.11 -y
conda activate recurrences
pip install -e sumpy-parametric
pip install --force-reinstall --no-deps git+https://github.com/inducer/pymbolic.git@45bea2e
pip install matplotlib scipy

# 4. Install a TeX distribution (needed for PGF figure export)
#    The matplotlib PGF backend requires pdflatex at figure generation time.
#    Ubuntu/Debian:
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
#    macOS:
# brew install --cask mactex
```

## Regenerate all figures

```bash
conda activate recurrences
python regenerate_all.py
```

This regenerates all figures as PDFs into `output/`. You can also regenerate a subset:

```bash
python regenerate_all.py heatmaps   # error heatmaps only
python regenerate_all.py qbx        # QBX convergence only
python regenerate_all.py cost       # flop count figures only
```

## Scripts

### `plot_recurrence_error.py` -- Error heatmaps (Figures 3--5)

3-panel heatmaps showing relative error of the n-th derivative computed via
large-|x1| recurrence, small-|x1| expansion, and the hybrid approach, compared
to SymPy.

- Figure 3: Laplace 2D, 9th derivative
- Figure 4: Helmholtz 2D, 8th derivative
- Figure 5: Biharmonic 2D, 8th derivative

```bash
python plot_recurrence_error.py [laplace|helmholtz|biharmonic|all]
```

Options:
- `--grid N` -- grid resolution (default: 60, increase for smoother plots)
- `--p-offaxis N` -- small-|x1| Taylor expansion order (default: 12)
- `--xi N` -- blending threshold xi (default: 10)

### `plot_bestfit_and_cost.py` -- Best-fit error slope (Figure 6)

Relative error due to rounding in a single large-|x1| recurrence step vs
|x1|/xbar, with a linear least-squares fit in log-log space.

```bash
python plot_bestfit_and_cost.py bestfit
```

### `plot_cost.py` -- Flop count comparison (Figures 9--10)

Counts arithmetic operations (after CSE) for the line-Taylor expansion under
three configurations: no rotation, rotation without recurrence, rotation with
recurrence.

- Figure 9: 2D Laplace and Helmholtz
- Figure 10: 3D Laplace and Helmholtz

```bash
python plot_cost.py [laplace|helmholtz|all]
```

Flop data is cached to `output/cost_data_*.json`. Delete these files to
force recomputation.

### `plot_qbx.py` -- QBX convergence (Figure 11)

QBX boundary evaluation error on an ellipse, comparing recurrence-based vs
standard QBX against an analytic solution.

```bash
python plot_qbx.py
```

Options:
- `--orders 5 7 9 11` -- QBX orders to sweep
- `--resolutions 300 2000 4000` -- mesh resolutions (number of points)

### `plot_appendix.py` -- Asymptotic bounds (Figures 12--13)

Heatmaps validating the asymptotic derivative growth assumptions for
odd-order and even-order derivatives.

- Figure 12: odd derivative order mu=5
- Figure 13: even derivative order nu=6

```bash
python plot_appendix.py [c5|d6|all]
```

Options:
- `--grid N` -- grid resolution (default: 60)
