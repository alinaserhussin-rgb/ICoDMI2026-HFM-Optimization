# ICoDMI 2026 — Optimization Experiments

Python implementation accompanying the comparative study of Gradient Descent with Momentum (GD+M), Fuzzy Logic Optimization (FLO), and Hybrid Fuzzy–Momentum (HFM).

## Requirements

Python 3.9 or newer, NumPy, SciPy, and Matplotlib.

## Installation

```bash
python -m pip install numpy scipy matplotlib
```

## Run

```bash
python optimization_full_code_HFM.py
```

The script performs 30 paired runs per benchmark with fixed random seeds, reports final objective values and Wilcoxon signed-rank tests, and generates benchmark figures.

## Experimental settings

- Random initialization seeds: 0–29 for 2D benchmarks and 1000–1029 for Rosenbrock 10D.
- Maximum iterations: 500.
- Gradient-norm stopping tolerance: 1e-8.
- GD+M learning rate: 0.001; momentum: 0.9.
- HFM minimum momentum: 0.3.
- Fuzzy gradient-norm thresholds: 50, 200, and 500.
- Fuzzy learning-rate singleton outputs: 0.005, 0.002, and 0.0005.

## Reproducibility note

The main script reproduces the baseline benchmark experiments. Additional experiments reported in the revised manuscript, including higher-dimensional and noise tests, require their corresponding scripts and data to be made available separately.

## Authors

Ali Nasser Al-Moussawi and Uday Abdulrazzaq.
