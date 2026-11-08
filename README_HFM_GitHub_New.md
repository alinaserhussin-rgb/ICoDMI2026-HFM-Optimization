# ICoDMI 2026 — GD+M, FLO and HFM benchmark experiments

This repository contains Python code for the comparative experiments on gradient descent with momentum (GD+M), fuzzy learning-rate optimization (FLO), and hybrid fuzzy–momentum optimization (HFM).

## Run

Requires Python 3.9+ and these packages:

```bash
python -m pip install numpy scipy matplotlib
python optimization_full_code_HFM.py
```

The script prints summary statistics and Wilcoxon tests and saves five PNG figures in the current directory. The program may take some time to finish.

## Reproducibility

- Six two-dimensional benchmark functions and a 10-dimensional Rosenbrock benchmark.
- 30 paired starting points per benchmark; seeds `0` through `29` for 2D, and `1000` through `1029` for 10D.
- Maximum 500 iterations, stopping threshold for gradient norm `1e-8`.
- GD+M: learning rate `0.001`, momentum `0.9`.
- FLO: weighted-singleton fuzzy learning-rate rules.
- HFM: fuzzy learning rate, cosine-adaptive momentum (minimum `0.3`), and EMA step attenuation.
- The Python function `hnf()` retains its historical identifier to preserve traceability to the original experimental code; its scientific designation is **HFM**, because no neural-network training occurs.

## Scope

This script reproduces the original 2D/10D benchmark experiments. Additional experiments reported in any revised manuscript (e.g., 50D/100D, gradient noise, tuning) require their corresponding separate scripts and are **not** reproduced by this file alone.

Non-finite optimizer states are handled using the original script's `safe()` sentinel convention; these should not be interpreted as successful convergence.
