# ICoDMI 2026 — HFM Benchmark Reproducibility

Companion code and experiment data for **“Comparative Analysis of Gradient-Based, Fuzzy, and Hybrid Fuzzy–Momentum Optimization Methods on Non-Convex Benchmark Functions”** (Paper #11).

## Complete reproducibility package

**[Download the complete code, experiment logs, and figures (ZIP)](./HFM-benchmark_code%20%281%29.zip)**

This is the **canonical package for the revised paper**. It contains the reference optimizers, experiment-generation scripts, numerical analysis, archived CSV results, and the plots used in the manuscript. The older `optimization_full_code.py` and `optimization_full_code_HFM.py` at the repository root are earlier standalone benchmark scripts; **use the ZIP package for the revised manuscript's full experiments**.

## Package contents

- `optimizers.py` — objective functions, analytic gradients, GD+M, FLO and HFM
- `experiments.py` — generates the experiment CSV files
- `analyze.py` and `tables_json.py` — numerical summaries, statistics and table data
- `figures.py` and `arch.py` — figures
- `results/E1_main.csv` through `results/E9_betamin.csv` (nine CSV datasets)
- `results/summary.md` and four PNG figures

## Run the code

Extract the ZIP file, open a terminal in the extracted folder, and run:

```bash
python -m pip install numpy scipy pandas matplotlib
python analyze.py results
python figures.py results
python arch.py
```

To **regenerate the experimental observations** rather than analyze the included archived CSV results, first run:

```bash
python experiments.py results
```

This overwrites the corresponding CSV files in `results/`. Retain a separate copy of the original ZIP for comparison.

## Experimental design

- Main benchmark protocol: 30 paired starting points, max 500 iterations and gradient-norm tolerance `1e-8`
- 2D initialization seeds: 0–29; higher-dimensional test seeds: 1000–1029
- Separate tuning seeds: 500–509 (2D) and 1500–1509 (higher dimensions)
- Himmelblau extension: 100 paired seeds (0–99)
- Noise RNG seed: initialization seed + 10000
- Gradient-based baseline: learning rate 0.001 and momentum 0.9
- Fuzzy membership breakpoints: 50, 200 and 500; singleton step sizes: 0.005, 0.002 and 0.0005
- HFM: minimum momentum 0.3, maximum momentum 0.9, EMA coefficient 0.95 and attenuation factor 0.5

The full protocol comprises **11,640 optimizer executions**, including **4,200 configuration-tuning executions**. The archive includes **7,440 individual-run CSV records** and **420 aggregated tuning-configuration records**; the individual tuning executions are not stored separately. The scripts permit regeneration.

The bounded-step result in the manuscript **is not a general convergence guarantee**. Training ANFIS parameters and mini-batch objectives are outside the study's scope.

## Contact

Ali Nasser Al-Moussawi and Audie Sabrie Abd Al-Razzak
