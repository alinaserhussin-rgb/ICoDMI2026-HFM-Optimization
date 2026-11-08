# Supplementary experiment outputs — ICoDMI 2026

The six CSV files are currently stored in the repository root (the analysis script also supports an optional `data/` directory). These archived CSVs document supplementary tests carried out during the revision of the comparative GD+M/FLO/HFM study. They are supplied for transparency and independent inspection.

## Included datasets

- `reviewer3_robustness_raw.csv`: 30 runs per algorithm for Rosenbrock 50D and 100D, and Rosenbrock 2D under gradient noise standard deviations 0, 0.01, 0.1 and 1.0.
- `reviewer3_robustness_summary.csv`: previously computed descriptive summary of those tests.
- `fuzzy_threshold_sensitivity_raw.csv` and `fuzzy_threshold_sensitivity_summary.csv`: recorded fuzzy-threshold sensitivity outcomes.
- `matched_1001_calls_raw.csv` and `matched_1001_calls_summary.csv`: instrumented objective-plus-gradient-call budget comparison.
- `analyze_additional_experiments.py`: Python standard-library script to inspect and summarize the archived raw records.

## Usage

```bash
python analyze_additional_experiments.py
```

The analysis script needs no external Python packages. It summarizes existing experiment records; **it does not regenerate the underlying optimization runs**. The full generation script for these supplementary experiments is not included, and these files should not be represented as complete end-to-end reproduction of the supplementary results. A recorded non-finite objective indicates numerical failure, not successful convergence. Instrumented function-call totals are not the same as equivalent computational cost for every algorithm.

The main optimizer source is `optimization_full_code_HFM.py` in the repository root. Its original experiment scope is 2D and 10D benchmarks. The HFM name replaces the historical HNF label; it does not introduce neural-network training.
