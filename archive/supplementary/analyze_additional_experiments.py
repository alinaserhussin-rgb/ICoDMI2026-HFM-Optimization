"""Summarize archived ICoDMI 2026 supplementary experiment CSV files.
This script analyzes stored outputs; it does NOT regenerate optimizer runs.
Run from the repository root; CSV files may reside in the root or in data/.
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(name):
    candidates = (ROOT / name, ROOT / 'data' / name)
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot find {name} in repository root or data/")
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def summary(rows, group_cols, value_col):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_cols)].append(row)
    for key, items in sorted(groups.items()):
        vals = [float(r[value_col]) for r in items]
        finite = [v for v in vals if math.isfinite(v)]
        mean = statistics.fmean(finite) if finite else float('nan')
        med = statistics.median(finite) if finite else float('nan')
        print(' | '.join(key), '| n=',len(vals),'nonfinite=',len(vals)-len(finite),
              'finite mean=',format(mean,'.6g'),'finite median=',format(med,'.6g'))

if __name__ == '__main__':
    print('=== High-dimensional and gradient-noise runs (archived data) ===')
    summary(load('reviewer3_robustness_raw.csv'),
            ('benchmark','gradient_noise_sd','algorithm'),'final_objective')
    print('\n=== Fuzzy threshold sensitivity (archived data) ===')
    summary(load('fuzzy_threshold_sensitivity_raw.csv'),
            ('threshold_multiplier',),'final_objective')
    print('\n=== Matched instrumented call budget (archived data) ===')
    summary(load('matched_1001_calls_raw.csv'),
            ('benchmark','algorithm'),'final_objective')
