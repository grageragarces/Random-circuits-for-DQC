"""
Mann-Whitney U test: do real, random, and generated circuits yield
statistically different partitioning cost distributions?

Reads partitioning_results.csv, groups normalised cut costs by circuit origin
category, then runs pairwise Mann-Whitney U tests with effect sizes.

Run from the repo root:
    python3 mannwhitney_test.py

Expected output (approximate):
    Rows loaded: 337988
    After filtering (no SKIPPED, no random-baseline): ...

    === Pairwise Mann-Whitney U tests on normalised cut cost ===

    Real vs Random
      U = ...  p = ...  r = ...  (large effect)

    Real vs Generated
      U = ...  p = ...  r = ...  (large/medium effect)

    Random vs Generated
      U = ...  p = ...  r = ...
"""

import os
import csv
import math
from collections import defaultdict

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    raise SystemExit("scipy is required: pip install scipy")

REPO_ROOT   = os.path.dirname(__file__)
RESULTS_CSV = os.path.join(REPO_ROOT, "partitioning_results.csv")

# Map raw origin tags → broad category
CATEGORY = {
    "mqt":                   "real",
    "quipper":               "real",
    "qgen":                  "generated",
    "randomqiskit":          "random",
    "randomfromgraphqiskit": "random",
}

# Strategies to exclude (random baseline is not a real partitioner)
EXCLUDE_STRATEGIES = {"random"}

# Value used by the benchmark to flag a failed/skipped run
SKIP_SENTINEL = "SKIPPED"


def rank_biserial(U, n1, n2):
    """Effect size r = 1 - 2U/(n1*n2), range [-1, 1]."""
    return 1.0 - (2.0 * U) / (n1 * n2)


def effect_label(r):
    a = abs(r)
    if a >= 0.5:
        return "large"
    if a >= 0.3:
        return "medium"
    if a >= 0.1:
        return "small"
    return "negligible"


def main():
    costs = defaultdict(list)   # category -> [normalised cost, ...]

    total = 0
    kept  = 0
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            strategy = row["partitioning_strategy"].strip().lower()
            origin   = row["origin_of_circuit"].strip().lower()
            cost_str = row["coms_cost"].strip()

            if strategy in EXCLUDE_STRATEGIES:
                continue
            if cost_str == SKIP_SENTINEL or cost_str == "":
                continue

            category = CATEGORY.get(origin)
            if category is None:
                continue

            try:
                cost = float(cost_str)
            except ValueError:
                continue

            # Normalise by circuit qubit count so costs are comparable
            # across different circuit sizes
            try:
                n_qubits = int(row["n_qubits_circuit"])
            except (ValueError, KeyError):
                n_qubits = 1  # fallback: no normalisation

            costs[category].append(cost / max(n_qubits, 1))
            kept += 1

    print(f"Rows loaded:  {total:,}")
    print(f"Rows kept:    {kept:,}  (excluded SKIPPED, random-baseline strategy, unknown origin)\n")

    for cat, vals in costs.items():
        print(f"  {cat:12s}: {len(vals):,} measurements  "
              f"mean={sum(vals)/len(vals):.4f}  "
              f"median={sorted(vals)[len(vals)//2]:.4f}")

    print()
    print("=" * 60)
    print("Pairwise Mann-Whitney U tests on normalised cut cost")
    print("(two-sided, alternative='two-sided')")
    print("=" * 60)

    pairs = [
        ("real",      "random"),
        ("real",      "generated"),
        ("random",    "generated"),
    ]

    for cat_a, cat_b in pairs:
        a = costs[cat_a]
        b = costs[cat_b]
        if not a or not b:
            print(f"\n{cat_a} vs {cat_b}: insufficient data")
            continue

        U, p = mannwhitneyu(a, b, alternative="two-sided")
        r    = rank_biserial(U, len(a), len(b))
        eff  = effect_label(r)

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        print(f"\n{cat_a.capitalize()} vs {cat_b.capitalize()}")
        print(f"  n₁={len(a):,}  n₂={len(b):,}")
        print(f"  U = {U:.3e}")
        print(f"  p = {p:.3e}  {sig}")
        print(f"  r = {r:+.3f}  ({eff} effect)")

    print()
    print("Effect size interpretation (rank-biserial r):")
    print("  |r| >= 0.5 → large   |r| >= 0.3 → medium   |r| >= 0.1 → small")


if __name__ == "__main__":
    main()
