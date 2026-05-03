"""
Two-qubit gate density analysis across circuit origins.

Only analyses circuits that appear in partitioning_results.csv (i.e. those
that passed all filters and were actually benchmarked).

Run from the repo root:
    python3 twoqubit_density.py


Circuits in results CSV: 6508
Unique monolithic circuits tested: 6508

Origin                           N circuits    Avg 2q%      Std      Min      Max
----------------------------------------------------------------------------------
Random (Qiskit)                        1815      22.6%    12.2%     0.0%    61.8%
Random (Graph-based)                   1815      66.1%     2.0%    43.9%    71.6%
Real (MQT)                             1760      76.0%    23.8%    21.7%    99.2%
Real (Quipper)                            2      66.2%    33.8%    32.4%   100.0%
Generated (QGen)                       1116       4.7%    10.9%     0.0%    33.3%

--- Random circuit 2q density by qubit count ---

Random (Qiskit):
  n= 10: 16.4% ± 14.2%  (15 circuits)
  n= 11: 20.2% ± 12.1%  (15 circuits)
  n= 12: 17.2% ± 11.3%  (15 circuits)
  n= 13: 18.6% ± 12.1%  (15 circuits)
  n= 14: 22.9% ± 12.1%  (15 circuits)
  n= 15: 23.8% ± 9.7%  (15 circuits)
  n= 16: 22.4% ± 13.3%  (15 circuits)
  n= 17: 21.3% ± 11.9%  (15 circuits)
  n= 18: 17.7% ± 7.9%  (15 circuits)
  n= 19: 20.8% ± 14.9%  (15 circuits)
  n= 20: 21.0% ± 10.2%  (15 circuits)
  n= 21: 24.8% ± 8.7%  (15 circuits)

Random (Graph-based):
  n= 10: 57.0% ± 6.7%  (15 circuits)
  n= 11: 57.9% ± 4.2%  (15 circuits)
  n= 12: 59.8% ± 4.8%  (15 circuits)
  n= 13: 61.3% ± 4.1%  (15 circuits)
  n= 14: 63.2% ± 2.2%  (15 circuits)
  n= 15: 63.2% ± 2.5%  (15 circuits)
  n= 16: 64.8% ± 2.1%  (15 circuits)
  n= 17: 64.5% ± 2.1%  (15 circuits)
  n= 18: 64.3% ± 2.5%  (15 circuits)
  n= 19: 64.3% ± 2.1%  (15 circuits)
  n= 20: 64.9% ± 2.7%  (15 circuits)
  n= 21: 64.4% ± 2.2%  (15 circuits)
"""

import os
import csv
import math
from collections import defaultdict

REPO_ROOT     = os.path.dirname(__file__)
CIRCUITS_DIR  = os.path.join(REPO_ROOT, "Circuits")
RESULTS_CSV   = os.path.join(REPO_ROOT, "partitioning_results.csv")

TWO_QUBIT_PREFIXES = [
    "cx ", "cz ", "cy ", "ch ",
    "crz", "crx", "cry",
    "cu1", "cu3", "cu ",
    "ccx",
    "swap",
    "rzz", "rxx", "ryy",
    "ecr", "dcx", "iswap",
]

ORIGIN_MAP = {
    "randomqiskit":          "Random (Qiskit)",
    "randomfromgraphqiskit": "Random (Graph-based)",
    "mqt":                   "Real (MQT)",
    "quipper":               "Real (Quipper)",
    "qgen":                  "Generated (QGen)",
}


def get_tested_circuits():
    """Return set of (circuit_partitioned, origin) pairs from the results CSV."""
    tested = {}  # circuit_partitioned -> origin_of_circuit
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            name = row["circuit_partitioned"]
            origin = row["origin_of_circuit"]
            tested[name] = origin
    return tested


def count_gates(filepath):
    """Return (total_gates, two_qubit_gates) for a QASM file."""
    skip_prefixes = ("//", "OPENQASM", "include", "qreg", "creg",
                     "measure", "barrier", "reset")
    total = 0
    two_q = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(p) for p in skip_prefixes):
                continue
            total += 1
            if any(line.startswith(p) for p in TWO_QUBIT_PREFIXES):
                two_q += 1
    return total, two_q


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    s = math.sqrt(sum((x - m) ** 2 for x in values) / n)
    return m, s


def main():
    tested = get_tested_circuits()
    print(f"Circuits in results CSV: {len(tested)}")

    # Only keep monolithic circuits (the input circuits, not distributed ones)
    monolithic_tested = {name: origin for name, origin in tested.items()
                         if name.startswith("monolithic")}
    print(f"Unique monolithic circuits tested: {len(monolithic_tested)}\n")

    densities = defaultdict(list)  # origin_tag -> list of densities
    skipped = 0

    for circuit_name, origin in monolithic_tested.items():
        fpath = os.path.join(CIRCUITS_DIR, circuit_name + ".qasm")
        if not os.path.exists(fpath):
            skipped += 1
            continue

        total, two_q = count_gates(fpath)
        if total == 0:
            skipped += 1
            continue

        densities[origin].append(two_q / total)

    print(f"{'Origin':<30} {'N circuits':>12} {'Avg 2q%':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 82)

    for tag, label in ORIGIN_MAP.items():
        vals = densities.get(tag, [])
        if not vals:
            print(f"{label:<30} {'0':>12}")
            continue
        m, s = mean_std(vals)
        print(f"{label:<30} {len(vals):>12} {m*100:>9.1f}% {s*100:>7.1f}%"
              f" {min(vals)*100:>7.1f}% {max(vals)*100:>7.1f}%")

    if skipped:
        print(f"\n(Skipped {skipped} circuits — file not found or empty)")

    # Per-qubit-size breakdown for random origins
    print("\n--- Random circuit 2q density by qubit count ---")
    for tag in ("randomqiskit", "randomfromgraphqiskit"):
        label = ORIGIN_MAP[tag]
        by_size = defaultdict(list)
        for circuit_name, origin in monolithic_tested.items():
            if origin != tag:
                continue
            fpath = os.path.join(CIRCUITS_DIR, circuit_name + ".qasm")
            if not os.path.exists(fpath):
                continue
            # Extract qubit count from filename (last numeric segment before .qasm)
            try:
                n_qubits = int(circuit_name.rstrip(".qasm").split("__")[-1])
            except ValueError:
                continue
            total, two_q = count_gates(fpath)
            if total > 0:
                by_size[n_qubits].append(two_q / total)

        print(f"\n{label}:")
        for n in sorted(by_size)[:12]:
            m, s = mean_std(by_size[n])
            print(f"  n={n:3d}: {m*100:.1f}% ± {s*100:.1f}%  ({len(by_size[n])} circuits)")


if __name__ == "__main__":
    main()
