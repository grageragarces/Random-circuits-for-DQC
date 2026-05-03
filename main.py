"""
Quantum Circuit Partitioning Pipeline
====================================
Currently uses Zoltan PHG (Parallel HyperGraph) partitioning via a compiled
C subprocess wrapper.  Random, greedy, and KaHyPar strategies are kept below
but commented out.

Strategy history: KaHyPar → greedy → random → zoltan_phg
Partitions all monolithic .qasm circuits in ./Circuits/ across QPU networks
of size k=2..10, subject to a minimum of 5 qubits per QPU.

Expected filename format:
  monolithic__{origin}__{circuit_identifier}__{n_qubits}.qasm

Circuit → Hypergraph:
  Vertices   = qubits
  Hyperedges = multi-qubit gates (single-qubit gates ignored)

Communication cost = number of hyperedges cut (kh.cut()).

Output CSV columns:
  partitioning_strategy | circuit_partitioned | origin_of_circuit |
  circuit_identifier    | n_qubits_circuit    | n_qpus | coms_cost
"""

import csv
import os
import pathlib
import random
import subprocess
import traceback
import multiprocessing
from typing import List, Tuple

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CIRCUITS_DIR       = "./Circuits"
OUTPUT_CSV         = "./partitioning_results.csv"
NUM_WORKERS        = 11
EPSILON            = 0.05
MIN_QUBITS_PER_QPU = 5
MAX_QUBITS         = 130    # circuits above this are skipped
MAX_HYPEREDGES     = 20000   # circuits with more multi-qubit gates than this are skipped
TIMEOUT_SECS       = 120    # if a job hangs longer than this, kill the worker and skip
STRATEGY_NAME      = "StocG"
RANDOM_SEED        = 42     # seed for reproducible random partitioning
# Path to the compiled Zoltan PHG wrapper binary (relative to working dir)
ZOLTAN_PHG_BIN     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "zoltan_phg_partition")

CSV_FIELDNAMES = [
    "partitioning_strategy",
    "circuit_partitioned",
    "origin_of_circuit",
    "circuit_identifier",
    "n_qubits_circuit",
    "n_qpus",
    "coms_cost",
]


# ─────────────────────────────────────────────────────────────────────────────
# WORKER POOL INITIALIZER — runs once per process at startup
# ─────────────────────────────────────────────────────────────────────────────

_QuantumCircuit = None


def _worker_init():
    """Import heavy dependencies once per worker process."""
    global _QuantumCircuit
    from qiskit import QuantumCircuit
    _QuantumCircuit = QuantumCircuit


# ── KaHyPar worker init (commented out) ──────────────────────────────────────
# _kh          = None
# _config_path = None
#
# def _worker_init_kahypar(config_path: str):
#     global _kh, _QuantumCircuit, _config_path
#     from kahypar import kahypar as kh
#     from qiskit import QuantumCircuit
#     _kh          = kh
#     _QuantumCircuit = QuantumCircuit
#     _config_path = config_path


# ─────────────────────────────────────────────────────────────────────────────
# K-WAY FIDUCCIA-MATTHEYSES (FM) HYPERGRAPH PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def fm_partition(n_vertices: int, hyperedges: List[List[int]], k: int,
                 epsilon: float, max_passes: int = 25) -> int:
    """
    k-way Fiduccia-Mattheyses hypergraph partitioning.

    Algorithm
    ---------
    Starts from a balanced round-robin assignment, then runs iterative FM
    passes until no improvement is found or max_passes is reached.

    Each pass:
      1. Compute the gain of moving every unlocked vertex to its best
         alternative block.
      2. Greedily pick the highest-gain legal move (balance-constrained),
         apply it, lock the vertex, and update affected neighbours' gains.
      3. Track the best cut seen during the pass.
      4. Roll back to the best state at the end of the pass.

    Gain of moving vertex v from block A to block B (per hyperedge e ∋ v):
      +1  if block_count[A][e] == 1 and no pins of e are outside {A, B}
          → e goes from cut to uncut
      -1  if block_count[A][e] == |e|
          → e goes from uncut to cut
       0  otherwise

    Balance constraint: each block may hold at most
        ⌊n_vertices / k⌋ * (1 + epsilon) + 1  vertices.

    Returns the number of cut hyperedges.
    """
    import heapq

    max_block = max(1, int(n_vertices / k * (1.0 + epsilon)) + 1)

    # ── Build vertex → hyperedge incidence list ──
    v_to_edges: List[List[int]] = [[] for _ in range(n_vertices)]
    for ei, edge in enumerate(hyperedges):
        for v in edge:
            v_to_edges[v].append(ei)

    # ── Initial balanced assignment (round-robin) ──
    assignment = [i % k for i in range(n_vertices)]
    load       = [0] * k
    for v in range(n_vertices):
        load[assignment[v]] += 1

    def _count_cut() -> int:
        return sum(
            1 for edge in hyperedges
            if len({assignment[v] for v in edge}) > 1
        )

    def _build_block_counts():
        """bc[ei][b] = # pins of hyperedge ei in block b."""
        bc: List[List[int]] = []
        for edge in hyperedges:
            counts = [0] * k
            for v in edge:
                counts[assignment[v]] += 1
            bc.append(counts)
        return bc

    def _gain(v: int, bc: List[List[int]]) -> tuple:
        """Return (best_gain, best_target_block) for moving v."""
        src = assignment[v]
        best_gain, best_tgt = -10**9, -1
        for tgt in range(k):
            if tgt == src or load[tgt] >= max_block:
                continue
            g = 0
            for ei in v_to_edges[v]:
                edge = hyperedges[ei]
                n_src = bc[ei][src]
                n_tgt = bc[ei][tgt]
                n_other = len(edge) - n_src - n_tgt
                if n_src == len(edge):      # edge uncut, will become cut
                    g -= 1
                elif n_src == 1 and n_other == 0:  # edge cut, will become uncut
                    g += 1
            if g > best_gain:
                best_gain, best_tgt = g, tgt
        return best_gain, best_tgt

    best_cut = _count_cut()
    best_assignment = list(assignment)

    for _ in range(max_passes):
        bc     = _build_block_counts()
        locked = [False] * n_vertices

        # Max-heap of (-gain, vertex) — Python heapq is min-heap
        heap: list = []
        gain_cache = [None] * n_vertices   # (gain, tgt) or None
        for v in range(n_vertices):
            g, tgt = _gain(v, bc)
            if tgt >= 0:
                gain_cache[v] = (g, tgt)
                heapq.heappush(heap, (-g, v))

        pass_assignment = list(assignment)
        pass_load       = list(load)
        pass_cut        = best_cut
        pass_best_cut   = best_cut
        pass_best_state = list(assignment)
        improved        = False

        while heap:
            neg_g, v = heapq.heappop(heap)
            if locked[v]:
                continue
            # Re-validate: cached gain may be stale
            cached = gain_cache[v]
            if cached is None:
                continue
            real_g, tgt = _gain(v, bc)
            gain_cache[v] = (real_g, tgt)
            if tgt < 0:
                locked[v] = True
                continue
            if real_g != -neg_g:
                # Gain changed; re-insert with updated value
                heapq.heappush(heap, (-real_g, v))
                continue

            # Apply move v: assignment[v] = src → tgt
            src = assignment[v]
            if load[tgt] >= max_block:
                locked[v] = True
                continue

            # Update block counts for all edges containing v
            for ei in v_to_edges[v]:
                bc[ei][src] -= 1
                bc[ei][tgt] += 1

            assignment[v] = tgt
            load[src] -= 1
            load[tgt] += 1

            # Update running cut
            for ei in v_to_edges[v]:
                edge = hyperedges[ei]
                was_cut_before = (bc[ei][src] + 1 > 0 and
                                  (bc[ei][tgt] - 1 > 0 or
                                   any(bc[ei][b] > 0 for b in range(k) if b != src and b != tgt)))
                # Simpler: recount for this edge only
            pass_cut = _count_cut()   # recount fully (correct but O(pins))
            if pass_cut < pass_best_cut:
                pass_best_cut   = pass_cut
                pass_best_state = list(assignment)
                improved        = True

            locked[v] = True
            # Invalidate gain cache for neighbours
            for ei in v_to_edges[v]:
                for u in hyperedges[ei]:
                    if not locked[u]:
                        gain_cache[u] = None   # mark stale; recomputed on pop

        # Roll back to best state in this pass
        if pass_best_cut < best_cut:
            best_cut        = pass_best_cut
            best_assignment = pass_best_state
        assignment = list(best_assignment)
        load = [0] * k
        for v in range(n_vertices):
            load[assignment[v]] += 1

        if not improved:
            break

    return best_cut


# ── FM partition_job (commented out — swap in to activate) ───────────────────
# def partition_job_fm(args: tuple) -> dict:
#     ...  (calls fm_partition)
#     result["coms_cost"] = fm_partition(qc.num_qubits, hyperedges, k, EPSILON)


# ─────────────────────────────────────────────────────────────────────────────
# EVOLUTIONARY / GENETIC ALGORITHM HYPERGRAPH PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def ea_partition(n_vertices: int, hyperedges: List[List[int]], k: int,
                 epsilon: float,
                 pop_size: int = 40,
                 max_generations: int = 80,
                 patience: int = 20,
                 elite_size: int = 2,
                 tournament_k: int = 3,
                 mut_rate: float = 0.04,
                 seed: int = 0) -> int:
    """
    Evolutionary / genetic algorithm for k-way hypergraph partitioning.

    Representation
    --------------
    Each individual is a flat integer array of length n_vertices where
    assignment[v] ∈ {0, …, k-1} gives the block for vertex v.

    Population initialisation
    -------------------------
    • One round-robin balanced individual (deterministic seed).
    • One greedy individual (reuse greedy_partition logic internally).
    • Remainder: random balanced individuals (block loads spread evenly).

    Fitness (minimise)
    ------------------
        f = cut_edges + α × imbalance_penalty

    where α = n_vertices (so even slight imbalance costs more than a cut edge)
    and imbalance_penalty = sum of max(0, load[b] − max_block) across blocks.
    This softly enforces balance rather than hard-rejecting bad individuals.

    Selection
    ---------
    Tournament selection of size `tournament_k` (smaller tournament → more
    diversity; larger → more pressure toward best).

    Crossover — Uniform partition crossover (UPX)
    --------------------------------------------
    For each vertex independently pick its allele from parent A or parent B
    with equal probability.  The offspring is then repaired for balance: any
    over-filled block donates least-valuable vertices (highest marginal cut
    gain for leaving) to the most under-filled block.

    Mutation
    --------
    Each vertex is independently mutated with probability `mut_rate`.  A
    mutated vertex is moved to a randomly chosen *different* block, subject
    to capacity.  If no valid target exists (all other blocks full) the
    vertex is skipped.

    Elitism
    -------
    The `elite_size` best individuals always survive unchanged to the next
    generation, preventing regression.

    Termination
    -----------
    Stops after `max_generations` generations, or earlier if the global best
    fitness has not improved for `patience` consecutive generations.

    Returns the number of cut hyperedges of the best individual found.
    """
    import random as _rnd

    rng = _rnd.Random(seed ^ (n_vertices * 17) ^ (k * 53))

    max_block = max(1, int(n_vertices / k * (1.0 + epsilon)) + 1)

    # ── Incidence list: vertex → [edge indices] ──────────────────────────────
    v_edges: List[List[int]] = [[] for _ in range(n_vertices)]
    for ei, edge in enumerate(hyperedges):
        for v in edge:
            v_edges[v].append(ei)
    E = len(hyperedges)

    # ── Fitness function ─────────────────────────────────────────────────────
    def _fitness(asgn: List[int]) -> int:
        """Cut edges + heavy penalty for imbalance."""
        load = [0] * k
        for b in asgn:
            load[b] += 1
        penalty = sum(max(0, load[b] - max_block) for b in range(k))
        cut = sum(
            1 for edge in hyperedges
            if len({asgn[v] for v in edge}) > 1
        )
        return cut + penalty * n_vertices

    def _cut_only(asgn: List[int]) -> int:
        return sum(
            1 for edge in hyperedges
            if len({asgn[v] for v in edge}) > 1
        )

    # ── Balance repair ───────────────────────────────────────────────────────
    def _repair(asgn: List[int]) -> List[int]:
        """Move over-quota vertices to under-filled blocks, guided by gain."""
        load = [0] * k
        for b in asgn:
            load[b] += 1

        # Build block counts per edge for gain computation
        bc = [[0] * k for _ in range(E)]
        for ei, edge in enumerate(hyperedges):
            for v in edge:
                bc[ei][asgn[v]] += 1

        def _move_gain(v: int, src: int, tgt: int) -> int:
            g = 0
            for ei in v_edges[v]:
                edge = hyperedges[ei]
                n_src = bc[ei][src]
                n_tgt = bc[ei][tgt]
                n_oth = len(edge) - n_src - n_tgt
                if n_src == len(edge):          # uncut → cut
                    g -= 1
                elif n_src == 1 and n_oth == 0:  # cut → uncut
                    g += 1
            return g

        for _ in range(n_vertices):             # at most n_vertices repair moves
            over  = [b for b in range(k) if load[b] > max_block]
            under = [b for b in range(k) if load[b] < max_block]
            if not over:
                break
            src = max(over,   key=lambda b: load[b])
            tgt = min(under,  key=lambda b: load[b])
            # pick vertex from src block with best move gain
            candidates = [v for v in range(n_vertices) if asgn[v] == src]
            if not candidates:
                break
            v_move = max(candidates, key=lambda v: _move_gain(v, src, tgt))
            # apply move
            for ei in v_edges[v_move]:
                bc[ei][src] -= 1
                bc[ei][tgt] += 1
            asgn[v_move] = tgt
            load[src] -= 1
            load[tgt] += 1

        return asgn

    # ── Individual factories ─────────────────────────────────────────────────
    def _make_roundrobin() -> List[int]:
        return [i % k for i in range(n_vertices)]

    def _make_random() -> List[int]:
        """Balanced random assignment."""
        asgn = [-1] * n_vertices
        load = [0] * k
        verts = list(range(n_vertices))
        rng.shuffle(verts)
        for v in verts:
            tgt = min(range(k), key=lambda b: load[b]
                      if load[b] < max_block else 10**9)
            asgn[v] = tgt
            load[tgt] += 1
        return asgn

    def _make_greedy_init() -> List[int]:
        """One greedy sweep for a good initial individual."""
        asgn = [-1] * n_vertices
        load = [0] * k
        for edge in hyperedges:
            votes = [0] * k
            for v in edge:
                if asgn[v] >= 0:
                    votes[asgn[v]] += 1
            tgt = min(
                sorted(range(k), key=lambda b: (-votes[b], load[b])),
                key=lambda b: load[b] if votes[b] == max(votes) else 10**9,
            )
            # simpler: plurality first, then load
            order = sorted(range(k), key=lambda b: (-votes[b], load[b]))
            tgt = next((b for b in order if load[b] < max_block), order[0])
            for v in edge:
                if asgn[v] < 0:
                    if load[tgt] < max_block:
                        asgn[v] = tgt
                        load[tgt] += 1
                    else:
                        fb = min(range(k), key=lambda b: load[b])
                        asgn[v] = fb
                        load[fb] += 1
        for v in range(n_vertices):
            if asgn[v] < 0:
                fb = min(range(k), key=lambda b: load[b])
                asgn[v] = fb
                load[fb] += 1
        return asgn

    # ── Initial population ───────────────────────────────────────────────────
    population = [
        _make_roundrobin(),
        _make_greedy_init(),
    ]
    while len(population) < pop_size:
        population.append(_make_random())

    fitnesses = [_fitness(ind) for ind in population]

    # ── Tournament selection ─────────────────────────────────────────────────
    def _tournament() -> List[int]:
        contestants = rng.sample(range(pop_size), min(tournament_k, pop_size))
        winner = min(contestants, key=lambda i: fitnesses[i])
        return list(population[winner])

    # ── Uniform partition crossover ──────────────────────────────────────────
    def _crossover(pa: List[int], pb: List[int]) -> List[int]:
        child = [pa[v] if rng.random() < 0.5 else pb[v] for v in range(n_vertices)]
        return _repair(child)

    # ── Mutation ─────────────────────────────────────────────────────────────
    def _mutate(asgn: List[int]) -> List[int]:
        ind = list(asgn)
        load = [0] * k
        for b in ind:
            load[b] += 1
        for v in range(n_vertices):
            if rng.random() >= mut_rate:
                continue
            src = ind[v]
            targets = [b for b in range(k) if b != src and load[b] < max_block]
            if not targets:
                continue
            tgt = rng.choice(targets)
            ind[v] = tgt
            load[src] -= 1
            load[tgt] += 1
        return ind

    # ── Main evolutionary loop ───────────────────────────────────────────────
    best_fitness     = min(fitnesses)
    best_idx         = fitnesses.index(best_fitness)
    best_individual  = list(population[best_idx])
    no_improve_count = 0

    for _gen in range(max_generations):
        # Sort to identify elites
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i])
        elites = [list(population[i]) for i in ranked[:elite_size]]

        new_pop = list(elites)
        new_fit = [fitnesses[ranked[i]] for i in range(elite_size)]

        while len(new_pop) < pop_size:
            pa = _tournament()
            pb = _tournament()
            child = _crossover(pa, pb)
            child = _mutate(child)
            new_pop.append(child)
            new_fit.append(_fitness(child))

        population = new_pop
        fitnesses  = new_fit

        gen_best = min(fitnesses)
        if gen_best < best_fitness:
            best_fitness    = gen_best
            best_idx        = fitnesses.index(gen_best)
            best_individual = list(population[best_idx])
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return _cut_only(best_individual)


# ── EA partition_job (commented out — swap in to activate) ───────────────────
# def partition_job_ea(args: tuple) -> dict:
#     ...  (calls ea_partition)
#     result["coms_cost"] = ea_partition(qc.num_qubits, hyperedges, k, EPSILON, seed=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC GREEDY SEARCH PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def stochastic_greedy_partition(n_vertices: int, hyperedges: List[List[int]],
                                k: int, epsilon: float,
                                time_cap: float = 30.0,
                                seed: int = 0) -> int:
    """
    Randomised iterated greedy hypergraph partitioning with a wall-clock cap.

    Each iteration independently:
      1. Shuffles the vertex ordering at random.
      2. Shuffles the hyperedge scan order at random.
      3. Runs a greedy pass in that order: for every unassigned vertex in each
         edge, place it in the block that currently holds the plurality of the
         edge's already-assigned vertices (capacity-constrained).
      4. Applies one round of greedy local-search improvement: iterates over
         every vertex once (in a freshly shuffled order) and moves it to the
         block that reduces the cut the most, provided the target is within
         balance.  This is repeated until no vertex can be moved profitably.

    The best cut found across all iterations within `time_cap` seconds is
    returned.  Because starting orders are drawn at random (not on a grid),
    each iteration explores a genuinely different part of the solution space.

    Balance constraint: each block holds at most
        ⌊n_vertices / k⌋ × (1 + epsilon) + 1  vertices.

    Returns the number of cut hyperedges of the best solution found.
    """
    import time
    import random as _rnd

    rng = _rnd.Random(seed ^ (n_vertices * 13) ^ (k * 37))

    max_block  = max(1, int(n_vertices / k * (1.0 + epsilon)) + 1)
    E          = len(hyperedges)

    # Vertex → edge incidence list
    v_edges: List[List[int]] = [[] for _ in range(n_vertices)]
    for ei, edge in enumerate(hyperedges):
        for v in edge:
            v_edges[v].append(ei)

    def _count_cut(asgn: List[int]) -> int:
        return sum(
            1 for edge in hyperedges
            if len({asgn[v] for v in edge}) > 1
        )

    def _run_one(rng_state) -> List[int]:
        """One random-order greedy construction + local improvement pass."""
        asgn = [-1] * n_vertices
        load = [0] * k

        # ── Construction: scan hyperedges in a random order ──────────────────
        edge_order = list(range(E))
        rng_state.shuffle(edge_order)

        for ei in edge_order:
            edge = hyperedges[ei]
            votes = [0] * k
            for v in edge:
                if asgn[v] >= 0:
                    votes[asgn[v]] += 1
            order = sorted(range(k), key=lambda b: (-votes[b], load[b]))
            tgt = next((b for b in order if load[b] < max_block), order[0])
            for v in edge:
                if asgn[v] >= 0:
                    continue
                if load[tgt] < max_block:
                    asgn[v] = tgt
                    load[tgt] += 1
                else:
                    fb = min(range(k), key=lambda b: load[b])
                    asgn[v] = fb
                    load[fb] += 1

        # Place any isolated vertices (no 2-qubit gate)
        for v in range(n_vertices):
            if asgn[v] < 0:
                fb = min(range(k), key=lambda b: load[b])
                asgn[v] = fb
                load[fb] += 1

        # ── Local improvement: scan vertices in random order ─────────────────
        # Build per-edge block counts
        bc = [[0] * k for _ in range(E)]
        for ei, edge in enumerate(hyperedges):
            for v in edge:
                bc[ei][asgn[v]] += 1

        def _move_gain(v: int, src: int, tgt: int) -> int:
            g = 0
            for ei in v_edges[v]:
                edge = hyperedges[ei]
                n_src = bc[ei][src]
                n_tgt = bc[ei][tgt]
                n_oth = len(edge) - n_src - n_tgt
                if n_src == len(edge):            # edge uncut → will become cut
                    g -= 1
                elif n_src == 1 and n_oth == 0:   # edge cut → will become uncut
                    g += 1
            return g

        improved = True
        while improved:
            improved = False
            vorder = list(range(n_vertices))
            rng_state.shuffle(vorder)
            for v in vorder:
                src = asgn[v]
                best_g, best_tgt = 0, -1          # only move if gain > 0
                for tgt in range(k):
                    if tgt == src or load[tgt] >= max_block:
                        continue
                    g = _move_gain(v, src, tgt)
                    if g > best_g:
                        best_g, best_tgt = g, tgt
                if best_tgt < 0:
                    continue
                # Apply move
                for ei in v_edges[v]:
                    bc[ei][src] -= 1
                    bc[ei][best_tgt] += 1
                asgn[v] = best_tgt
                load[src] -= 1
                load[best_tgt] += 1
                improved = True

        return asgn

    best_cut  = n_vertices + E          # upper bound
    best_asgn = None
    iters     = 0
    deadline  = time.monotonic() + time_cap

    while time.monotonic() < deadline:
        candidate = _run_one(rng)
        cut = _count_cut(candidate)
        iters += 1
        if cut < best_cut:
            best_cut  = cut
            best_asgn = candidate
        if best_cut == 0:               # can't do better
            break

    return best_cut


# ── Stochastic-greedy partition_job (commented out — swap in to activate) ────
# def partition_job_stochastic(args: tuple) -> dict:
#     ...
#     result["coms_cost"] = stochastic_greedy_partition(
#         qc.num_qubits, hyperedges, k, EPSILON,
#         time_cap=30.0, seed=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY HYPERGRAPH PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def greedy_partition(n_vertices: int, hyperedges: List[List[int]], k: int,
                     epsilon: float) -> int:
    """
    Greedy hypergraph partitioning that minimises cut hyperedges.

    Algorithm
    ---------
    Process hyperedges (gates) in circuit order.  For each edge, identify the
    block that already contains the *plurality* of its vertices, then assign
    all still-unassigned vertices of that edge to that block — provided the
    block has capacity left.  Overflow vertices go to the least-loaded block.
    Isolated qubits (never appearing in a 2-qubit gate) are packed at the end.

    Balance constraint: each block may hold at most
        floor(n_vertices / k * (1 + epsilon)) + 1  vertices.

    Returns the number of cut hyperedges (edges whose vertices span > 1 block).
    """
    max_block = max(1, int(n_vertices / k * (1.0 + epsilon)) + 1)
    assignment = [-1] * n_vertices   # qubit → block id (-1 = unassigned)
    load       = [0]  * k            # current occupancy per block

    def _target_block(edge: List[int]) -> int:
        """Block with plurality of already-assigned edge vertices,
        tie-broken by lowest load.  Falls back to least-loaded block."""
        votes = [0] * k
        for v in edge:
            b = assignment[v]
            if b >= 0:
                votes[b] += 1
        # Sort candidates: most votes first, then lowest load
        order = sorted(range(k), key=lambda b: (-votes[b], load[b]))
        for b in order:
            if load[b] < max_block:
                return b
        # All full — return absolute minimum-load block
        return min(range(k), key=lambda b: load[b])

    for edge in hyperedges:
        target = _target_block(edge)
        for v in edge:
            if assignment[v] >= 0:
                continue                          # already placed
            if load[target] < max_block:
                assignment[v] = target
                load[target] += 1
            else:
                fb = min(range(k), key=lambda b: load[b])
                assignment[v] = fb
                load[fb] += 1

    # Place any qubit that never appeared in a 2-qubit gate
    for v in range(n_vertices):
        if assignment[v] < 0:
            fb = min(range(k), key=lambda b: load[b])
            assignment[v] = fb
            load[fb] += 1

    # Count cut hyperedges
    cut = sum(
        1 for edge in hyperedges
        if len({assignment[v] for v in edge}) > 1
    )
    return cut


# ─────────────────────────────────────────────────────────────────────────────
# ZOLTAN PHG PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def zoltan_phg_partition(n_vertices: int, hyperedges: List[List[int]],
                         k: int) -> int:
    """
    Hypergraph partitioning via Zoltan's PHG algorithm.

    Calls the compiled C subprocess `zoltan_phg_partition` with the
    hypergraph encoded on stdin.  Returns the number of cut hyperedges.

    Input wire format:
        Line 1 : <n_vertices> <n_hyperedges> <k>
        Lines 2+: one hyperedge per line — space-separated 0-based vertex indices
    """
    lines = [f"{n_vertices} {len(hyperedges)} {k}"]
    for edge in hyperedges:
        lines.append(" ".join(map(str, edge)))
    stdin_data = "\n".join(lines) + "\n"

    proc = subprocess.run(
        [ZOLTAN_PHG_BIN],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECS,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"zoltan_phg_partition exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return int(proc.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM HYPERGRAPH PARTITIONER
# ─────────────────────────────────────────────────────────────────────────────

def random_partition(n_vertices: int, hyperedges: List[List[int]], k: int,
                     seed: int = 42) -> int:
    """
    Baseline random hypergraph partitioning.

    Assigns every qubit to one of k blocks uniformly at random, then counts
    the number of hyperedges (gates) whose vertices span more than one block.
    The seed is offset by the circuit size and k so that each (circuit, k)
    combination gets a distinct but reproducible shuffle.

    Returns the number of cut hyperedges.
    """
    rng = random.Random(seed ^ (n_vertices * 31) ^ (k * 97))
    assignment = [rng.randrange(k) for _ in range(n_vertices)]
    return sum(
        1 for edge in hyperedges
        if len({assignment[v] for v in edge}) > 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# JOB FUNCTION  (zoltan_phg — active)
# ─────────────────────────────────────────────────────────────────────────────

def partition_job(args: tuple) -> dict:
    qasm_path, stem, origin, identifier, n_qubits, k = args

    result = dict(
        partitioning_strategy = STRATEGY_NAME,
        circuit_partitioned   = stem,
        origin_of_circuit     = origin,
        circuit_identifier    = identifier,
        n_qubits_circuit      = n_qubits,
        n_qpus                = k,
        coms_cost             = "ERROR",
        _error                = None,
    )

    try:
        qc = _QuantumCircuit.from_qasm_file(qasm_path)

        qubit_index = {q: i for i, q in enumerate(qc.qubits)}
        hyperedges: List[List[int]] = []
        for item in qc.data:
            if item.operation.name in {"barrier", "snapshot", "delay", "label", "measure"}:
                continue
            if len(item.qubits) >= 2:
                edge = list(dict.fromkeys(qubit_index[q] for q in item.qubits))
                if len(edge) >= 2:
                    hyperedges.append(edge)

        if not hyperedges:
            result["coms_cost"] = 0
            return result

        if len(hyperedges) > MAX_HYPEREDGES:
            result["coms_cost"] = "SKIPPED_TOO_MANY_GATES"
            result["_error"]    = f"{len(hyperedges)} hyperedges exceeds limit of {MAX_HYPEREDGES}"
            return result

        result["coms_cost"] = stochastic_greedy_partition(qc.num_qubits, hyperedges, k, EPSILON, time_cap=30.0, seed=97)
        #if seed always 97

    except Exception:
        result["_error"] = traceback.format_exc()

    return result


# ── Random partition_job (commented out) ─────────────────────────────────────
# def partition_job_random(args: tuple) -> dict:
#     ...  (calls random_partition)
#     result["coms_cost"] = random_partition(qc.num_qubits, hyperedges, k, RANDOM_SEED)


# ── Greedy partition_job (commented out) ─────────────────────────────────────
# def partition_job_greedy(args: tuple) -> dict:
#     ...  (calls greedy_partition)
#     result["coms_cost"] = greedy_partition(qc.num_qubits, hyperedges, k, EPSILON)


# ── KaHyPar partition_job (commented out) ────────────────────────────────────
# def partition_job_kahypar(args: tuple) -> dict:
#     qasm_path, stem, origin, identifier, n_qubits, k = args
#     ...
#     hg  = _kh.Hypergraph(n_vertices, len(hyperedges), he_indices, he_flat, k,
#                           [1]*len(hyperedges), [1]*n_vertices)
#     ctx = _kh.Context()
#     ctx.loadINIconfiguration(_config_path)
#     ctx.setK(k); ctx.setEpsilon(EPSILON); ctx.setSeed(42); ctx.suppressOutput(True)
#     _kh.partition(hg, ctx)
#     result["coms_cost"] = _kh.cut(hg)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# ── _find_config (KaHyPar — commented out) ───────────────────────────────────
# def _find_config() -> str:
#     from kahypar import kahypar as kh
#     env = os.environ.get("KAHYPAR_CONFIG")
#     if env and os.path.isfile(env): return env
#     kahypar_pkg_dir = os.path.dirname(kh.__file__)
#     for name in ("cut_kKaHyPar_sea20.ini", "cut_rKaHyPar_sea20.ini"):
#         p = os.path.join(kahypar_pkg_dir, "config", name)
#         if os.path.isfile(p): return p
#     for root, _, files in os.walk(kahypar_pkg_dir):
#         for fn in sorted(files):
#             if fn.endswith(".ini") and "-E_" not in fn and "mypy" not in fn:
#                 return os.path.join(root, fn)
#     raise FileNotFoundError("No KaHyPar .ini config found.")


def parse_filename(stem: str) -> Tuple[str, str, int]:
    parts = stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"'{stem}' does not match monolithic__<origin>__<id>__<n_qubits>")
    _, origin, identifier, qubits_str = parts
    try:
        return origin, identifier, int(qubits_str)
    except ValueError:
        raise ValueError(f"Cannot parse qubit count from '{qubits_str}' in '{stem}'")


def valid_k_values(n_qubits: int) -> List[int]:
    return [k for k in range(2, 11) if n_qubits >= k * MIN_QUBITS_PER_QPU]


def _load_completed_jobs(csv_path: str, strategy: str) -> set:
    """
    Read the output CSV and return the set of (stem, k) pairs that have
    already been recorded for *strategy*.  Used to resume an interrupted run.
    """
    completed = set()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("partitioning_strategy") != strategy:
                    continue
                try:
                    k = int(row["n_qpus"])
                except (ValueError, TypeError, KeyError):
                    continue
                stem = row.get("circuit_partitioned", "")
                if stem:
                    completed.add((stem, k))
    except FileNotFoundError:
        pass
    return completed


def _make_pool():
    return multiprocessing.Pool(
        processes   = NUM_WORKERS,
        initializer = _worker_init,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    circuits_path = pathlib.Path(CIRCUITS_DIR)
    qasm_files    = sorted(circuits_path.rglob("*.qasm"))

    if not qasm_files:
        print(f"[!] No .qasm files found in '{CIRCUITS_DIR}'")
        return

    print(f"[+] Found {len(qasm_files)} circuit(s) in '{CIRCUITS_DIR}'")
    print(f"[+] Strategy       : {STRATEGY_NAME}")
    print(f"[+] QPU range      : k=2..10  (min {MIN_QUBITS_PER_QPU} qubits/QPU)")
    print(f"[+] Qubit limit    : <= {MAX_QUBITS}")
    print(f"[+] Hyperedge limit: <= {MAX_HYPEREDGES} multi-qubit gates")
    print(f"[+] Workers        : {NUM_WORKERS}")
    print(f"[+] Output CSV     : {OUTPUT_CSV}\n")

    # Build job list
    jobs    = []
    skipped = []
    for qasm_path in qasm_files:
        stem = qasm_path.stem
        try:
            origin, identifier, n_qubits = parse_filename(stem)
        except ValueError as exc:
            skipped.append(str(exc))
            continue

        ks = valid_k_values(n_qubits)
        if not ks:
            skipped.append(f"{qasm_path.name}: {n_qubits} qubits — too small")
            continue

        if n_qubits > MAX_QUBITS:
            skipped.append(f"{qasm_path.name}: {n_qubits} qubits — too large")
            continue

        if origin == "bloated":
            skipped.append(f"{qasm_path.name}: bloated circuit — skipped")
            continue

        for k in ks:
            jobs.append((str(qasm_path), stem, origin, identifier, n_qubits, k))

    if skipped:
        print(f"[SKIP] {len(skipped)} file(s) excluded (too small or too large).\n")

    jobs.sort(key=lambda x: (x[4], x[5]))  # smallest circuits + smallest k first

    # ── Resume: drop jobs already recorded in the CSV for this strategy ──
    completed = _load_completed_jobs(OUTPUT_CSV, STRATEGY_NAME)
    if completed:
        before = len(jobs)
        jobs = [j for j in jobs if (j[1], j[5]) not in completed]
        print(f"[+] Resuming     : {len(completed)} jobs already in CSV — skipping them")
        print(f"[+] Remaining    : {len(jobs)} / {before} jobs to run")
    # ─────────────────────────────────────────────────────────────────────

    print(f"[+] Total jobs  : {len(jobs)}")
    print(f"[+] Spawning {NUM_WORKERS} workers...", flush=True)

    errors = []
    pool   = _make_pool()

    with open(OUTPUT_CSV, "a", newline="") as csv_file:
        write_header = csv_file.tell() == 0
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        csv_file.flush()

        with tqdm(total=len(jobs), unit="job", dynamic_ncols=True) as pbar:
            # Submit up to NUM_WORKERS jobs at a time so all workers stay busy.
            # Using a sliding-window approach: maintain a queue of in-flight
            # (future, job_meta) pairs; once the window is full, collect the
            # oldest result before submitting the next job.
            in_flight = []   # list of (AsyncResult, job_tuple)
            job_iter  = iter(jobs)

            def _submit_next():
                """Submit the next job from the iterator; returns False when exhausted."""
                try:
                    job = next(job_iter)
                    future = pool.apply_async(partition_job, (job,))
                    in_flight.append((future, job))
                    return True
                except StopIteration:
                    return False

            def _collect_oldest(timed_out_pool):
                """
                Wait for the oldest in-flight future and write the result.
                Returns the (possibly restarted) pool.
                """
                future, job = in_flight.pop(0)
                _, stem, origin, identifier, n_qubits, k = job

                try:
                    res = future.get(timeout=TIMEOUT_SECS)

                except multiprocessing.TimeoutError:
                    # Worker is stuck in KaHyPar C++ — restart only once per
                    # detected hung pool; the other in-flight futures on the
                    # same pool are also lost, so drain them as TIMEOUT too.
                    timed_out_pool.terminate()
                    timed_out_pool.join()
                    timed_out_pool = _make_pool()

                    # Mark remaining in-flight futures (now orphaned) as TIMEOUT
                    for _, orphan_job in list(in_flight):
                        _, os_, oo, oi, on, ok = orphan_job
                        orphan_res = dict(
                            partitioning_strategy = STRATEGY_NAME,
                            circuit_partitioned   = os_,
                            origin_of_circuit     = oo,
                            circuit_identifier    = oi,
                            n_qubits_circuit      = on,
                            n_qpus                = ok,
                            coms_cost             = "TIMEOUT",
                            _error                = f"Timed out after {TIMEOUT_SECS}s — pool restarted",
                        )
                        errors.append(f"{os_} k={ok}: {orphan_res['_error']}")
                        orphan_res.pop("_error", None)
                        writer.writerow(orphan_res)
                        csv_file.flush()
                        pbar.set_description(f"{os_} k={ok} cut=TIMEOUT")
                        pbar.update(1)
                    in_flight.clear()

                    res = dict(
                        partitioning_strategy = STRATEGY_NAME,
                        circuit_partitioned   = stem,
                        origin_of_circuit     = origin,
                        circuit_identifier    = identifier,
                        n_qubits_circuit      = n_qubits,
                        n_qpus                = k,
                        coms_cost             = "TIMEOUT",
                        _error                = f"Timed out after {TIMEOUT_SECS}s — pool restarted",
                    )

                except Exception as e:
                    res = dict(
                        partitioning_strategy = STRATEGY_NAME,
                        circuit_partitioned   = stem,
                        origin_of_circuit     = origin,
                        circuit_identifier    = identifier,
                        n_qubits_circuit      = n_qubits,
                        n_qpus                = k,
                        coms_cost             = "ERROR",
                        _error                = str(e),
                    )

                error_msg = res.pop("_error", None)
                writer.writerow(res)
                csv_file.flush()

                if error_msg:
                    errors.append(f"{stem} k={k}: {error_msg}")

                pbar.set_description(f"{stem} k={k} cut={res['coms_cost']}")
                pbar.update(1)
                return timed_out_pool

            # Fill the initial window
            for _ in range(NUM_WORKERS):
                if not _submit_next():
                    break

            # Slide the window until all jobs are done
            while in_flight:
                pool = _collect_oldest(pool)
                _submit_next()   # submit next job as soon as a slot opens

    pool.terminate()
    pool.join()

    print(f"\n[OK] Results written to '{OUTPUT_CSV}' ({len(jobs)} rows)")

    if errors:
        log_path = OUTPUT_CSV.replace(".csv", "_errors.log")
        with open(log_path, "w") as lf:
            lf.write("\n\n".join(errors))
        print(f"[!]  {len(errors)} error(s) — see '{log_path}'")
    else:
        print("[OK] No errors.")


if __name__ == "__main__":
    main()