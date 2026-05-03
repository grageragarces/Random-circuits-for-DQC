#!/usr/bin/env python3
"""
analysis.py — Partitioning Strategy Comparison
===============================================
Evaluates how KaHyPar, Greedy, Zoltan PHG, FM, EA, StocG, and Random partitioning
perform across three circuit-origin categories:

  • Real       — MQT Bench circuits + Quipper circuits
  • Random     — randomly generated circuits (randomqiskit, randomfromgraphqiskit)
  • Generated  — procedurally generated circuits (qgen)

Core question: does the *gap* between techniques change with circuit origin,
and is "random partitioning" specifically good or bad depending on origin?

All improvements are measured relative to the random partitioning baseline:

  improvement (%) = (random_cut − strategy_cut) / max(random_cut, 1) × 100

Positive = better than random assignment; 0 = same; negative = worse.

Usage: venv/bin/python analysis.py
       python analysis.py
"""

import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive; saves as SVG cleanly
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = "./partitioning_results.csv"
OUT_DIR  = "./analysis_figures"

STRATEGY_LABELS: dict[str, str] = {
    "kahypar_default": "KaHyPar",
    "greedy":          "Greedy",
    "zoltan_phg":      "Zoltan PHG",
    "fm":              "FM",
    "ea":              "EA",
    "StocG":           "StocG",
    "random":          "Random",
}

# Display order for strategies (Random last since it's the baseline)
STRATEGY_ORDER = ["KaHyPar", "Greedy", "Zoltan PHG", "FM", "EA", "StocG"]
STRATEGY_ORDER_WITH_RANDOM = STRATEGY_ORDER + ["Random"]

STRATEGY_COLORS: dict[str, str] = {
    "KaHyPar":    "#2196F3",   # blue
    "Greedy":     "#4CAF50",   # green
    "Zoltan PHG": "#FF9800",   # orange
    "FM":         "#9C27B0",   # purple
    "EA":         "#E91E63",   # pink/red
    "StocG":      "#00BCD4",   # cyan
    "Random":     "#9E9E9E",   # grey (baseline)
}

STRATEGY_MARKERS: dict[str, str] = {
    "KaHyPar":    "o",
    "Greedy":     "s",
    "Zoltan PHG": "^",
    "FM":         "D",
    "EA":         "P",
    "StocG":      "*",
    "Random":     "x",
}

# Origin → category mapping
ORIGIN_GROUPS: dict[str, str] = {
    "mqt":                   "Real",
    "quipper":               "Real",
    "randomqiskit":          "Random",
    "randomfromgraphqiskit": "Random",
    "qgen":                  "Generated",
}

ORIGIN_PRETTY: dict[str, str] = {
    "mqt":                   "MQT Bench",
    "quipper":               "Quipper",
    "randomqiskit":          "Random (Qiskit)",
    "randomfromgraphqiskit": "Random (Graph)",
    "qgen":                  "QGen",
}

GROUP_ORDER    = ["Real", "Random", "Generated"]
GROUP_COLORS   = {"Real": "#1565C0", "Random": "#2E7D32", "Generated": "#6A1B9A"}

ORIGIN_ORDER = ["mqt", "quipper", "randomqiskit", "randomfromgraphqiskit", "qgen"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV, drop non-numeric rows (TIMEOUT / SKIPPED / ERROR)."""
    rows = []
    skipped = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cost = row["coms_cost"]
            if not cost.lstrip("-").isdigit():
                skipped += 1
                continue
            rows.append({
                "strategy":   row["partitioning_strategy"],
                "circuit":    row["circuit_partitioned"],
                "origin":     row["origin_of_circuit"],
                "identifier": row["circuit_identifier"],
                "n_qubits":   int(row["n_qubits_circuit"]),
                "k":          int(row["n_qpus"]),
                "cut":        int(cost),
            })

    print(f"  Loaded {len(rows):,} numeric rows  ({skipped:,} non-numeric skipped)")

    df = pd.DataFrame(rows)
    df["label"]        = df["strategy"].map(STRATEGY_LABELS)
    df["origin_group"] = df["origin"].map(ORIGIN_GROUPS)
    return df


def compute_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (circuit, k) pair, compute improvement over the random baseline:

        improvement (%) = (random_cut - strategy_cut) / max(random_cut, 1) × 100

    Only keeps rows that can be matched to a random baseline row.
    """
    rand_df = (
        df[df["strategy"] == "random"][["circuit", "k", "cut"]]
        .rename(columns={"cut": "random_cut"})
    )
    merged = df.merge(rand_df, on=["circuit", "k"], how="inner")
    merged["improvement"] = (
        (merged["random_cut"] - merged["cut"]) / merged["random_cut"].clip(lower=1) * 100
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _legend_handles(strategies=None, include_random=False):
    pool = STRATEGY_ORDER_WITH_RANDOM if include_random else STRATEGY_ORDER
    if strategies is not None:
        pool = [s for s in pool if s in strategies]
    return [
        Line2D([0], [0], color=STRATEGY_COLORS[s], lw=2,
               marker=STRATEGY_MARKERS[s], markersize=6, label=s)
        for s in pool
    ]


def _strategy_subset(imp_df: pd.DataFrame, include_random=False) -> pd.DataFrame:
    pool = STRATEGY_ORDER_WITH_RANDOM if include_random else STRATEGY_ORDER
    labels_present = set(imp_df["label"].unique())
    pool = [s for s in pool if s in labels_present]
    return imp_df[imp_df["label"].isin(pool)].copy()


def _mean_improvement(imp_df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    return (
        imp_df.groupby(group_by)["improvement"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "imp_mean", "std": "imp_std", "count": "n"})
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Per-circuit improvement distributions (violins)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_distributions(imp_df: pd.DataFrame, out_dir: str):
    """
    Single violin plot — strategies on the x-axis, one overlapping violin per
    origin group at each strategy position.  Colour encodes origin category
    (Real / Random / Generated); semi-transparent so overlaps are visible.

    Answers: "Do the improvement distributions differ by circuit origin,
              and is a technique consistently good across origins?"
    """
    sub = _strategy_subset(imp_df, include_random=False)
    strategies = [s for s in STRATEGY_ORDER if s in sub["label"].unique()]

    # Slight x-offsets so violin edges remain distinguishable when overlapping
    # offsets = [-0.18, 0.0, 0.18]
    violin_w = 0.45   # wide enough that the three violins overlap noticeably

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.suptitle(
    #     "Distribution of per-circuit improvements over Random partitioning\n"
    #     "colour = circuit origin  |  overlapping violins show cross-origin similarity",
    #     fontsize=20, fontweight="bold",
    # )

    for gi, group in enumerate(GROUP_ORDER):
        gdata = sub[sub["origin_group"] == group]
        plot_data = []
        positions = []

        for si, strat in enumerate(strategies):
            d = gdata[gdata["label"] == strat]["improvement"].dropna().values
            if len(d) >= 2:
                clipped = np.clip(d, np.percentile(d, 1), np.percentile(d, 99))
                plot_data.append(clipped)
                positions.append(si + 0.0)  # No offset for now

        if not plot_data:
            continue

        color = GROUP_COLORS[group]
        parts = ax.violinplot(plot_data, positions=positions,
                              showmedians=True, showextrema=False, widths=violin_w)

        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.42)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(2.0)
        parts["cmedians"].set_alpha(0.95)

    ax.axhline(0, color="#555", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=20)
    ax.set_ylabel("Improvement over Random (%)", fontsize=20)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[g], alpha=0.6,
                       edgecolor=GROUP_COLORS[g], linewidth=1.2, label=g)
        for g in GROUP_ORDER
    ]
    ax.legend(handles=legend_handles, title="Circuit origin", fontsize=20,
              title_fontsize=20, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_distributions.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Technique spread (how much do techniques diverge by origin/k?)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_technique_spread(imp_df: pd.DataFrame, out_dir: str):
    """
   Heatmap: std deviation of strategy means for every (origin_group, k)
            cell.  Shows if differentiation is k-dependent.

    Directly answers the user's core hypothesis:
    "The gap between techniques is larger for certain circuit types."
    """
    sub  = _strategy_subset(imp_df, include_random=False)
    data = _mean_improvement(sub, ["origin_group", "k", "label"])

    # For each (origin_group, k), compute std across strategies
    spread = (
        data.groupby(["origin_group", "k"])["imp_mean"]
        .std()
        .reset_index()
        .rename(columns={"imp_mean": "spread"})
    )
    # Average spread over k per group
    spread_avg = spread.groupby("origin_group")["spread"].mean().reset_index()

    fig, (ax_heat) = plt.subplots(1, 1, figsize=(13, 5.5))
    # fig.suptitle(
    #     "Technique differentiation by circuit origin\n"
    #     "(std deviation across strategies' mean improvements — higher = more differentiated)",
    #     fontsize=20, fontweight="bold",
    # )

    # ── Right: heatmap (origin_group × k) ────────────────────────────────────
    pivot = spread.pivot(index="origin_group", columns="k", values="spread")
    pivot = pivot.reindex(index=GROUP_ORDER)
    ks    = sorted(spread["k"].unique())

    im = ax_heat.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=pivot.values.max())
    ax_heat.set_xticks(range(len(ks)))
    ax_heat.set_xticklabels([str(k) for k in ks], fontsize=20)
    ax_heat.set_yticks(range(len(GROUP_ORDER)))
    ax_heat.set_yticklabels(GROUP_ORDER, fontsize=20)
    ax_heat.set_xlabel("Number of QPUs", fontsize=20)
    ax_heat.set_title("By origin group and k", fontsize=20)

    for row_i in range(len(GROUP_ORDER)):
        for col_i in range(len(ks)):
            val = pivot.values[row_i, col_i]
            if np.isfinite(val):
                ax_heat.text(col_i, row_i, f"{val:.1f}",
                             ha="center", va="center",
                             fontsize=20, color="black" if val < pivot.values.max() * 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04)
    cbar.set_label("Std across strategies (%)", fontsize=20)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_technique_spread.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Individual origins breakdown
# ─────────────────────────────────────────────────────────────────────────────

def fig5_individual_origins(imp_df: pd.DataFrame, out_dir: str):
    """
    5-panel bar chart — one panel per individual circuit origin.
    Within each panel: strategies on x-axis, mean improvement on y-axis.
    Error bars = ±1 std.

    Answers: "Within 'Real' circuits, does MQT vs Quipper differ?
              Within 'Random', does structure (graph-based) matter?"
    """
    sub  = _strategy_subset(imp_df, include_random=False)
    data = _mean_improvement(sub, ["origin", "label"])
    strategies = [s for s in STRATEGY_ORDER if s in sub["label"].unique()]

    present_origins = [o for o in ORIGIN_ORDER if o in sub["origin"].unique()]
    n_panels = len(present_origins)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    # fig.suptitle(
    #     "Mean improvement over Random partitioning — individual circuit origins",
    #     fontsize=13, fontweight="bold", y=1.01,
    # )

    x     = np.arange(len(strategies))
    bar_w = 0.6

    for ax, origin in zip(axes, present_origins):
        odata = data[data["origin"] == origin]
        label_to_row = {r["label"]: r for _, r in odata.iterrows()}

        means  = [label_to_row.get(s, {}).get("imp_mean", np.nan) for s in strategies]
        stds   = [label_to_row.get(s, {}).get("imp_std",  np.nan) for s in strategies]
        counts = [label_to_row.get(s, {}).get("n", 0) for s in strategies]
        colors = [STRATEGY_COLORS[s] for s in strategies]

        bars = ax.bar(x, means, bar_w, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.8,
                      error_kw={"elinewidth": 1.2, "alpha": 0.6})

        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=20)
        ax.set_title(ORIGIN_PRETTY.get(origin, origin), fontsize=20,
                     fontweight="bold",
                     color="black")
        ax.set_xlabel("Strategy", fontsize=20)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", alpha=0.35, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # n= annotation
        n_tot = max(counts) if counts else 0
        ax.text(0.98, 0.97, f"n={n_tot:,}\ncircuit×k pairs",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=20, color="#666")

    axes[0].set_ylabel("Improvement over Random (%)", fontsize=20)

    # Colour-coded group legend
    group_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[g], alpha=0.5,
                       edgecolor="none", label=g)
        for g in GROUP_ORDER
    ]
    fig.legend(handles=group_patches + _legend_handles(strategies),
               loc="lower center", ncol=len(GROUP_ORDER) + len(strategies),
               bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=20)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_individual_origins.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Random partitioning's absolute cut vs k (is it ever competitive?)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_random_vs_rest(imp_df: pd.DataFrame, out_dir: str):
    """
    Specifically focuses on Random partitioning as a baseline.
    Line plots (one per origin group) showing mean *normalised cut*:

        norm_cut = strategy_cut / random_cut    (1.0 = same as random)

    Values < 1 = better than random.  This gives an absolute sense of how
    far random is from the optimum — and whether clever methods close that gap.
    """
    all_strats = _strategy_subset(imp_df, include_random=True)

    # Also add random itself (improvement = 0 by definition, norm_cut = 1.0)
    rand_rows = imp_df[imp_df["strategy"] == "random"].copy()
    rand_rows["label"] = "Random"
    rand_rows["improvement"] = 0.0

    combined = pd.concat([all_strats, rand_rows], ignore_index=True)

    # Use normalised cut = 1 − improvement/100
    combined["norm_cut"] = 1.0 - combined["improvement"] / 100.0

    data = (
        combined.groupby(["origin_group", "k", "label"])["norm_cut"]
        .mean()
        .reset_index()
        .rename(columns={"norm_cut": "nc_mean"})
    )

    strategies_full = [s for s in STRATEGY_ORDER_WITH_RANDOM if s in combined["label"].unique()]
    ks = sorted(combined["k"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # fig.suptitle(
    #     "Normalised cut vs network size (Random partitioning = 1.0 baseline)\n"
    #     "Lower is better; 0.0 = zero communication cost",
    #     fontsize=20, fontweight="bold", y=1.01,
    # )

    for ax, group in zip(axes, GROUP_ORDER):
        gdata = data[data["origin_group"] == group]
        for strat in strategies_full:
            sdata = gdata[gdata["label"] == strat].sort_values("k")
            if sdata.empty:
                continue
            lw = 1.5 if strat != "Random" else 2.0
            ls = "--" if strat == "Random" else "-"
            ax.plot(sdata["k"], sdata["nc_mean"],
                    color=STRATEGY_COLORS[strat],
                    marker=STRATEGY_MARKERS[strat],
                    linewidth=lw, linestyle=ls, markersize=6, label=strat)

        ax.axhline(1.0, color=STRATEGY_COLORS["Random"], linewidth=1.2,
                   linestyle="--", alpha=0.5)
        ax.set_title(group, fontsize=20, fontweight="bold",
                     color="black")
        ax.set_xlabel("Number of QPUs", fontsize=20)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Normalized Cut cost", fontsize=20)

    handles = _legend_handles(strategies_full, include_random=True)
    fig.legend(handles=handles, loc="lower center", ncol=len(strategies_full),
               bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_normalised_cut_vs_k.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def fig6_random_vs_rest_v2(imp_df: pd.DataFrame, out_dir: str):
    """
    Specifically focuses on Random partitioning as a baseline.
    Line plots (one per origin group) showing mean *normalised cut*:

        norm_cut = strategy_cut / random_cut    (1.0 = same as random)

    Values < 1 = better than random.  This gives an absolute sense of how
    far random is from the optimum — and whether clever methods close that gap.
    """
    all_strats = _strategy_subset(imp_df, include_random=True)

    # Also add random itself (improvement = 0 by definition, norm_cut = 1.0)
    rand_rows = imp_df[imp_df["strategy"] == "random"].copy()
    rand_rows["label"] = "Random"
    rand_rows["improvement"] = 0.0

    combined = pd.concat([all_strats, rand_rows], ignore_index=True)

    # Use normalised cut = 1 − improvement/100
    combined["norm_cut"] = 1.0 - combined["improvement"] / 100.0

    data = (
        combined.groupby(["origin_group", "k", "label"])["norm_cut"]
        .mean()
        .reset_index()
        .rename(columns={"norm_cut": "nc_mean"})
    )

    # Average norm_cut over all k values, group by circuit size
    data = (
        combined.groupby(["origin_group", "n_qubits", "label"])["norm_cut"]
        .mean()
        .reset_index()
        .rename(columns={"norm_cut": "nc_mean"})
    )

    strategies_full = [s for s in STRATEGY_ORDER_WITH_RANDOM if s in combined["label"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # fig.suptitle(
    #     "Normalised cut vs circuit size (Random partitioning = 1.0 baseline)\n"
    #     "Lower is better; 0.0 = zero communication cost",
    #     fontsize=20, fontweight="bold", y=1.01,
    # )

    for ax, group in zip(axes, GROUP_ORDER):
        gdata = data[data["origin_group"] == group]
        sizes = sorted(gdata["n_qubits"].unique())
        for strat in strategies_full:
            sdata = gdata[gdata["label"] == strat].sort_values("n_qubits")
            if sdata.empty:
                continue
            lw = 1.5 if strat != "Random" else 2.0
            ls = "--" if strat == "Random" else "-"
            ax.plot(sdata["n_qubits"], sdata["nc_mean"],
                    color=STRATEGY_COLORS[strat],
                    marker=STRATEGY_MARKERS[strat] if len(sizes) <= 20 else None,
                    linewidth=lw, linestyle=ls, markersize=5, label=strat)

        ax.axhline(1.0, color=STRATEGY_COLORS["Random"], linewidth=1.2,
                   linestyle="--", alpha=0.5)
        ax.set_title(group, fontsize=20, fontweight="bold",
                     color="black")
        ax.set_xlabel("Circuit size (n_qubits)", fontsize=20)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Normalized Cut cost", fontsize=20)

    handles = _legend_handles(strategies_full, include_random=True)
    fig.legend(handles=handles, loc="lower center", ncol=len(strategies_full),
               bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_normalised_cut_vs_k_v2.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS FOR ABSOLUTE-CUT FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _all_strategies(df: pd.DataFrame, include_random=True) -> list[str]:
    pool = STRATEGY_ORDER_WITH_RANDOM if include_random else STRATEGY_ORDER
    present = set(df["label"].dropna().unique())
    return [s for s in pool if s in present]


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Total cut cost per strategy, grouped by origin category
# ─────────────────────────────────────────────────────────────────────────────

def fig7_total_cut_by_strategy(df: pd.DataFrame, out_dir: str):
    """
    Grouped bar chart.
    x-axis : partitioning strategy
    y-axis : total sum of cut costs (all circuits × all k in that origin group)
    Colour-coded groups: Real / Random / Generated appear side-by-side for
    each strategy, making cross-origin comparison easy.
    """
    sub = df[df["label"].notna()].copy()
    strategies = _all_strategies(sub)
    groups = GROUP_ORDER

    # Sum of cut per (strategy_label, origin_group)
    totals = (
        sub.groupby(["label", "origin_group"])["cut"]
        .sum()
        .reset_index()
        .rename(columns={"cut": "total_cut"})
    )

    n_strat = len(strategies)
    n_group = len(groups)
    group_w = 0.7
    bar_w   = group_w / n_group
    x       = np.arange(n_strat)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_strat), 5.5))
    # fig.suptitle(
    #     "Total communication cost (sum of cut hyperedges) per strategy\n"
    #     "grouped by circuit origin category (all k=2..10 included)",
    #     fontsize=13, fontweight="bold",
    # )

    for gi, group in enumerate(groups):
        offsets = x + (gi - (n_group - 1) / 2) * bar_w
        vals = [
            totals.loc[
                (totals["label"] == s) & (totals["origin_group"] == group), "total_cut"
            ].sum()
            for s in strategies
        ]
        bars = ax.bar(offsets, vals, bar_w * 0.92,
                      color=GROUP_COLORS[group], alpha=0.82,
                      label=group, edgecolor="white", linewidth=0.6)
        # for bar, v in zip(bars, vals):
        #     if v > 0:
        #         ax.text(bar.get_x() + bar.get_width() / 2, v,
        #                 f"{v:,.0f}", ha="center", va="bottom",
        #                 fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=20)
    ax.set_ylabel("Total cut cost (sum)", fontsize=20)
    ax.set_xlabel("Partitioning strategy", fontsize=20)
    #ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="Circuit origin", fontsize=20, title_fontsize=20,
              loc="upper right", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig7_total_cut_by_strategy.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Mean cut vs circuit size (n_qubits), 3 panels by origin group
# ─────────────────────────────────────────────────────────────────────────────

def fig11_cut_vs_circuit_size(df: pd.DataFrame, out_dir: str):
    """
    For each circuit size (n_qubits) and each strategy, compute the mean cut
    across all k values and all circuits of that size.
    3 panels — one per origin group.
    x = n_qubits, y = mean cut per circuit, one line per strategy.
    """
    sub = df[df["label"].notna()].copy()
    strategies = _all_strategies(sub, include_random=True)

    # Mean cut per (origin_group, n_qubits, strategy_label)
    means = (
        sub.groupby(["origin_group", "n_qubits", "label"])["cut"]
        .mean()
        .reset_index()
        .rename(columns={"cut": "mean_cut"})
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    # fig.suptitle(
    #     "Mean cut cost vs circuit size (n_qubits)\n"
    #     "averaged over all k=2..10 values",
    #     fontsize=13, fontweight="bold", y=1.01,
    # )

    for ax, group in zip(axes, GROUP_ORDER):
        gdata = means[means["origin_group"] == group]
        sizes = sorted(gdata["n_qubits"].unique())

        for strat in strategies:
            sdata = gdata[gdata["label"] == strat].sort_values("n_qubits")
            if sdata.empty:
                continue
            lw = 2.2 if strat == "Random" else 1.6
            ls = "--" if strat == "Random" else "-"
            ax.plot(sdata["n_qubits"], sdata["mean_cut"],
                    color=STRATEGY_COLORS[strat],
                    marker=STRATEGY_MARKERS[strat] if len(sizes) <= 20 else None,
                    linewidth=lw, linestyle=ls, markersize=4, alpha=0.9)

        ax.set_title(group, fontsize=20, fontweight="bold",
                     color="black")
        ax.set_xlabel("Circuit size (n_qubits)", fontsize=20)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean cut per circuit", fontsize=20)
    handles = _legend_handles(strategies, include_random=True)
    fig.legend(handles=handles, loc="lower center", ncol=len(strategies),
               bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig11_cut_vs_circuit_size.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RANKING TABLES — best-to-worst strategy per qubit-size bin and per k
# ─────────────────────────────────────────────────────────────────────────────

def make_ranking_tables(imp_df: pd.DataFrame, out_dir: str):
    """
    Produce two CSV tables showing strategies ranked best→worst:

    1. table_rankings_by_qubit_bin.csv
       Rows = (origin_group, qubit-size bin every 15 qubits)
       Columns = Rank 1 … Rank N  (strategy name + mean improvement)

    2. table_rankings_by_k.csv
       Rows = (origin_group, k)
       Columns = Rank 1 … Rank N

    Having all three origin groups side-by-side makes it easy to see whether
    Random-circuit or Generated-circuit rankings mirror the Real-circuit order.
    """
    sub = _strategy_subset(imp_df, include_random=False).copy()
    strategies = [s for s in STRATEGY_ORDER if s in sub["label"].unique()]
    n_strats = len(strategies)

    def _rank_block(gdata, key_col, key_val, group):
        """Return an ordered-rank dict for one (origin_group, key) cell."""
        kdata = gdata[gdata[key_col] == key_val]
        means = {s: kdata[kdata["label"] == s]["improvement"].mean()
                 for s in strategies}
        ranked = sorted(strategies, key=lambda s: means.get(s, float("-inf")), reverse=True)
        row = {"origin_group": group, key_col: key_val}
        for ri, strat in enumerate(ranked, 1):
            v = means.get(strat, float("nan"))
            row[f"rank_{ri}"] = f"{strat} ({v:+.1f}%)" if np.isfinite(v) else f"{strat} (N/A)"
        return row

    # ── Table 1: qubit-size bins ──────────────────────────────────────────────
    bin_width = 15
    max_q = int(sub["n_qubits"].max()) + 1
    bins   = list(range(0, max_q + bin_width, bin_width))
    labels = [f"{b}–{b+bin_width-1}" for b in bins[:-1]]
    sub["qubit_bin"] = pd.cut(sub["n_qubits"], bins=bins, labels=labels, right=False)

    rows_qbin = []
    for group in GROUP_ORDER:
        gdata = sub[sub["origin_group"] == group]
        for qbin in labels:
            if gdata[gdata["qubit_bin"] == qbin].empty:
                continue
            rows_qbin.append(_rank_block(gdata, "qubit_bin", qbin, group))

    rank_cols = [f"rank_{i}" for i in range(1, n_strats + 1)]
    df_qbin = pd.DataFrame(rows_qbin, columns=["origin_group", "qubit_bin"] + rank_cols)
    path1 = os.path.join(out_dir, "table_rankings_by_qubit_bin.csv")
    df_qbin.to_csv(path1, index=False)
    print(f"  Saved {path1}")

    # ── Table 2: QPU count k ──────────────────────────────────────────────────
    rows_k = []
    for group in GROUP_ORDER:
        gdata = sub[sub["origin_group"] == group]
        for k in sorted(sub["k"].unique()):
            rows_k.append(_rank_block(gdata, "k", k, group))

    df_k = pd.DataFrame(rows_k, columns=["origin_group", "k"] + rank_cols)
    path2 = os.path.join(out_dir, "table_rankings_by_k.csv")
    df_k.to_csv(path2, index=False)
    print(f"  Saved {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG — Origin-group rank similarity
# ─────────────────────────────────────────────────────────────────────────────

def fig_origin_rank_similarity(imp_df: pd.DataFrame, out_dir: str):
    """
    Directly answers: is 'Random' or 'Generated' closer to 'Real' circuits
    in terms of *how strategies rank against each other*?

    Three panels:
      Left   — Scatter: Real vs Random  (one point per strategy)
      Centre — Scatter: Real vs Generated
      Right  — Spearman rank-correlation heatmap between all three origin groups

    In the scatter panels the dashed diagonal = perfect agreement; strategies
    above the line are relatively better on the y-axis origin type.
    In the heatmap, ρ near 1 means the two origin types agree on which
    strategies are best, ρ near 0 means they disagree.
    """
    sub = _strategy_subset(imp_df, include_random=False)
    strategies = [s for s in STRATEGY_ORDER if s in sub["label"].unique()]
    data = _mean_improvement(sub, ["origin_group", "label"])

    # Mean improvement vector per origin group (indexed by STRATEGY_ORDER)
    group_vec: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        gd = data[data["origin_group"] == group]
        lut = {r["label"]: r["imp_mean"] for _, r in gd.iterrows()}
        group_vec[group] = np.array([lut.get(s, np.nan) for s in strategies])

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        ra = pd.Series(a[mask]).rank().values
        rb = pd.Series(b[mask]).rank().values
        return float(np.corrcoef(ra, rb)[0, 1])

    corr = np.array([[_spearman(group_vec[g1], group_vec[g2])
                      for g2 in GROUP_ORDER] for g1 in GROUP_ORDER])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── Left / Centre: scatter panels ────────────────────────────────────────
    comparisons = [("Random", axes[0]), ("Generated", axes[1])]
    real_vec = group_vec["Real"]

    for comp_group, ax in comparisons:
        comp_vec = group_vec[comp_group]
        for si, strat in enumerate(strategies):
            ax.scatter(real_vec[si], comp_vec[si],
                       color=STRATEGY_COLORS[strat],
                       marker=STRATEGY_MARKERS[strat],
                       s=100, zorder=3, linewidths=0.5, edgecolors="white")
            ax.annotate(strat, (real_vec[si], comp_vec[si]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=20, color=STRATEGY_COLORS[strat], fontweight="bold")

        # Diagonal y = x
        all_vals = np.concatenate([real_vec, comp_vec])
        lo, hi = np.nanmin(all_vals) - 2, np.nanmax(all_vals) + 2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real circuits — mean improvement (%)", fontsize=20)
        ax.set_ylabel(f"{comp_group} circuits — mean improvement (%)", fontsize=20)
        ax.set_title(f"Real  vs  {comp_group}", fontsize=20, fontweight="bold",
                     color="black")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        # Annotate Spearman ρ
        rho = _spearman(real_vec, comp_vec)
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=20, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Strategy colour legend on left scatter
    axes[0].legend(handles=_legend_handles(strategies), fontsize=20,
                   framealpha=0.85, loc="lower right")

    # ── Right: Spearman heatmap ───────────────────────────────────────────────
    ax_hm = axes[2]
    im = ax_hm.imshow(corr, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax_hm.set_xticks(range(3))
    ax_hm.set_yticks(range(3))
    ax_hm.set_xticklabels(GROUP_ORDER, fontsize=20)
    ax_hm.set_yticklabels(GROUP_ORDER, fontsize=20)
    ax_hm.set_title("Spearman ρ — strategy ranking agreement\nbetween origin groups",
                    fontsize=20, fontweight="bold")
    for i in range(3):
        for j in range(3):
            v = corr[i, j]
            ax_hm.text(j, i, f"{v:.3f}", ha="center", va="center",
                       fontsize=20, fontweight="bold",
                       color="black" if abs(v) < 0.7 else "white")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=20)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_origin_rank_similarity.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(imp_df: pd.DataFrame):
    """Print a concise text summary to stdout."""
    sub  = _strategy_subset(imp_df, include_random=False)
    data = _mean_improvement(sub, ["origin_group", "label"])

    print("\n" + "═" * 68)
    print(" SUMMARY: mean improvement over Random partitioning (%)")
    print("═" * 68)
    print(f"  {'Strategy':<18}  {'Real':>10}  {'Random':>10}  {'Generated':>10}")
    print("  " + "─" * 52)

    strategies = [s for s in STRATEGY_ORDER if s in sub["label"].unique()]
    for strat in strategies:
        sdata = data[data["label"] == strat]
        vals  = {r["origin_group"]: r["imp_mean"] for _, r in sdata.iterrows()}
        r  = vals.get("Real",      float("nan"))
        rn = vals.get("Random",    float("nan"))
        g  = vals.get("Generated", float("nan"))
        def _fmt(v): return f"{v:+.1f}%" if np.isfinite(v) else "   N/A"
        print(f"  {strat:<18}  {_fmt(r):>10}  {_fmt(rn):>10}  {_fmt(g):>10}")

    print()
    print(" KEY OBSERVATIONS:")

    # Which strategy wins overall?
    best_per_group = (
        data.loc[data.groupby("origin_group")["imp_mean"].idxmax()]
        .set_index("origin_group")["label"]
        .to_dict()
    )
    for grp in GROUP_ORDER:
        best = best_per_group.get(grp, "?")
        row = data[(data["origin_group"] == grp) & (data["label"] == best)]
        if not row.empty:
            val = float(row["imp_mean"].iloc[0])
            print(f"  • Best in {grp:<11}: {best} ({val:+.1f}%)")

    # Spread
    spread = (
        data.groupby("origin_group")["imp_mean"]
        .std()
        .sort_values(ascending=False)
    )
    most_diff = spread.index[0]
    least_diff = spread.index[-1]
    print(f"  • Most   differentiated origin: {most_diff} (σ={spread[most_diff]:.1f}%)")
    print(f"  • Least  differentiated origin: {least_diff} (σ={spread[least_diff]:.1f}%)")
    print("═" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n[+] Loading data from '{CSV_PATH}' ...")
    df = load_data(CSV_PATH)

    print(f"[+] Computing improvement over Random baseline ...")
    imp_df = compute_improvement(df)

    n_strats   = imp_df["label"].nunique()
    n_circuits = imp_df["circuit"].nunique()
    n_rows     = len(imp_df)
    print(f"    {n_rows:,} rows  |  {n_circuits:,} unique circuits  |  {n_strats} strategies")
    print(f"    Strategies: {sorted(imp_df['label'].unique(), key=lambda x: str(x))}")
    print(f"    Origins:    {sorted(imp_df['origin'].unique(), key=lambda x: str(x))}")

    print(f"\n[+] Generating figures → '{OUT_DIR}/' ...")
    # ── Improvement-over-random figures ──
    fig3_distributions(imp_df, OUT_DIR)
    fig4_technique_spread(imp_df, OUT_DIR)
    fig5_individual_origins(imp_df, OUT_DIR)
    fig6_random_vs_rest(imp_df, OUT_DIR)
    fig6_random_vs_rest_v2(imp_df, OUT_DIR)
    # ── Absolute cut-cost figures ──
    fig7_total_cut_by_strategy(df, OUT_DIR)
    fig11_cut_vs_circuit_size(df, OUT_DIR)
    # ── Origin similarity ──
    fig_origin_rank_similarity(imp_df, OUT_DIR)
    # ── Ranking tables (CSV) ──
    make_ranking_tables(imp_df, OUT_DIR)

    print_summary(imp_df)
    print(f"\n[OK] All figures saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
