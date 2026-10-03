#!/usr/bin/env python3
"""Draws the benchmark charts in docs/figures/ from the CSV files in results/.

Run from the repository root: `make plots`. Each point is the mean of the runs
in the CSV; error bars show the lowest and highest run (for both recall and speed).
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("docs/figures")

# Colours: the first three slots of a colour-blind-checked categorical palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def read(name):
    """Rows of a results CSV as dicts, skipping the '#' header lines."""
    with open(RESULTS / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def machine(name):
    """'cpu, os' from the '#' header lines, for chart captions."""
    info = {}
    with open(RESULTS / name) as f:
        for line in f:
            if line.startswith("# ") and ": " in line:
                key, value = line[2:].strip().split(": ", 1)
                info[key] = value
    return f"{info.get('cpu', '?')}, {info.get('ram', '?')} RAM, single thread"


def by(rows, *keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def summary(rows, column):
    values = [float(r[column]) for r in rows]
    return mean(values), min(values), max(values)


def curve(rows):
    """Per ef_search (sorted): ef, recall, qps, lowest qps, highest qps, lowest/highest recall."""
    points = []
    for (ef,), group in sorted(by(rows, "ef_search").items(), key=lambda kv: int(kv[0][0])):
        recall, recall_lo, recall_hi = summary(group, "recall")
        qps, lo, hi = summary(group, "qps")
        points.append((int(ef), recall, qps, lo, hi, recall_lo, recall_hi))
    return points


def recall_vs_qps(ax, rows, series_column, label, ef_labels_for=None):
    """One line per value of series_column: recall@10 (x) against queries/s (y)."""
    hnsw = [r for r in rows if r["algo"] == "hnsw"]
    groups = by(hnsw, series_column)
    keys = sorted(groups, key=lambda k: (not k[0].isdigit(), int(k[0]) if k[0].isdigit() else k[0]))
    for color, key in zip(SERIES, keys):
        points = curve(groups[key])
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        yerr = [[p[2] - p[3] for p in points], [p[4] - p[2] for p in points]]
        xerr = [[p[1] - p[5] for p in points], [p[6] - p[1] for p in points]]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=color, linewidth=2, marker="o", markersize=5,
                    capsize=2, elinewidth=1, label=label(key[0], groups[key]))
        if ef_labels_for == key[0]:
            for ef, x, y, *_ in points:
                ax.annotate(f"ef {ef}", (x, y), textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=MUTED)
    exact = [r for r in rows if r["algo"] == "exact"]
    if exact:
        qps = summary(exact, "qps")[0]
        ax.axhline(qps, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"exact search: {qps:,.0f} queries/s", (ax.get_xlim()[0], qps),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, color=MUTED)
    ax.set_yscale("log")
    log_axis(ax.yaxis)
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (log scale)")
    ax.legend(loc="upper right")


def log_axis(axis):
    """Log scale with readable ticks: 1,000  2,000  5,000  10,000 ..."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_locator(LogLocator(subs="auto"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def build_label(prefix):
    return lambda key, group: f"{prefix}{key} (build {summary(group, 'build_s')[0]:.1f} s)"


def chart_recall_vs_qps():
    rows = read("sift-100k.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    recall_vs_qps(ax, rows, "M", build_label("M = "), ef_labels_for="16")
    ax.set_title(f"HNSW on SIFT-100k: recall vs speed\n{machine('sift-100k.csv')}", fontsize=10)
    save(fig, "recall_vs_qps.png")


def chart_select():
    rows = read("sift-100k-select.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    recall_vs_qps(ax, rows, "select", build_label(""))
    ax.set_title(
        "Choosing links: paper's heuristic vs simply the M closest (M = 16)\n"
        f"SIFT-100k, {machine('sift-100k-select.csv')}",
        fontsize=10,
    )
    save(fig, "select.png")


def chart_visited():
    rows = [r for r in read("sift-100k-visited.csv") if r["algo"] == "hnsw"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = {"tags": "visit-tag array", "hash": "new unordered_set per search"}
    for color, ((visited,), group) in zip(SERIES, sorted(by(rows, "visited").items(),
                                                         reverse=True)):
        points = curve(group)
        ax.errorbar([p[0] for p in points], [p[2] for p in points],
                    yerr=[[p[2] - p[3] for p in points], [p[4] - p[2] for p in points]],
                    color=color, linewidth=2, marker="o", markersize=5, capsize=2, elinewidth=1,
                    label=names[visited])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    log_axis(ax.yaxis)
    efs = sorted({int(r["ef_search"]) for r in rows})
    ax.set_xticks(efs, [str(e) for e in efs])
    ax.minorticks_off()
    ax.set_xlabel("ef_search")
    ax.set_ylabel("queries per second (log scale)")
    ax.legend(loc="upper right")
    ax.set_title(
        f"Visited set: visit tags vs hash set (M = 16)\nSIFT-100k, {machine('sift-100k-visited.csv')}",
        fontsize=10,
    )
    save(fig, "visited.png")


def chart_memory():
    rows = [r for r in read("sift-100k.csv") if r["algo"] == "hnsw"]
    dim = 128
    groups = sorted(by(rows, "M").items(), key=lambda kv: int(kv[0][0]))
    ms = [k[0] for k, _ in groups]
    per_vector = [summary(g, "bytes_per_vector")[0] for _, g in groups]
    vector_bytes = [4 * dim] * len(ms)
    graph_bytes = [b - v for b, v in zip(per_vector, vector_bytes)]
    recall40 = [summary([r for r in g if r["ef_search"] == "40"], "recall")[0] for _, g in groups]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.8))
    x = range(len(ms))
    left.bar(x, vector_bytes, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
             label="vector (4 bytes x 128)")
    left.bar(x, graph_bytes, width=0.6, bottom=vector_bytes, color=SERIES[1], edgecolor=SURFACE,
             linewidth=2, label="graph links")
    for i, total in enumerate(per_vector):
        left.annotate(f"{total:.0f}", (i, total), textcoords="offset points", xytext=(0, 3),
                      ha="center", fontsize=9, color=TEXT)
    left.set_xticks(list(x), [f"M = {m}" for m in ms])
    left.set_ylabel("bytes per vector")
    left.set_ylim(0, max(per_vector) * 1.15)
    left.legend(loc="upper left", fontsize=8)
    left.grid(axis="x", visible=False)
    left.set_title("Memory", fontsize=10)

    right.bar(x, recall40, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for i, value in enumerate(recall40):
        right.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=9, color=TEXT)
    right.set_xticks(list(x), [f"M = {m}" for m in ms])
    right.set_ylabel("recall@10 at ef_search = 40")
    right.set_ylim(0, 1.08)
    right.grid(axis="x", visible=False)
    right.set_title("Accuracy", fontsize=10)
    fig.suptitle(f"Memory vs accuracy on SIFT-100k\n{machine('sift-100k.csv')}", fontsize=10)
    save(fig, "memory_vs_m.png")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


if __name__ == "__main__":
    chart_recall_vs_qps()
    chart_select()
    chart_visited()
    chart_memory()
