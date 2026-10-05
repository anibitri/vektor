#!/usr/bin/env python3
"""Draws the benchmark charts in docs/figures/ from the CSV files in results/.

Run from the repository root: `make plots`. Each point is the mean of the runs
in the CSV; error bars show the lowest and highest run (for both recall and speed).
"""

import csv
import math
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
# An ordered (light to dark) blue ramp for the synthetic datasets, smallest r first.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
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


REAL = [("sift-100k", "SIFT-100k", "s"), ("glove-100k", "GloVe-100k", "^"),
        ("fashion-mnist-30k", "Fashion-MNIST-30k", "D")]
SYNTHETIC = [4, 8, 16, 32, 64]
# Where each point's label goes in the intrinsic-dimension chart, so labels do not collide.
# An offset can also be a dict {estimator: offset} if one label needs a different
# place in each column.
LABEL_OFFSET = {"sift-100k": (8, 6), "glove-100k": (8, 5), "fashion-mnist-30k": (-8, 6),
                "synthetic-4": (8, -3), "synthetic-8": (8, -3), "synthetic-16": (-8, -12),
                "synthetic-32": (-8, -12), "synthetic-64": (-8, 6)}


def m16_hnsw(name):
    """HNSW rows with M = 16 (the default) from results/<name>.csv."""
    return [r for r in read(f"{name}.csv") if r["algo"] == "hnsw" and r["M"] == "16"]


def qps_at_recall(points, target):
    """Queries/s at a target recall, interpolated between ef_search settings
    (log of queries/s, straight line in recall). Returns (qps, is_lower_bound);
    is_lower_bound is True when even the smallest ef_search beats the target.
    Returns (None, False) if the target is never reached."""
    if points[0][1] >= target:
        return points[0][2], True
    for a, b in zip(points, points[1:]):
        if a[1] < target <= b[1]:
            t = (target - a[1]) / (b[1] - a[1])
            return math.exp(math.log(a[2]) + t * (math.log(b[2]) - math.log(a[2]))), False
    return None, False


def chart_datasets():
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for (name, label, marker), color in zip(REAL, SERIES):
        points = curve(m16_hnsw(name))
        left.plot([p[1] for p in points], [p[2] for p in points], color=color, linewidth=2,
                  marker=marker, markersize=5, label=label)
    for r, color in zip(SYNTHETIC, RAMP):
        points = curve(m16_hnsw(f"synthetic-{r}"))
        right.plot([p[1] for p in points], [p[2] for p in points], color=color, linewidth=2,
                   marker="o", markersize=5, label=f"r = {r}")
    for ax, title in ((left, "Real datasets"), (right, "Synthetic, 128 dimensions, intrinsic r")):
        ax.set_yscale("log")
        log_axis(ax.yaxis)
        ax.set_xlabel("recall@10")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
    left.set_ylabel("queries per second (log scale)")
    fig.suptitle(f"HNSW (M = 16) on every dataset\n{machine('glove-100k.csv')}", fontsize=10)
    save(fig, "recall_vs_qps_datasets.png")


def chart_intrinsic_dimension(targets=(0.9, 0.99)):
    """Queries/s at fixed recall against estimated intrinsic dimension: one row
    per recall target, one column per estimator. Hollow markers are lower
    bounds (the smallest ef_search already beats the target)."""
    id_rows = read("intrinsic-dimension.csv")
    estimates = {r["dataset"]: r for r in id_rows}
    k = id_rows[0]["mle_k"]
    datasets = REAL + [(f"synthetic-{r}", f"r = {r}", "o") for r in SYNTHETIC]
    speed = {}  # (name, target) -> (queries/s or None, is lower bound)
    for name, _, _ in datasets:
        points = curve(m16_hnsw(name))
        for target in targets:
            speed[(name, target)] = qps_at_recall(points, target)

    print(f"| dataset | TwoNN | MLE (k={k}) | "
          + " | ".join(f"queries/s at recall {t}" for t in targets) + " |")
    for name, _, _ in datasets:
        cells = []
        for target in targets:
            qps, bound = speed[(name, target)]
            cells.append("not reached" if qps is None else f"{'>= ' if bound else ''}{qps:,.0f}")
        est = estimates[name]
        print(f"| {name} | {float(est['twonn']):.1f} | {float(est['mle']):.1f} | "
              + " | ".join(cells) + " |")

    fig, axes = plt.subplots(len(targets), 2, figsize=(10, 3.9 * len(targets)), sharey="row")
    sample = id_rows[0]["sample"]
    estimators = (("twonn", f"TwoNN, {int(sample):,}-point sample (as in the paper)"),
                  ("mle", f"Levina-Bickel MLE, {k} neighbours, same sample"))
    for row_axes, target in zip(axes, targets):
        for ax, (column, title) in zip(row_axes, estimators):
            synth = [(name, label) for name, label, _ in datasets if name.startswith("synthetic")]
            xs = [float(estimates[name][column]) for name, _ in synth]
            ys = [speed[(name, target)][0] for name, _ in synth]
            ax.plot(xs, ys, color=GRID, linewidth=2, zorder=1)
            for (name, label), x, y, color in zip(synth, xs, ys, RAMP):
                bound = speed[(name, target)][1]
                ax.scatter(x, y, s=50, facecolor=SURFACE if bound else color, edgecolor=color,
                           linewidth=2, zorder=2)
                dx, dy = LABEL_OFFSET[name]
                ax.annotate(("at least " if bound else "") + label, (x, y),
                            textcoords="offset points", xytext=(dx, dy),
                            ha="left" if dx > 0 else "right", fontsize=8, color=MUTED)
            for name, label, marker in REAL:
                qps, bound = speed[(name, target)]
                if qps is None:
                    continue
                x = float(estimates[name][column])
                ax.scatter(x, qps, s=60, marker=marker, facecolor=SURFACE if bound else TEXT,
                           edgecolor=TEXT, linewidth=1.5, zorder=3)
                offset = LABEL_OFFSET[name]
                dx, dy = offset[column] if isinstance(offset, dict) else offset
                ax.annotate(("at least " if bound else "") + label, (x, qps),
                            textcoords="offset points", xytext=(dx, dy),
                            ha="left" if dx > 0 else "right", fontsize=8, color=TEXT)
            ax.set_yscale("log")
            log_axis(ax.yaxis)
            ax.set_xlim(left=0)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("estimated intrinsic dimension")
        row_axes[0].set_ylabel(f"queries/s at recall@10 = {target}")
    fig.suptitle("Search speed against intrinsic dimension (HNSW, M = 16). Synthetic data in "
                 "blue, real datasets in black;\nhollow markers are lower bounds. "
                 f"{machine('glove-100k.csv')}", fontsize=10)
    save(fig, "intrinsic_dimension.png")


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
    chart_datasets()
    chart_intrinsic_dimension()
