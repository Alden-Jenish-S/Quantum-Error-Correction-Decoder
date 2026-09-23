"""Generate diagnostics for the deployed L=3 MAP / L=5-7 MWPM hybrid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("circuit_mwpm", "pre_hybrid_data_only", "hybrid")
LABELS = {
    "circuit_mwpm": "supplied circuit MWPM",
    "pre_hybrid_data_only": "pre-hybrid data-only MWPM",
    "hybrid": "deployed hybrid",
}
COLORS = {"circuit_mwpm": "#0072B2", "pre_hybrid_data_only": "#D55E00", "hybrid": "#009E73"}
HATCHES = dict(zip(VARIANTS, ("//", "..", "")))
MARKERS = dict(zip(VARIANTS, ("s", "^", "o")))


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("seed", "L", "p", "xi"))
        if key in seen or row["shots"] <= 0:
            raise ValueError("duplicate stratum or nonpositive shots")
        seen.add(key)
        for variant in VARIANTS:
            if row["decoders"][variant]["status"] != "ok":
                raise ValueError("Figures require a passing receipt; failed runs must be reported separately")
    expected = {(L, p, xi) for L in (3, 5, 7) for p in (0.005, 0.01) for xi in (0, 2, 5, 10)}
    if not rows:
        raise ValueError("empty receipt")
    for seed in {r["seed"] for r in rows}:
        if {k[1:] for k in seen if k[0] == seed} != expected:
            raise ValueError("incomplete challenge grid")
    return rows


def paired_effect(rows: list[dict], reference: str) -> tuple[float, float]:
    """Saved errors/M and approximate 95% CI half-width for fixed strata.

    Shot difference is +1 for rescue, -1 for harm, 0 for agreement. Variances
    are summed across independent seed/point strata; zero discordance does not
    establish population equality. These CIs are descriptive, not simultaneous.
    """
    n = sum(r["shots"] for r in rows)
    saved = 0
    variance_sum = 0.0
    for row in rows:
        pair = row["paired"][reference]
        a, b, m = pair["rescue"], pair["harm"], row["shots"]
        saved += a - b
        if m > 1:
            variance_sum += (a + b - (a - b)**2 / m) * m / (m - 1)
    return saved * 1e6 / n, 1.96 * np.sqrt(variance_sum) * 1e6 / n


def aggregate(rows: list[dict], variant: str, keys: tuple[str, ...]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = {}
    for key, group in groups.items():
        decoder_rows = [row["decoders"][variant] for row in group]
        shots = sum(row["shots"] for row in group)
        result[key] = {
            "errors": sum(item["errors"] for item in decoder_rows),
            "shots": shots,
            "errors_per_million": sum(item["errors"] for item in decoder_rows) * 1e6 / shots,
            "max_elapsed_s": max(item["build_decode_s"] for item in decoder_rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    by_point = {variant: aggregate(rows, variant, ("L", "p", "xi")) for variant in VARIANTS}
    points = sorted(by_point["hybrid"])
    labels = [f"L{L}\np={p:g}\nxi={xi:g}" for L, p, xi in points]

    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    x = np.arange(len(points))
    width = 0.26
    for offset, variant in zip((-width, 0, width), VARIANTS):
        values = [by_point[variant][point]["errors_per_million"] for point in points]
        ax.bar(x + offset, values, width, label=LABELS[variant], color=COLORS[variant], hatch=HATCHES[variant], edgecolor="white", linewidth=0.2)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("logical errors per million")
    ax.set_title(f"{len({r['seed'] for r in rows})}-seed validation across all 24 points (pooled counts)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(args.outdir / "hybrid_errors_by_point.png", dpi=180)
    plt.close(fig)

    deltas = {}
    for L, p, xi in points:
        deltas[(L, p, xi)] = by_point["pre_hybrid_data_only"][(L, p, xi)]["errors_per_million"] - by_point["hybrid"][(L, p, xi)]["errors_per_million"]
    matrix = np.array([[deltas[(L, p, xi)] for xi in (0.0, 2.0, 5.0, 10.0)] for L in (3, 5, 7) for p in (0.005, 0.01)])
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    vmax = max(abs(float(matrix.min())), abs(float(matrix.max())), 1.0)
    image = ax.imshow(matrix, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(4), ["0", "2", "5", "10"])
    ax.set_yticks(range(6), [f"L={L}, p={p:g}" for L in (3, 5, 7) for p in (0.005, 0.01)])
    ax.set_xlabel("correlation length xi")
    ax.set_ylabel("parameter stratum")
    ax.set_title("Errors saved by hybrid relative to pre-hybrid MWPM")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="errors/M saved (positive is better)")
    fig.savefig(args.outdir / "hybrid_improvement_heatmap.png", dpi=180)
    plt.close(fig)

    by_seed = aggregate(rows, "hybrid", ("seed",))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    seeds = sorted(by_seed)
    for offset, variant in zip((-0.14, 0, 0.14), VARIANTS):
        seed_values = aggregate(rows, variant, ("seed",))
        ax.plot(np.arange(len(seeds)) + offset, [seed_values[seed]["errors_per_million"] for seed in seeds], linestyle="none", marker=MARKERS[variant], label=LABELS[variant], color=COLORS[variant])
    ax.set_xticks(np.arange(len(seeds)), [str(seed[0]) for seed in seeds])
    ax.set_xlabel("validation seed (categorical; not a numerical trend)")
    ax.set_ylabel("pooled logical errors per million")
    ax.set_title("Multi-seed aggregate variability")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(args.outdir / "hybrid_seed_variability.png", dpi=180)
    plt.close(fig)

    runtime = {variant: aggregate(rows, variant, ("L",)) for variant in VARIANTS}
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    x = np.arange(3)
    for offset, variant in zip((-width, 0, width), VARIANTS):
        values = [runtime[variant][(L,)]["max_elapsed_s"] for L in (3, 5, 7)]
        bars = ax.bar(x + offset, values, width, label=LABELS[variant], color=COLORS[variant], hatch=HATCHES[variant], edgecolor="white", linewidth=0.2)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
    ax.axhline(2.5, color="#C00000", linestyle="--", label="2.5 s limit")
    ax.set_xticks(x, ["L=3", "L=5", "L=7"])
    ax.set_ylabel("maximum build + decode seconds")
    ax.set_title("Decoder timing margin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(args.outdir / "hybrid_runtime_by_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    changed = [(3, p, xi) for p in (0.005, 0.01) for xi in (5.0, 10.0)]
    means, margins = [], []
    for point in changed:
        subset = [r for r in rows if (r["L"], r["p"], r["xi"]) == point]
        mean, margin = paired_effect(subset, "pre_hybrid_data_only")
        means.append(mean)
        margins.append(margin)
    ax.errorbar(range(4), means, yerr=margins, fmt="o", capsize=5, color=COLORS["hybrid"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(4), [f"p={p:g}, xi={xi:g}" for _, p, xi in changed])
    ax.set_ylabel("errors/M saved vs pre-hybrid MWPM")
    ax.set_title("L=3 paired gain: approximate pointwise 95% intervals")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(args.outdir / "hybrid_paired_gain.png", dpi=180)
    plt.close(fig)

    # Tabular data alternative and source binding for every retained figure.
    with (args.outdir / "hybrid_figure_data.csv").open("w", newline="\n") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["L", "p", "xi", "decoder", "errors", "shots", "errors_per_million", "max_elapsed_s"])
        for point in points:
            for variant in VARIANTS:
                r = by_point[variant][point]
                writer.writerow([*point, variant, r["errors"], r["shots"], r["errors_per_million"], r["max_elapsed_s"]])
    metadata = {
        "input_name": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(rows), "shots_per_decoder": sum(r["shots"] for r in rows),
        "aggregation": "sum(errors)/sum(shots); no rows excluded; unrounded rates",
        "uncertainty": "paired fixed-stratum normal 95% intervals, not simultaneous; see paired_effect",
        "runtime": "maximum construction + decode, sampling excluded, recorded host only",
        "seed_axis": "categorical, no interpolation",
    }
    (args.outdir / "hybrid_figure_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
