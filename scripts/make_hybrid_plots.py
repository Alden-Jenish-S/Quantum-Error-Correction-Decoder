"""Generate diagnostics for the deployed L=3 MAP / L=5-7 MWPM hybrid."""

from __future__ import annotations

import argparse
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


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        ax.bar(x + offset, values, width, label=LABELS[variant], color=COLORS[variant])
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("logical errors per million")
    ax.set_title("Five-seed, 1M-shot validation across all 24 parameter points")
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
    for variant in VARIANTS:
        seed_values = aggregate(rows, variant, ("seed",))
        ax.plot(seeds, [seed_values[seed]["errors_per_million"] for seed in seeds], marker="o", label=LABELS[variant], color=COLORS[variant])
    ax.set_xlabel("validation seed")
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
        ax.bar(x + offset, values, width, label=LABELS[variant], color=COLORS[variant])
    ax.axhline(2.5, color="#C00000", linestyle="--", label="2.5 s limit")
    ax.set_xticks(x, ["L=3", "L=5", "L=7"])
    ax.set_ylabel("maximum build + decode seconds")
    ax.set_title("Decoder timing margin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(args.outdir / "hybrid_runtime_by_distance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
