"""Generate publication-style diagnostics from JSONL experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    base, improved = load(args.baseline), load(args.improved)
    key = lambda r: (r["seed"], r["L"], r["p"], r["xi"])
    b = {key(r): r for r in base}
    i = {key(r): r for r in improved}
    keys = list(b)
    labels = [f"L{l}, p={p:g}, xi={x:g}" for _, l, p, x in keys]
    br = np.array([b[k]["error_rate"] * 1e6 for k in keys])
    ir = np.array([i[k]["error_rate"] * 1e6 for k in keys])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(keys)); w = .38
    ax.bar(x-w/2, br, w, label="supplied MWPM")
    ax.bar(x+w/2, ir, w, label="data-only iid MWPM")
    ax.set_ylabel("errors per million")
    ax.set_xlabel("parameter point (seed 42 first, then validation seeds)")
    ax.set_title("Logical error comparison")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.outdir / "error_comparison.png", dpi=160); plt.close(fig)

    grouped = {}
    for r in improved:
        grouped.setdefault((r["L"], r["xi"]), []).append(r["error_rate"] * 1e6)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (L, xi), vals in sorted(grouped.items()):
        ax.plot([0.005, 0.01], [np.mean([r["error_rate"]*1e6 for r in improved if r["L"]==L and r["xi"]==xi and r["p"]==p]) for p in [0.005,0.01]], marker="o", label=f"L={L}, xi={xi:g}")
    ax.set_xlabel("physical error rate p"); ax.set_ylabel("errors per million")
    ax.set_title("Improved decoder across noise regimes"); ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.outdir / "improved_regimes.png", dpi=160); plt.close(fig)

    times = [i[k]["elapsed_s"] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, marker="."); ax.axhline(2.5, color="r", ls="--", label="2.5 s limit")
    ax.set_xlabel("point/seed record"); ax.set_ylabel("build + decode seconds"); ax.set_title("Improved decoder runtime"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.outdir / "runtime.png", dpi=160); plt.close(fig)

    reductions = (br-ir) / np.maximum(br, 1e-12) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(reductions, bins=20, color="#4c78a8", edgecolor="white")
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("relative error reduction (%)"); ax.set_ylabel("records")
    ax.set_title("Per-point improvement variability"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.outdir / "variability.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
