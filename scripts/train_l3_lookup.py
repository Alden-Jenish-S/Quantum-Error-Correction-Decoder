"""Fit finite-syndrome MAP tables for L=3 from independent simulated samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qec_benchmark.config import challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = SurfaceCodeExperiment(distance=3)
    result = {}
    for point in challenge_grid():
        if point.L != 3:
            continue
        rng = np.random.default_rng(args.seed + int(point.p * 1_000_000) + int(point.xi * 100))
        syndrome, truth = experiment.sample_correlated(shots=args.shots, p=point.p, xi=point.xi, rng=rng)
        codes = syndrome.astype(np.uint16) @ (1 << np.arange(syndrome.shape[1], dtype=np.uint16))
        counts = np.zeros((1 << syndrome.shape[1], 2), dtype=np.int64)
        np.add.at(counts, codes, np.column_stack((1 - truth, truth)))
        result[point.key()] = counts.argmax(axis=1).astype(int).tolist()
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
