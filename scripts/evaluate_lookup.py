"""Evaluate a saved L=3 syndrome MAP table while using MWPM elsewhere."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qec_benchmark.baselines import MWPMDecoder
from qec_benchmark.config import challenge_grid
from qec_benchmark.models import ParameterPoint
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


class LookupHybrid:
    def __init__(self, point: ParameterPoint, tables: dict):
        self.point = point
        self.tables = tables
        self.experiment = SurfaceCodeExperiment(point.L)
        self.mwpm = MWPMDecoder(point, True)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        if self.point.L != 3:
            return self.mwpm.decode(syndrome)
        table = np.asarray(self.tables[self.point.key()], dtype=np.uint8)
        codes = syndrome.astype(np.uint16) @ (1 << np.arange(syndrome.shape[1], dtype=np.uint16))
        return table[codes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = json.loads(args.tables.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    rows = []
    experiments = {}
    for point in challenge_grid():
        experiment = experiments.setdefault(point.L, SurfaceCodeExperiment(point.L))
        syndrome, truth = experiment.sample_correlated(shots=args.shots, p=point.p, xi=point.xi, rng=rng)
        pred = LookupHybrid(point, tables).decode(syndrome)
        rows.append({"L": point.L, "p": point.p, "xi": point.xi, "errors": int(np.sum(pred != truth)), "shots": args.shots})
    args.output.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    print(sum(r["errors"] for r in rows), sum(r["shots"] for r in rows))


if __name__ == "__main__":
    main()
