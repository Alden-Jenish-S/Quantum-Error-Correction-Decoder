"""Verify solve.py with the unmodified official evaluator and small contract probes.

The evaluator's time limit includes wrapper overhead. Reported elapsed_s spans
construction through decode return/raise; build_elapsed_s and decode_elapsed_s
measure the individual calls. Full-batch output checks and small-batch input
checks run after the benchmark, outside both timing windows. No I/O or stdout
redirection is added to the submission's decode method.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from qec_benchmark.config import (
    DEFAULT_SEED, DEFAULT_SHOTS, DEFAULT_TIME_LIMIT, VALIDATE_SEEDS,
    challenge_grid, tiny_grid,
)
from qec_benchmark.evaluation import run_benchmark
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment

if __package__:
    from .run_experiment import _contract_error, _load_final_build_decoder, _maxrss_bytes, _provenance
else:
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_experiment import _contract_error, _load_final_build_decoder, _maxrss_bytes, _provenance


GRIDS = {"challenge": challenge_grid, "tiny": tiny_grid}
TIMING_METHOD = (
    "perf_counter wall time from build_decoder entry through decode return/raise, "
    "including construction and small wrapper overhead; imports, sampling, "
    "output validation and input probes excluded. The unmodified run_benchmark "
    "enforces its own monotonic 2.5s window including all wrapper overhead."
)
MEMORY_METHOD = (
    "resource.getrusage(RUSAGE_SELF).ru_maxrss normalized to bytes "
    "(Darwin bytes; Linux KiB); whole-process high-water, including sampling, "
    "retained predictions and contract probes, not isolated decoder memory."
)


class _InstrumentedDecoder:
    def __init__(self, decoder, record: dict, started: float):
        self.decoder = decoder
        self.record = record
        self.started = started

    def decode(self, syndromes):
        started = time.perf_counter()
        try:
            try:
                predictions = self.decoder.decode(syndromes)
            finally:
                ended = time.perf_counter()
                self.record["decode_elapsed_s"] = ended - started
                self.record["elapsed_s"] = ended - self.started
        except Exception as exc:
            self.record.update(
                exception=repr(exc), exception_type=type(exc).__name__, failure_stage="decode",
            )
            raise
        # Retain predictions only (not million-shot syndrome arrays/copies).
        # Their shape/dtype/values are checked after the official timer stops.
        self.record["_predictions"] = predictions
        return predictions


def _instrumented_factory(build_decoder, records: list[dict], seed: int):
    def factory(point):
        record = {
            "seed": seed, "L": point.L, "p": point.p, "xi": point.xi,
            "exception": None, "exception_type": None, "failure_stage": None,
            "build_elapsed_s": 0.0, "decode_elapsed_s": 0.0, "elapsed_s": 0.0,
        }
        records.append(record)
        started = time.perf_counter()
        try:
            try:
                decoder = build_decoder(point)
            finally:
                ended = time.perf_counter()
                record["build_elapsed_s"] = ended - started
                record["elapsed_s"] = ended - started
        except Exception as exc:
            record.update(
                exception=repr(exc), exception_type=type(exc).__name__, failure_stage="build",
            )
            raise
        return _InstrumentedDecoder(decoder, record, started)

    return factory


def _check_input_contract(build_decoder, point) -> dict:
    """Separate 8-shot probes: both accepted input dtypes, immutable and empty input."""
    check = {"L": point.L, "p": point.p, "xi": point.xi, "passed": False, "exception": None}
    try:
        experiment = SurfaceCodeExperiment(distance=point.L)
        syndromes, _ = experiment.sample_correlated(
            shots=8, p=point.p, xi=point.xi, rng=np.random.default_rng(DEFAULT_SEED),
        )
        decoder = build_decoder(point)
        expected = None
        for dtype in (np.bool_, np.uint8):
            for readonly in (False, True):
                batch = syndromes.astype(dtype)
                before = batch.copy()
                batch.setflags(write=not readonly)
                predictions = decoder.decode(batch)
                error = _contract_error(predictions, len(before))
                if error is not None:
                    raise ValueError(error)
                if (batch.shape != before.shape or batch.dtype != before.dtype
                        or not np.array_equal(batch, before)
                        or batch.flags.writeable != (not readonly)):
                    raise ValueError("decode modified its syndrome input or writeability")
                if expected is not None and not np.array_equal(predictions, expected):
                    raise ValueError("decode is not repeatable across equivalent inputs")
                expected = predictions.copy()
            empty = np.empty((0, experiment.num_detectors), dtype=dtype)
            empty.setflags(write=False)
            error = _contract_error(decoder.decode(empty), 0)
            if error is not None:
                raise ValueError(error)
            if empty.shape != (0, experiment.num_detectors) or empty.flags.writeable:
                raise ValueError("decode modified its empty input")
        check["passed"] = True
    except Exception as exc:
        check.update(exception=repr(exc), exception_type=type(exc).__name__)
    return check


def verify(*, shots: int = DEFAULT_SHOTS, seed: int = DEFAULT_SEED,
           validate: bool = False, grid_name: str = "challenge") -> dict:
    if shots <= 0:
        raise ValueError("shots must be positive")
    seeds = list(VALIDATE_SEEDS) if validate else [seed]
    report = {
        "grid": grid_name, "shots": shots, "seed": seed, "seeds": seeds,
        "points": [], "seed_scores": [], "input_contract_checks": [],
        "score": None, "score_errors_per_million": None,
        "time_limit": DEFAULT_TIME_LIMIT, "passed": False,
        "timed_out": False, "crashed": False,
        "timing_method": TIMING_METHOD, "memory_method": MEMORY_METHOD,
        "input_check_method": "Separate post-benchmark 8-shot bool/uint8, writable/read-only, repeatability and empty-batch probes; no full-batch immutability claim",
        "exception": None, "exception_type": None,
    }
    try:
        report.update(_provenance())
        grid = GRIDS[grid_name]()
        build_decoder = _load_final_build_decoder()
        for current_seed in seeds:
            records: list[dict] = []
            result = run_benchmark(
                build_decoder_fn=_instrumented_factory(build_decoder, records, current_seed),
                grid=grid, shots_per_point=shots, seed=current_seed,
                time_limit=DEFAULT_TIME_LIMIT,
            )
            if len(records) != len(grid) or len(result.point_results) != len(grid):
                raise RuntimeError("instrumentation did not observe every benchmark point")
            for official, record in zip(result.point_results, records):
                if (record["L"], record["p"], record["xi"]) != (official.L, official.p, official.xi):
                    raise RuntimeError("instrumentation point order differs from benchmark")
                predictions = record.pop("_predictions", None)
                contract_error = _contract_error(predictions, shots) if record["exception"] is None else None
                record.update(
                    shots=official.shots, errors=official.errors, error_rate=official.error_rate,
                    # Preserve the official failure flag, which also covers exceptions.
                    timed_out=official.timed_out,
                    time_limit_exceeded=record["elapsed_s"] > DEFAULT_TIME_LIMIT,
                    crashed=record["exception"] is not None,
                    contract_valid=record["exception"] is None and contract_error is None,
                    contract_error=contract_error,
                    process_maxrss_bytes=_maxrss_bytes(),
                    process_rss_bytes=_maxrss_bytes(),
                )
                report["points"].append(record)
            report["seed_scores"].append({"seed": current_seed, "score": result.score})

        # These probes come last so they cannot warm the timed decoder builds.
        report["input_contract_checks"] = [_check_input_contract(build_decoder, point) for point in grid]
        total_errors = sum(point["errors"] for point in report["points"])
        total_shots = sum(point["shots"] for point in report["points"])
        score = round(total_errors * 1_000_000 / total_shots) if total_shots else 0
        report.update(
            total_errors=total_errors, total_shots=total_shots,
            score=score, score_errors_per_million=score,
            mean_seed_score=round(sum(row["score"] for row in report["seed_scores"]) / len(seeds)),
            size_valid=report["solve_size_bytes"] < 200_000,
        )
        report["timed_out"] = any(point["timed_out"] or point["time_limit_exceeded"] for point in report["points"])
        report["crashed"] = any(point["crashed"] for point in report["points"])
        report["passed"] = (
            report["size_valid"]
            and all(not point["timed_out"] and not point["time_limit_exceeded"]
                    and not point["crashed"] and point["contract_valid"]
                    for point in report["points"])
            and all(check["passed"] for check in report["input_contract_checks"])
        )
    except Exception as exc:
        report.update(exception=repr(exc), exception_type=type(exc).__name__, crashed=True)
    report["process_maxrss_bytes"] = _maxrss_bytes()
    report["process_rss_bytes"] = report["process_maxrss_bytes"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify solve.py with the official benchmark")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate", action="store_true", help="use the official validation seeds")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="challenge")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    report = verify(shots=args.shots, seed=args.seed, validate=args.validate, grid_name=args.grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
