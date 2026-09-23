"""Run reproducible pointwise decoder experiments without changing the challenge evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from qec_benchmark.config import DEFAULT_TIME_LIMIT, VALIDATE_SEEDS, challenge_grid, tiny_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment

if __package__:
    from .candidates import DECODERS
else:
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from candidates import DECODERS


ROOT = Path(__file__).resolve().parents[1]


def _load_final_build_decoder():
    """Load the root submission once, outside all per-point timing windows."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("final_submission", ROOT / "solve.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load root solve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_decoder


def _maxrss_bytes() -> int:
    """Return this process's high-water RSS in bytes on Darwin and Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    # Linux reports ru_maxrss in KiB.  This is deliberately process-wide
    # high-water memory, not an estimate of decoder-only allocation.
    return value * 1024


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _provenance() -> dict:
    source = (ROOT / "solve.py").read_bytes()
    return {
        "solve_sha256": hashlib.sha256(source).hexdigest(),
        "solve_size_bytes": len(source),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "stim": _package_version("stim"),
            "pymatching": _package_version("pymatching"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "git_revision": _git_revision(),
    }


def _git_revision() -> str | None:
    """Return HEAD when present, including None for an unborn repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _contract_error(predictions: object, shots: int) -> str | None:
    if not isinstance(predictions, np.ndarray):
        return "decode must return a numpy.ndarray"
    if predictions.shape != (shots,):
        return f"decode returned shape {predictions.shape!r}; expected {(shots,)!r}"
    if predictions.dtype != np.uint8:
        return f"decode returned dtype {predictions.dtype!s}; expected uint8"
    if not np.all((predictions == 0) | (predictions == 1)):
        return "decode returned non-binary predictions"
    return None


def run(decoder_name: str, shots: int, seeds: list[int], grid_name: str,
        time_limit: float | None = DEFAULT_TIME_LIMIT) -> list[dict]:
    if shots <= 0 or not seeds:
        raise ValueError("shots must be positive and seeds must not be empty")
    if grid_name not in {"challenge", "tiny"}:
        raise ValueError("unknown grid")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive, or None")
    grid = challenge_grid() if grid_name == "challenge" else tiny_grid()
    # Importing solve.py can perform setup and filesystem work.  Do it once
    # before any point is timed; build_decoder itself remains per-point.
    decoder_factory = _load_final_build_decoder() if decoder_name == "final" else DECODERS[decoder_name]
    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        experiments: dict[int, SurfaceCodeExperiment] = {}
        for point in grid:
            if point.L not in experiments:
                experiments[point.L] = SurfaceCodeExperiment(distance=point.L)
            experiment = experiments[point.L]
            syndromes, truth = experiment.sample_correlated(shots=shots, p=point.p, xi=point.xi, rng=rng)
            rss_before = _maxrss_bytes()
            build_elapsed = 0.0
            decode_elapsed = 0.0
            exception = None
            exception_type = None
            crashed = False
            failure_stage = None
            contract_valid = None
            stage = "build"
            started = time.perf_counter()
            try:
                try:
                    decoder = decoder_factory(point)
                finally:
                    ended = time.perf_counter()
                    build_elapsed = ended - started
                stage = "decode"
                decode_start = time.perf_counter()
                try:
                    predictions = decoder.decode(syndromes)
                finally:
                    ended = time.perf_counter()
                    decode_elapsed = ended - decode_start
            except Exception as exc:  # match challenge all-wrong behavior, while recording the cause
                exception = repr(exc)
                exception_type = type(exc).__name__
                crashed = True
                failure_stage = stage
            elapsed = ended - started
            # Scoring and contract validation are outside construction/decode
            # timing, including for a completed decode over the time limit.
            if not crashed:
                contract_error = _contract_error(predictions, shots)
                contract_valid = contract_error is None
                if contract_error is not None:
                    exception = repr(ValueError(contract_error))
                    exception_type = "ValueError"
                    failure_stage = "contract"
            time_limit_exceeded = time_limit is not None and elapsed > time_limit
            timed_out = time_limit_exceeded or exception is not None
            errors = shots if timed_out else int(np.sum(predictions != truth))
            rss_after = _maxrss_bytes()
            rows.append({
                "decoder": decoder_name,
                "seed": seed,
                "L": point.L,
                "p": point.p,
                "xi": point.xi,
                "shots": shots,
                "errors": errors,
                "error_rate": errors / shots if shots else 0.0,
                "elapsed_s": elapsed,
                "build_elapsed_s": build_elapsed,
                "decode_elapsed_s": decode_elapsed,
                "timed_out": timed_out,
                "time_limit_exceeded": time_limit_exceeded,
                "crashed": crashed,
                "contract_valid": contract_valid,
                "exception": exception,
                "exception_type": exception_type,
                "failure_stage": failure_stage,
                "process_maxrss_before_bytes": rss_before,
                "process_maxrss_bytes": rss_after,
                "process_rss_bytes": rss_after,
                "process_maxrss_delta_bytes": max(0, rss_after - rss_before),
                "python": platform.python_version(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=sorted((*DECODERS, "final")), default="baseline")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--grid", choices=["tiny", "challenge"], default="challenge")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--no-time-limit", action="store_true", help="disable the per-point time limit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive")
    seeds = VALIDATE_SEEDS if args.validate else [42 if args.seed is None else args.seed]
    time_limit = None if args.no_time_limit else args.time_limit
    provenance = _provenance()
    rows = run(args.decoder, args.shots, seeds, args.grid, time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    process_rss_bytes = _maxrss_bytes()
    summary = {
        "decoder": args.decoder,
        "grid": args.grid,
        "shots": args.shots,
        "seeds": seeds,
        "time_limit": time_limit,
        "timing_method": "perf_counter wall time from build_decoder entry through decode return/raise; imports, sampling, contract checks and scoring excluded",
        "memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss, normalized to bytes (Darwin bytes; Linux KiB); whole-process high-water including sampling, not isolated decoder memory",
        "rows": len(rows),
        "total_errors": sum(r.get("errors", 0) for r in rows),
        "total_shots": sum(r.get("shots", 0) for r in rows),
        "score_errors_per_million": round(sum(r.get("errors", 0) for r in rows) * 1_000_000 / sum(r.get("shots", 0) for r in rows)),
        "process_maxrss_bytes": process_rss_bytes,
        "process_rss_bytes": process_rss_bytes,
        **provenance,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
