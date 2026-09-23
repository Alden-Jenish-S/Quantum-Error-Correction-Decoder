"""Reproducible, paired hybrid validation; no benchmark infrastructure changes.

Smoke: uv run python scripts/validate_hybrid.py --shots 1000 --seeds 42 --label smoke
Full:  uv run python scripts/validate_hybrid.py --label official_1m_5seeds --compatibility

The defaults are the official five seeds and 1M shots at each of 24 points.
One RNG per seed, one unchanged sampler call per point, in challenge_grid order.
All three variants see the same read-only shot array. Only construction and
decode are timed; contract checks, sampling, scoring and reporting are outside.
Timeouts are measured after return (the official harness's soft-limit semantics),
with all-wrong scored errors and separately retained observed errors. Paired
rescues/harms describe returned predictions, even if a soft timeout occurred.

The pre-hybrid baseline is the explicit, unchanged DataOnlyMWPM class, never a
dynamic snapshot of build_decoder. Its injection circuit is independently checked.
Memory is bounded by one point of <=1M shots; only small records persist. RSS is
normalized to bytes and is the entire process high-water mark, including sampling
and all decoders, not incremental memory attributable to an individual decoder.
The optional unchanged run.py validation is a separate subprocess after the
paired suite; its complete report is retained outside all decoder timing regions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qec_benchmark.baselines import MWPMDecoder
from qec_benchmark.config import DEFAULT_TIME_LIMIT, VALIDATE_SEEDS, challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment
from solve import DataOnlyMWPM, _data_only_circuit, build_decoder


VARIANTS = ("circuit_mwpm", "pre_hybrid_data_only", "hybrid")
REFERENCES = VARIANTS[:2]
MEMORY_LIMIT_BYTES = 16 * 1024**3
FROZEN_SHA256 = "debb61816d64cb556cc58c17a3a1a9cf5d0a8c90c08fe3d480d9d931cf29c119"
SOURCE_PATHS = (
    "solve.py", "run.py", "scripts/validate_hybrid.py",
    "tests/test_submission.py", "tests/test_hybrid_validation.py",
    "src/qec_benchmark/config.py", "src/qec_benchmark/models.py",
    "src/qec_benchmark/baselines.py", "src/qec_benchmark/evaluation.py",
    "src/qec_benchmark/noise.py", "src/qec_benchmark/stim_surface_code.py",
    "experiments/tracks/A/frozen_tables.py", "experiments/tracks/A/tables_qmc.json",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def normalized_rss_bytes(raw, system):
    """Darwin reports bytes; Linux/BSD resource reports KiB."""
    return int(raw * (1 if system == "Darwin" else 1024))


def process_rss_bytes():
    return normalized_rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                platform.system())


def provenance():
    return {
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "stim", "pymatching")},
        "source_sha256": source_hashes(),
        "submission_bytes": (ROOT / "solve.py").stat().st_size,
        "memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; Darwin bytes, "
                         "otherwise KiB converted to bytes; cumulative whole-process "
                         "high-water RSS including sampling and all three variants; "
                         "not per-decoder incremental memory; excludes compatibility child",
    }


def verify_pre_hybrid_circuits(experiments, grid):
    """Verify the retained baseline against the original benchmark split/order."""
    checked = []
    for L, p in dict.fromkeys((point.L, point.p) for point in grid):
        experiment = experiments[L]
        original = stim.Circuit()
        original += experiment._prefix
        for qubit in experiment._data_qubits:
            original.append("X_ERROR", [int(qubit)], float(p))
        original += experiment._suffix
        if _data_only_circuit(L, p) != original:
            raise AssertionError("pre-hybrid injection circuit changed")
        checked.append({"L": L, "p": p})
    return checked


def check_contract(predictions, syndromes, before, shots):
    if not isinstance(predictions, np.ndarray):
        raise ValueError("predictions must be a numpy array")
    if predictions.shape != (shots,) or predictions.dtype != np.uint8:
        raise ValueError("predictions must have shape (shots,) and dtype uint8")
    if not np.all((predictions == 0) | (predictions == 1)):
        raise ValueError("predictions must be binary")
    if syndromes.flags.writeable or not np.array_equal(syndromes, before):
        raise ValueError("decoder changed its input")


def measure_variant(factory, point, syndromes, truth, before, time_limit,
                    *, clock=time.perf_counter):
    shots = len(truth)
    prediction = None
    failure = None
    built = None
    start = clock()
    try:
        decoder = factory(point)
        built = clock()
        prediction = decoder.decode(syndromes)
    except Exception as exc:
        # Exception types identify the failure without leaking private paths.
        failure = type(exc).__name__
    ended = clock()
    elapsed = ended - start
    over_limit = time_limit is not None and elapsed > time_limit
    status = "exception" if failure else "timeout" if over_limit else "ok"
    # This entire block is deliberately outside the clock interval.
    wrong = None
    if failure is None:
        try:
            check_contract(prediction, syndromes, before, shots)
        except ValueError as exc:
            failure = str(exc)
            status = "contract_error"
        else:
            wrong = prediction != truth
    errors = int(np.count_nonzero(wrong)) if wrong is not None else None
    scored_errors = errors if status == "ok" else shots
    return {
        "status": status, "failure": failure,
        "errors": errors, "error_rate": errors / shots if errors is not None else None,
        "scored_errors": scored_errors, "scored_error_rate": scored_errors / shots,
        "timed_out": over_limit, "official_all_wrong": status != "ok",
        "build_s": built - start if built is not None else None,
        "decode_s": ended - built if built is not None else None,
        "build_decode_s": elapsed,
        "whole_process_high_water_rss_bytes": process_rss_bytes(),
    }, wrong


def paired_counts(hybrid_wrong, reference_wrong):
    if hybrid_wrong is None or reference_wrong is None:
        return None
    rescue = int(np.count_nonzero(reference_wrong & ~hybrid_wrong))
    harm = int(np.count_nonzero(hybrid_wrong & ~reference_wrong))
    return {"rescue": rescue, "harm": harm, "net_saved_errors": rescue - harm,
            "disagreements": rescue + harm,
            "both_wrong": int(np.count_nonzero(reference_wrong & hybrid_wrong)),
            "both_correct": int(np.count_nonzero(~reference_wrong & ~hybrid_wrong))}


def group_summary(rows):
    shots = sum(row["shots"] for row in rows)
    decoders = {}
    for name in VARIANTS:
        values = [row["decoders"][name] for row in rows]
        observed_shots = sum(row["shots"] for row in rows
                             if row["decoders"][name]["errors"] is not None)
        errors = sum(v["errors"] for v in values if v["errors"] is not None)
        scored = sum(v["scored_errors"] for v in values)
        decoders[name] = {
            "observed_errors": errors, "observed_shots": observed_shots,
            "observed_error_rate": errors / observed_shots if observed_shots else None,
            "scored_errors": scored, "score_errors_per_million": round(scored * 1e6 / shots),
            "timeouts": sum(v["timed_out"] for v in values),
            "failed_contract_or_exception": sum(v["status"] in ("exception", "contract_error") for v in values),
            "max_build_decode_s": max(v["build_decode_s"] for v in values),
            "mean_build_decode_s": sum(v["build_decode_s"] for v in values) / len(values),
            "total_build_s": sum(v["build_s"] for v in values if v["build_s"] is not None),
            "total_decode_s": sum(v["decode_s"] for v in values if v["decode_s"] is not None),
        }
    paired = {}
    for name in REFERENCES:
        valid = [row for row in rows if row["paired"][name] is not None]
        paired_shots = sum(row["shots"] for row in valid)
        rescue = sum(row["paired"][name]["rescue"] for row in valid)
        harm = sum(row["paired"][name]["harm"] for row in valid)
        paired[name] = {
            "rescue": rescue, "harm": harm, "net_saved_errors": rescue - harm,
            "shots": paired_shots, "unavailable_rows": len(rows) - len(valid),
            "net_saved_error_rate": (rescue - harm) / paired_shots if paired_shots else None,
            "improved_point_seed_rows": sum(row["paired"][name]["net_saved_errors"] > 0 for row in valid),
            "regressed_point_seed_rows": sum(row["paired"][name]["net_saved_errors"] < 0 for row in valid),
            "equal_point_seed_rows": sum(row["paired"][name]["net_saved_errors"] == 0 for row in valid),
        }
    return {"rows": len(rows), "shots_per_decoder": shots, "decoders": decoders, "paired": paired}


def summarize(rows):
    result = {"overall": group_summary(rows)}
    for field in ("seed", "L", "p", "xi"):
        result[f"by_{field}"] = {
            str(value): group_summary([r for r in rows if r[field] == value])
            for value in dict.fromkeys(row[field] for row in rows)
        }
    result["by_point"] = {
        point.key(): group_summary([r for r in rows
                                   if (r["L"], r["p"], r["xi"]) == (point.L, point.p, point.xi)])
        for point in challenge_grid()
        if any((r["L"], r["p"], r["xi"]) == (point.L, point.p, point.xi) for r in rows)
    }
    result["regressions_by_point"] = {
        name: {key: group["paired"][name]["net_saved_errors"]
               for key, group in result["by_point"].items()
               if group["paired"][name]["net_saved_errors"] < 0}
        for name in REFERENCES
    }
    return result


def evaluate(*, shots, seeds, time_limit, emit, progress=False):
    grid = challenge_grid()
    experiments = {L: SurfaceCodeExperiment(L) for L in dict.fromkeys(p.L for p in grid)}
    circuit_checks = verify_pre_hybrid_circuits(experiments, grid)
    factories = {
        "circuit_mwpm": lambda point: MWPMDecoder(point, weighted=True),
        "pre_hybrid_data_only": DataOnlyMWPM,
        "hybrid": build_decoder,
    }
    rows = []
    start = time.perf_counter()
    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for point_index, point in enumerate(grid):
            point_start = time.perf_counter()
            syndromes, truth = experiments[point.L].sample_correlated(
                shots=shots, p=point.p, xi=point.xi, rng=rng)
            sampled = time.perf_counter()
            if process_rss_bytes() >= MEMORY_LIMIT_BYTES:
                raise MemoryError("whole-process RSS exceeded the 16 GiB validation budget")
            syndromes.setflags(write=False)
            before = syndromes.copy()
            # Rotate timing position without consuming any sampling RNG state.
            offset = (seed_index + point_index) % len(VARIANTS)
            order = VARIANTS[offset:] + VARIANTS[:offset]
            row = {"seed": seed, "point_index": point_index, "L": point.L,
                   "p": point.p, "xi": point.xi, "shots": shots,
                   "sampling_s": sampled - point_start,
                   "variant_order": list(order), "decoders": {}, "paired": {}}
            wrong = {}
            for name in order:
                row["decoders"][name], wrong[name] = measure_variant(
                    factories[name], point, syndromes, truth, before, time_limit)
            for name in REFERENCES:
                row["paired"][name] = paired_counts(wrong["hybrid"], wrong[name])
            row["point_wall_s"] = time.perf_counter() - point_start
            rows.append(row)
            emit(row)
            if process_rss_bytes() >= MEMORY_LIMIT_BYTES:
                raise MemoryError("whole-process RSS exceeded the 16 GiB validation budget")
            # Release each batch before sampling the next point; no raw data saved.
            del syndromes, truth, before, wrong
        if progress:
            seed_rows = [row for row in rows if row["seed"] == seed]
            print(json.dumps({"seed_completed": seed, "scored_errors": {
                name: sum(r["decoders"][name]["scored_errors"] for r in seed_rows)
                for name in VARIANTS}}), flush=True)
    return rows, {"paired_wall_s": time.perf_counter() - start,
                  "pre_hybrid_circuit_checks": circuit_checks}


def parse_compatibility_report(text):
    seeds = {int(seed): int(score.replace(",", "")) for seed, score in
             re.findall(r"seed\s+(\d+):\s+([\d,]+) errors/M", text)}
    score = re.search(r"Score\s*:\s*([\d,]+) errors per million \(mean\)", text)
    timeouts = re.search(r"Timeouts\s*:\s*(\d+)\s*/\s*(\d+)", text)
    if score is None or timeouts is None or set(seeds) != set(VALIDATE_SEEDS):
        raise ValueError("incomplete standard validation report")
    return {"per_seed_scores": {str(k): v for k, v in seeds.items()},
            "mean_score": int(score.group(1).replace(",", "")),
            "timeouts": int(timeouts.group(1)), "point_seed_count": int(timeouts.group(2))}


def run_compatibility(shots, report_path, *, timeout_s=1800):
    command = ["run.py", "--validate", "--shots", str(shots)]
    start = time.perf_counter()
    result = {"command": ["python", *command], "timeout_s": timeout_s,
              "report": report_path.name, "status": "ok"}
    try:
        completed = subprocess.run([sys.executable, *command], cwd=ROOT,
                                   text=True, capture_output=True, timeout=timeout_s, check=False)
        text = completed.stdout + completed.stderr
        result["returncode"] = completed.returncode
        if completed.returncode:
            result["status"] = "failed"
    except subprocess.TimeoutExpired as exc:
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        text = as_text(exc.stdout) + as_text(exc.stderr)
        result["status"] = "subprocess_timeout"
    result["wall_s"] = time.perf_counter() - start
    # Preserve output but redact local paths if a subprocess failure printed any.
    report_path.write_text(text.replace(str(ROOT), "<workspace>").replace(str(Path.home()), "<home>"))
    if result["status"] == "ok":
        try:
            result.update(parse_compatibility_report(text))
        except ValueError:
            result["status"] = "unparseable_report"
    result["terminated_children_high_water_rss_bytes"] = normalized_rss_bytes(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, platform.system())
    result["memory_method"] = (
        "RUSAGE_CHILDREN high-water RSS for terminated children; "
        "normalized to bytes; separate from paired parent; not summed"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=VALIDATE_SEEDS)
    parser.add_argument("--label", default="official_1m_5seeds")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--compatibility", action="store_true")
    parser.add_argument("--compatibility-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if not 0 < args.shots <= 1_000_000:
        parser.error("--shots must be in [1, 1000000] for the one-point memory bound")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label must contain only letters, digits, underscores or hyphens")
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive")
    if not np.isfinite(args.compatibility_timeout) or args.compatibility_timeout <= 0:
        parser.error("--compatibility-timeout must be finite and positive")
    if args.compatibility and (args.seeds != VALIDATE_SEEDS or args.time_limit != DEFAULT_TIME_LIMIT):
        parser.error("--compatibility requires official seeds/order and default time limit")

    folder = ROOT / "experiments/hybrid"
    folder.mkdir(parents=True, exist_ok=True)
    records_path = folder / f"{args.label}.jsonl"
    summary_path = folder / f"{args.label}.summary.json"
    report_path = folder / f"{args.label}.standard_validate.txt"
    for path in (records_path, summary_path, report_path):
        if path.exists():
            parser.error(f"receipt already exists: {path.name}; choose a new --label")

    started = time.perf_counter()
    summary = {"schema_version": 1, "label": args.label, "seeds": args.seeds,
               "shots_per_point": args.shots, "time_limit_s": args.time_limit,
               "expected_point_seed_rows": 24 * len(args.seeds),
               "provenance_before": provenance(),
               "rng_semantics": "one default_rng(seed), exact challenge_grid order, one sampler call per point, shared shots",
               "baseline": "explicit retained DataOnlyMWPM class; independent circuit equality verified",
               "table_method": "offline frozen approximate QMC-MAP; no benchmark training or runtime integration",
               "paired_semantics": "rescue=reference wrong/hybrid right; harm=hybrid wrong/reference right; positive net saved is improvement; raw predictions even on soft timeout",
               "memory_limit_bytes": MEMORY_LIMIT_BYTES,
               "status": "running"}
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = []
    try:
        if summary["provenance_before"]["source_sha256"]["experiments/tracks/A/frozen_tables.py"] != FROZEN_SHA256:
            raise ValueError("reviewed frozen table source hash changed")
        with records_path.open("x") as stream:
            def emit(row):
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
            _, timing = evaluate(shots=args.shots, seeds=args.seeds, time_limit=args.time_limit,
                                 emit=emit, progress=True)
        summary.update(timing)
        summary.update(summarize(rows))
        summary["status"] = "completed"
        summary["source_sha256_after_paired"] = source_hashes()
        # Save the paired result before starting the independent compatibility run.
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if args.compatibility:
            compatibility = run_compatibility(args.shots, report_path,
                                              timeout_s=args.compatibility_timeout)
            expected = {seed: result["decoders"]["hybrid"]["score_errors_per_million"]
                        for seed, result in summary["by_seed"].items()}
            compatibility["matches_paired_per_seed_scores"] = compatibility.get("per_seed_scores") == expected
            compatibility["expected_paired_per_seed_scores"] = expected
            compatibility["expected_mean_of_rounded_scores"] = round(sum(expected.values()) / len(expected))
            compatibility["matches_paired_mean_score"] = compatibility.get("mean_score") == compatibility["expected_mean_of_rounded_scores"]
            summary["standard_compatibility"] = compatibility
    except (Exception, KeyboardInterrupt) as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["failure_type"] = type(exc).__name__
        summary["completed_rows"] = len(rows)
        if rows:
            summary.update(summarize(rows))
    finally:
        summary["provenance_after"] = provenance()
        before = summary["provenance_before"]["source_sha256"]
        summary["sources_unchanged"] = before == summary["provenance_after"]["source_sha256"]
        summary["wall_s_including_compatibility"] = time.perf_counter() - started
        summary["whole_process_high_water_rss_bytes"] = process_rss_bytes()
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    valid = summary["status"] == "completed" and summary["sources_unchanged"]
    if rows:
        valid &= all(not v["official_all_wrong"] for row in rows for v in row["decoders"].values())
    if args.compatibility:
        check = summary.get("standard_compatibility", {})
        valid &= (check.get("status") == "ok" and check.get("timeouts") == 0
                  and check.get("point_seed_count") == 120
                  and check.get("matches_paired_per_seed_scores", False)
                  and check.get("matches_paired_mean_score", False))
    print(json.dumps({"summary": f"experiments/hybrid/{summary_path.name}",
                      "valid": bool(valid), "status": summary["status"],
                      "sources_unchanged": summary["sources_unchanged"],
                      "whole_process_high_water_rss_bytes": summary["whole_process_high_water_rss_bytes"],
                      "wall_s": summary["wall_s_including_compatibility"]}), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
