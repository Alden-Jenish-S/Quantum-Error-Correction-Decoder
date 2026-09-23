"""Reproducible Track-C feature, paired-batch, fitting, and runtime runner."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from candidate_c import (  # noqa: E402
    MODEL_FEATURES,
    MODEL_INTERCEPT,
    MODEL_MEAN,
    MODEL_SCALE,
    MODEL_THRESHOLD,
    MODEL_COEFFICIENTS,
    DECODERS,
    FeatureBatch,
    ResidualDecoder,
    build_geometry,
    extract_features,
)
from qec_benchmark.baselines import MWPMDecoder  # noqa: E402
from qec_benchmark.config import VALIDATE_SEEDS, challenge_grid  # noqa: E402
from qec_benchmark.models import ParameterPoint  # noqa: E402
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment  # noqa: E402


def _load_final():
    spec = importlib.util.spec_from_file_location("track_c_final_submission", ROOT / "solve.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load solve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_decoder


def _output_path(name: str | Path) -> Path:
    path = (HERE / name).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("Track-C outputs must stay under experiments/tracks/C")
    return path


def _maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _current_rss_bytes() -> int:
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(output.strip()) * 1024


def _source_hashes() -> dict[str, str]:
    paths = [
        "solve.py",
        "src/qec_benchmark/stim_surface_code.py",
        "src/qec_benchmark/baselines.py",
        "src/qec_benchmark/config.py",
        "experiments/tracks/C/candidate_c.py",
        "experiments/tracks/C/run_track_c.py",
    ]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def _versions() -> dict[str, str | None]:
    names = ("numpy", "scipy", "stim", "pymatching")
    return {name: importlib.metadata.version(name) for name in names}


def _contract(prediction: object, shots: int) -> str | None:
    if not isinstance(prediction, np.ndarray):
        return "not ndarray"
    if prediction.shape != (shots,):
        return f"shape {prediction.shape}, expected {(shots,)}"
    if prediction.dtype != np.uint8:
        return f"dtype {prediction.dtype}, expected uint8"
    if not np.all((prediction == 0) | (prediction == 1)):
        return "non-binary output"
    return None


def _strata(features: FeatureBatch, truth: np.ndarray, base: np.ndarray, pred: np.ndarray) -> list[dict]:
    f = features.values
    weight = f["syndrome_weight"].astype(np.int16)
    classes = f["cluster_class"].astype(np.int16)
    keys = np.stack([np.minimum(weight, 8), classes], axis=1)
    rows = []
    for weight_bin, cluster_class in sorted(set(map(tuple, keys.tolist()))):
        mask = (keys[:, 0] == weight_bin) & (keys[:, 1] == cluster_class)
        if not np.any(mask):
            continue
        base_wrong = base[mask] != truth[mask]
        candidate_wrong = pred[mask] != truth[mask]
        rows.append({
            "weight_bin": int(weight_bin),
            "cluster_class": int(cluster_class),
            "shots": int(mask.sum()),
            "base_errors": int(base_wrong.sum()),
            "candidate_errors": int(candidate_wrong.sum()),
            "rescued": int((base_wrong & ~candidate_wrong).sum()),
            "harmed": int((~base_wrong & candidate_wrong).sum()),
        })
    return rows


def _paired(base_wrong: np.ndarray, candidate_wrong: np.ndarray) -> dict:
    rescued = int((base_wrong & ~candidate_wrong).sum())
    harmed = int((~base_wrong & candidate_wrong).sum())
    n = int(base_wrong.size)
    delta = (harmed - rescued) / n if n else 0.0
    discordant = rescued + harmed
    se = np.sqrt(max(0.0, discordant / n - delta * delta) / n) if n else 0.0
    return {
        "shots": n,
        "rescued": rescued,
        "harmed": harmed,
        "delta_errors": harmed - rescued,
        "delta_error_rate": delta,
        "normal_ci95": [delta - 1.95996398454 * se, delta + 1.95996398454 * se],
        "mcnemar_exact_p": float(binomtest(rescued, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def fit_classifier(shots: int, seeds: list[int], ridge: float = 10.0) -> dict:
    """Fit a small frozen linear score using only explicitly supplied seeds."""
    if not seeds:
        raise ValueError("at least one training seed is required")
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    experiments = {L: SurfaceCodeExperiment(L) for L in (3, 5, 7)}
    geometries = {L: build_geometry(L) for L in (3, 5, 7)}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for point in challenge_grid():
            syndrome, truth = experiments[point.L].sample_correlated(
                shots=shots, p=point.p, xi=point.xi, rng=rng
            )
            base_decoder = ResidualDecoder(point, "noop")
            base = base_decoder._base_decode(syndrome)
            features = extract_features(syndrome, geometries[point.L], base, point)
            X_parts.append(features.matrix())
            y_parts.append((base != truth).astype(np.float64))
    X = np.concatenate(X_parts, axis=0).astype(np.float64, copy=False)
    y = np.concatenate(y_parts, axis=0)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    Z = (X - mean[None, :]) / scale[None, :]
    gram = Z.T @ Z
    gram.flat[:: gram.shape[0] + 1] += ridge
    intercept = float(y.mean())
    coefficients = np.linalg.solve(gram, Z.T @ (y - intercept))
    raw_score = Z @ coefficients
    # Pick the operating point only on the training shots.  Ties prefer fewer
    # changes, so the selected rule is conservative when several are equivalent.
    candidates = np.unique(np.r_[raw_score, np.inf])
    best = (int(y.sum()), np.inf, np.inf)
    best_threshold = np.inf
    for threshold in candidates:
        flip = raw_score >= threshold
        errors = int(np.where(flip, 1.0 - y, y).sum())
        key = (errors, int(flip.sum()), float(threshold))
        if key < best:
            best = key
            best_threshold = float(threshold)
    flip = raw_score >= best_threshold
    return {
        "features": list(MODEL_FEATURES),
        "intercept": intercept,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "threshold": best_threshold,
        "ridge": ridge,
        "training_seeds": seeds,
        "training_shots_per_point": shots,
        "training_rows": int(len(y)),
        "training_base_errors": int(y.sum()),
        "training_flips": int(flip.sum()),
        "training_score_errors": int(np.where(flip, 1.0 - y, y).sum()),
        "selection": "ridge least-squares risk score; threshold minimizes training errors, then flip count",
    }


def run(shots: int, seeds: list[int], variants: list[str], output: str) -> dict:
    if shots <= 0 or not seeds:
        raise ValueError("shots must be positive and seeds must not be empty")
    if "baseline" not in variants:
        variants = ["baseline", *variants]
    if "final" not in variants:
        variants = [*variants, "final"]
    final_factory = _load_final()
    factories = {
        "baseline": lambda point: MWPMDecoder(point, weighted=True),
        "final": final_factory,
        **{name: (lambda point, name=name: ResidualDecoder(point, name)) for name in DECODERS},
    }
    rows: list[dict] = []
    experiments = {L: SurfaceCodeExperiment(L) for L in (3, 5, 7)}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for point_index, point in enumerate(challenge_grid()):
            syndrome, truth = experiments[point.L].sample_correlated(
                shots=shots, p=point.p, xi=point.xi, rng=rng
            )
            batch_sha = hashlib.sha256(syndrome.tobytes() + truth.tobytes()).hexdigest()
            predictions: dict[str, np.ndarray] = {}
            wrong: dict[str, np.ndarray] = {}
            base_prediction = None
            for name in variants:
                rss_before = _current_rss_bytes()
                maxrss_before = _maxrss_bytes()
                started = time.perf_counter()
                build_elapsed = 0.0
                exception = None
                decoder = None
                try:
                    decoder = factories[name](point)
                    build_elapsed = time.perf_counter() - started
                    prediction = decoder.decode(syndrome)
                except Exception as exc:  # benchmark-compatible all-wrong row
                    exception = repr(exc)
                    prediction = np.zeros(shots, dtype=np.uint8)
                ended = time.perf_counter()
                if decoder is not None and build_elapsed == 0.0:
                    build_elapsed = ended - started
                decode_elapsed = max(0.0, ended - started - build_elapsed)
                contract_error = _contract(prediction, shots)
                if contract_error is not None:
                    exception = contract_error
                timed_out = exception is not None or ended - started > 2.5
                predictions[name] = prediction
                wrong[name] = prediction != truth
                if name == "final":
                    base_prediction = prediction.copy()
                feature_strata = None
                flips = None
                if isinstance(decoder, ResidualDecoder):
                    if decoder.last_features is not None and decoder.last_base_prediction is not None:
                        feature_strata = _strata(
                            decoder.last_features, truth, decoder.last_base_prediction, prediction
                        )
                        flips = int(np.sum(prediction != decoder.last_base_prediction))
                        decoder.last_features = None
                        decoder.last_base_prediction = None
                row = {
                    "decoder": name,
                    "variant": name,
                    "seed": seed,
                    "point_index": point_index,
                    "L": point.L,
                    "p": point.p,
                    "xi": point.xi,
                    "shots": shots,
                    "errors": shots if timed_out else int(wrong[name].sum()),
                    "raw_errors": int(wrong[name].sum()),
                    "error_rate": float(wrong[name].mean()),
                    "elapsed_s": ended - started,
                    "build_elapsed_s": build_elapsed,
                    "decode_elapsed_s": decode_elapsed,
                    "timed_out": timed_out,
                    "contract_valid": contract_error is None,
                    "exception": exception,
                    "process_rss_before_bytes": rss_before,
                    "process_rss_after_bytes": _current_rss_bytes(),
                    "process_maxrss_before_bytes": maxrss_before,
                    "process_maxrss_bytes": _maxrss_bytes(),
                    "batch_sha256": batch_sha,
                    "flips_vs_base": flips,
                    "feature_strata": feature_strata,
                }
                rows.append(row)
                print(
                    f"seed={seed} L={point.L} p={point.p:g} xi={point.xi:g} "
                    f"{name}: {row['raw_errors']}/{shots} {row['elapsed_s']:.4f}s",
                    flush=True,
                )
            if base_prediction is None:
                raise RuntimeError("final decoder was not evaluated")
            # The Track-C base must agree bit-for-bit with the current final on
            # every paired batch; this catches geometry or graph drift.
            for name in ("noop", "rule", "classifier"):
                candidate_rows = [r for r in rows if r["seed"] == seed and r["point_index"] == point_index and r["variant"] == name]
                if candidate_rows and name == "noop":
                    if not np.array_equal(predictions[name], base_prediction):
                        raise AssertionError("Track-C noop is not current final MWPM")
                if candidate_rows and name != "noop":
                    base_wrong = predictions["final"] != truth
                    candidate_wrong = predictions[name] != truth
                    candidate_rows[-1]["paired_vs_final"] = _paired(base_wrong, candidate_wrong)
    path = _output_path(output)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(rows)
    summary.update({
        "decoder_variants": variants,
        "seeds": seeds,
        "shots_per_point": shots,
        "grid_points": 24,
        "time_limit_s": 2.5,
        "timing": "fresh build_decoder plus decode wall time; sampling/scoring/output excluded",
        "memory": "current RSS via ps and whole-process high-water RSS via getrusage",
        "rng_semantics": "one default_rng(seed), all 24 challenge_grid points in order, shared batches",
        "source_sha256": _source_hashes(),
        "versions": _versions(),
    })
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    aggregate = {}
    for variant, subset in sorted(grouped.items()):
        shots = sum(row["shots"] for row in subset)
        errors = sum(row["errors"] for row in subset)
        aggregate[variant] = {
            "shots": shots,
            "errors": errors,
            "score_errors_per_million": round(errors * 1_000_000 / shots) if shots else 0,
            "error_rate": errors / shots if shots else 0.0,
            "max_elapsed_s": max(row["elapsed_s"] for row in subset),
            "max_build_elapsed_s": max(row["build_elapsed_s"] for row in subset),
            "max_decode_elapsed_s": max(row["decode_elapsed_s"] for row in subset),
            "timeouts": sum(bool(row["timed_out"]) for row in subset),
            "contract_failures": sum(not row["contract_valid"] for row in subset),
            "max_process_rss_bytes": max(row["process_rss_after_bytes"] for row in subset),
            "max_process_maxrss_bytes": max(row["process_maxrss_bytes"] for row in subset),
        }
    comparisons = {}
    by_stratum = {}
    if "final" in grouped:
        final_rows = {(r["seed"], r["point_index"]): r for r in grouped["final"]}
        for variant in sorted(set(grouped) - {"final", "baseline"}):
            wins = losses = 0
            by_l_xi = defaultdict(lambda: [0, 0, 0])
            for row in grouped[variant]:
                final = final_rows[(row["seed"], row["point_index"])]
                wins += row.get("paired_vs_final", {}).get("rescued", 0)
                losses += row.get("paired_vs_final", {}).get("harmed", 0)
                key = (row["L"], row["p"], row["xi"])
                by_l_xi[key][0] += row.get("paired_vs_final", {}).get("rescued", 0)
                by_l_xi[key][1] += row.get("paired_vs_final", {}).get("harmed", 0)
                by_l_xi[key][2] += row["shots"]
            comparisons[variant] = {
                "rescued": wins,
                "harmed": losses,
                "delta_errors": losses - wins,
                "mcnemar_exact_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
                "by_L_p_xi": {
                    f"L{L}_p{p:g}_xi{xi:g}": {
                        "rescued": value[0], "harmed": value[1], "shots": value[2],
                        "delta_errors": value[1] - value[0],
                    }
                    for (L, p, xi), value in sorted(by_l_xi.items())
                },
            }
    # Aggregate the saved/harmed strata emitted by candidate rows.
    for variant in sorted(set(grouped) - {"baseline", "final", "noop"}):
        strata = defaultdict(lambda: [0, 0, 0, 0, 0])
        for row in grouped[variant]:
            for item in row.get("feature_strata") or []:
                key = (row["L"], row["p"], row["xi"], item["weight_bin"], item["cluster_class"])
                strata[key][0] += item["shots"]
                strata[key][1] += item["base_errors"]
                strata[key][2] += item["candidate_errors"]
                strata[key][3] += item["rescued"]
                strata[key][4] += item["harmed"]
        by_stratum[variant] = [
            {
                "L": key[0], "p": key[1], "xi": key[2], "weight_bin": key[3], "cluster_class": key[4],
                "shots": value[0], "base_errors": value[1], "candidate_errors": value[2],
                "rescued": value[3], "harmed": value[4], "delta_errors": value[4] - value[3],
            }
            for key, value in sorted(strata.items())
        ]
    return {"rows": len(rows), "aggregate": aggregate, "paired_vs_final": comparisons, "strata": by_stratum}


def feature_diagnostics() -> dict:
    full_feature_names = [
        "syndrome_weight", "layer0_weight", "layer1_weight", "layer_weight_imbalance",
        "temporal_disagreement", "spatial_active_weight", "adjacency_pairs",
        "component_count", "largest_component", "second_component", "component_density",
        "boundary_count", "boundary_fraction", "boundary_min_distance", "near_boundary_count",
        "row_occupied", "col_occupied", "row_max", "col_max", "row_span", "col_span",
        "edge_count", "edge_balance", "cluster_class", "predicted_logical",
        "predicted_cluster_mass", "residual_signed_weight", "residual_signed_adjacency",
        "residual_signed_boundary", "residual_signed_component", "prediction_residual_density",
        "xi", "p", "L",
    ]
    report = {"detectors": {}, "description": "Stim coordinate/order and precomputed Track-C matrices"}
    for L in (3, 5, 7):
        geometry = build_geometry(L)
        experiment = SurfaceCodeExperiment(L)
        coords = experiment.circuit.get_detector_coordinates()
        report["detectors"][str(L)] = {
            "num_detectors": experiment.num_detectors,
            "num_data_qubits": experiment.num_data_qubits,
            "detector_order": [list(coords[i]) for i in range(experiment.num_detectors)],
            "layer0_indices": geometry.layer0.tolist(),
            "layer1_indices": geometry.layer1.tolist(),
            "spatial_adjacency_edges": np.argwhere(np.triu(geometry.spatial_adjacency, 1)).tolist(),
            "spatial_adjacency_edge_count": int(geometry.spatial_adjacency.sum() // 2),
            "boundary_detector_indices": np.flatnonzero(geometry.boundary_mask).tolist(),
            "row_ids": geometry.row_ids.tolist(),
            "col_ids": geometry.col_ids.tolist(),
            "boundary_distance_layer0": geometry.boundary_distance[: geometry.layer0.size].tolist(),
            "feature_names": full_feature_names,
            "fast_feature_names": sorted(set(MODEL_FEATURES) | {"residual_signed_weight", "residual_signed_adjacency", "residual_signed_boundary"}),
        }
    report["model"] = {
        "features": list(MODEL_FEATURES),
        "intercept": MODEL_INTERCEPT,
        "threshold": MODEL_THRESHOLD,
        "coefficients": MODEL_COEFFICIENTS.tolist(),
        "mean": MODEL_MEAN.tolist(),
        "scale": MODEL_SCALE.tolist(),
    }
    return report


def validate_deliverables() -> None:
    required = (
        "README.md", "candidate_c.py", "run_track_c.py", "test_track_c.py",
        "analysis.md", "feature_diagnostics.json", "classifier_fit.json",
        "results_controlled_5k.jsonl", "results_controlled_5k.summary.json",
        "results_runtime_1m.jsonl", "results_runtime_1m.summary.json",
    )
    assert all((_output_path(name).is_file() and _output_path(name).stat().st_size > 0) for name in required)
    for name, expected_shots in (("results_controlled_5k.jsonl", 5000), ("results_runtime_1m.jsonl", 1_000_000)):
        rows = [json.loads(line) for line in _output_path(name).read_text().splitlines()]
        assert len(rows) == 96, (name, len(rows))
        assert {row["variant"] for row in rows} == {"baseline", "final", "rule", "classifier"}
        assert {row["shots"] for row in rows} == {expected_shots}
        assert all(row["contract_valid"] and not row["exception"] for row in rows)
    runtime = [json.loads(line) for line in _output_path("results_runtime_1m.jsonl").read_text().splitlines()]
    assert max(row["elapsed_s"] for row in runtime if row["variant"] in {"rule", "classifier"}) < 2.5
    print("TRACK C DELIVERABLES PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--feature-diagnostics", action="store_true")
    parser.add_argument("--fit-classifier", action="store_true")
    parser.add_argument("--validate-deliverables", action="store_true")
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9029])
    parser.add_argument("--variants", nargs="+", choices=["baseline", "final", *DECODERS], default=["rule", "classifier"])
    parser.add_argument("--output", default="results_controlled_5k.jsonl")
    args = parser.parse_args()
    if args.feature_diagnostics:
        path = _output_path("feature_diagnostics.json")
        path.write_text(json.dumps(feature_diagnostics(), indent=2, sort_keys=True) + "\n")
        print(f"FEATURE_DIAGNOSTICS_OK: {path}")
    elif args.fit_classifier:
        result = fit_classifier(args.shots, args.seeds)
        path = _output_path("classifier_fit.json")
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.self_test:
        for L, detectors in ((3, 8), (5, 24), (7, 48)):
            point = ParameterPoint(L, 0.01, 5.0)
            rng = np.random.default_rng(1234 + L)
            experiment = SurfaceCodeExperiment(L)
            syndrome, truth = experiment.sample_correlated(shots=128, p=point.p, xi=point.xi, rng=rng)
            decoder = ResidualDecoder(point, "rule")
            prediction = decoder.decode(syndrome)
            assert syndrome.shape == (128, detectors)
            assert prediction.shape == (128,) and prediction.dtype == np.uint8
            assert np.all((prediction == 0) | (prediction == 1))
            assert decoder.last_features is not None
            empty = decoder.decode(syndrome[:0])
            assert empty.shape == (0,) and empty.dtype == np.uint8
            assert decoder.geometry.num_detectors == detectors
        print("TRACK C TESTS PASS")
    elif args.validate_deliverables:
        validate_deliverables()
    else:
        if args.shots <= 0:
            parser.error("--shots must be positive")
        summary = run(args.shots, args.seeds, args.variants, args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
