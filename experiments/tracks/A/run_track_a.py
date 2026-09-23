"""Offline integration, covariance audit and exactly paired official-grid evaluation."""
from __future__ import annotations

import argparse
import io
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import binomtest
from scipy.stats._qmvnt import _qmvn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qec_benchmark.config import challenge_grid, VALIDATE_SEEDS
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment
from qec_benchmark.baselines import MWPMDecoder
from qec_benchmark.noise import _cached_cholesky, sample_correlated_bernoulli
from solve import build_decoder as final_decoder


def key(p, xi):
    return f"p{p:g}_xi{xi:g}"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if platform.system() == "Darwin" else 1024))


def provenance():
    paths = ["solve.py", "src/qec_benchmark/noise.py", "src/qec_benchmark/config.py",
             "src/qec_benchmark/stim_surface_code.py", "src/qec_benchmark/evaluation.py",
             "experiments/tracks/A/run_track_a.py", "experiments/tracks/A/candidate_a.py",
             "experiments/tracks/A/frozen_tables.py"]
    return {"dependencies": {n: importlib.metadata.version(n) for n in
                             ("numpy", "scipy", "stim", "pymatching")},
            "python": sys.version, "platform": platform.platform(),
            "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in paths if (ROOT / p).exists()},
            "memory_method": "whole-process high-water RSS, includes sampling and integration"}


def enumerate_masks():
    ex = SurfaceCodeExperiment(3)
    masks = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(bool)
    syndromes, logical = ex.sample_from_mask(masks)
    ids = syndromes @ (1 << np.arange(8))
    # Confirm linearity over GF(2), not assumed detector or logical indexing.
    assert np.array_equal((masks @ syndromes[1 << np.arange(9)].astype(int)) % 2, syndromes)
    assert np.array_equal((masks @ logical[1 << np.arange(9)].astype(int)) % 2, logical)
    assert ex.num_detectors == 8 and ex.num_data_qubits == 9
    return ex, masks, ids, logical.astype(np.uint8)


def covariance(positions, xi):
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    return np.eye(len(positions)) if xi <= 0 else np.exp(-distance / xi) + np.eye(len(positions)) * 1e-12


def integrate_masks(p, xi, budget, seed, batches=8):
    ex, masks, ids, logical = enumerate_masks()
    t0 = time.perf_counter()
    probs, errors, samples = np.zeros(512), np.zeros(512), np.zeros(512, dtype=int)
    if xi == 0:
        weight = masks.sum(axis=1)
        probs = p ** weight * (1-p) ** (9-weight)
        method = "exact iid products (only xi=0)"
    else:
        cov = covariance(ex.data_positions, xi)
        threshold = ndtri(1-p)
        for i, mask in enumerate(masks):
            low = np.where(mask, threshold, -np.inf)
            high = np.where(mask, np.inf, threshold)
            # Independent streams per physical mask; independent budget runs.
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            probs[i], errors[i], samples[i] = _qmvn(
                budget, cov.copy(), low, high, rng=rng, n_batches=batches)
        method = "SciPy 1.15.3 _qmvn randomized-lattice QMC, Genz conditional transformation"
    assert np.all(probs >= 0) and np.isfinite(probs).all()
    joint = np.bincount(ids * 2 + logical, weights=probs, minlength=512).reshape(256, 2)
    joint_se = np.sqrt(np.bincount(ids * 2 + logical, weights=(errors/3)**2,
                                 minlength=512).reshape(256, 2))
    table = (joint[:, 1] > joint[:, 0]).astype(np.uint8)
    active = np.unique(ids)
    final_predictions = final_decoder(type("Point", (), {"L": 3, "p": p, "xi": xi})()).decode(
        ((active[:, None] >> np.arange(8)) & 1).astype(np.uint8))
    margin_se = np.sqrt((joint_se**2).sum(axis=1))
    return {"p": p, "xi": xi, "method": method, "seed": seed, "batches": batches,
            "requested_budget_per_mask": budget, "actual_samples_per_mask": samples.tolist(),
            "build_elapsed_s": time.perf_counter()-t0, "raw_probability_sum": float(probs.sum()),
            "probability_sum_3se": float(np.linalg.norm(errors)),
            "mask_probabilities": probs.tolist(), "mask_probability_3se": errors.tolist(),
            "marginals": (probs @ masks).tolist(), "syndrome_logical_joint": joint.tolist(),
            "joint_3se": (3*joint_se).tolist(), "table": table.tolist(),
            "table_hex": np.packbits(table, bitorder="little").tobytes().hex(),
            "active_syndromes": active.tolist(), "margin": (joint[:, 1]-joint[:, 0]).tolist(),
            "margin_3se": (3*margin_se).tolist(),
            "unresolved_active_syndromes_3se": active[
                np.abs(joint[active, 1]-joint[active, 0]) <= 3*margin_se[active]].tolist(),
            "estimated_bayes_error": float(np.minimum(joint[:, 0], joint[:, 1]).sum()),
            "estimated_final_error": float(joint[active, 1-final_predictions].sum()),
            "differ_from_final": active[table[active] != final_predictions].tolist(),
            "process_maxrss_bytes": rss()}


def build(args):
    ex, masks, ids, logical = enumerate_masks()
    report = {"provenance": provenance(), "integration_seeds": [args.seed, args.seed+1],
              "training": "model integration only; zero sampled benchmark training shots",
              "error_method": "3*SE over 8 randomized lattice shifts per mask; not rigorous bounds",
              "probabilities_normalized": False, "qubits": ex._data_qubits.tolist(),
              "positions": ex.data_positions.tolist(), "mask_ids": list(range(512)),
              "mask_syndromes": ids.tolist(), "mask_logicals": logical.tolist(), "points": {}}
    for point in challenge_grid()[:8]:
        low = integrate_masks(point.p, point.xi, args.budget, args.seed)
        high = integrate_masks(point.p, point.xi, args.refine_budget, args.seed+1)
        low_p, high_p = np.array(low["mask_probabilities"]), np.array(high["mask_probabilities"])
        report["points"][key(point.p, point.xi)] = {
            "coarse": low, "refined": high,
            "convergence": {"mask_l1_change": float(np.abs(low_p-high_p).sum()),
                            "max_mask_absolute_change": float(np.abs(low_p-high_p).max()),
                            "table_changes": np.flatnonzero(np.array(low["table"]) != high["table"]).tolist(),
                            "risk_change": high["estimated_bayes_error"]-low["estimated_bayes_error"]}}
        print(key(point.p, point.xi), "risk", high["estimated_bayes_error"], "final",
              high["estimated_final_error"], "seconds", high["build_elapsed_s"], flush=True)
        dump(HERE / args.output, report)
    constants = {k: v["refined"]["table_hex"] for k, v in report["points"].items()}
    if not args.no_freeze:
        (HERE / "frozen_tables.py").write_text(
            '"""Generated by run_track_a.py build; QMC provenance in tables_qmc.json."""\n'
            + "TABLE_HEX = " + repr(constants) + "\n")
    # Refresh the final receipt after writing the frozen source, so its source
    # hash describes the exact table imported by candidate_a.
    report["provenance"] = provenance()
    dump(HERE / args.output, report)


def binary_covariance(p, rho):
    """Plackett identity: Phi_2(t,t;rho)-Phi(t)^2, avoiding cancellation."""
    t = ndtri(1-p)
    return quad(lambda r: np.exp(-t*t/(1+r))/(2*np.pi*np.sqrt(1-r*r)),
                0, rho, epsabs=1e-14, epsrel=1e-11)


def audit_covariance(args):
    ex, masks, ids, logical = enumerate_masks()
    report = {"provenance": provenance(), "seed": args.seed, "shots_per_point": args.shots,
              "positions": ex.data_positions.tolist(), "qubits": ex._data_qubits.tolist(),
              "nearest_neighbor_spacing": 2.0, "points": {}}
    for point in challenge_grid()[:8]:
        cov = covariance(ex.data_positions, point.xi)
        if point.xi:
            chol = _cached_cholesky(tuple(map(tuple, ex.data_positions)), point.xi)
            assert np.allclose(chol @ chol.T, cov, atol=2e-15, rtol=0)
        variance = cov[0, 0]
        effective_p = float(ndtr(-ndtri(1-point.p)/np.sqrt(variance))) if point.xi else point.p
        binary = np.zeros((9, 9))
        pairs = []
        sampled = sample_correlated_bernoulli(positions=ex.data_positions, shots=args.shots,
                    p=point.p, xi=point.xi, rng=np.random.default_rng(args.seed))
        joint_empirical = sampled.astype(float).T @ sampled / args.shots
        for i in range(9):
            binary[i, i] = effective_p*(1-effective_p)
            for j in range(i+1, 9):
                rho = cov[i, j]/variance
                value, error = binary_covariance(effective_p, rho)
                binary[i, j] = binary[j, i] = value
                joint = value + effective_p**2
                se = np.sqrt(joint*(1-joint)/args.shots)
                pairs.append({"i": i, "j": j, "distance": float(np.linalg.norm(ex.data_positions[i]-ex.data_positions[j])),
                              "latent_rho": float(rho), "binary_covariance": value,
                              "binary_correlation": value/(effective_p*(1-effective_p)),
                              "quadrature_absolute_error_estimate": error,
                              "joint_probability": joint, "empirical_joint": float(joint_empirical[i,j]),
                              "joint_mc_se": float(se),
                              "joint_z": float((joint_empirical[i,j]-joint)/se)})
        report["points"][key(point.p, point.xi)] = {
            "latent_covariance_with_jitter": cov.tolist(), "binary_covariance": binary.tolist(),
            "effective_marginal_p": effective_p, "empirical_marginals": sampled.mean(axis=0).tolist(),
            "latent_eigenvalues": np.linalg.eigvalsh(cov).tolist(), "pairs": pairs}
    dump(HERE / "covariance_validation.json", report)
    print("covariance audit complete", flush=True)


def wilson(errors, shots):
    p = errors/shots
    z = 1.959963984540054
    center = (p+z*z/(2*shots))/(1+z*z/shots)
    half = z*np.sqrt(p*(1-p)/shots+z*z/(4*shots*shots))/(1+z*z/shots)
    return [max(0., center-half), min(1., center+half)]


def paired(wins, losses, n):
    # Positive delta means fewer errors for A. D = other_wrong - A_wrong.
    delta = (wins-losses)/n
    se = np.sqrt(max(0., (wins+losses)/n-delta*delta)/(n-1)) if n > 1 else 0.
    return {"a_wins": wins, "a_losses": losses, "saved_errors": wins-losses,
            "saved_error_rate": delta, "paired_se": float(se),
            "paired_normal_ci95": [float(delta-1.95996398454*se), float(delta+1.95996398454*se)],
            "mcnemar_exact_p": float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.0,
            "normal_ci_caution": "approximate; few discordances need exact McNemar test; zero-width with zero discordances is not population certainty"}


def evaluate(args, *, write_records=True):
    from experiments.tracks.A.candidate_a import build_decoder
    rows = []
    seeds = VALIDATE_SEEDS if args.validate else [args.seed]
    grid = challenge_grid()
    experiments = {L: SurfaceCodeExperiment(L) for L in (3, 5, 7)}
    factories = {"candidate_a": build_decoder, "final": final_decoder,
                 "baseline": lambda point: MWPMDecoder(point, weighted=True)}
    output = HERE / f"results_{args.label}.jsonl"
    with (output.open("w") if write_records else io.StringIO()) as file:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for point in grid:
                syndrome, truth = experiments[point.L].sample_correlated(
                    shots=args.shots, p=point.p, xi=point.xi, rng=rng)
                row = {"seed": seed, "L": point.L, "p": point.p, "xi": point.xi,
                       "shots": args.shots, "decoders": {}, "paired": {}}
                wrong = {}
                # Alternate timing order by seed/point; data are identical for all decoders.
                order = list(factories) if (seed+grid.index(point)) % 2 else list(reversed(factories))
                for name in order:
                    start = time.perf_counter()
                    decoder = factories[name](point)
                    built = time.perf_counter()
                    prediction = decoder.decode(syndrome)
                    ended = time.perf_counter()
                    assert prediction.shape == truth.shape and prediction.dtype == np.uint8
                    assert np.isin(prediction, [0, 1]).all()
                    wrong[name] = prediction != truth
                    nerr = int(wrong[name].sum())
                    timed_out = ended-start > 2.5
                    row["decoders"][name] = {"errors": nerr, "accuracy": 1-nerr/args.shots,
                        "error_rate": nerr/args.shots, "error_rate_wilson_ci95": wilson(nerr, args.shots),
                        "build_elapsed_s": built-start, "decode_elapsed_s": ended-built,
                        "elapsed_s": ended-start, "timed_out": timed_out,
                        "scored_errors": args.shots if timed_out else nerr,
                        "table_numpy_bytes": decoder.table.nbytes if hasattr(decoder, "table") else None,
                        "process_maxrss_bytes": rss()}
                for name in ("final", "baseline"):
                    row["paired"][name] = paired(
                        int((wrong[name] & ~wrong["candidate_a"]).sum()),
                        int((wrong["candidate_a"] & ~wrong[name]).sum()), args.shots)
                rows.append(row)
                file.write(json.dumps(row, allow_nan=False)+"\n")
                file.flush()
            print("seed", seed, "finished", flush=True)
    summary = summarize(rows)
    summary.update({"provenance": provenance(), "seeds": seeds, "shots_per_point": args.shots,
                    "rng_semantics": "one default_rng(seed); all 24 points in challenge_grid order, one sampler call per point; shared shots across decoders",
                    "training": "frozen model-only tables; independent integration seed; no benchmark observations used to select entries",
                    "time_limit": 2.5, "offline_integration_in_timer": False})
    if write_records:
        dump(HERE / f"results_{args.label}.json", summary)
        print(json.dumps(summary["aggregates"], indent=2), flush=True)
    return rows, summary


def summarize(rows):
    def group(rs):
        n = sum(r["shots"] for r in rs)
        decoders = {}
        for name in ("candidate_a", "final", "baseline"):
            errors = sum(r["decoders"][name]["errors"] for r in rs)
            # Fixed-stratum variance, not iid over the heterogeneous official grid.
            variance = sum(r["shots"]*(r["decoders"][name]["error_rate"])*(1-r["decoders"][name]["error_rate"])
                           for r in rs)/n**2
            decoders[name] = {"errors": errors, "shots": n, "error_rate": errors/n,
                              "accuracy": 1-errors/n, "score": round(errors*1e6/n),
                              "stratified_normal_ci95": [errors/n-1.96*np.sqrt(variance), errors/n+1.96*np.sqrt(variance)],
                              "max_build_decode_s": max(r["decoders"][name]["elapsed_s"] for r in rs),
                              "mean_decode_s": float(np.mean([r["decoders"][name]["decode_elapsed_s"] for r in rs])),
                              "timeouts": sum(r["decoders"][name]["timed_out"] for r in rs),
                              "official_scored_errors": sum(r["decoders"][name]["scored_errors"] for r in rs)}
        comparisons = {}
        for name in ("final", "baseline"):
            result = paired(sum(r["paired"][name]["a_wins"] for r in rs),
                            sum(r["paired"][name]["a_losses"] for r in rs), n)
            se = np.sqrt(sum((r["shots"]*r["paired"][name]["paired_se"])**2 for r in rs))/n
            result["stratified_paired_ci95"] = [result["saved_error_rate"]-1.96*se, result["saved_error_rate"]+1.96*se]
            comparisons[name] = result
        return {"decoders": decoders, "paired": comparisons}
    result = {"rows": len(rows), "aggregates": {
        "L3": group([r for r in rows if r["L"] == 3]), "full_grid": group(rows)},
        "by_seed": {str(seed): {"L3": group([r for r in rows if r["seed"] == seed and r["L"] == 3]),
                               "full_grid": group([r for r in rows if r["seed"] == seed])}
                    for seed in sorted({r["seed"] for r in rows})},
        "by_point": {f'L{p.L}_{key(p.p,p.xi)}': group([r for r in rows if (r["L"],r["p"],r["xi"]) == (p.L,p.p,p.xi)])
                     for p in challenge_grid()}}
    for name in ("final", "baseline"):
        family = [v["paired"][name] for k,v in result["by_point"].items() if k.startswith("L3_")]
        ordered = sorted(family, key=lambda x: x["mcnemar_exact_p"])
        running = 0.
        for rank, item in enumerate(ordered):
            running = max(running, min(1., (len(ordered)-rank)*item["mcnemar_exact_p"]))
            item["mcnemar_holm_p_8points"] = running
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--budget", type=int, default=4096)
    builder.add_argument("--refine-budget", type=int, default=32768)
    builder.add_argument("--seed", type=int, default=902100)
    builder.add_argument("--output", default="tables_qmc.json")
    builder.add_argument("--no-freeze", action="store_true")
    cov = sub.add_parser("covariance")
    cov.add_argument("--shots", type=int, default=500000)
    cov.add_argument("--seed", type=int, default=902101)
    run = sub.add_parser("evaluate")
    run.add_argument("--shots", type=int, default=10000)
    run.add_argument("--seed", type=int, default=8675309)
    run.add_argument("--validate", action="store_true")
    run.add_argument("--label", default="pilot_10k")
    args = parser.parse_args()
    if hasattr(args, "shots") and args.shots <= 0:
        parser.error("--shots must be positive")
    {"build": build, "covariance": audit_covariance, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
