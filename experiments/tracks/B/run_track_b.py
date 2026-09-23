"""Reproducible shared-batch evaluation; writes only beneath track B."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pymatching
import scipy
from scipy.stats import binomtest
import stim

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from candidate_b import DECODERS, VARIANTS, GraphDecoder, joint_probability, physical_columns, _append_event
from qec_benchmark.config import VALIDATE_SEEDS, challenge_grid
from qec_benchmark.models import ParameterPoint
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


def maxrss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def current_rss():
    # ps works on both local Darwin and Linux; outside all decoding timers.
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(out.strip()) * 1024


def output_path(name):
    path = (HERE / name).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("outputs must stay inside experiments/tracks/B")
    return path


def write_json(name, obj):
    output_path(name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def audit():
    raw = stim.DetectorErrorModel("error(0.01) D0 D1 D2 D3 L0")
    decomposed = stim.DetectorErrorModel("error(0.01) D0 D1 L0 ^ D2 D3")
    split = pymatching.Matching.from_detector_error_model(decomposed, enable_correlations=True)
    overlap = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel("error(0.01) D0 D1 L0 ^ D1 D2"), enable_correlations=True)
    b = pymatching.Matching()
    b.add_boundary_edge(0, fault_ids={2}, weight=2)
    merge = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel("error(0.01) D0 L0\nerror(0.02) D0 L1"))
    return dict(
        pymatching_version=pymatching.__version__, python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, stim=stim.__version__,
        raw_hyperedge_edges=pymatching.Matching.from_detector_error_model(raw).num_edges,
        decomposed_hyperedge_edges=split.num_edges,
        decomposed_correlated_prediction=split.decode([1, 1, 1, 1], enable_correlations=True).tolist(),
        overlapping_components_prediction=overlap.decode([1, 0, 1], enable_correlations=True).tolist(),
        boundary_prediction_fault_id_2=b.decode([1]).tolist(),
        boundary_num_fault_ids=b.num_fault_ids,
        parallel_merge_keeps_first_fault_ids=sorted(merge.edges()[0][2]["fault_ids"]),
        parallel_merged_probability=merge.edges()[0][2]["error_probability"],
        true_hypergraph_matching=False,
        correlation_support="two-pass heuristic for graphlike DEM decompositions; enable on load AND decode",
    )


def self_test():
    from solve import DataOnlyMWPM
    for L in (5, 7):
        exp = SurfaceCodeExperiment(L)
        columns = physical_columns(exp)
        assert max(CounterColumns(columns)) == 2
        rng = np.random.default_rng(321)
        masks = rng.random((2000, L * L)) < .13
        syndrome, truth = exp.sample_from_mask(masks)
        singles, single_obs = exp.sample_from_mask(np.eye(L * L, dtype=bool))
        point0 = ParameterPoint(L, .01, 0)
        iid = GraphDecoder(point0, "iid")
        reference = DataOnlyMWPM(point0).decode(syndrome)
        assert np.array_equal(iid.decode(syndrome), reference)
        original_edges = [(u, v, a["fault_ids"]) for u, v, a in iid._matching.edges()]
        for name in VARIANTS:
            dec0 = GraphDecoder(point0, name)
            assert np.array_equal(dec0.decode(syndrome), reference), (L, name, "xi0")
            for xi in (2, 10):
                decoder = GraphDecoder(ParameterPoint(L, .01, xi), name)
                pred = decoder.decode(syndrome)
                assert pred.shape == (2000,) and pred.dtype == np.uint8
                assert np.all((pred == 0) | (pred == 1))
                assert decoder.decode(syndrome[:0]).shape == (0,)
                assert np.array_equal(decoder.decode(singles), single_obs), (L, name, "single")
                assert decoder._matching.num_detectors == exp.num_detectors
                assert decoder._matching.num_fault_ids == 1
                new_edges = [(u, v, a["fault_ids"]) for u, v, a in decoder._matching.edges()]
                if name.startswith("pair_shortcut"):
                    assert set(map(str, original_edges)) <= set(map(str, new_edges)), (L, name, "fault ids")
                else:
                    assert sorted(map(str, original_edges)) == sorted(map(str, new_edges)), (L, name, "fault ids")
                assert decoder.model_details.get("physical_marginal_max_abs_error", 0) < 1e-14
        # Every encoded pair's detector/observable parity must equal its
        # deterministic circuit columns, including identical boundary columns.
        for i in range(L * L):
            for j in range(i + 1, L * L):
                for collapse in (False, True):
                    dem = stim.DetectorErrorModel()
                    _append_event(dem, .01, columns, (i, j), collapse_graphlike=collapse)
                    parity_d = set()
                    parity_o = 0
                    for inst in dem:
                        for target in inst.targets_copy():
                            if target.is_relative_detector_id():
                                parity_d.symmetric_difference_update({target.val})
                            elif target.is_logical_observable_id():
                                parity_o ^= 1
                    assert parity_d == set(columns[i][0]) ^ set(columns[j][0])
                    assert parity_o == columns[i][1] ^ columns[j][1]
    assert joint_probability(.01, 2, 0) == .0001
    assert .0001 < joint_probability(.01, 2, 2) < joint_probability(.01, 2, 10) < .01
    # Analytic two-site latent source check prevents rate-factor mistakes.
    p = .01
    joint = joint_probability(p, 2, 5)
    rate = .5 * np.log1p(4 * (joint - p * p) / (1 - 2 * p) ** 2)
    pair_p = -np.expm1(-rate) / 2
    single_p = (1 - (1 - 2 * p) / (1 - 2 * pair_p)) / 2
    recovered_joint = pair_p * (1 - single_p) ** 2 + (1 - pair_p) * single_p ** 2
    assert abs(recovered_joint - joint) < 1e-14
    info = audit()
    assert info["raw_hyperedge_edges"] == 0
    assert info["decomposed_hyperedge_edges"] == 2
    assert info["boundary_prediction_fault_id_2"] == [0, 0, 1]
    assert info["parallel_merge_keeps_first_fault_ids"] == [0]
    print("SELF_TEST_OK: exact xi=0 equivalence, graph/fault IDs, single errors, contract, marginals, runtime audit")


def CounterColumns(columns):
    from collections import Counter
    return Counter(columns).values()


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["L"], row["xi"])].append(row)
    aggregates = []
    for (name, L, xi), rs in sorted(groups.items()):
        shots = sum(r["shots"] for r in rs)
        rescued = sum(r["rescued_vs_iid"] for r in rs)
        harmed = sum(r["harmed_vs_iid"] for r in rs)
        delta = (harmed - rescued) / shots
        se = np.sqrt(max(0, (harmed + rescued) / shots - delta ** 2) / shots)
        aggregates.append(dict(
            variant=name, L=L, xi=xi, shots=shots,
            errors=sum(r["raw_errors"] for r in rs), scored_errors=sum(r["errors"] for r in rs),
            iid_errors=sum(r["iid_errors"] for r in rs), rescued=rescued, harmed=harmed,
            delta_errors=harmed-rescued, delta_rate=delta,
            delta_rate_95ci_approx=[delta - 1.96 * se, delta + 1.96 * se],
            paired_exact_p=binomtest(rescued, rescued+harmed, .5).pvalue if rescued+harmed else 1.0,
            max_elapsed_s=max(r["elapsed_s"] for r in rs),
            max_build_s=max(r["build_elapsed_s"] for r in rs),
            max_decode_s=max(r["decode_elapsed_s"] for r in rs),
            max_process_rss_bytes=max(r["process_rss_after_bytes"] for r in rs),
            max_process_maxrss_bytes=max(r["process_maxrss_bytes"] for r in rs),
            timeouts=sum(r["timed_out"] for r in rs),
        ))
    # Holm control across non-null L/xi/candidate exploratory comparisons.
    comparisons = [a for a in aggregates if a["variant"] not in {"iid", "noop"} and a["xi"] > 0]
    order = sorted(comparisons, key=lambda a: a["paired_exact_p"])
    running = 0
    for rank, a in enumerate(order):
        running = max(running, min(1, (len(order) - rank) * a["paired_exact_p"]))
        a["paired_p_holm"] = running
    return dict(
        rows=len(rows), variants=sorted({r["variant"] for r in rows}),
        shots_per_point=sorted({r["shots"] for r in rows}), seeds=sorted({r["seed"] for r in rows}),
        time_limit_s=2.5, aggregates=aggregates,
        timing="monotonic build+decode wall time; imports/sampling/diagnostics/scoring excluded",
        memory="current RSS via ps, process high-water via getrusage; includes sampling and previous candidates",
        sampling="full official challenge-grid sequential RNG, including L=3 draws; only L=5/7 decoded",
        uncertainty="paired exact binomial/McNemar and approximate paired rate-difference CI; strata descriptive; seed42 selection is exploratory",
        audit=audit(), candidate_sha256=hashlib.sha256((HERE / "candidate_b.py").read_bytes()).hexdigest(),
    )


def run(shots, seeds, variants, output):
    if "iid" not in variants:
        variants = ["iid", *variants]
    path = output_path(output)
    rows = []
    with path.open("w") as stream:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            experiments = {L: SurfaceCodeExperiment(L) for L in (3, 5, 7)}
            for point in challenge_grid():
                syndrome, truth = experiments[point.L].sample_correlated(
                    shots=shots, p=point.p, xi=point.xi, rng=rng)
                if point.L == 3:
                    continue
                batch_sha = hashlib.sha256(syndrome.tobytes() + truth.tobytes()).hexdigest()
                baseline = None
                for name in variants:
                    rss_before, peak_before = current_rss(), maxrss()
                    decoder = None
                    exception = None
                    start = time.monotonic()
                    build_end = start
                    try:
                        decoder = DECODERS[name](point)
                        build_end = time.monotonic()
                        predictions = decoder.decode(syndrome)
                    except Exception as exc:
                        exception = repr(exc)
                        predictions = np.zeros(shots, dtype=np.uint8)
                    end = time.monotonic()
                    if predictions.shape != (shots,) or predictions.dtype != np.uint8:
                        exception = "invalid output contract"
                    wrong = predictions != truth
                    if name == "iid":
                        baseline = wrong.copy()
                    assert baseline is not None
                    timed_out = end - start > 2.5 or exception is not None
                    raw_errors = int(wrong.sum())
                    row = dict(
                        variant=name, seed=seed, L=point.L, p=point.p, xi=point.xi,
                        shots=shots, errors=shots if timed_out else raw_errors, raw_errors=raw_errors,
                        iid_errors=int(baseline.sum()),
                        rescued_vs_iid=int(np.sum(baseline & ~wrong)),
                        harmed_vs_iid=int(np.sum(~baseline & wrong)),
                        elapsed_s=end-start, build_elapsed_s=build_end-start, decode_elapsed_s=end-build_end,
                        timed_out=timed_out, time_limit_s=2.5, exception=exception,
                        process_rss_before_bytes=rss_before, process_rss_after_bytes=current_rss(),
                        process_maxrss_bytes=maxrss(), process_maxrss_delta_bytes=maxrss()-peak_before,
                        batch_sha256=batch_sha,
                        graph=decoder.diagnostics() if decoder is not None else None,
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(f"seed={seed} {point.key()} {name}: {raw_errors}/{shots}, {end-start:.4f}s timeout={timed_out}", flush=True)
    summary = summarize(rows)
    write_json(path.with_suffix(".summary.json").name, summary)
    return summary


def load_rows(name):
    return [json.loads(line) for line in output_path(name).read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--audit-json", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--quick-results", action="store_true")
    p.add_argument("--official-results", action="store_true")
    p.add_argument("--validate-deliverables", action="store_true")
    p.add_argument("--shots", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--output", default="results_seed42_5k.jsonl")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.audit_json:
        print(json.dumps(audit(), indent=2))
    elif args.diagnostics:
        graphs = [GraphDecoder(point, name).diagnostics(include_edges=True)
                  for point in challenge_grid() if point.L in (5, 7) for name in VARIANTS]
        write_json("graph_diagnostics.json", dict(audit=audit(), graphs=graphs))
        print(f"GRAPH_DIAGNOSTICS_OK: {len(graphs)} graphs")
    elif args.quick_results:
        rows = load_rows("results_seed42_5k.jsonl")
        assert len(rows) == 16 * len(VARIANTS)
        assert all(r["seed"] == 42 and r["shots"] == 5000 for r in rows)
        assert all(not r["exception"] for r in rows)
        assert all(r["harmed_vs_iid"] == r["rescued_vs_iid"] == 0 for r in rows if r["xi"] == 0)
        print(json.dumps(dict(rows=len(rows), shots=5000, xi0_exact=True)))
    elif args.official_results:
        rows = load_rows("results_official_100k.jsonl")
        assert {r["seed"] for r in rows} == set(VALIDATE_SEEDS)
        assert {r["L"] for r in rows} == {5, 7}
        print(f"OFFICIAL_RESULTS_OK: {len(rows)} rows")
    elif args.validate_deliverables:
        names = ["README.md", "candidate_b.py", "run_track_b.py", "analysis.md", "graph_diagnostics.json",
                 "results_seed42_5k.jsonl", "results_seed42_10k.jsonl", "results_official_100k.jsonl"]
        assert all((HERE / name).is_file() and (HERE / name).stat().st_size > 0 for name in names)
        print("DELIVERABLES_OK")
    else:
        if args.shots <= 0:
            p.error("shots must be positive")
        run(args.shots, args.seeds, args.variants, args.output)


if __name__ == "__main__":
    main()
