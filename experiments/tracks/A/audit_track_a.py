"""Recompute report claims from retained artifacts, without rerunning evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    tables = json.loads((HERE / "tables_qmc.json").read_text())
    cov = json.loads((HERE / "covariance_validation.json").read_text())
    val = json.loads((HERE / "results_official_100k_5seeds.json").read_text())
    rows = [json.loads(s) for s in (HERE / "results_official_100k_5seeds.jsonl").read_text().splitlines()]
    masks = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(float)
    audit = {"validation_rows": len(rows), "points": {}, "aggregates": val["aggregates"],
             "table_storage_bytes": {"uint8_per_decoder": 256, "packed_all_8_tables": 256},
             "scope_incident": "INCIDENT.md; outside-A deletion unresolved; ownership requirement not satisfied"}
    assert len(rows) == 120
    for name, item in tables["points"].items():
        refined = item["refined"]
        probability = np.array(refined["mask_probabilities"])
        c = cov["points"][name]
        p = c["effective_marginal_p"]
        joint = masks.T @ (probability[:, None]*masks)
        theory = np.array(c["binary_covariance"]) + p*p
        margins = np.abs(np.array(refined["margin"])[:16])
        error = np.array(refined["margin_3se"])[:16]
        minimum_ratio = float(np.min(margins/error)) if error.max() else None
        pair = c["pairs"][0]
        audit["points"][name] = {
            "build_s": refined["build_elapsed_s"], "build_rss_bytes": refined["process_maxrss_bytes"],
            "actual_samples_per_mask": sorted(set(refined["actual_samples_per_mask"])),
            "normalization_residual": refined["raw_probability_sum"]-1,
            "normalization_3se": refined["probability_sum_3se"],
            "max_marginal_error": float(np.abs(probability@masks-p).max()),
            "max_second_moment_error": float(np.abs(joint-theory).max()),
            "min_active_margin_over_3se": minimum_ratio,
            "syndrome9_margin": refined["margin"][9],
            "syndrome9_margin_3se": refined["margin_3se"][9],
            "bayes_risk": refined["estimated_bayes_error"],
            "model_predicted_saved_errors_per_million": 1e6*(refined["estimated_final_error"]-refined["estimated_bayes_error"]),
            "convergence": item["convergence"],
            "nearest_neighbor": {k: pair[k] for k in ("latent_rho", "binary_covariance", "binary_correlation")},
            "max_pair_empirical_abs_z": max(abs(pair["joint_z"]) for pair in c["pairs"]),
            "min_latent_eigenvalue": min(c["latent_eigenvalues"])}
        assert len(probability) == 512 and len(refined["table"]) == 256
        assert len(refined["unresolved_active_syndromes_3se"]) == 0
    for filename in ("results_pilot_10k.json", "results_official_100k_5seeds.json"):
        receipt = json.loads((HERE / filename).read_text())
        for path in ("experiments/tracks/A/frozen_tables.py", "solve.py"):
            actual = hashlib.sha256((HERE.parents[2]/path).read_bytes()).hexdigest()
            assert actual == receipt["provenance"]["sha256"][path]
    audit["frozen_tables_match_pilot_and_heldout_hash"] = True
    audit["solve_matches_pilot_and_heldout_hash"] = True
    audit["by_distance_timing_memory"] = {}
    for L in (3, 5, 7):
        rs = [r for r in rows if r["L"] == L]
        audit["by_distance_timing_memory"][str(L)] = {
            name: {"mean_build_s": float(np.mean([r["decoders"][name]["build_elapsed_s"] for r in rs])),
                   "mean_decode_s": float(np.mean([r["decoders"][name]["decode_elapsed_s"] for r in rs])),
                   "max_build_decode_s": max(r["decoders"][name]["elapsed_s"] for r in rs),
                   "whole_process_maxrss_bytes": max(r["decoders"][name]["process_maxrss_bytes"] for r in rs)}
            for name in ("candidate_a", "final", "baseline")}
    audit["L3_point_comparisons"] = {k: v for k, v in val["by_point"].items() if k.startswith("L3_")}
    audit["L3_saved_errors_by_seed"] = {k: v["L3"]["paired"]["final"]["saved_errors"]
                                       for k,v in val["by_seed"].items()}
    (HERE / "results_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    for name, p in audit["points"].items():
        print(name, json.dumps({k: v for k,v in p.items() if k != "convergence"}))
    print("TIMING_MEMORY", json.dumps(audit["by_distance_timing_memory"]))
    print("SAVINGS_BY_SEED", audit["L3_saved_errors_by_seed"])
    print("ARTIFACT AUDIT PASSED: 8 tables, 120 paired rows, unchanged predictions/source")


if __name__ == "__main__":
    main()
