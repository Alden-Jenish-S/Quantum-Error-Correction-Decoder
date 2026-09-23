"""Reproducible offline training, freeze, paired evaluation and runtime stress."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qec_benchmark.config import challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment
from experiments.tracks.D.candidate_d import VARIANTS, build_decoder, point_key
from experiments.tracks.D.reference_iid import IIDDecoder
from experiments.tracks.D.geometry import (geometry, ids_from_syndrome,
    syndrome_from_ids, canonicalize, orbit_ids, ambiguous_ids)

TRAIN_SEED = 2718281
TEST_SEEDS = [104729, 130363, 155921]
PILOT_SEED = 32452843
HASH_PATHS = [
    "experiments/tracks/D/" + name for name in
    ("candidate_d.py", "reference_iid.py", "geometry.py", "run_track_d.py",
     "PROTOCOL.md", "frozen_tables.py", "training_counts.npz", "training.json")
] + ["experiments/tracks/A/" + name for name in ("candidate_a.py", "frozen_tables.py")
] + ["src/qec_benchmark/" + name for name in
     ("noise.py", "stim_surface_code.py", "config.py", "models.py")]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in HASH_PATHS if (ROOT / p).exists()}


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
               * (1 if platform.system() == "Darwin" else 1024))


def current_rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "dependencies": {n: importlib.metadata.version(n) for n in
                             ("numpy", "scipy", "stim", "pymatching")},
            "memory_method": "RSS bytes normalized for Darwin/Linux; whole process incl sampling and allocators, not decoder incremental allocation"}


def add_counts(counts, ids, labels):
    values, frequencies = np.unique(ids.astype(np.uint64) * 2 + labels, return_counts=True)
    for value, frequency in zip(values, frequencies):
        k = int(value)
        counts[k] = counts.get(k, 0) + int(frequency)


def array_counts(counts):
    keys = np.array(sorted({k // 2 for k in counts}), dtype=np.uint32)
    labels = np.array([[counts.get(2*int(k), 0), counts.get(2*int(k)+1, 0)]
                       for k in keys], dtype=np.uint32)
    return keys, labels


def lookup_counts(keys, counts, ids):
    position = np.searchsorted(keys, ids)
    safe = np.minimum(position, len(keys)-1)
    return np.where((keys[safe] == ids)[:, None], counts[safe], 0)


def posterior_decisions(base, counts, *, conservative):
    n = counts.sum(axis=1)
    wrong = counts[np.arange(len(base)), 1-base]
    right = counts[np.arange(len(base)), base]
    if conservative:
        lower = beta.ppf(.01, wrong+1., right+9.)
        flip = (n >= 20) & (lower > .5)
    else:
        flip = wrong > right
    return base ^ flip.astype(np.uint8)


def serialize(array):
    import zlib
    return base64.b85encode(zlib.compress(array.tobytes(), 9)).decode("ascii")


def materialize(point, geom, raw_keys, raw_counts, sym_keys, sym_counts):
    ids = np.arange(4096, dtype=np.uint32) if point.L == 5 else orbit_ids(raw_keys, geom)
    base = IIDDecoder(point).decode(syndrome_from_ids(ids, geom))
    canonical, action = canonicalize(ids, geom)
    pooled = lookup_counts(sym_keys, sym_counts, canonical)
    # Convert the canonical logical posterior back to original observable labels.
    pooled = np.where(action[:, None] != 0, pooled[:, ::-1], pooled)
    unpooled = lookup_counts(raw_keys, raw_counts, ids)
    ambiguous = ambiguous_ids(ids, geom)
    record = {"active": geom["active"]}
    diagnostics = {}
    for variant in ("raw", "supported", "supported_nosym"):
        counts = unpooled if variant == "supported_nosym" else pooled
        prediction = posterior_decisions(base, counts, conservative=variant != "raw")
        if variant != "supported_nosym":
            prediction[ambiguous] = base[ambiguous]
        flips = ids[prediction != base]
        payload = prediction if point.L == 5 else flips.astype("<u4")
        record[variant] = serialize(payload)
        diagnostics[variant] = {"flipped_ids": len(flips), "payload_bytes": payload.nbytes,
            "encoded_chars": len(record[variant]),
            "changed_cell_min_support": int(counts[prediction != base].sum(axis=1).min()) if len(flips) else None}
    return record, diagnostics


def train(args):
    if (HERE / "freeze.json").exists():
        raise RuntimeError("frozen run cannot be retrained; preserve held-out design")
    start = time.perf_counter()
    tables, archive, points = {}, {}, {}
    geoms = {L: geometry(L) for L in (3, 5, 7)}
    dump(HERE / "geometry_audit.json", geoms)
    for point in challenge_grid():
        if point.L == 3 or point.xi == 0:
            continue
        tick = time.perf_counter()
        geom = geoms[point.L]
        ex = SurfaceCodeExperiment(point.L)
        entropy = [TRAIN_SEED, point.L, round(point.p*1e6), round(point.xi)]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        raw, sym = {}, {}
        for offset in range(0, args.shots, 25000):
            s, truth = ex.sample_correlated(shots=min(25000, args.shots-offset),
                                          p=point.p, xi=point.xi, rng=rng)
            ids = ids_from_syndrome(s, geom["active"])
            canon, action = canonicalize(ids, geom)
            add_counts(raw, ids, truth.astype(np.uint8))
            add_counts(sym, canon, truth.astype(np.uint8) ^ action)
        raw_keys, raw_counts = array_counts(raw)
        sym_keys, sym_counts = array_counts(sym)
        assert raw_counts.sum() == args.shots == sym_counts.sum()
        k = point_key(point)
        for name, value in (("raw_keys", raw_keys), ("raw_counts", raw_counts),
                            ("sym_keys", sym_keys), ("sym_counts", sym_counts)):
            archive[k + "_" + name] = value
        tables[k], diagnostics = materialize(point, geom, raw_keys, raw_counts, sym_keys, sym_counts)
        points[k] = {"seed_sequence_entropy": entropy, "shots": args.shots,
            "raw_unique_syndromes": len(raw_keys), "canonical_unique_syndromes": len(sym_keys),
            "canonical_cells_support_ge20": int((sym_counts.sum(axis=1) >= 20).sum()),
            "variants": diagnostics, "elapsed_s": time.perf_counter()-tick,
            "maxrss_bytes": rss()}
        print(k, json.dumps(diagnostics), flush=True)
    np.savez_compressed(HERE / "training_counts.npz", **archive)
    (HERE / "frozen_tables.py").write_text(
        '"""Generated offline; research only. See training.json and freeze.json."""\n'
        + "TABLES = " + repr(tables) + "\n")
    dump(HERE / "training.json", {"shots_per_point": args.shots, "seed": TRAIN_SEED,
        "batch": 25000, "points": points, "environment": environment(),
        "elapsed_s": time.perf_counter()-start, "maxrss_bytes": rss(),
        "prior": "Beta(wrong+1,right+9); >=20 samples, lower 1% posterior quantile > .5",
        "symmetry_counts": "one canonical observation per independent mask; no augmented pseudo-replicates"})


def freeze(args):
    target = HERE / "freeze.json"
    if target.exists():
        raise RuntimeError("already frozen")
    dump(target, {"hashes": hashes(), "test_seeds": TEST_SEEDS, "shots_per_point": 100000,
        "variants": VARIANTS, "primary": "supported vs IID, separately L5 and L7; Holm family 2",
        "test_sampled_before_freeze": False, "environment": environment(),
        "training_shots": json.loads((HERE / "training.json").read_text())["shots_per_point"],
        "status": "research-only; awaiting user review; no deployment authorization"})
    print("Design frozen", flush=True)


def verify_freeze():
    frozen = json.loads((HERE / "freeze.json").read_text())
    actual = hashes()
    assert frozen["hashes"] == actual, "frozen source/data changed"
    return hashlib.sha256((HERE / "freeze.json").read_bytes()).hexdigest()


def paired(rescue, harm, n):
    d = (rescue-harm)/n
    se = np.sqrt(max(0., (rescue+harm)/n-d*d)/(n-1)) if n > 1 else 0.
    return {"rescue": rescue, "harm": harm, "saved": rescue-harm,
            "saved_rate": d, "paired_se": float(se),
            "ci95": [float(d-1.95996398454*se), float(d+1.95996398454*se)],
            "mcnemar_p": float(binomtest(rescue, rescue+harm).pvalue) if rescue+harm else 1.,
            "zero_discordance_rate_upper95": float(-np.expm1(np.log(.05)/n)) if rescue+harm == 0 else None}


def aggregate(rows):
    n = sum(r["shots"] for r in rows)
    out = {"shots": n, "decoders": {}, "paired_vs_iid": {}, "paired_vs_a": {}}
    for variant in VARIANTS:
        values = [r["decoders"][variant] for r in rows]
        errors = sum(v["errors"] for v in values)
        out["decoders"][variant] = {"errors": errors, "error_rate": errors/n,
            "max_build_decode_s": max(v["elapsed_s"] for v in values),
            "max_build_s": max(v["build_s"] for v in values),
            "max_decode_s": max(v["decode_s"] for v in values),
            "timeouts": sum(v["timeout"] for v in values)}
        for reference in ("iid", "a"):
            field = "paired_vs_" + reference
            parts = [r[field][variant] for r in rows]
            result = paired(sum(p["rescue"] for p in parts), sum(p["harm"] for p in parts), n)
            se = np.sqrt(sum((r["shots"]*p["paired_se"])**2 for r,p in zip(rows,parts)))/n
            result["stratified_ci95"] = [result["saved_rate"]-1.96*se, result["saved_rate"]+1.96*se]
            out[field][variant] = result
    return out


def holm(items):
    running = 0.
    for rank, item in enumerate(sorted(items, key=lambda x: x["mcnemar_p"])):
        running = max(running, min(1., (len(items)-rank)*item["mcnemar_p"]))
        item["holm_p"] = running


def summarize(rows):
    by_l = {str(L): aggregate([r for r in rows if r["L"] == L]) for L in sorted({r["L"] for r in rows})}
    by_point = {k: aggregate([r for r in rows if r["key"] == k]) for k in dict.fromkeys(r["key"] for r in rows)}
    holm([v["paired_vs_iid"]["supported"] for k,v in by_l.items() if k in ("5", "7")])
    holm([v["paired_vs_iid"]["supported"] for k,v in by_point.items() if not k.startswith("L3_")])
    return {"full_grid" if len(by_l) == 3 else "stress_points": aggregate(rows),
        "by_L": by_l, "by_point": by_point,
        "by_seed": {str(seed): aggregate([r for r in rows if r["seed"] == seed]) for seed in sorted({r["seed"] for r in rows})}}


def support_diagnostics(point, syndrome, geom, archive):
    if point.L == 3 or point.xi == 0:
        return {"mode": "unchanged A" if point.L == 3 else "structural IID"}
    k = point_key(point)
    ids = ids_from_syndrome(syndrome, geom["active"])
    canon, _ = canonicalize(ids, geom)
    result = {}
    for mode, lookup in (("raw", ids), ("sym", canon)):
        counts = lookup_counts(archive[k+"_"+mode+"_keys"], archive[k+"_"+mode+"_counts"], lookup).sum(axis=1)
        result[mode] = {"unseen_shots": int((counts == 0).sum()),
            "support_ge20_shots": int((counts >= 20).sum()),
            "nonzero_syndrome_shots": int((ids != 0).sum()),
            "unseen_nonzero_rate": float(((counts == 0) & (ids != 0)).sum()/max(1,(ids != 0).sum()))}
    return result


def evaluate(args):
    kind = args.mode
    freeze_hash = verify_freeze() if kind in ("test", "stress") else None
    seeds = TEST_SEEDS if kind == "test" else ([49979687] if kind == "stress" else [PILOT_SEED])
    shots = 100000 if kind == "test" else (1000000 if kind == "stress" else 50000)
    label = args.label or kind
    output = HERE / f"results_{label}.jsonl"
    if output.exists():
        raise RuntimeError("result already exists; choose a new label to preserve receipts")
    archive = np.load(HERE / "training_counts.npz")
    geoms = {L: geometry(L) for L in (3, 5, 7)}
    experiments = {L: SurfaceCodeExperiment(L) for L in (3, 5, 7)}
    rows = []
    with output.open("w") as file:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for index, point in enumerate(challenge_grid()):
                # Unconditional one-call advance at every point, including skipped L3.
                syndrome, truth = experiments[point.L].sample_correlated(
                    shots=shots, p=point.p, xi=point.xi, rng=rng)
                if kind == "stress" and not (point.L in (5,7) and point.p == .01 and point.xi == 10):
                    continue
                row = {"key": point_key(point), "L": point.L, "p": point.p, "xi": point.xi,
                    "seed": seed, "shots": shots, "phase": kind, "freeze_sha256": freeze_hash,
                    "decoders": {}, "paired_vs_iid": {}, "paired_vs_a": {}}
                predictions, wrong = {}, {}
                order = VARIANTS if (seed+index) % 2 else VARIANTS[::-1]
                for variant in order:
                    before = current_rss()
                    start = time.perf_counter()
                    decoder = build_decoder(point, variant)
                    built = time.perf_counter()
                    prediction = decoder.decode(syndrome)
                    end = time.perf_counter()
                    assert prediction.shape == truth.shape and prediction.dtype == np.uint8
                    assert np.all(prediction <= 1)
                    predictions[variant] = prediction
                    wrong[variant] = prediction != truth
                    row["decoders"][variant] = {"errors": int(wrong[variant].sum()),
                        "build_s": built-start, "decode_s": end-built, "elapsed_s": end-start,
                        "timeout": end-start > 2.5, "storage_bytes": getattr(decoder,"storage_bytes",None),
                        "rss_before_bytes": before, "rss_after_bytes": current_rss(), "maxrss_bytes": rss()}
                for variant in VARIANTS:
                    row["decoders"][variant]["changed_shots_vs_iid"] = int((predictions[variant] != predictions["iid"]).sum())
                    for reference, name in (("iid", "iid"), ("a", "a_reference")):
                        row["paired_vs_"+reference][variant] = paired(
                            int((wrong[name] & ~wrong[variant]).sum()),
                            int((wrong[variant] & ~wrong[name]).sum()), shots)
                row["support"] = support_diagnostics(point, syndrome, geoms[point.L], archive)
                rows.append(row)
                file.write(json.dumps(row, allow_nan=False)+"\n")
                file.flush()
            print("finished", kind, seed, flush=True)
    summary = summarize(rows)
    summary.update({"phase": kind, "seeds": seeds, "shots_per_point": shots,
        "hashes": hashes(), "freeze_sha256": freeze_hash, "environment": environment(),
        "rng": "one default_rng(seed); all 24 points in official order; exactly one official sampler call per point incl skipped L3",
        "uncertainty": "paired fixed-stratum normal CIs; exact two-sided McNemar; zero discordance upper bound supplied; no CI establishes pointwise noninferiority",
        "maxrss_bytes": rss()})
    dump(HERE / f"results_{label}.summary.json", summary)
    print(json.dumps(summary["by_L"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--shots", type=int, default=250000)
    sub.add_parser("freeze")
    p = sub.add_parser("evaluate")
    p.add_argument("--mode", choices=("pilot", "test", "stress"), required=True)
    p.add_argument("--label")
    args = parser.parse_args()
    if args.command == "train" and not 0 < args.shots <= 1000000:
        parser.error("training budget must be 1..1M/point")
    {"train": train, "freeze": freeze, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
