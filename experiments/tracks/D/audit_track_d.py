"""Audit retained receipts, report coverage/storage and render numeric tables.

Does not train, evaluate new shots, modify frozen sources, or select a model.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qec_benchmark.config import challenge_grid
from experiments.tracks.D.candidate_d import VARIANTS, build_decoder, point_key
from experiments.tracks.D.reference_iid import IIDDecoder
from experiments.tracks.D.geometry import geometry, syndrome_from_ids, canonicalize
from experiments.tracks.D.run_track_d import (verify_freeze, summarize, dump,
    lookup_counts, TEST_SEEDS)
from experiments.tracks.D.frozen_tables import TABLES


def main():
    freeze_hash = verify_freeze()
    all_rows = {}
    for label, size in (("pilot_250k",24),("pilot_1mtrain",24),("test",72),("stress",2)):
        rows = [json.loads(line) for line in (HERE / f"results_{label}.jsonl").read_text().splitlines()]
        assert len(rows) == size
        expected = summarize(rows)
        summary = json.loads((HERE / f"results_{label}.summary.json").read_text())
        for key, value in expected.items():
            assert value == summary[key]
        for row in rows:
            assert set(row["decoders"]) == set(VARIANTS)
            for variant in VARIANTS:
                comp = row["paired_vs_iid"][variant]
                assert comp["saved"] == row["decoders"]["iid"]["errors"]-row["decoders"][variant]["errors"]
                assert comp["rescue"]+comp["harm"] == row["decoders"][variant]["changed_shots_vs_iid"]
            if label in ("test","stress"):
                assert row["freeze_sha256"] == freeze_hash
        all_rows[label] = rows
    assert {r["seed"] for r in all_rows["test"]} == set(TEST_SEEDS)
    assert all(r["shots"] == 100000 for r in all_rows["test"])
    assert all(r["shots"] == 1000000 for r in all_rows["stress"])
    assert len({(r["seed"],r["key"]) for r in all_rows["test"]}) == 72

    archive = np.load(HERE / "training_counts.npz")
    training = json.loads((HERE / "training.json").read_text())
    supported_cells = {}
    totals = {v: {"payload_bytes": 0,"encoded_chars": 0,"L7_payload_bytes": 0} for v in VARIANTS[2:]}
    for point in challenge_grid():
        if point.L == 3 or point.xi == 0:
            continue
        k = point_key(point)
        geom = geometry(point.L)
        for v in VARIANTS[2:]:
            payload = training["points"][k]["variants"][v]["payload_bytes"]
            totals[v]["payload_bytes"] += payload
            totals[v]["encoded_chars"] += len(TABLES[k][v])
            if point.L == 7:
                totals[v]["L7_payload_bytes"] += payload
        decoder = build_decoder(point,"supported")
        if point.L == 5:
            ids = np.arange(4096,dtype=np.uint32)
            base = IIDDecoder(point).decode(syndrome_from_ids(ids,geom))
            ids = ids[base != decoder.table]
        else:
            ids = decoder.flip_ids
        s = syndrome_from_ids(ids,geom)
        base = IIDDecoder(point).decode(s)
        canonical, action = canonicalize(ids,geom)
        c = lookup_counts(archive[k+"_sym_keys"],archive[k+"_sym_counts"],canonical)
        c = np.where(action[:,None] != 0,c[:,::-1],c)
        details = []
        for i, sid in enumerate(ids):
            wrong, right = int(c[i,1-base[i]]), int(c[i,base[i]])
            details.append({"syndrome_id":int(sid),"iid":int(base[i]),
                "candidate":int(1-base[i]),"canonical_id":int(canonical[i]),
                "training_wrong":wrong,"training_right":right,
                "posterior_lower01":float(beta.ppf(.01,wrong+1,right+9))})
        supported_cells[k] = details
    assert totals["supported"]["L7_payload_bytes"] < 200000

    summary = json.loads((HERE / "results_test.summary.json").read_text())
    lines = ["# Generated numeric audit tables", "", "Positive saved = fewer errors than IID.", "",
             "## TEST: 300,000 shots per point", "",
             "| Point | IID errors | raw saved | supported saved | no-sym saved | supported rescue/harm | supported Holm p (16 points) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    coverage = {}
    for k,value in summary["by_point"].items():
        if k.startswith("L3_"):
            continue
        p = value["paired_vs_iid"]
        c = p["supported"]
        lines.append(f'| {k} | {value["decoders"]["iid"]["errors"]} | {p["raw"]["saved"]} | {c["saved"]} | {p["supported_nosym"]["saved"]} | {c["rescue"]}/{c["harm"]} | {c["holm_p"]:.6g} |')
        rows = [r for r in all_rows["test"] if r["key"] == k]
        if "sym" in rows[0]["support"]:
            coverage[k] = {}
            for mode in ("raw","sym"):
                fields = ("unseen_shots","support_ge20_shots","nonzero_syndrome_shots")
                totals_count = {f:sum(r["support"][mode][f] for r in rows) for f in fields}
                totals_count["unseen_nonzero_rate"] = totals_count["unseen_shots"]/totals_count["nonzero_syndrome_shots"]
                coverage[k][mode] = totals_count
    lines += ["", "## TEST coverage on nonzero-xi points", "",
              "| Point | nonzero syndrome shots | unseen raw | unseen pooled | pooled n>=20 shots (all) | supported changed / all shots |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, value in coverage.items():
        rows = [r for r in all_rows["test"] if r["key"] == k]
        changed = sum(r["decoders"]["supported"]["changed_shots_vs_iid"] for r in rows)
        lines.append(f'| {k} | {value["sym"]["nonzero_syndrome_shots"]} | {value["raw"]["unseen_shots"]} | {value["sym"]["unseen_shots"]} | {value["sym"]["support_ge20_shots"]} | {changed}/300000 |')
    lines += ["", "## Supported cell training posteriors", "",
              "| Point | syndrome ID | canonical ID | IID→D | training wrong/right | posterior 1% lower bound |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, entries in supported_cells.items():
        for cell in entries:
            lines.append(f'| {k} | {cell["syndrome_id"]} | {cell["canonical_id"]} | {cell["iid"]}→{cell["candidate"]} | {cell["training_wrong"]}/{cell["training_right"]} | {cell["posterior_lower01"]:.6f} |')
    (HERE / "numeric_tables.md").write_text("\n".join(lines)+"\n")
    memory = {}
    for label,rows in all_rows.items():
        readings = [d for row in rows for d in row["decoders"].values()]
        current = [d["rss_after_bytes"] for d in readings if d["rss_after_bytes"] is not None]
        memory[label] = {"max_rss_after_bytes":max(current) if current else None,
            "max_high_water_bytes":max(d["maxrss_bytes"] for d in readings),
            "interpretation":"whole process, includes sampling, references, previous decoder work and retained allocators"}
    result = {"status":"PASS", "freeze_sha256":freeze_hash,
        "rows_audited":{k:len(v) for k,v in all_rows.items()}, "serialization":totals,
        "frozen_tables_source_bytes":(HERE / "frozen_tables.py").stat().st_size,
        "coverage":coverage,"supported_cells":supported_cells,"memory":memory,
        "training_seconds":training["elapsed_s"], "training_maxrss_bytes":training["maxrss_bytes"],
        "supported_seed_point_regressions":[{"seed":r["seed"],"key":r["key"],
            "saved":r["paired_vs_iid"]["supported"]["saved"],
            "rescue":r["paired_vs_iid"]["supported"]["rescue"],
            "harm":r["paired_vs_iid"]["supported"]["harm"]}
            for r in all_rows["test"] if r["L"] > 3 and r["paired_vs_iid"]["supported"]["saved"] < 0],
        "pointwise_regressions":{v:[k for k,p in summary["by_point"].items()
             if not k.startswith("L3_") and p["paired_vs_iid"][v]["saved"] < 0]
             for v in VARIANTS[2:]}}
    dump(HERE / "audit.json",result)
    # Includes test/report scripts and receipts; manifest intentionally excludes itself.
    manifest = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.rglob("*")) if p.is_file()
                and "__pycache__" not in str(p) and p.name != "artifact_manifest.json"}
    dump(HERE / "artifact_manifest.json",manifest)
    print(json.dumps({k:result[k] for k in ("status","rows_audited","serialization",
        "frozen_tables_source_bytes","memory","training_seconds","training_maxrss_bytes",
        "pointwise_regressions","supported_seed_point_regressions")},indent=2))
    print("\n".join(lines[:23]))


if __name__ == "__main__":
    main()
