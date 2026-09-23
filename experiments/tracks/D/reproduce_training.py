"""Replay the frozen training seeds without replacing any model artifact."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qec_benchmark.config import challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment
from experiments.tracks.D.run_track_d import (verify_freeze, TRAIN_SEED, add_counts,
    array_counts, materialize, dump, environment, rss)
from experiments.tracks.D.geometry import geometry, canonicalize, ids_from_syndrome
from experiments.tracks.D.candidate_d import point_key
from experiments.tracks.D.frozen_tables import TABLES


def main():
    freeze_hash = verify_freeze()
    training = json.loads((HERE / "training.json").read_text())
    archive = np.load(HERE / "training_counts.npz")
    start = time.perf_counter()
    points = []
    for point in challenge_grid():
        if point.L == 3 or point.xi == 0:
            continue
        k = point_key(point)
        geom, ex = geometry(point.L), SurfaceCodeExperiment(point.L)
        entropy = [TRAIN_SEED,point.L,round(point.p*1e6),round(point.xi)]
        assert entropy == training["points"][k]["seed_sequence_entropy"]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        raw, sym = {}, {}
        shots, batch = training["shots_per_point"], training["batch"]
        for offset in range(0,shots,batch):
            s,y = ex.sample_correlated(shots=min(batch,shots-offset),p=point.p,xi=point.xi,rng=rng)
            ids = ids_from_syndrome(s,geom["active"])
            canon, action = canonicalize(ids,geom)
            add_counts(raw,ids,y.astype(np.uint8))
            add_counts(sym,canon,y.astype(np.uint8)^action)
        raw_keys, raw_counts = array_counts(raw)
        sym_keys, sym_counts = array_counts(sym)
        for name, value in (("raw_keys",raw_keys),("raw_counts",raw_counts),
                            ("sym_keys",sym_keys),("sym_counts",sym_counts)):
            assert np.array_equal(value,archive[k+"_"+name]), (k,name)
        record, _ = materialize(point,geom,raw_keys,raw_counts,sym_keys,sym_counts)
        assert record == TABLES[k], k
        points.append(k)
    verify_freeze()
    report = {"status":"PASS", "freeze_sha256":freeze_hash,
        "points":points, "shots_replayed":len(points)*training["shots_per_point"],
        "counts_equal":True,"constants_equal":True,"model_files_overwritten":False,
        "elapsed_s":time.perf_counter()-start,"maxrss_bytes":rss(),"environment":environment()}
    dump(HERE / "reproducibility.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
