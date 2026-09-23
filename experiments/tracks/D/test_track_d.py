"""Scientific-contract tests: geometry, symmetry labels, reference, freeze, RNG."""
from pathlib import Path
import ast
import inspect
import json
import numpy as np
import pytest

from qec_benchmark.config import challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment
from experiments.tracks.A.candidate_a import DataOnlyFallback, build_decoder as build_a
from experiments.tracks.D.reference_iid import IIDDecoder
from experiments.tracks.D.candidate_d import VARIANTS, build_decoder, packed_ids, point_key
from experiments.tracks.D.geometry import (geometry, canonicalize, parity, gf2_solve,
    ids_from_syndrome, syndrome_from_ids, ambiguous_ids)
from experiments.tracks.D.run_track_d import (posterior_decisions, materialize,
    add_counts, array_counts, lookup_counts, TEST_SEEDS, TRAIN_SEED, PILOT_SEED,
    verify_freeze)

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("L", [3, 5, 7])
def test_linear_geometry_and_physical_symmetries(L):
    geom = geometry(L)
    ex = SurfaceCodeExperiment(L)
    # Basis spans physical mask vector space; dense random masks exercise parity.
    masks = np.concatenate([np.eye(L*L, dtype=bool),
                            np.random.default_rng(611953+L).integers(0,2,(1000,L*L)).astype(bool)])
    s, truth = ex.sample_from_mask(masks)
    h = np.array(geom["h"], dtype=np.uint8)
    logical = np.array(geom["logical"], dtype=np.uint8)
    assert np.array_equal((masks @ h) % 2, s[:, geom["active"]])
    assert np.array_equal((masks @ logical) % 2, truth)
    # Every active-check target is reachable; all 2^m syndromes are legal.
    for row in np.eye(h.shape[1], dtype=np.uint8):
        assert gf2_solve(h.T, row) is not None
    ids = ids_from_syndrome(s, geom["active"])
    assert np.array_equal(packed_ids(s, geom["active"]), ids)
    canonical, action = canonicalize(ids, geom)
    assert len(geom["symmetries"]) == 2
    assert len(geom["rejected"]) == 6
    for sym in geom["symmetries"]:
        transformed = np.empty_like(masks)
        transformed[:, sym["physical_perm"]] = masks
        changed_s, changed_t = ex.sample_from_mask(transformed)
        assert np.array_equal(changed_s[:, geom["active"]], s[:, geom["active"]][:, sym["syndrome_perm"]])
        assert np.array_equal(changed_t.astype(np.uint8) ^ truth, parity(ids & sym["logical_mask"]))
        canon2, action2 = canonicalize(ids_from_syndrome(changed_s, geom["active"]), geom)
        assert np.array_equal(canon2, canonical)
        valid = ~ambiguous_ids(ids, geom)
        assert np.array_equal((changed_t ^ action2)[valid], (truth ^ action)[valid])


@pytest.mark.parametrize("point", challenge_grid())
def test_reference_and_variant_contracts(point):
    ex = SurfaceCodeExperiment(point.L)
    s, _ = ex.sample_correlated(shots=400, p=point.p, xi=point.xi,
                               rng=np.random.default_rng(67867967))
    iid = IIDDecoder(point).decode(s)
    assert np.array_equal(iid, DataOnlyFallback(point).decode(s))
    for variant in VARIANTS:
        decoder = build_decoder(point, variant)
        prediction = decoder.decode(s)
        assert prediction.dtype == np.uint8 and prediction.shape == (400,)
        assert np.all(prediction <= 1)
        assert decoder.decode(s[:0]).shape == (0,)
        assert np.array_equal(decoder.decode(s), prediction)
        if point.L == 3 and variant != "iid":
            assert np.array_equal(prediction, build_a(point).decode(s))
        if point.xi == 0:
            assert np.array_equal(prediction, iid)


def test_prior_suppresses_rare_overfit_and_can_accept_evidence():
    counts = np.array([[0,1],[0,19],[0,20],[10,90],[90,10],[100,100],[0,0]])
    base = np.zeros(len(counts), dtype=np.uint8)
    assert posterior_decisions(base, counts, conservative=False).tolist() == [1,1,1,1,0,0,0]
    assert posterior_decisions(base, counts, conservative=True).tolist() == [0,0,0,1,0,0,0]


def test_counts_single_observation_and_determinism():
    for L in (5,7):
        geom = geometry(L)
        ex = SurfaceCodeExperiment(L)
        results = []
        for _ in range(2):
            s, y = ex.sample_correlated(shots=2000, p=.01, xi=10,
                                       rng=np.random.default_rng(np.random.SeedSequence([TRAIN_SEED,L,10000,10])))
            ids, action = canonicalize(ids_from_syndrome(s, geom["active"]), geom)
            counts = {}
            add_counts(counts, ids, y ^ action)
            keys, array = array_counts(counts)
            assert array.sum() == len(y)  # not two observations per symmetry orbit
            assert np.array_equal(lookup_counts(keys,array,keys), array)
            results.append((keys,array))
        assert all(np.array_equal(a,b) for a,b in zip(*results))


def test_materialized_constants_reproduce_counts():
    from experiments.tracks.D.frozen_tables import TABLES
    archive = np.load(HERE / "training_counts.npz")
    for point in challenge_grid():
        if point.L == 3 or point.xi == 0:
            continue
        k = point_key(point)
        record, _ = materialize(point, geometry(point.L),
            *(archive[k+"_"+name] for name in ("raw_keys","raw_counts","sym_keys","sym_counts")))
        assert record == TABLES[k]
    # Entire L7 supported representation, uncompressed, must be under 200 KB.
    assert sum(build_decoder(p,"supported").storage_bytes for p in challenge_grid()
               if p.L == 7 and p.xi > 0) < 200000


def test_all_l5_states_lookup_and_iid_reference():
    for point in challenge_grid():
        if point.L != 5:
            continue
        geom = geometry(5)
        s = syndrome_from_ids(np.arange(4096,dtype=np.uint32),geom)
        assert np.array_equal(IIDDecoder(point).decode(s),DataOnlyFallback(point).decode(s))
        for v in ("raw","supported","supported_nosym"):
            decoder = build_decoder(point,v)
            if point.xi:
                assert np.array_equal(decoder.decode(s),decoder.table)


def test_rng_order_includes_l3_and_seed_separation():
    assert not set(TEST_SEEDS) & {TRAIN_SEED,PILOT_SEED,42,137,256,1729,31415,49979687}
    # Compare iterator ordering (including L3 advances) against explicit official loop.
    reference, observed = [], []
    for destination in (reference, observed):
        rng = np.random.default_rng(86028121)
        for point in challenge_grid():
            s,y = SurfaceCodeExperiment(point.L).sample_correlated(shots=7,p=point.p,xi=point.xi,rng=rng)
            if destination is reference or point.L > 3:
                destination.append((s.copy(),y.copy()))
    assert len(observed) == 16
    assert all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(reference[8:],observed))
    # Audit actual runner ordering: sample call must precede stress skip.
    from experiments.tracks.D.run_track_d import evaluate
    source = inspect.getsource(evaluate)
    assert source.index("sample_correlated(") < source.index('if kind == "stress" and')


def test_decode_no_io_training_or_randomness():
    from experiments.tracks.D.candidate_d import PosteriorDecoder
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(PosteriorDecoder.decode)))
    forbidden = {"open","read","write","load","fit","train","sample","random"}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name = node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,"attr","")
            assert name not in forbidden


def test_frozen_provenance_if_present():
    if (HERE / "freeze.json").exists():
        assert len(verify_freeze()) == 64
