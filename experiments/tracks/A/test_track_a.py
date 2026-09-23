from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri

from experiments.tracks.A.candidate_a import build_decoder, TableDecoder
from experiments.tracks.A.run_track_a import binary_covariance, enumerate_masks, evaluate, final_decoder
from qec_benchmark.evaluation import run_benchmark
from qec_benchmark.config import challenge_grid
from qec_benchmark.baselines import MWPMDecoder
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment

HERE = Path(__file__).resolve().parent


def test_mask_enumeration_preserves_surface_code_contract():
    experiment, masks, syndrome_ids, logical = enumerate_masks()
    assert masks.shape == (512, 9)
    assert syndrome_ids.shape == (512,)
    assert logical.shape == (512,)
    assert experiment.num_detectors == 8
    assert np.array_equal(np.unique(syndrome_ids), np.arange(16))
    assert logical[0] == 0


def test_l3_table_is_full_and_binary():
    decoder = build_decoder(SimpleNamespace(L=3, p=0.01, xi=10.0))
    assert decoder.table.shape == (256,)
    assert decoder.table.dtype == np.uint8
    syndromes = np.zeros((3, 8), dtype=np.uint8)
    syndromes[1, 0] = 1
    syndromes[2, 7] = 1
    predictions = decoder.decode(syndromes)
    assert predictions.shape == (3,)
    assert set(predictions.tolist()) <= {0, 1}


def test_fallback_matches_interface_for_each_official_distance():
    for distance, detectors in ((5, 24), (7, 48)):
        decoder = build_decoder(SimpleNamespace(L=distance, p=0.005, xi=2.0))
        predictions = decoder.decode(np.zeros((2, detectors), dtype=np.uint8))
        assert predictions.shape == (2,)
        assert predictions.dtype == np.uint8


def test_binary_covariance_is_not_latent_rho():
    p = 0.01
    rho = np.exp(-2.0 / 5.0)
    covariance = binary_covariance(p, rho)[0]
    assert 0.0 < covariance < rho
    assert abs(binary_covariance(p, 0.0)[0]) < 1e-14


def test_covariance_formula_tail_convention():
    p, rho = .01, .67
    t = ndtri(1-p)
    actual = binary_covariance(p, rho)[0]
    # Independent deterministic 1D conditional-Gaussian integration.
    from scipy.integrate import quad
    from scipy.special import ndtr
    joint, _ = quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi) *
                   ndtr((rho*z-t)/np.sqrt(1-rho*rho)), t, np.inf,
                   epsabs=1e-13)
    assert abs(actual-(joint-p*p)) < 1e-12
    assert (1-p)**2-p*p > .9  # The originally requested lower-tail formula is wrong.


def test_same_inputs_match_unmodified_official_evaluator():
    args = SimpleNamespace(shots=128, seed=137, validate=False, label="unused_test")
    rows, summary = evaluate(args, write_records=False)
    factories = {"candidate_a": build_decoder, "final": final_decoder,
                 "baseline": lambda p: MWPMDecoder(p, weighted=True)}
    for name, factory in factories.items():
        expected = run_benchmark(factory, challenge_grid(), args.shots, args.seed, time_limit=2.5)
        assert [r["decoders"][name]["errors"] for r in rows] == [p.errors for p in expected.point_results]
        assert not any(r["decoders"][name]["timed_out"] for r in rows)


def test_lookup_endianness_input_immutability_and_empty_batch():
    table = np.arange(256, dtype=np.uint8) % 2
    decoder = TableDecoder(table)
    for dtype in (bool, np.uint8):
        inputs = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(dtype)
        before = inputs.copy()
        inputs.setflags(write=False)
        assert np.array_equal(decoder.decode(inputs), table)
        assert np.array_equal(inputs, before)
    assert decoder.decode(np.zeros((0, 8), dtype=np.uint8)).shape == (0,)
    with pytest.raises(ValueError):
        decoder.decode(np.zeros((1, 9)))


def test_fallback_predictions_identical_on_official_noise():
    for point in challenge_grid()[8:]:
        syndromes, _ = SurfaceCodeExperiment(point.L).sample_correlated(
            shots=500, p=point.p, xi=point.xi, rng=np.random.default_rng(87000))
        assert np.array_equal(build_decoder(point).decode(syndromes), final_decoder(point).decode(syndromes))


def test_integrated_tables_are_valid_and_frozen_predictions_did_not_change():
    report = json.loads((HERE / "tables_qmc.json").read_text())
    for point in challenge_grid()[:8]:
        item = report["points"][f"p{point.p:g}_xi{point.xi:g}"]
        refined = item["refined"]
        assert len(refined["mask_probabilities"]) == 512
        assert all(p >= 0 for p in refined["mask_probabilities"])
        assert abs(refined["raw_probability_sum"]-1) < 1e-5
        assert item["convergence"]["table_changes"] == []
        assert refined["unresolved_active_syndromes_3se"] == []
        assert np.array_equal(build_decoder(point).table, refined["table"])
    digest = hashlib.sha256((HERE / "frozen_tables.py").read_bytes()).hexdigest()
    for filename in ("results_pilot_10k.json", "results_official_100k_5seeds.json"):
        result = json.loads((HERE / filename).read_text())
        assert result["provenance"]["sha256"]["experiments/tracks/A/frozen_tables.py"] == digest
