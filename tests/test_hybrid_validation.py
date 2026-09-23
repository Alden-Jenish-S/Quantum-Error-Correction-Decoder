"""Paired harness semantics, independent official scoring, and failure receipts."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from qec_benchmark.config import VALIDATE_SEEDS, challenge_grid
from qec_benchmark.evaluation import run_benchmark


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("hybrid_validation", ROOT / "scripts/validate_hybrid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_run(harness):
    emitted = []
    rows, timing = harness.evaluate(shots=32, seeds=[42, 137], time_limit=2.5, emit=emitted.append)
    assert emitted == rows
    return rows, timing


def test_all_variants_match_independent_official_evaluator(harness, small_run):
    rows, timing = small_run
    assert len(rows) == 48
    assert timing["pre_hybrid_circuit_checks"] == [
        {"L": L, "p": p} for L in (3, 5, 7) for p in (0.005, 0.01)]
    factories = {"circuit_mwpm": lambda p: harness.MWPMDecoder(p, weighted=True),
                 "pre_hybrid_data_only": harness.DataOnlyMWPM, "hybrid": harness.build_decoder}
    for seed in (42, 137):
        seed_rows = [row for row in rows if row["seed"] == seed]
        assert [(r["L"], r["p"], r["xi"]) for r in seed_rows] == [
            (p.L, p.p, p.xi) for p in challenge_grid()]
        assert [r["point_index"] for r in seed_rows] == list(range(24))
        for name, factory in factories.items():
            official = run_benchmark(factory, challenge_grid(), 32, seed, time_limit=2.5)
            assert [r["decoders"][name]["scored_errors"] for r in seed_rows] == [
                result.errors for result in official.point_results]
            assert all(r["decoders"][name]["status"] == "ok" for r in seed_rows)
        for row in seed_rows:
            for name in harness.REFERENCES:
                pair = row["paired"][name]
                assert pair["net_saved_errors"] == (row["decoders"][name]["errors"]
                                                    - row["decoders"]["hybrid"]["errors"])
                assert sum(pair[k] for k in ("rescue", "harm", "both_wrong", "both_correct")) == 32
            if row["L"] in (5, 7) or row["xi"] in (0, 2):
                assert row["paired"]["pre_hybrid_data_only"]["disagreements"] == 0


def test_shared_batch_rng_order_and_explicit_baseline(harness, monkeypatch):
    calls = []
    samples = []
    batches = []

    class Experiment:
        def __init__(self, L):
            self.L = L

        def sample_correlated(self, *, shots, p, xi, rng):
            batch = rng.integers(0, 2, (shots, 8), dtype=np.uint8)
            samples.append((self.L, p, xi, batch.copy()))
            batches.append(batch)
            return batch, batch[:, 0].copy()

    class Decoder:
        def __init__(self, name, point):
            self.name, self.point = name, point

        def decode(self, batch):
            assert batch is batches[-1]
            assert not batch.flags.writeable
            calls.append((self.name, self.point, batch))
            return batch[:, 0].copy()

    monkeypatch.setattr(harness, "SurfaceCodeExperiment", Experiment)
    monkeypatch.setattr(harness, "verify_pre_hybrid_circuits", lambda *args: [])
    monkeypatch.setattr(harness, "MWPMDecoder", lambda p, weighted: Decoder("circuit_mwpm", p))
    monkeypatch.setattr(harness, "DataOnlyMWPM", lambda p: Decoder("pre_hybrid_data_only", p))
    monkeypatch.setattr(harness, "build_decoder", lambda p: Decoder("hybrid", p))
    rows, _ = harness.evaluate(shots=3, seeds=[42, 137], time_limit=2.5, emit=lambda row: None)
    assert len(calls) == 144
    for si, seed in enumerate((42, 137)):
        rng = np.random.default_rng(seed)
        for pi, point in enumerate(challenge_grid()):
            index = si * 24 + pi
            L, p, xi, batch = samples[index]
            assert (L, p, xi) == (point.L, point.p, point.xi)
            np.testing.assert_array_equal(batch, rng.integers(0, 2, (3, 8), dtype=np.uint8))
            point_calls = calls[3 * index:3 * index + 3]
            assert [call[0] for call in point_calls] == rows[index]["variant_order"]
            assert set(call[0] for call in point_calls) == set(harness.VARIANTS)
            assert all(call[1] == point for call in point_calls)


@pytest.mark.parametrize("case", ["ok", "timeout", "build_error", "decode_error",
                                  "shape", "dtype", "binary", "not_array", "mutate"])
def test_timer_contract_and_all_wrong_failure_semantics(harness, monkeypatch, case):
    now = [10.0]
    batch = np.zeros((4, 8), dtype=np.uint8)
    before = batch.copy()
    batch.setflags(write=False)
    truth = np.array([0, 1, 0, 1], dtype=bool)

    class Decoder:
        def decode(self, batch):
            now[0] += 3.0 if case == "timeout" else 0.5
            if case == "decode_error":
                raise RuntimeError("a private path must not be saved")
            if case == "shape":
                return np.zeros((4, 1), dtype=np.uint8)
            if case == "dtype":
                return np.zeros(4, dtype=bool)
            if case == "binary":
                return np.full(4, 2, dtype=np.uint8)
            if case == "not_array":
                return [0, 0, 0, 0]
            if case == "mutate":
                batch.setflags(write=True)
                batch[0, 0] = 1
                batch.setflags(write=False)
            return np.zeros(4, dtype=np.uint8)

    def factory(point):
        now[0] += 0.25
        if case == "build_error":
            raise RuntimeError("a private path must not be saved")
        return Decoder()

    original = harness.check_contract

    def expensive_check(*args):
        now[0] += 100.0
        return original(*args)

    monkeypatch.setattr(harness, "check_contract", expensive_check)
    record, wrong = harness.measure_variant(factory, None, batch, truth, before, 2.5,
                                             clock=lambda: now[0])
    assert record["build_decode_s"] == (0.25 if case == "build_error" else 3.25 if case == "timeout" else 0.75)
    assert record["timed_out"] == (case == "timeout")
    if case in ("ok", "timeout"):
        assert record["errors"] == 2 and record["error_rate"] == 0.5
        np.testing.assert_array_equal(wrong, truth)
        assert record["scored_errors"] == (2 if case == "ok" else 4)
    else:
        assert record["errors"] is None and wrong is None
        assert record["scored_errors"] == 4
        assert record["status"] == ("exception" if case.endswith("error") else "contract_error")
        assert "private" not in record["failure"]


def test_paired_direction_and_unavailable(harness):
    pair = harness.paired_counts(np.array([0, 0, 1, 1, 0], dtype=bool),
                                 np.array([1, 1, 0, 1, 0], dtype=bool))
    assert pair == {"rescue": 2, "harm": 1, "net_saved_errors": 1, "disagreements": 3,
                    "both_wrong": 1, "both_correct": 1}
    assert harness.paired_counts(None, np.zeros(2, dtype=bool)) is None
    assert harness.paired_counts(np.zeros(2, dtype=bool), None) is None


def test_summary_totals_strata_and_regressions(harness, small_run):
    rows, _ = small_run
    summary = harness.summarize(rows)
    overall = summary["overall"]
    assert overall["shots_per_decoder"] == 48 * 32
    for axis in ("L", "p", "xi", "seed", "point"):
        groups = summary[f"by_{axis}"]
        assert sum(g["shots_per_decoder"] for g in groups.values()) == 48 * 32
        for name in harness.VARIANTS:
            assert sum(g["decoders"][name]["observed_errors"] for g in groups.values()) == overall["decoders"][name]["observed_errors"]
        for name in harness.REFERENCES:
            assert sum(g["paired"][name]["net_saved_errors"] for g in groups.values()) == overall["paired"][name]["net_saved_errors"]
    for name in harness.REFERENCES:
        assert summary["regressions_by_point"][name] == {
            key: value["paired"][name]["net_saved_errors"]
            for key, value in summary["by_point"].items()
            if value["paired"][name]["net_saved_errors"] < 0}


def test_rss_units_and_path_free_provenance(harness):
    assert harness.normalized_rss_bytes(123, "Darwin") == 123
    assert harness.normalized_rss_bytes(123, "Linux") == 123 * 1024
    receipt = harness.provenance()
    assert receipt["source_sha256"]["experiments/tracks/A/frozen_tables.py"] == harness.FROZEN_SHA256
    assert receipt["submission_bytes"] < 200_000
    assert "whole-process" in receipt["memory_method"]
    text = json.dumps(receipt)
    assert str(ROOT) not in text and str(Path.home()) not in text
    assert all(not Path(name).is_absolute() for name in receipt["source_sha256"])


def _standard_report():
    return ("\n".join(f"    seed {s:>5}:  3,141 errors/M" for s in VALIDATE_SEEDS)
            + "\n Score: 3,141 errors per million (mean)\n Timeouts: 0 / 120\n")


@pytest.mark.parametrize("failure", [None, "timeout", "returncode", "truncated"])
def test_compatibility_subprocess_and_report_preservation(harness, monkeypatch, tmp_path, failure):
    text = _standard_report()
    if failure == "truncated":
        text = text.split("Score")[0]

    def run(command, **kwargs):
        assert command == [sys.executable, "run.py", "--validate", "--shots", "1000000"]
        assert kwargs["cwd"] == ROOT and kwargs["timeout"] == 1800
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1800, output=text.encode())
        return SimpleNamespace(stdout=text, stderr="", returncode=1 if failure == "returncode" else 0)

    monkeypatch.setattr(harness.subprocess, "run", run)
    path = tmp_path / "standard.txt"
    result = harness.run_compatibility(1_000_000, path)
    assert path.read_text() == text
    assert result["status"] == {None: "ok", "timeout": "subprocess_timeout",
                                "returncode": "failed", "truncated": "unparseable_report"}[failure]
    if failure is None:
        assert result["per_seed_scores"] == {str(s): 3141 for s in VALIDATE_SEEDS}
        assert result["point_seed_count"] == 120 and result["timeouts"] == 0
    assert str(ROOT) not in json.dumps(result)


@pytest.mark.parametrize("changed", [False, True])
def test_main_receipts_and_source_change_failure(harness, monkeypatch, tmp_path, small_run, changed):
    rows, timing = small_run
    frozen = {"experiments/tracks/A/frozen_tables.py": harness.FROZEN_SHA256, "solve.py": "before"}
    after = {**frozen, "solve.py": "after" if changed else "before"}
    snapshots = iter([{"source_sha256": frozen}, {"source_sha256": after}])
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "provenance", lambda: next(snapshots))
    monkeypatch.setattr(harness, "source_hashes", lambda: after)

    def evaluate(**kwargs):
        for row in rows:
            kwargs["emit"](row)
        return rows, timing

    monkeypatch.setattr(harness, "evaluate", evaluate)
    code = harness.main(["--shots", "32", "--seeds", "42", "137", "--label", "receipt_test"])
    assert code == int(changed)
    folder = tmp_path / "experiments/hybrid"
    receipt = json.loads((folder / "receipt_test.summary.json").read_text())
    assert receipt["sources_unchanged"] == (not changed)
    lines = (folder / "receipt_test.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == rows
    with pytest.raises(SystemExit) as exc:
        harness.main(["--label", "receipt_test"])
    assert exc.value.code == 2


@pytest.mark.parametrize("args", [["--shots", "0"], ["--shots", "1000001"],
                                  ["--seeds", "42", "42"], ["--label", "../escape"],
                                  ["--time-limit", "nan"],
                                  ["--seeds", "42", "--compatibility"]])
def test_invalid_cli_parameters(harness, args):
    with pytest.raises(SystemExit) as exc:
        harness.main(args)
    assert exc.value.code == 2
