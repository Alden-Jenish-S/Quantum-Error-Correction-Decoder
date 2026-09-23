"""Small-shot CLI, evaluator equivalence and failure-path regression checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qec_benchmark import evaluation
from qec_benchmark.config import DEFAULT_SHOTS, VALIDATE_SEEDS, tiny_grid


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    modules = []
    for name in ("run_experiment", "verify_submission"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return SimpleNamespace(research=modules[0], verifier=modules[1])


def _cli(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{script}.py"), *map(str, args)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False, timeout=60,
    )


def test_final_loads_once_across_points_and_seeds(tools, monkeypatch):
    harness = tools.research
    calls = []
    original = harness._load_final_build_decoder

    def loader():
        calls.append("load")
        return original()

    monkeypatch.setattr(harness, "_load_final_build_decoder", loader)
    rows = harness.run("final", 16, [42, 137], "tiny")
    assert calls == ["load"]
    assert len(rows) == 2 * len(tiny_grid())
    assert all(row["exception"] is None and row["contract_valid"] for row in rows)
    assert all(row["elapsed_s"] >= row["build_elapsed_s"] + row["decode_elapsed_s"] for row in rows)


@pytest.mark.parametrize("candidate", ["baseline", "uniform", "data_iid", "final"])
def test_research_counts_match_official_evaluator(tools, candidate):
    harness = tools.research
    factory = harness._load_final_build_decoder() if candidate == "final" else harness.DECODERS[candidate]
    expected = evaluation.run_benchmark(factory, tiny_grid(), 32, 42, time_limit=2.5)
    rows = harness.run(candidate, 32, [42], "tiny")
    assert [row["errors"] for row in rows] == [point.errors for point in expected.point_results]
    assert not any(row["timed_out"] for row in rows)


@pytest.mark.parametrize("stage", ["build", "decode"])
def test_exception_has_one_complete_row_and_correct_durations(tools, monkeypatch, stage):
    harness = tools.research
    now = [0.0]
    monkeypatch.setattr(harness, "time", SimpleNamespace(perf_counter=lambda: now[0]))

    class BrokenDecoder:
        def decode(self, syndromes):
            now[0] += 0.75
            raise RuntimeError("intentional failure")

    def factory(point):
        now[0] += 0.25
        if stage == "build":
            raise RuntimeError("intentional failure")
        return BrokenDecoder()

    monkeypatch.setitem(harness.DECODERS, "broken", factory)
    rows = harness.run("broken", 8, [42, 137], "tiny")
    assert len(rows) == 6
    for row in rows:
        assert row["seed"] in (42, 137)
        assert row["L"] in (3, 5)
        assert row["p"] > 0
        assert row["xi"] >= 0
        assert row["crashed"] and row["timed_out"]
        assert row["errors"] == row["shots"] == 8
        assert row["failure_stage"] == stage
        assert row["exception_type"] == "RuntimeError"
        assert row["build_elapsed_s"] == 0.25
        assert row["decode_elapsed_s"] == (0.75 if stage == "decode" else 0.0)
        assert row["elapsed_s"] == (1.0 if stage == "decode" else 0.25)


@pytest.mark.parametrize("bad_output", ["list", "shape", "dtype", "binary"])
def test_both_harnesses_reject_malformed_output(tools, monkeypatch, bad_output):
    class BadDecoder:
        def decode(self, syndromes):
            n = len(syndromes)
            return {
                "list": [0] * n,
                "shape": np.zeros((n, 1), dtype=np.uint8),
                "dtype": np.zeros(n, dtype=np.float64),
                "binary": np.full(n, 2, dtype=np.uint8),
            }[bad_output]

    factory = lambda point: BadDecoder()
    monkeypatch.setitem(tools.research.DECODERS, "bad", factory)
    rows = tools.research.run("bad", 8, [42], "tiny")
    assert len(rows) == 3
    assert all(row["errors"] == 8 and row["failure_stage"] == "contract" for row in rows)
    monkeypatch.setattr(tools.verifier, "_load_final_build_decoder", lambda: factory)
    report = tools.verifier.verify(shots=8, grid_name="tiny")
    assert not report["passed"]
    assert len(report["points"]) == 3
    assert all(not point["contract_valid"] and point["contract_error"] for point in report["points"])


def test_validation_checks_are_outside_research_timer(tools, monkeypatch):
    now = [0.0]
    harness = tools.research
    monkeypatch.setattr(harness, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    original = harness._contract_error

    def expensive_check(predictions, shots):
        now[0] += 100.0
        return original(predictions, shots)

    monkeypatch.setattr(harness, "_contract_error", expensive_check)
    rows = harness.run("final", 8, [42], "tiny")
    assert all(row["elapsed_s"] == 0.0 and not row["timed_out"] for row in rows)


@pytest.mark.parametrize("limit", [2.5, None])
def test_research_timeout_counts_build_and_decode(tools, monkeypatch, limit):
    harness = tools.research
    now = [0.0]
    monkeypatch.setattr(harness, "time", SimpleNamespace(perf_counter=lambda: now[0]))

    class Decoder:
        def decode(self, syndromes):
            now[0] += 2.0
            return np.zeros(len(syndromes), dtype=np.uint8)

    def build(point):
        now[0] += 0.75
        return Decoder()

    monkeypatch.setitem(harness.DECODERS, "slow", build)
    rows = harness.run("slow", 8, [42], "tiny", time_limit=limit)
    for row in rows:
        assert row["build_elapsed_s"] == 0.75
        assert row["decode_elapsed_s"] == 2.0
        assert row["elapsed_s"] == 2.75
        assert row["timed_out"] == (limit is not None)
        assert not row["crashed"] and row["contract_valid"]
        if limit is not None:
            assert row["errors"] == 8


@pytest.mark.parametrize("exploratory", [False, True])
def test_research_cli_provenance_and_time_limit(tmp_path, exploratory):
    output = tmp_path / "result.jsonl"
    extra = ["--no-time-limit"] if exploratory else []
    completed = _cli("run_experiment", "--decoder", "final", "--grid", "tiny", "--shots", 8,
                     "--output", output, *extra)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert len(rows) == summary["rows"] == 3
    assert summary["time_limit"] == (None if exploratory else 2.5)
    assert summary["total_shots"] == 24
    assert summary["total_errors"] == sum(row["errors"] for row in rows)
    source = (ROOT / "solve.py").read_bytes()
    assert summary["solve_sha256"] == hashlib.sha256(source).hexdigest()
    assert summary["solve_size_bytes"] == len(source)
    assert all(summary["dependencies"][name] for name in ("python", "numpy", "scipy", "stim", "pymatching"))
    assert "git_revision" in summary
    assert str(ROOT) not in json.dumps(summary)
    assert "whole-process" in summary["memory_method"]
    assert summary["process_rss_bytes"] == summary["process_maxrss_bytes"]


def test_unborn_head_is_nullable(tools, monkeypatch):
    def unborn(*args, **kwargs):
        return SimpleNamespace(returncode=128, stdout="HEAD\n")

    monkeypatch.setattr(tools.research.subprocess, "run", unborn)
    assert tools.research._provenance()["git_revision"] is None


@pytest.mark.parametrize("system,multiplier", [("Darwin", 1), ("Linux", 1024)])
def test_process_rss_is_normalized_to_bytes(tools, monkeypatch, system, multiplier):
    monkeypatch.setattr(tools.research.platform, "system", lambda: system)
    monkeypatch.setattr(tools.research.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1234))
    assert tools.research._maxrss_bytes() == 1234 * multiplier


def test_verifier_cli_matches_unmodified_evaluator(tmp_path, tools):
    output = tmp_path / "verification.json"
    completed = _cli("verify_submission", "--grid", "tiny", "--shots", 32, "--output", output)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    expected = evaluation.run_benchmark(tools.research._load_final_build_decoder(), tiny_grid(), 32, 42, 2.5)
    assert report["passed"]
    assert report["seeds"] == [42]
    assert report["time_limit"] == 2.5
    assert report["score"] == expected.score
    assert [point["errors"] for point in report["points"]] == [point.errors for point in expected.point_results]
    assert all(point["contract_valid"] for point in report["points"])
    assert all(point["elapsed_s"] >= point["build_elapsed_s"] + point["decode_elapsed_s"] for point in report["points"])
    assert all(check["passed"] for check in report["input_contract_checks"])
    assert report["solve_size_bytes"] < 200_000
    assert len(report["solve_sha256"]) == 64
    assert isinstance(report["process_maxrss_bytes"], int) and report["process_maxrss_bytes"] > 0
    assert report["process_rss_bytes"] == report["process_maxrss_bytes"]
    assert "whole-process" in report["memory_method"]
    assert "construction" in report["timing_method"]
    assert str(ROOT) not in json.dumps(report)


def test_verifier_validate_uses_official_seeds_and_one_load(tools, monkeypatch):
    original = tools.verifier._load_final_build_decoder
    loads = []

    def loader():
        loads.append(1)
        return original()

    monkeypatch.setattr(tools.verifier, "_load_final_build_decoder", loader)
    report = tools.verifier.verify(shots=8, seed=999, validate=True, grid_name="tiny")
    assert report["passed"]
    assert loads == [1]
    assert report["seeds"] == VALIDATE_SEEDS
    assert len(report["points"]) == 3 * len(VALIDATE_SEEDS)
    assert [row["seed"] for row in report["seed_scores"]] == VALIDATE_SEEDS


def test_verifier_checks_are_outside_official_timer(tools, monkeypatch):
    verifier = tools.verifier
    now = [0.0]
    monkeypatch.setattr(verifier, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(monotonic=lambda: now[0]))
    original_check = verifier._contract_error
    original_probe = verifier._check_input_contract

    def expensive_check(predictions, shots):
        now[0] += 100.0
        return original_check(predictions, shots)

    def expensive_probe(build, point):
        now[0] += 100.0
        return original_probe(build, point)

    monkeypatch.setattr(verifier, "_contract_error", expensive_check)
    monkeypatch.setattr(verifier, "_check_input_contract", expensive_probe)
    assert verifier.run_benchmark is evaluation.run_benchmark
    report = verifier.verify(shots=8, grid_name="tiny")
    assert report["passed"]
    assert now[0] > 100.0
    assert all(point["elapsed_s"] == 0.0 and not point["timed_out"] for point in report["points"])


def test_verifier_small_shots_cover_full_challenge_grid(tools):
    report = tools.verifier.verify(shots=16)
    assert report["passed"]
    assert len(report["points"]) == 24
    assert report["total_shots"] == 24 * 16
    assert {point["L"] for point in report["points"]} == {3, 5, 7}
    assert len(report["input_contract_checks"]) == 24


def test_verifier_detects_input_mutation(tools, monkeypatch):
    class MutatingDecoder:
        def decode(self, syndromes):
            syndromes[:] = ~syndromes
            return np.zeros(len(syndromes), dtype=np.uint8)

    monkeypatch.setattr(tools.verifier, "_load_final_build_decoder", lambda: lambda point: MutatingDecoder())
    report = tools.verifier.verify(shots=8, grid_name="tiny")
    assert not report["passed"]
    assert all(not check["passed"] for check in report["input_contract_checks"])
    assert all("modified its syndrome input" in check["exception"] for check in report["input_contract_checks"])


@pytest.mark.parametrize("failure", ["timeout", "build", "decode", "load"])
def test_verifier_cli_nonzero_with_saved_failure(tools, monkeypatch, tmp_path, failure):
    verifier = tools.verifier
    now = [0.0]
    monkeypatch.setattr(verifier, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Decoder:
        def decode(self, syndromes):
            now[0] += 2.25 if failure == "timeout" else 0.75
            if failure == "decode":
                raise RuntimeError("decode failed")
            return np.zeros(len(syndromes), dtype=np.uint8)

    def build(point):
        now[0] += 0.5
        if failure == "build":
            raise RuntimeError("build failed")
        return Decoder()

    def load():
        if failure == "load":
            raise ImportError("load failed")
        return build

    monkeypatch.setattr(verifier, "_load_final_build_decoder", load)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["verify_submission.py", "--grid", "tiny", "--shots", "8", "--output", str(output)])
    assert verifier.main() == 1
    report = json.loads(output.read_text())
    assert not report["passed"]
    if failure == "load":
        assert report["exception_type"] == "ImportError"
    else:
        assert len(report["points"]) == 3
        assert report["score"] == 1_000_000
        for point in report["points"]:
            assert point["timed_out"] and point["errors"] == 8
            assert point["build_elapsed_s"] == 0.5
            assert point["decode_elapsed_s"] == {"build": 0.0, "decode": 0.75, "timeout": 2.25}[failure]
            assert point["elapsed_s"] == point["build_elapsed_s"] + point["decode_elapsed_s"]


def test_verifier_cli_defaults_without_expensive_run(tools, monkeypatch, tmp_path):
    captured = {}

    def verify(**kwargs):
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(tools.verifier, "verify", verify)
    monkeypatch.setattr(sys, "argv", ["verify_submission.py", "--output", str(tmp_path / "default.json")])
    assert tools.verifier.main() == 0
    assert captured == {"shots": DEFAULT_SHOTS, "seed": 42, "validate": False, "grid_name": "challenge"}


@pytest.mark.parametrize("script", ["run_experiment", "verify_submission"])
def test_cli_rejects_nonpositive_shots(tmp_path, script):
    output = tmp_path / "bad.json"
    completed = _cli(script, "--shots", 0, "--output", output)
    assert completed.returncode != 0
    assert "--shots must be positive" in completed.stderr
    assert not output.exists()
