"""Submission compatibility, equivalence, and scoped no-I/O regression checks."""

import ast
import importlib.util
from itertools import product
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pymatching
import pytest
import stim

from qec_benchmark.config import challenge_grid
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


SOLVE_PATH = Path(__file__).resolve().parents[1] / "solve.py"
GRID = challenge_grid()


@pytest.fixture(scope="module")
def submission():
    spec = importlib.util.spec_from_file_location("submission_under_test", SOLVE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiments():
    return {distance: SurfaceCodeExperiment(distance=distance) for distance in (3, 5, 7)}


def _original_circuit(experiment, p):
    """The data-only construction from solve.py before it was made standalone."""
    circuit = stim.Circuit()
    circuit += experiment._prefix
    for qubit in experiment._data_qubits:
        circuit.append("X_ERROR", [int(qubit)], float(p))
    circuit += experiment._suffix
    return circuit


def _original_predictions(circuit, syndromes):
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    predictions = matching.decode_batch(syndromes.astype(np.uint8, copy=False))
    if predictions.ndim == 2:
        predictions = predictions[:, 0]
    return predictions.astype(np.uint8, copy=False)


@pytest.fixture(scope="module")
def diagnostic_syndromes(experiments):
    """Exhaust L=3 masks; exercise single errors and dense/tied cases at L=5,7."""
    batches = {}
    for distance, experiment in experiments.items():
        width = experiment.num_data_qubits
        if distance == 3:
            masks = np.array(list(product((False, True), repeat=width)), dtype=bool)
        else:
            rng = np.random.default_rng(1729 + distance)
            masks = np.concatenate(
                [
                    np.zeros((1, width), dtype=bool),
                    np.ones((1, width), dtype=bool),
                    np.eye(width, dtype=bool),
                    rng.random((512, width)) < 0.5,
                ]
            )
        batches[distance] = experiment.sample_from_mask(masks)[0]
    return batches


def _assert_interface(predictions, shots):
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (shots,)
    assert predictions.dtype == np.uint8
    assert np.all((predictions == 0) | (predictions == 1))


def test_grid_covers_all_24_points():
    assert len(GRID) == 24
    assert {(p.L, p.p, p.xi) for p in GRID} == set(
        product((3, 5, 7), (0.005, 0.01), (0.0, 2.0, 5.0, 10.0))
    )


@pytest.mark.parametrize("point", GRID, ids=lambda point: point.key())
def test_exact_data_only_circuit(submission, experiments, point):
    # Equality includes the circuit split, injected qubit order, and all readouts.
    assert submission._data_only_circuit(point.L, point.p) == _original_circuit(
        experiments[point.L], point.p
    )


@pytest.mark.parametrize("point", GRID, ids=lambda point: point.key())
@pytest.mark.parametrize("dtype", [np.uint8, np.bool_], ids=["uint8", "bool"])
def test_predictions_interface_determinism_and_immutability(
    submission, experiments, diagnostic_syndromes, point, dtype
):
    experiment = experiments[point.L]
    correlated, _ = experiment.sample_correlated(
        shots=256, p=point.p, xi=point.xi, rng=np.random.default_rng(31415)
    )
    syndromes = np.concatenate([diagnostic_syndromes[point.L], correlated]).astype(dtype)
    before = syndromes.copy()
    expected = _original_predictions(_original_circuit(experiment, point.p), syndromes)
    decoder = submission.build_decoder(point)

    predictions = decoder.decode(syndromes)
    _assert_interface(predictions, len(syndromes))
    np.testing.assert_array_equal(predictions, expected)
    np.testing.assert_array_equal(syndromes, before)

    # Read-only input also detects transient writes that might later be restored.
    syndromes.setflags(write=False)
    for repeated in (
        decoder.decode(syndromes),
        submission.build_decoder(point).decode(syndromes),
    ):
        _assert_interface(repeated, len(syndromes))
        np.testing.assert_array_equal(repeated, predictions)
    np.testing.assert_array_equal(syndromes, before)
    assert not syndromes.flags.writeable


@pytest.mark.parametrize("point", GRID, ids=lambda point: point.key())
def test_empty_shots(submission, experiments, point):
    syndromes = np.empty((0, experiments[point.L].num_detectors), dtype=np.uint8)
    syndromes.setflags(write=False)
    expected = _original_predictions(
        _original_circuit(experiments[point.L], point.p), syndromes
    )
    predictions = submission.build_decoder(point).decode(syndromes)
    _assert_interface(predictions, 0)
    np.testing.assert_array_equal(predictions, expected)
    assert not syndromes.flags.writeable


def test_submission_size_and_imports():
    source = SOLVE_PATH.read_bytes()
    assert len(source) < 200_000
    allowed = {"numpy", "pymatching", "stim", "typing"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module in allowed


# This is intentionally a fresh interpreter: audit hooks cannot be removed.
# Only decode calls are monitored, including the first call and empty batches.
# It detects Python-audited filesystem/network activity and the extra guarded
# os APIs below, including attempted activity whose exception is swallowed.
# It is not a syscall sandbox and cannot rule out uninstrumented native I/O.
_ISOLATED_CHECK = textwrap.dedent(
    """
    import importlib.abc
    import importlib.util
    import os
    import socket
    import sys
    from itertools import product
    from types import SimpleNamespace

    assert not any(name.split('.')[0] == 'qec_benchmark' for name in sys.modules)

    class NoBenchmark(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] == 'qec_benchmark':
                raise ImportError('qec_benchmark is deliberately unavailable')

    sys.meta_path.insert(0, NoBenchmark())
    try:
        import qec_benchmark
    except ImportError:
        pass
    else:
        raise AssertionError('benchmark import blocker did not work')

    spec = importlib.util.spec_from_file_location('standalone_submission', sys.argv[1])
    submission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission)
    import numpy as np

    audited = sys.argv[2] == 'audit'
    decoding = False
    attempts = []

    def deny(event):
        attempts.append(event)
        raise AssertionError('I/O during decode: ' + event)

    def audit(event, args):
        if decoding and (
            event == 'open'
            or event.startswith(('socket.', 'os.', 'subprocess.', 'mmap.'))
        ):
            deny(event)

    def guarded(name, original):
        def call(*args, **kwargs):
            if decoding:
                deny('os.' + name)
            return original(*args, **kwargs)
        return call

    if audited:
        sys.addaudithook(audit)
        # These file APIs do not all emit audit events on supported Python builds.
        for name in (
            'read', 'write', 'pread', 'pwrite', 'readv', 'writev', 'sendfile',
            'stat', 'lstat', 'fstat', 'access', 'readlink', 'getcwd', 'getcwdb',
        ):
            if hasattr(os, name):
                setattr(os, name, guarded(name, getattr(os, name)))

        # Positive controls establish that the instruments actually reject I/O.
        decoding = True
        try:
            for operation in (
                lambda: open(sys.argv[1], 'rb'),
                lambda: socket.socket(),
                lambda: os.stat(sys.argv[1]),
            ):
                try:
                    operation()
                except AssertionError as exc:
                    assert str(exc).startswith('I/O during decode: ')
                else:
                    raise AssertionError('I/O guard did not fire')
        finally:
            decoding = False
        assert attempts == ['open', 'socket.__new__', 'os.stat'], attempts
        attempts.clear()

    checked = 0
    for distance, p, xi in product((3, 5, 7), (0.005, 0.01), (0.0, 2.0, 5.0, 10.0)):
        point = SimpleNamespace(L=distance, p=p, xi=xi)
        circuit = submission._data_only_circuit(distance, p)
        samples = circuit.compile_detector_sampler(seed=137).sample(shots=64)
        decoder = submission.build_decoder(point)
        for batch in (samples.astype(np.uint8), samples, samples[:0]):
            before = batch.copy()
            batch.setflags(write=False)
            decoding = audited
            try:
                predictions = decoder.decode(batch)
            finally:
                decoding = False
            assert not attempts, attempts
            assert predictions.shape == (len(batch),)
            assert predictions.dtype == np.uint8
            assert np.all((predictions == 0) | (predictions == 1))
            np.testing.assert_array_equal(batch, before)
        checked += 1
    assert checked == 24
    assert not any(name.split('.')[0] == 'qec_benchmark' for name in sys.modules)
    print('24 isolated points passed; audited=' + str(audited))
    """
)


@pytest.mark.parametrize("mode", ["plain", "audit"], ids=["load_without_benchmark", "decode_no_io"])
def test_isolated_submission(tmp_path, mode):
    isolated_solve = tmp_path / "solve.py"
    shutil.copyfile(SOLVE_PATH, isolated_solve)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _ISOLATED_CHECK, str(isolated_solve), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "24 isolated points passed" in result.stdout
