"""Offline Gaussian-copula Bayes tables for L=3; data-only MWPM for L=5/7.

Import as experiments.tracks.A.candidate_a and call build_decoder(point).
Frozen tables are produced by run_track_a.py build, independently of test shots.
"""
from __future__ import annotations

import numpy as np
import pymatching
import stim

from experiments.tracks.A.frozen_tables import TABLE_HEX


def point_key(p: float, xi: float) -> str:
    return f"p{p:g}_xi{xi:g}"


class TableDecoder:
    def __init__(self, table: np.ndarray):
        table = np.asarray(table, dtype=np.uint8)
        if table.shape != (256,) or np.any(table > 1):
            raise ValueError("table must contain 256 binary logical predictions")
        self.table = table.copy()

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2 or syndrome_array.shape[1] != 8:
            raise ValueError("L=3 requires shape (shots, 8)")
        # Detector k is bit k, including the four physically inactive detectors.
        ids = np.packbits(syndrome_array, axis=1, bitorder="little")[:, 0]
        return self.table[ids]


class DataOnlyFallback:
    """Same data-only graph and injection order as current solve.DataOnlyMWPM."""
    def __init__(self, point):
        noiseless = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=point.L, rounds=1,
            after_clifford_depolarization=0.0,
            before_round_data_depolarization=0.0,
            before_measure_flip_probability=0.0, after_reset_flip_probability=0.0,
        )
        candidates = []
        for inst in noiseless:
            if inst.name in {"MX", "MY", "MZ", "M"}:
                candidates.append([int(t.value) for t in inst.targets_copy()
                                   if not (t.is_combiner or t.is_measurement_record_target
                                           or t.is_sweep_bit_target)])
        data = sorted(set(max(candidates, key=len)))
        circuit = stim.Circuit()
        injected = False
        for inst in noiseless:
            circuit.append(inst)
            if not injected and inst.name == "TICK":
                for qubit in data:
                    circuit.append("X_ERROR", [qubit], float(point.p))
                injected = True
        self._matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True))

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2:
            raise ValueError("syndrome_array must be rank 2")
        result = self._matching.decode_batch(syndrome_array.astype(np.uint8, copy=False))
        return result.reshape(-1).astype(np.uint8, copy=False)


def build_decoder(point):
    if point.L == 3:
        raw = bytes.fromhex(TABLE_HEX[point_key(point.p, point.xi)])
        return TableDecoder(np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little"))
    if point.L in (5, 7):
        return DataOnlyFallback(point)
    raise ValueError("track A supports official L=3,5,7 only")
