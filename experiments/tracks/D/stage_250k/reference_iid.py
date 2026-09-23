"""Immutable data-only IID reference, copied explicitly from A's graph construction.

No dependency on solve.py. Injection order, fault IDs and PyMatching tie breaking
are identical to Track A DataOnlyFallback (2026-09-23 snapshot).
"""
import numpy as np
import pymatching
import stim


class IIDDecoder:
    def __init__(self, point):
        noiseless = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=point.L, rounds=1,
            after_clifford_depolarization=0., before_round_data_depolarization=0.,
            before_measure_flip_probability=0., after_reset_flip_probability=0.)
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
                for q in data:
                    circuit.append("X_ERROR", [q], float(point.p))
                injected = True
        self._matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True))

    def decode(self, syndrome_array):
        result = self._matching.decode_batch(syndrome_array.astype(np.uint8, copy=False))
        return result.reshape(-1).astype(np.uint8, copy=False)
