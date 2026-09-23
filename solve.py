"""Challenge decoder.

The benchmark samples data-qubit X errors directly.  The supplied MWPM baseline
builds its graph from a circuit-level depolarizing model, so this implementation
uses the matching graph for the same data-only iid model.  It deliberately does
not attempt to infer ``xi``: the graph remains a fast, robust reference decoder
for the independent component of the challenge distribution.
"""

from typing import Protocol

import numpy as np
import pymatching
import stim


class _ParameterPoint(Protocol):
    L: int
    p: float
    xi: float


def _data_only_circuit(distance: int, p: float) -> stim.Circuit:
    """Reproduce the benchmark's default circuit and data-error injection order."""
    if distance < 3:
        raise ValueError("distance must be >=3")
    noiseless = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=1,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )

    split = None
    for i, inst in enumerate(noiseless):
        if inst.name == "TICK":
            split = i + 1
            break
    if split is None:
        raise ValueError("circuit had no TICK; cannot find safe injection point")

    prefix = stim.Circuit()
    suffix = stim.Circuit()
    for i, inst in enumerate(noiseless):
        if i < split:
            prefix.append(inst)
        else:
            suffix.append(inst)

    candidates: list[list[int]] = []
    for inst in noiseless:
        if inst.name in {"MX", "MY", "MZ", "M"}:
            targets = [
                int(t.value)
                for t in inst.targets_copy()
                if not (
                    t.is_combiner
                    or t.is_measurement_record_target
                    or t.is_sweep_bit_target
                )
            ]
            if targets:
                candidates.append(targets)
    if not candidates:
        raise ValueError("could not infer data qubits from measurement instructions")
    data_qubits = sorted(set(max(candidates, key=len)))

    circuit = stim.Circuit()
    circuit += prefix
    for qubit in data_qubits:
        circuit.append("X_ERROR", [qubit], float(p))
    circuit += suffix
    return circuit


class DataOnlyMWPM:
    """MWPM calibrated to the benchmark's data-only X-error process."""

    def __init__(self, point: _ParameterPoint) -> None:
        circuit = _data_only_circuit(point.L, point.p)
        dem = circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2:
            raise ValueError("syndrome_array must be rank 2")
        predictions = self._matching.decode_batch(
            syndrome_array.astype(np.uint8, copy=False)
        )
        if predictions.ndim == 2:
            predictions = predictions[:, 0]
        return predictions.astype(np.uint8, copy=False)


def build_decoder(point: _ParameterPoint):
    """Return a decoder for the given parameter point.

    Your decoder must implement: decode(syndrome_array: np.ndarray) -> np.ndarray
    where syndrome_array is (shots, num_detectors) uint8 and return is (shots,) uint8.
    """
    return DataOnlyMWPM(point)
