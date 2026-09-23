"""Frozen L=3 QMC-MAP lookup plus data-only MWPM for the challenge.

The lookup is an offline, approximate QMC model-integration result, not an exact
MAP certificate. It is used only at L=3, p in {0.005, 0.01}, xi in {0, 2, 5, 10}.
Every other parameter point conservatively uses DataOnlyMWPM, which ignores xi;
there is no interpolation, rounding, or extrapolation of table parameters.
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


# Reviewed source: experiments/tracks/A/frozen_tables.py (model-only integration).
# SHA256: debb61816d64cb556cc58c17a3a1a9cf5d0a8c90c08fe3d480d9d931cf29c119
# Each hex value packs 256 labels, little-endian within each byte. Only IDs 0..15
# are physically reachable: detectors 4..7 are inactive for data-only X errors.
_L3_TABLE_HEX = {
    (0.005, 0.0): "044b000000000000000000000000000000000000000000000000000000000000",
    (0.005, 2.0): "044b000000000000000000000000000000000000000000000000000000000000",
    (0.005, 5.0): "0449000000000000000000000000000000000000000000000000000000000000",
    (0.005, 10.0): "0449000000000000000000000000000000000000000000000000000000000000",
    (0.01, 0.0): "044b000000000000000000000000000000000000000000000000000000000000",
    (0.01, 2.0): "044b000000000000000000000000000000000000000000000000000000000000",
    (0.01, 5.0): "0449000000000000000000000000000000000000000000000000000000000000",
    (0.01, 10.0): "0449000000000000000000000000000000000000000000000000000000000000",
}


class _FrozenL3QMC:
    """Lookup for binary (uint8/bool) eight-detector input in the data-X domain.

    Nonzero inactive detectors are outside the integrated model and rejected;
    the zero placeholders for unreachable IDs are never used as predictions.
    """

    def __init__(self, table_hex: str) -> None:
        self._table = np.unpackbits(
            np.frombuffer(bytes.fromhex(table_hex), dtype=np.uint8), bitorder="little"
        )
        self._table.setflags(write=False)

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2 or syndrome_array.shape[1] != 8:
            raise ValueError("L=3 syndrome_array must have shape (shots, 8)")
        ids = np.packbits(syndrome_array, axis=1, bitorder="little")[:, 0]
        if np.any(ids >= 16):
            raise ValueError("nonzero inactive detectors are outside the data-X domain")
        return self._table[ids]


def build_decoder(point: _ParameterPoint):
    """Return a decoder for the given parameter point.

    Your decoder must implement: decode(syndrome_array: np.ndarray) -> np.ndarray
    where syndrome_array is (shots, num_detectors) uint8 and return is (shots,) uint8.
    """
    if point.L == 3:
        table_hex = _L3_TABLE_HEX.get((point.p, point.xi))
        if table_hex is not None:
            return _FrozenL3QMC(table_hex)
    return DataOnlyMWPM(point)
