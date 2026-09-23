"""Experimental decoder candidates; the challenge submission remains solve.py."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from qec_benchmark.baselines import MWPMDecoder
from qec_benchmark.models import ParameterPoint
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


class CurrentMWPM:
    """The supplied weighted circuit-level MWPM baseline."""

    def __init__(self, point: ParameterPoint):
        self._decoder = MWPMDecoder(point=point, weighted=True)

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        return self._decoder.decode(syndrome_array)


class UniformMWPM:
    """MWPM with uniform edge weights, testing whether baseline calibration hurts."""

    def __init__(self, point: ParameterPoint):
        experiment = SurfaceCodeExperiment(distance=point.L)
        self._matching = experiment.build_matching(p=point.p, weighted=False)

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        pred = self._matching.decode_batch(syndrome_array.astype(np.uint8, copy=False))
        return pred[:, 0].astype(np.uint8, copy=False) if pred.ndim == 2 else pred.astype(np.uint8, copy=False)


class DataOnlyIIDMWPM:
    """MWPM graph generated from the benchmark's data-only iid X noise model."""

    def __init__(self, point: ParameterPoint):
        experiment = SurfaceCodeExperiment(distance=point.L)
        circuit = stim.Circuit()
        circuit += experiment._prefix
        for q in experiment._data_qubits:
            circuit.append("X_ERROR", [int(q)], float(point.p))
        circuit += experiment._suffix
        dem = circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        pred = self._matching.decode_batch(syndrome_array.astype(np.uint8, copy=False))
        return pred[:, 0].astype(np.uint8, copy=False) if pred.ndim == 2 else pred.astype(np.uint8, copy=False)


DECODERS = {
    "baseline": CurrentMWPM,
    "uniform": UniformMWPM,
    "data_iid": DataOnlyIIDMWPM,
}
