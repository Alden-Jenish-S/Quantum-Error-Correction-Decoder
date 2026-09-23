"""Track B: data-only matching and explicitly approximate burst DEMs.

All point-dependent probabilities are computed in construction. Decode performs
no file/network I/O, training, truth access, or physical-mask enumeration.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Protocol

import numpy as np
import pymatching
import stim
from scipy.integrate import quad
from scipy.special import ndtri

from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


class Point(Protocol):
    L: int
    p: float
    xi: float


VARIANTS = (
    "iid", "noop", "parity", "parity_reg", "pair_uncorr", "pair_corr",
    "pair_corr90", "cluster_corr", "pair_shortcut", "pair_shortcut_corr",
)


def joint_probability(p: float, distance: float, xi: float) -> float:
    """Exact bivariate Gaussian-copula tail, using deterministic quadrature.

    Plackett's identity: J=p^2 + integral_0^rho phi_2(t,t;r) dr.
    Benchmark adds 1e-12 to the latent diagonal; its effect is below 1e-12
    here. We use the requested marginal p and rho=exp(-distance/xi).
    """
    if xi <= 0:
        return p * p
    if distance == 0:
        return p
    rho = math.exp(-distance / xi)
    threshold = float(ndtri(1 - p))
    integral, _ = quad(
        lambda r: math.exp(-threshold * threshold / (1 + r))
        / (2 * math.pi * math.sqrt(1 - r * r)),
        0, rho, epsabs=1e-14, epsrel=1e-11,
    )
    return p * p + integral


def data_dem(experiment: SurfaceCodeExperiment, p: float) -> stim.DetectorErrorModel:
    circuit = experiment._prefix.copy()
    for qubit in experiment._data_qubits:
        circuit.append("X_ERROR", [int(qubit)], float(p))
    circuit += experiment._suffix
    return circuit.detector_error_model(decompose_errors=True)


def physical_columns(experiment: SurfaceCodeExperiment):
    """Deterministic one-error circuit probes, not benchmark truth or training."""
    detectors, observable = experiment.sample_from_mask(
        np.eye(experiment.num_data_qubits, dtype=np.bool_)
    )
    columns = [(tuple(map(int, np.flatnonzero(s))), int(o))
               for s, o in zip(detectors, observable)]
    if any(len(d) not in (1, 2) for d, _ in columns):
        raise ValueError("data-only X columns must be graphlike")
    # Parallel columns must agree on logical labels; merging otherwise loses
    # logical information. This invariant holds for these L>=3 geometries.
    labels = {}
    for d, o in columns:
        if d in labels and labels[d] != o:
            raise ValueError("parallel columns with different observables")
        labels[d] = o
    return columns


def _append_event(dem, probability, columns, indices, *, collapse_graphlike=False):
    # A pair source is one correlated mechanism. The ^ separators are essential:
    # they preserve the source correlation for PyMatching's two-pass algorithm.
    # Identical columns cancel exactly (including their observable labels), so
    # that source has no syndrome or logical effect and is omitted.
    counts = Counter(columns[i] for i in indices)
    selected = [col for col, count in counts.items() if count % 2]
    if not selected or probability <= 0:
        return False
    if collapse_graphlike:
        detectors = Counter(d for ds, _ in selected for d in ds)
        detectors = tuple(sorted(d for d, count in detectors.items() if count % 2))
        observable = sum(o for _, o in selected) % 2
        if len(detectors) <= 2:
            if not detectors and not observable:
                return False
            selected = [(detectors, observable)]
    targets = []
    for component_index, (detectors, observable) in enumerate(selected):
        if component_index:
            targets.append(stim.target_separator())
        targets.extend(stim.target_relative_detector_id(d) for d in detectors)
        if observable:
            targets.append(stim.target_logical_observable_id(0))
    dem.append("error", probability, targets)
    return True


class GraphDecoder:
    def __init__(self, point: Point, variant: str = "iid"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown track-B variant {variant}")
        if not 0 < point.p < .5:
            raise ValueError("track-B candidates require 0 < p < 0.5")
        self.point = point
        self.variant = variant
        self.correlations = False
        self.model_details = {}
        experiment = SurfaceCodeExperiment(distance=point.L)
        self._dem = data_dem(experiment, point.p)
        self._matching = pymatching.Matching.from_detector_error_model(self._dem)
        self._num_detectors = experiment.num_detectors

        # Structural xi=0 protection, including baseline tie-breaking.
        if point.xi <= 0 or variant in {"iid", "noop"}:
            return
        columns = physical_columns(experiment)
        positions = experiment.data_positions
        if variant in {"parity", "parity_reg"}:
            groups = {}
            for i, (d, o) in enumerate(columns):
                groups.setdefault((d, o), []).append(i)
            alpha = 1.0 if variant == "parity" else .5
            changes = []
            for (detectors, observable), indices in groups.items():
                if len(indices) == 1:
                    continue
                if len(indices) != 2:
                    raise ValueError("exact pair parity requires group size <=2")
                distance = float(np.linalg.norm(positions[indices[0]] - positions[indices[1]]))
                joint = joint_probability(point.p, distance, point.xi)
                prob_iid = 2 * point.p * (1 - point.p)
                prob_parity = 2 * (point.p - joint)
                old_weight = math.log((1 - prob_iid) / prob_iid)
                new_weight = math.log((1 - prob_parity) / prob_parity)
                weight = (1 - alpha) * old_weight + alpha * new_weight
                args = dict(fault_ids={0} if observable else set(), weight=weight,
                            error_probability=1 / (1 + math.exp(weight)), merge_strategy="replace")
                if len(detectors) == 1:
                    self._matching.add_boundary_edge(detectors[0], **args)
                else:
                    self._matching.add_edge(*detectors, **args)
                changes.append(dict(detectors=detectors, distance=distance,
                                    joint=joint, probability=prob_parity, weight=weight))
            self.model_details = dict(alpha=alpha, changed_edges=changes)
        else:
            self._build_burst(experiment, columns)

    def _build_burst(self, experiment, columns):
        p, xi = self.point.p, self.point.xi
        n = experiment.num_data_qubits
        positions = experiment.data_positions
        diff = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        budget = -math.log1p(-2 * p)
        events = []
        rates = []
        if self.variant == "cluster_corr":
            # A four-site connected burst surrogate. The tree probability
            # p*(J_nn/p)^3 is a Markov-tree approximation, NOT a Gaussian
            # four-variate probability. Subtract its iid value to vanish at xi=0.
            joint = joint_probability(p, 2.0, xi)
            raw_q = max(0.0, p * (joint / p) ** 3 - p ** 4)
            for y in range(self.point.L - 1):
                for x in range(self.point.L - 1):
                    i = y * self.point.L + x
                    events.append((i, i + 1, i + self.point.L, i + self.point.L + 1))
                    rates.append(-math.log1p(-2 * raw_q))
        else:
            # Pair latent-XOR model matches the two-spin moment in isolation.
            # Shared-node rate budgeting shrinks pair dependence to keep every
            # residual singleton valid and the physical marginal exactly p.
            for i in range(n):
                for j in range(i + 1, n):
                    if distance[i, j] <= 2.000001:
                        joint = joint_probability(p, float(distance[i, j]), xi)
                        # r is the log-rate of an independent XOR pair source.
                        # Each endpoint has factor exp(-r), while the shared
                        # source cancels in the endpoint XOR: C_ij =
                        # (1-2p)^2 exp(2r).
                        rate = .5 * math.log1p(4 * (joint - p * p) / (1 - 2 * p) ** 2)
                        events.append((i, j))
                        rates.append(rate)
        incident = np.zeros(n)
        for indices, rate in zip(events, rates):
            incident[list(indices)] += rate
        fraction = .9 if self.variant == "pair_corr90" else .5
        scale = min(1.0, fraction * budget / max(float(incident.max()), 1e-300))
        singleton_rates = budget - scale * incident
        singleton_prob = -np.expm1(-singleton_rates) / 2
        dem = stim.DetectorErrorModel()
        for i, prob in enumerate(singleton_prob):
            _append_event(dem, float(prob), columns, (i,))
        visible_bursts = 0
        for indices, rate in zip(events, rates):
            visible_bursts += _append_event(
                dem, -math.expm1(-scale * rate) / 2, columns, indices,
                collapse_graphlike=self.variant.startswith("pair_shortcut"),
            )
        # Preserve trailing isolated circuit detectors and observable width.
        dem.append("detector", [], [stim.target_relative_detector_id(self._num_detectors - 1)])
        dem.append("logical_observable", [], [stim.target_logical_observable_id(0)])
        self._dem = dem
        self.correlations = self.variant not in {"pair_uncorr", "pair_shortcut"}
        self._matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=self.correlations
        )
        reconstructed_p = -np.expm1(-(singleton_rates + scale * incident)) / 2
        self.model_details = dict(
            scale=scale, rate_budget=budget, burst_budget_fraction=fraction,
            events=len(events), visible_bursts=visible_bursts,
            singleton_probability_min=float(singleton_prob.min()),
            singleton_probability_max=float(singleton_prob.max()),
            physical_marginal_max_abs_error=float(np.max(abs(reconstructed_p - p))),
            event_probability_min=min((-math.expm1(-scale * r) / 2 for r in rates), default=0),
            event_probability_max=max((-math.expm1(-scale * r) / 2 for r in rates), default=0),
        )

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2 or syndrome_array.shape[1] != self._num_detectors:
            raise ValueError(f"expected rank-2 syndrome with width {self._num_detectors}")
        predictions = self._matching.decode_batch(
            syndrome_array.astype(np.uint8, copy=False),
            enable_correlations=self.correlations,
        )
        return predictions[:, 0].astype(np.uint8, copy=False)

    def diagnostics(self, include_edges=False):
        edges = self._matching.edges()
        weights = [float(a["weight"]) for _, _, a in edges]
        out = dict(
            variant=self.variant, L=self.point.L, p=self.point.p, xi=self.point.xi,
            pymatching_version=pymatching.__version__,
            detectors=self._matching.num_detectors, fault_ids=self._matching.num_fault_ids,
            edges=len(edges), boundary_edges=sum(v is None for _, v, _ in edges),
            weight_min=min(weights), weight_max=max(weights),
            weight_mean=float(np.mean(weights)), correlated_decode=self.correlations,
            model_details=self.model_details,
        )
        if include_edges:
            out["edge_list"] = [dict(u=u, v=v, fault_ids=sorted(a["fault_ids"]),
                                     weight=a["weight"], error_probability=a["error_probability"])
                                for u, v, a in edges]
        return out


def build_decoder(point: Point, variant: str = "iid") -> GraphDecoder:
    """Parent-harness API; default is conservative until validation approves one."""
    return GraphDecoder(point, variant)


DECODERS = {name: (lambda point, name=name: GraphDecoder(point, name)) for name in VARIANTS}
