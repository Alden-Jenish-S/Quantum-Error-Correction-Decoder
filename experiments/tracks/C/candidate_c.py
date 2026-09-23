"""Track C: vectorized syndrome morphology and conservative residual filters.

The decoder first makes the exact data-only MWPM prediction used by the current
submission.  It then extracts fixed-geometry syndrome features in one NumPy
batch and may flip that prediction using either a conservative analytic rule or
an offline-frozen linear score.  The score constants are deliberately embedded
in this module: decoding does not read files, access truth, fit parameters, or
iterate over individual shots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pymatching
import stim

from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


class Point(Protocol):
    L: int
    p: float
    xi: float


# This is intentionally a small model.  Values are replaced only by the
# offline training command in run_track_c.py; they are not estimated in decode.
MODEL_FEATURES = (
    "predicted_logical",
    "syndrome_weight",
    "adjacency_pairs",
    "component_count",
    "largest_component",
    "boundary_fraction",
    "row_max",
    "col_max",
    "temporal_disagreement",
    "cluster_class",
    "xi",
    "p",
    "L",
)
MODEL_MEAN = np.array(
    [0.0272875, 0.2523208333333333, 0.10099583333333334,
     0.1529375, 0.2218875, 0.0688945482862182,
     0.14444583333333333, 0.1385125, 0.2523208333333333,
     0.2998375, 4.25, 0.007499999832361937, 5.0],
    dtype=np.float64,
)
MODEL_SCALE = np.array(
    [0.16291989548134547, 0.7694565379117994, 0.43262263192001077,
     0.4401109190799717, 0.6561466584107664, 0.23116626554336378,
     0.4018783828043764, 0.37890428080198785, 0.7694565379117994,
     0.8694116632880552, 3.766629793329841, 0.002499999944127017,
     1.632993161855452],
    dtype=np.float64,
)
MODEL_COEFFICIENTS = np.array(
    [0.0030021019278999796, 0.026629150068479383, -0.016642939545778227,
     -0.010746803537415238, 0.0032086440015477173, 0.0036610939033493014,
     -0.00553247328261814, -0.0025252106759562913, 0.026629150068466827,
     -0.017944807694988605, 0.0024943448533480232, 2.857937640890077e-05,
     -0.0030205018473594466],
    dtype=np.float64,
)
MODEL_INTERCEPT = 0.0030708333333333334
MODEL_THRESHOLD = 0.27603908220245355
MODEL_TRAINING = "ridge score fit on seeds 8011 and 8017, 5000 shots per point"
DIAGNOSTIC_FEATURES = ("syndrome_weight", "cluster_class")


@dataclass(slots=True)
class DetectorGeometry:
    """All fixed indices and masks needed by the batched feature extractor."""

    L: int
    num_detectors: int
    layer0: np.ndarray
    layer1: np.ndarray
    spatial_adjacency: np.ndarray
    spatial_adjacency_with_self: np.ndarray
    component_edge_u: np.ndarray
    component_edge_v: np.ndarray
    propagation_rounds: int
    full_adjacency: np.ndarray
    row_indicator: np.ndarray
    col_indicator: np.ndarray
    edge_indicator: np.ndarray
    boundary_mask: np.ndarray
    boundary_distance: np.ndarray
    spatial_x: np.ndarray
    spatial_y: np.ndarray
    row_ids: np.ndarray
    col_ids: np.ndarray


@dataclass(slots=True)
class FeatureBatch:
    """Column-oriented features; every value has one entry per shot."""

    values: dict[str, np.ndarray]

    def matrix(self, names: tuple[str, ...] = MODEL_FEATURES) -> np.ndarray:
        return np.column_stack([self.values[name] for name in names])


def _data_only_dem(experiment: SurfaceCodeExperiment, p: float) -> stim.DetectorErrorModel:
    circuit = experiment._prefix.copy()
    for qubit in experiment._data_qubits:
        circuit.append("X_ERROR", [int(qubit)], float(p))
    circuit += experiment._suffix
    return circuit.detector_error_model(decompose_errors=True)


def _detector_graph(experiment: SurfaceCodeExperiment, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Return graphlike detector edges and detector-to-boundary flags.

    The graph is extracted from the deterministic data-only circuit once during
    construction.  The supplied one-round circuit has one active spatial layer;
    the same geometric graph is replicated for the second detector layer.
    """
    n = experiment.num_detectors
    adjacency = np.zeros((n, n), dtype=np.bool_)
    boundary = np.zeros(n, dtype=np.bool_)
    dem = _data_only_dem(experiment, p)
    for instruction in dem:
        if instruction.type != "error":
            continue
        detectors = [
            int(target.val)
            for target in instruction.targets_copy()
            if target.is_relative_detector_id()
        ]
        if len(detectors) == 1:
            boundary[detectors[0]] = True
        elif len(detectors) == 2:
            first, second = detectors
            adjacency[first, second] = True
            adjacency[second, first] = True
    return adjacency, boundary


def build_geometry(L: int, p: float = 0.01) -> DetectorGeometry:
    """Build authoritative Stim detector order and vectorization matrices."""
    experiment = SurfaceCodeExperiment(distance=L)
    n = experiment.num_detectors
    coordinates = experiment.circuit.get_detector_coordinates()
    xy = np.asarray([coordinates[i][:2] for i in range(n)], dtype=np.float64)
    layer_values = np.asarray([coordinates[i][2] for i in range(n)], dtype=np.float64)
    unique_layers = np.unique(layer_values)
    if unique_layers.size != 2:
        raise ValueError(f"expected two detector layers, found {unique_layers}")
    layer0 = np.flatnonzero(layer_values == unique_layers[0]).astype(np.intp)
    layer1 = np.flatnonzero(layer_values == unique_layers[1]).astype(np.intp)
    if layer0.size != layer1.size:
        raise ValueError("detector layers have different sizes")

    raw_adjacency, raw_boundary = _detector_graph(experiment, p)
    m = layer0.size
    spatial_adjacency = np.zeros((m, m), dtype=np.bool_)
    boundary_mask = np.zeros(m, dtype=np.bool_)
    layer0_position = {int(detector): i for i, detector in enumerate(layer0)}
    for i, detector_i in enumerate(layer0):
        boundary_mask[i] = raw_boundary[detector_i]
        for j, detector_j in enumerate(layer0):
            spatial_adjacency[i, j] = raw_adjacency[detector_i, detector_j]
    component_edge_u, component_edge_v = np.where(np.triu(spatial_adjacency, 1))
    distances = np.where(spatial_adjacency, 1, np.inf).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    for middle in range(m):
        distances = np.minimum(distances, distances[:, middle, None] + distances[None, middle, :])
    finite_distances = distances[np.isfinite(distances)]
    propagation_rounds = max(1, int(finite_distances.max(initial=0.0)))
    # The detector order is not assumed to be a rectangular integer grid.  The
    # second layer is matched by its spatial coordinate, not by arithmetic.
    if any(tuple(xy[a]) != tuple(xy[b]) for a, b in zip(layer0, layer1)):
        layer1_by_xy = {tuple(xy[int(detector)]): int(detector) for detector in layer1}
        layer1_order = np.asarray(
            [layer1_by_xy[tuple(xy[int(detector)])] for detector in layer0],
            dtype=np.intp,
        )
    else:
        layer1_order = layer1

    full_adjacency = np.zeros((n, n), dtype=np.bool_)
    full_adjacency[np.ix_(layer0, layer0)] = spatial_adjacency
    full_adjacency[np.ix_(layer1_order, layer1_order)] = spatial_adjacency
    full_adjacency |= np.eye(n, dtype=np.bool_)
    full_boundary = np.zeros(n, dtype=np.bool_)
    full_boundary[layer0] = boundary_mask
    full_boundary[layer1_order] = boundary_mask

    spatial_xy = xy[layer0]
    xs = np.unique(spatial_xy[:, 0])
    ys = np.unique(spatial_xy[:, 1])
    col_ids = np.searchsorted(xs, spatial_xy[:, 0]).astype(np.intp)
    row_ids = np.searchsorted(ys, spatial_xy[:, 1]).astype(np.intp)
    row_indicator = np.eye(ys.size, dtype=np.float32)[row_ids]
    col_indicator = np.eye(xs.size, dtype=np.float32)[col_ids]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    scale = max(x_max - x_min, y_max - y_min, 1.0)
    distance = np.minimum.reduce(
        [
            spatial_xy[:, 0] - x_min,
            x_max - spatial_xy[:, 0],
            spatial_xy[:, 1] - y_min,
            y_max - spatial_xy[:, 1],
        ]
    ) / scale
    edge_indicator = np.column_stack(
        [
            np.isclose(spatial_xy[:, 0], x_min),
            np.isclose(spatial_xy[:, 0], x_max),
            np.isclose(spatial_xy[:, 1], y_min),
            np.isclose(spatial_xy[:, 1], y_max),
        ]
    ).astype(np.float32)

    return DetectorGeometry(
        L=L,
        num_detectors=n,
        layer0=layer0,
        layer1=layer1_order,
        spatial_adjacency=spatial_adjacency,
        spatial_adjacency_with_self=spatial_adjacency | np.eye(m, dtype=np.bool_),
        component_edge_u=component_edge_u.astype(np.intp, copy=False),
        component_edge_v=component_edge_v.astype(np.intp, copy=False),
        propagation_rounds=propagation_rounds,
        full_adjacency=full_adjacency,
        row_indicator=row_indicator,
        col_indicator=col_indicator,
        edge_indicator=edge_indicator,
        boundary_mask=full_boundary,
        boundary_distance=np.tile(distance, 2),
        spatial_x=spatial_xy[:, 0],
        spatial_y=spatial_xy[:, 1],
        row_ids=row_ids,
        col_ids=col_ids,
    )


def _component_features(active: np.ndarray, geometry: DetectorGeometry) -> dict[str, np.ndarray]:
    """Vectorized connected components on all shots at once.

    Label propagation has a fixed maximum of ``m`` rounds, independent of the
    number of shots.  This is a Python loop over the precomputed geometry only;
    there is no per-shot Python control flow.
    """
    m = active.shape[1]
    weight = active.sum(axis=1, dtype=np.int16)
    component_count = np.zeros(len(active), dtype=np.float32)
    largest = np.zeros(len(active), dtype=np.float32)
    second = np.zeros(len(active), dtype=np.float32)
    one = weight == 1
    two = weight == 2
    component_count[one] = 1.0
    largest[one] = 1.0
    pair_count = np.einsum(
        "bi,ij,bj->b", active.astype(np.float32), geometry.spatial_adjacency, active.astype(np.float32)
    ) / 2.0
    connected_pair = two & (pair_count == 1)
    disconnected_pair = two & ~connected_pair
    component_count[connected_pair] = 1.0
    largest[connected_pair] = 2.0
    component_count[disconnected_pair] = 2.0
    largest[disconnected_pair] = 1.0
    second[disconnected_pair] = 1.0

    # Most benchmark shots are empty, singleton, or pair syndromes.  Restrict
    # the exact label propagation to the remaining shots so the million-shot
    # path stays O(shots * nodes) in memory and avoids work on trivial rows.
    complex_rows = weight >= 3
    if np.any(complex_rows):
        complex_active = active[complex_rows]
        labels = np.where(complex_active, np.arange(m, dtype=np.int16)[None, :], m)
        for _ in range(geometry.propagation_rounds):
            first = geometry.component_edge_u
            second_node = geometry.component_edge_v
            propagated = labels.copy()
            for first_node, second_node in zip(first, second_node):
                both_active = complex_active[:, first_node] & complex_active[:, second_node]
                pair_minimum = np.where(
                    both_active,
                    np.minimum(propagated[:, first_node], propagated[:, second_node]),
                    m,
                )
                propagated[:, first_node] = np.minimum(propagated[:, first_node], pair_minimum)
                propagated[:, second_node] = np.minimum(propagated[:, second_node], pair_minimum)
            labels = np.where(complex_active, propagated, m).astype(np.int16, copy=False)
        root_counts = np.zeros((len(complex_active), m), dtype=np.int16)
        for root in range(m):
            root_counts[:, root] = np.sum(labels == root, axis=1, dtype=np.int16)
        occupied_roots = root_counts > 0
        complex_component_count = occupied_roots.sum(axis=1, dtype=np.float32)
        sorted_counts = np.sort(root_counts, axis=1)
        complex_largest = sorted_counts[:, -1].astype(np.float32, copy=False)
        complex_second = sorted_counts[:, -2].astype(np.float32, copy=False) if m > 1 else np.zeros(len(complex_active), dtype=np.float32)
        component_count[complex_rows] = complex_component_count
        largest[complex_rows] = complex_largest
        second[complex_rows] = complex_second
    return {
        "component_count": component_count,
        "largest_component": largest,
        "second_component": second,
    }


def extract_features(
    syndrome_array: np.ndarray,
    geometry: DetectorGeometry,
    base_prediction: np.ndarray,
    point: Point,
    *,
    fast: bool = False,
) -> FeatureBatch:
    """Extract morphology and residual features from a complete syndrome batch.

    ``fast=True`` retains the complete connected-component calculation but
    omits diagnostic-only spans, edge balances, and minimum-distance columns.
    The default returns the full feature set requested by the experiment.
    """
    syndrome = np.asarray(syndrome_array, dtype=np.uint8)
    if syndrome.ndim != 2 or syndrome.shape[1] != geometry.num_detectors:
        raise ValueError(
            f"expected syndrome shape (shots, {geometry.num_detectors}), got {syndrome.shape}"
        )
    base = np.asarray(base_prediction, dtype=np.uint8).reshape(-1)
    if base.shape[0] != syndrome.shape[0] or not np.all((base == 0) | (base == 1)):
        raise ValueError("base prediction shape or values are invalid")

    active = syndrome.astype(np.bool_, copy=False)
    layer0 = active[:, geometry.layer0]
    layer1 = active[:, geometry.layer1]
    spatial = layer0 | layer1
    spatial_count = spatial.sum(axis=1, dtype=np.float32)
    total_weight = active.sum(axis=1, dtype=np.float32)
    layer0_weight = layer0.sum(axis=1, dtype=np.float32)
    layer1_weight = layer1.sum(axis=1, dtype=np.float32)
    temporal_disagreement = np.logical_xor(layer0, layer1).sum(axis=1, dtype=np.float32)

    adjacent = np.einsum(
        "bi,ij,bj->b", spatial.astype(np.float32), geometry.spatial_adjacency, spatial.astype(np.float32)
    ) / 2.0
    components = _component_features(spatial, geometry)
    component_count = components["component_count"]
    largest = components["largest_component"]
    second = components["second_component"]
    active_nonzero = np.maximum(spatial_count, 1.0)

    boundary_count = np.sum(spatial * geometry.boundary_mask[: geometry.layer0.size], axis=1, dtype=np.float32)
    boundary_fraction = boundary_count / active_nonzero

    row_counts = spatial.astype(np.float32) @ geometry.row_indicator
    col_counts = spatial.astype(np.float32) @ geometry.col_indicator
    row_max = row_counts.max(axis=1)
    col_max = col_counts.max(axis=1)
    cluster_class = np.zeros(len(spatial), dtype=np.float32)
    cluster_class[(spatial_count == 1) & (component_count == 1)] = 1.0
    cluster_class[(spatial_count > 1) & (largest == 1)] = 2.0
    cluster_class[(largest > 1) & (component_count == 1)] = 3.0
    cluster_class[(largest > 1) & (component_count > 1)] = 4.0
    sign = 1.0 - 2.0 * base.astype(np.float32)
    if fast:
        return FeatureBatch({
            "syndrome_weight": total_weight,
            "adjacency_pairs": adjacent.astype(np.float32),
            "component_count": component_count,
            "largest_component": largest,
            "boundary_fraction": boundary_fraction,
            "row_max": row_max.astype(np.float32),
            "col_max": col_max.astype(np.float32),
            "temporal_disagreement": temporal_disagreement,
            "cluster_class": cluster_class,
            "predicted_logical": base.astype(np.float32),
            "residual_signed_weight": sign * total_weight,
            "residual_signed_adjacency": sign * adjacent,
            "residual_signed_boundary": sign * boundary_fraction,
            "xi": np.full(len(spatial), float(point.xi), dtype=np.float32),
            "p": np.full(len(spatial), float(point.p), dtype=np.float32),
            "L": np.full(len(spatial), float(point.L), dtype=np.float32),
        })

    boundary_distance = geometry.boundary_distance[: geometry.layer0.size]
    min_boundary_distance = np.min(
        np.where(spatial, boundary_distance[None, :], np.inf), axis=1
    )
    min_boundary_distance = np.where(np.isfinite(min_boundary_distance), min_boundary_distance, 0.0).astype(np.float32)
    near_boundary_count = np.sum(
        spatial & (boundary_distance[None, :] <= 0.25), axis=1, dtype=np.float32
    )

    row_present = row_counts > 0
    col_present = col_counts > 0
    row_max = row_counts.max(axis=1)
    col_max = col_counts.max(axis=1)
    row_occupied = row_present.sum(axis=1, dtype=np.float32)
    col_occupied = col_present.sum(axis=1, dtype=np.float32)
    row_index = np.arange(row_counts.shape[1], dtype=np.float32)
    col_index = np.arange(col_counts.shape[1], dtype=np.float32)
    row_min = np.min(np.where(row_present, row_index[None, :], np.inf), axis=1)
    row_max_index = np.max(np.where(row_present, row_index[None, :], -np.inf), axis=1)
    col_min = np.min(np.where(col_present, col_index[None, :], np.inf), axis=1)
    col_max_index = np.max(np.where(col_present, col_index[None, :], -np.inf), axis=1)
    row_span = np.where(spatial_count > 0, row_max_index - row_min, 0.0).astype(np.float32)
    col_span = np.where(spatial_count > 0, col_max_index - col_min, 0.0).astype(np.float32)

    edge_counts = spatial.astype(np.float32) @ geometry.edge_indicator
    edge_count = edge_counts.sum(axis=1)
    edge_balance = np.abs(edge_counts[:, 0] - edge_counts[:, 1]) + np.abs(edge_counts[:, 2] - edge_counts[:, 3])

    predicted_cluster_mass = largest * base.astype(np.float32)
    values = {
        "syndrome_weight": total_weight,
        "layer0_weight": layer0_weight,
        "layer1_weight": layer1_weight,
        "layer_weight_imbalance": np.abs(layer0_weight - layer1_weight),
        "temporal_disagreement": temporal_disagreement,
        "spatial_active_weight": spatial_count,
        "adjacency_pairs": adjacent.astype(np.float32),
        "component_count": component_count,
        "largest_component": largest,
        "second_component": second,
        "component_density": adjacent / active_nonzero,
        "boundary_count": boundary_count,
        "boundary_fraction": boundary_fraction,
        "boundary_min_distance": min_boundary_distance,
        "near_boundary_count": near_boundary_count,
        "row_occupied": row_occupied,
        "col_occupied": col_occupied,
        "row_max": row_max.astype(np.float32),
        "col_max": col_max.astype(np.float32),
        "row_span": row_span,
        "col_span": col_span,
        "edge_count": edge_count.astype(np.float32),
        "edge_balance": edge_balance.astype(np.float32),
        "cluster_class": cluster_class,
        "predicted_logical": base.astype(np.float32),
        "predicted_cluster_mass": predicted_cluster_mass,
        "residual_signed_weight": sign * total_weight,
        "residual_signed_adjacency": sign * adjacent,
        "residual_signed_boundary": sign * boundary_fraction,
        "residual_signed_component": sign * largest,
        "prediction_residual_density": sign * adjacent / active_nonzero,
        "xi": np.full(len(spatial), float(point.xi), dtype=np.float32),
        "p": np.full(len(spatial), float(point.p), dtype=np.float32),
        "L": np.full(len(spatial), float(point.L), dtype=np.float32),
    }
    return FeatureBatch(values)


def _make_matching(point: Point) -> pymatching.Matching:
    experiment = SurfaceCodeExperiment(distance=point.L)
    dem = _data_only_dem(experiment, point.p)
    return pymatching.Matching.from_detector_error_model(dem)


class ResidualDecoder:
    """Data-only MWPM plus one vectorized post-processing policy."""

    def __init__(self, point: Point, variant: str = "rule"):
        if variant not in {"noop", "rule", "classifier"}:
            raise ValueError(f"unknown Track-C variant {variant}")
        self.point = point
        self.variant = variant
        self.geometry = build_geometry(point.L, point.p)
        self._matching = _make_matching(point)
        self.last_features: FeatureBatch | None = None
        self.last_base_prediction: np.ndarray | None = None

    def _base_decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        prediction = self._matching.decode_batch(
            np.asarray(syndrome_array, dtype=np.uint8)
        )
        if prediction.ndim == 2:
            prediction = prediction[:, 0]
        return prediction.astype(np.uint8, copy=False)

    def _rule_flip(self, features: FeatureBatch) -> np.ndarray:
        f = features.values
        if float(self.point.xi) < 5.0:
            return np.zeros(f["predicted_logical"].shape, dtype=np.bool_)
        # Deliberately high-specificity: only a connected interior burst with a
        # baseline logical-1 prediction can be reversed, and never at xi=0.
        return (
            (float(self.point.xi) >= 5.0)
            & (f["predicted_logical"] == 1)
            & (f["syndrome_weight"] >= 4)
            & (f["largest_component"] >= 3)
            & (f["adjacency_pairs"] >= 2)
            & (f["boundary_fraction"] <= 0.5)
        )

    def _classifier_flip(self, features: FeatureBatch) -> np.ndarray:
        if float(self.point.xi) < 5.0:
            return np.zeros(features.values["predicted_logical"].shape, dtype=np.bool_)
        matrix = features.matrix()
        standardized = (matrix - MODEL_MEAN[None, :]) / MODEL_SCALE[None, :]
        score = MODEL_INTERCEPT + standardized @ MODEL_COEFFICIENTS
        return score >= MODEL_THRESHOLD

    def decode(self, syndrome_array: np.ndarray) -> np.ndarray:
        if syndrome_array.ndim != 2:
            raise ValueError("syndrome_array must be rank 2")
        base = self._base_decode(syndrome_array)
        features = extract_features(syndrome_array, self.geometry, base, self.point, fast=True)
        if self.variant == "noop":
            flip = np.zeros(base.shape, dtype=np.bool_)
        elif self.variant == "rule":
            flip = self._rule_flip(features)
        else:
            flip = self._classifier_flip(features)
        # Keep only the two columns needed by the external experiment runner;
        # the complete feature dictionary is released before the next point.
        self.last_features = FeatureBatch(
            {name: features.values[name] for name in DIAGNOSTIC_FEATURES}
        )
        self.last_base_prediction = base
        return np.bitwise_xor(base, flip.astype(np.uint8, copy=False))


VARIANTS = ("noop", "rule", "classifier")
DECODERS = {name: (lambda point, name=name: ResidualDecoder(point, name)) for name in VARIANTS}


def build_decoder(point: Point, variant: str = "rule") -> ResidualDecoder:
    """Parent-harness API; ``rule`` is the conservative default."""
    return ResidualDecoder(point, variant)
