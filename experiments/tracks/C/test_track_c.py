"""Independent geometry and vectorization checks for Track C."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from candidate_c import ResidualDecoder, build_geometry, extract_features  # noqa: E402
from qec_benchmark.models import ParameterPoint  # noqa: E402
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment  # noqa: E402


EXPECTED = {
    3: {"detectors": 8, "data": 9, "edges": 3},
    5: {"detectors": 24, "data": 25, "edges": 15},
    7: {"detectors": 48, "data": 49, "edges": 35},
}


def reference_components(active: np.ndarray, adjacency: np.ndarray) -> tuple[int, int, int]:
    """Plain graph traversal used only as an independent test oracle."""
    seen: set[int] = set()
    sizes = []
    for start in np.flatnonzero(active):
        start = int(start)
        if start in seen:
            continue
        todo = [start]
        seen.add(start)
        size = 0
        while todo:
            node = todo.pop()
            size += 1
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if active[neighbor] and neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)
    return len(sizes), sizes[0] if sizes else 0, sizes[1] if len(sizes) > 1 else 0


def test_geometry_and_components() -> None:
    for L, expected in EXPECTED.items():
        experiment = SurfaceCodeExperiment(L)
        geometry = build_geometry(L)
        assert experiment.num_detectors == expected["detectors"]
        assert experiment.num_data_qubits == expected["data"]
        assert geometry.spatial_adjacency.sum() // 2 == expected["edges"]
        assert geometry.layer0.size * 2 == expected["detectors"]
        expected_xy = []
        for x_index in range(L + 1):
            y_values = range(4, 2 * L, 4) if x_index % 2 == 0 else range(2, 2 * L - 1, 4)
            expected_xy.extend((2 * x_index, y) for y in y_values)
        assert np.array_equal(
            np.column_stack([geometry.spatial_x, geometry.spatial_y]),
            np.asarray(expected_xy, dtype=np.float64),
        )

        rng = np.random.default_rng(6000 + L)
        syndrome = rng.integers(0, 2, size=(37, expected["detectors"]), dtype=np.uint8)
        base = rng.integers(0, 2, size=37, dtype=np.uint8)
        point = ParameterPoint(L, 0.01, 5.0)
        features = extract_features(syndrome, geometry, base, point).values
        active = syndrome[:, geometry.layer0].astype(bool) | syndrome[:, geometry.layer1].astype(bool)
        for shot in range(len(syndrome)):
            count, largest, second = reference_components(active[shot], geometry.spatial_adjacency)
            assert features["component_count"][shot] == count
            assert features["largest_component"][shot] == largest
            assert features["second_component"][shot] == second
        expected_weight = active.sum(axis=1) + (
            syndrome[:, geometry.layer0].astype(bool) & syndrome[:, geometry.layer1].astype(bool)
        ).sum(axis=1)
        assert np.array_equal(features["syndrome_weight"], expected_weight.astype(np.float32))
        assert np.all(features["adjacency_pairs"] >= 0)


def test_decoder_contract_and_no_per_shot_decode_loop() -> None:
    source = (HERE / "candidate_c.py").read_text()
    tree = ast.parse(source)
    decode_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    ]
    assert decode_functions
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(decode_functions[-1]))
    for L, expected in EXPECTED.items():
        decoder = ResidualDecoder(ParameterPoint(L, 0.01, 2.0), "rule")
        syndrome = np.zeros((5, expected["detectors"]), dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert result.shape == (5,)
        assert result.dtype == np.uint8
        assert np.all(result == 0)


if __name__ == "__main__":
    test_geometry_and_components()
    test_decoder_contract_and_no_per_shot_decode_loop()
    print("TRACK C TESTS PASS")
