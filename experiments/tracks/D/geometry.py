"""Offline GF(2) symmetry audit and syndrome canonicalization.

Each independent physical mask contributes exactly ONCE after canonicalization;
symmetry copies are never counted as independent statistical observations.
"""
import numpy as np
from qec_benchmark.stim_surface_code import SurfaceCodeExperiment


def gf2_solve(a, b):
    a = np.column_stack([a, b]).astype(np.uint8).copy()
    row = 0
    pivots = []
    for col in range(a.shape[1] - 1):
        hits = np.flatnonzero(a[row:, col])
        if not len(hits):
            continue
        j = row + hits[0]
        a[[row, j]] = a[[j, row]]
        for k in np.flatnonzero(a[:, col]):
            if k != row:
                a[k] ^= a[row]
        pivots.append(col)
        row += 1
    if np.any(a[row:, -1]):
        return None
    x = np.zeros(a.shape[1] - 1, dtype=np.uint8)
    x[pivots] = a[:row, -1]
    return x


def geometry(L):
    ex = SurfaceCodeExperiment(L)
    s, logical = ex.sample_from_mask(np.eye(L * L, dtype=bool))
    active = np.flatnonzero(s.any(axis=0))
    h = s[:, active].astype(np.uint8)
    logical = logical.astype(np.uint8)
    assert len(active) == (L * L - 1) // 2
    assert len(gf2_solve(h, np.zeros(L * L, dtype=np.uint8))) == len(active)
    xy = ex.data_positions
    centered = xy - xy.mean(axis=0)
    coord_index = {tuple(c): i for i, c in enumerate(centered)}
    support_index = {tuple(h[:, j]): j for j in range(h.shape[1])}
    symmetries, rejected = [], []
    # All eight square isometries; retain only exact check-space permutations.
    for swap in (False, True):
        for sx, sy in ((1, 1), (-1, -1), (-1, 1), (1, -1)):
            transformed = centered[:, ::-1] if swap else centered
            transformed = transformed * [sx, sy]
            perm = np.array([coord_index[tuple(c)] for c in transformed])
            name = f"swap{int(swap)}_x{sx}_y{sy}"
            hp = h[perm]
            if any(tuple(hp[:, j]) not in support_index for j in range(h.shape[1])):
                rejected.append(name)
                continue
            syndrome_perm = [support_index[tuple(hp[:, j])] for j in range(h.shape[1])]
            action = gf2_solve(h, logical[perm] ^ logical)
            if action is None:
                rejected.append(name)
                continue
            assert np.array_equal(h[:, syndrome_perm], hp)
            assert np.array_equal((h @ action) % 2, logical[perm] ^ logical)
            assert np.allclose(np.linalg.norm(xy[:, None]-xy[None, :], axis=2),
                               np.linalg.norm(xy[perm, None]-xy[None, perm], axis=2))
            symmetries.append({"name": name, "physical_perm": perm.tolist(),
                               "syndrome_perm": syndrome_perm,
                               "logical_mask": int(action @ (1 << np.arange(len(active))))})
    return {"L": L, "active": active.tolist(), "detectors": ex.num_detectors,
            "h": h.tolist(), "logical": logical.tolist(),
            "symmetries": symmetries, "rejected": rejected,
            "audit": "all single-error columns; linearity extends to all masks; Euclidean covariance invariant"}


def ids_from_syndrome(syndrome, active):
    return syndrome[:, active].astype(np.uint32) @ (np.uint32(1) << np.arange(len(active), dtype=np.uint32))


def syndrome_from_ids(ids, geom):
    s = np.zeros((len(ids), geom["detectors"]), dtype=np.uint8)
    s[:, geom["active"]] = (np.asarray(ids)[:, None] >> np.arange(len(geom["active"]))) & 1
    return s


def parity(x):
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    return ((np.uint32(0x6996) >> (x & 15)) & 1).astype(np.uint8)


def canonicalize(ids, geom):
    ids = np.asarray(ids, dtype=np.uint32)
    best = ids.copy()
    action = np.zeros(len(ids), dtype=np.uint8)
    for sym in geom["symmetries"]:
        changed = np.zeros(len(ids), dtype=np.uint32)
        for output, source in enumerate(sym["syndrome_perm"]):
            changed |= ((ids >> source) & 1) << output
        better = changed < best
        best[better] = changed[better]
        action[better] = parity(ids[better] & sym["logical_mask"])
    return best, action


def orbit_ids(ids, geom):
    outputs = []
    for sym in geom["symmetries"]:
        changed = np.zeros(len(ids), dtype=np.uint32)
        for output, source in enumerate(sym["syndrome_perm"]):
            changed |= ((ids >> source) & 1) << output
        outputs.append(changed)
    return np.unique(np.concatenate(outputs))


def ambiguous_ids(ids, geom):
    """A stabilizer changing logical but fixing syndrome forces P(logical)=1/2."""
    ambiguous = np.zeros(len(ids), dtype=bool)
    for sym in geom["symmetries"]:
        changed = np.zeros(len(ids), dtype=np.uint32)
        for output, source in enumerate(sym["syndrome_perm"]):
            changed |= ((ids >> source) & 1) << output
        ambiguous |= (changed == ids) & (parity(ids & sym["logical_mask"]) != 0)
    return ambiguous
