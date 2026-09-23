"""Frozen full-syndrome posterior architecture, research only.

L5: direct 4096-entry table. L7: sorted uint32 sparse flip IDs + IID backoff.
Offline constants are imported once; build/decode never train or read data files.
L3 is exactly the frozen Track-A QMC table for all non-IID variants.
"""
import base64
import zlib
import numpy as np
from experiments.tracks.A.candidate_a import build_decoder as build_a
from experiments.tracks.D.reference_iid import IIDDecoder

VARIANTS = ("iid", "a_reference", "raw", "supported", "supported_nosym")


def point_key(point):
    return f"L{point.L}_p{point.p:g}_xi{point.xi:g}"


def unpack(blob, dtype):
    return np.frombuffer(zlib.decompress(base64.b85decode(blob)), dtype=dtype).copy()


def packed_ids(syndrome, active):
    packed = np.packbits(syndrome[:, active], axis=1, bitorder="little")
    result = np.zeros(len(syndrome), dtype=np.uint32)
    for byte in range(packed.shape[1]):
        result |= packed[:, byte].astype(np.uint32) << (8 * byte)
    return result


class PosteriorDecoder:
    def __init__(self, point, variant, record):
        self.active = record["active"]
        self.width = point.L * point.L - 1
        self.L = point.L
        if point.L == 5:
            self.table = unpack(record[variant], "u1")
            self.storage_bytes = self.table.nbytes
        else:
            self.base = IIDDecoder(point)
            self.flip_ids = unpack(record[variant], "<u4")
            self.storage_bytes = self.flip_ids.nbytes

    def decode(self, syndrome_array):
        if syndrome_array.ndim != 2 or syndrome_array.shape[1] != self.width:
            raise ValueError("wrong syndrome shape")
        ids = packed_ids(syndrome_array, self.active)
        if self.L == 5:
            return self.table[ids]
        result = self.base.decode(syndrome_array)
        if len(self.flip_ids):
            indices = np.searchsorted(self.flip_ids, ids)
            indices = np.minimum(indices, len(self.flip_ids) - 1)
            result ^= (self.flip_ids[indices] == ids).astype(np.uint8)
        return result


def build_decoder(point, variant="supported"):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant == "iid":
        return IIDDecoder(point)
    if variant == "a_reference" or point.L == 3:
        return build_a(point)
    if point.xi == 0:
        return IIDDecoder(point)
    from experiments.tracks.D.frozen_tables import TABLES
    return PosteriorDecoder(point, variant, TABLES[point_key(point)])
