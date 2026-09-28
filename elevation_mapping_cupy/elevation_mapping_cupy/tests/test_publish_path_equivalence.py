"""The rewritten publish path must put the same bytes on the wire.

Old path (per layer): m -> m.T -> flip(0) -> flip(1) -> host copy -> node
encodes with encode_layer_to_multiarray(..., "gridmap_column"), which
transposes again and packs C order. New path: rot180(m) in C order straight
into the message. Both are pure permutations, so equality is exact, NaNs
included. The device half (copy_layer_rot180) is checked when cupy is
present; the host half always.
"""
import numpy as np
import pytest

from elevation_mapping_cupy.gridmap_utils import (
    encode_layer_to_multiarray,
    encode_rot180_as_gridmap_column,
)


def _old_bytes(m: np.ndarray):
    data = np.flip(np.flip(m.T, 0), 1).astype(np.float32)   # what get_map_with_name_ref produced
    msg = encode_layer_to_multiarray(data, layout="gridmap_column")
    return bytes(msg.data.tobytes()), [(d.label, d.size, d.stride) for d in msg.layout.dim]


def _new_bytes(m: np.ndarray):
    host = np.ascontiguousarray(m[::-1, ::-1], dtype=np.float32)
    msg = encode_rot180_as_gridmap_column(host)
    return bytes(msg.data.tobytes()), [(d.label, d.size, d.stride) for d in msg.layout.dim]


@pytest.mark.parametrize("n", [4, 7, 198])
def test_host_encoding_identical(n):
    rng = np.random.default_rng(n)
    m = rng.standard_normal((n, n)).astype(np.float32)
    m[rng.random((n, n)) < 0.3] = np.nan
    ob, od = _old_bytes(m)
    nb, nd = _new_bytes(m)
    assert od == nd
    assert ob == nb


def test_device_path_identical():
    cp = pytest.importorskip("cupy")
    from elevation_mapping_cupy.elevation_mapping import ElevationMap  # noqa: F401  (import guard)
    n = 198
    rng = np.random.default_rng(1)
    m_np = rng.standard_normal((n + 2, n + 2)).astype(np.float32)
    m_np[rng.random((n + 2, n + 2)) < 0.3] = np.nan
    m = cp.asarray(m_np)[1:-1, 1:-1]

    class Stub:
        pass

    stub = Stub()
    stub._publish_buf = cp.empty((n, n), dtype=cp.float32)
    import threading
    stub.map_lock = threading.Lock()
    stub._layer_for_publish = lambda name: m
    host = np.empty((n, n), dtype=np.float32)
    ElevationMap.copy_layer_rot180(stub, "x", host)
    ob, _ = _old_bytes(m_np[1:-1, 1:-1])
    nb = bytes(encode_rot180_as_gridmap_column(host).data.tobytes())
    assert ob == nb
