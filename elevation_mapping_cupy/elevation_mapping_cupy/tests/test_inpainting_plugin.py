import cupy as cp
import numpy as np

from elevation_mapping_cupy.plugins.inpainting import Inpainting


def _make_elevation_map(size: int = 9):
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    elevation = xx + 2.0 * yy
    valid = np.ones((size, size), dtype=np.float32)
    elevation_map = cp.zeros((7, size, size), dtype=cp.float32)
    elevation_map[0] = cp.asarray(elevation)
    elevation_map[2] = cp.asarray(valid)
    return elevation_map


def test_inpainting_only_fills_small_holes():
    elevation_map = _make_elevation_map()
    elevation_map[0, 2, 2] = cp.nan
    elevation_map[2, 2, 2] = 0.0
    elevation_map[0, 5:8, 5:8] = cp.nan
    elevation_map[2, 5:8, 5:8] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = cp.asnumpy(plugin(elevation_map, [], cp.zeros((0, 9, 9), dtype=cp.float32), []))

    assert np.isfinite(result[2, 2])
    assert np.isnan(result[5:8, 5:8]).all()
    assert np.isclose(result[1, 1], 3.0)
    assert np.isclose(result[4, 4], 12.0)


def test_inpainting_flat_terrain_does_not_broadcast_large_invalid_regions():
    size = 9
    elevation_map = cp.zeros((7, size, size), dtype=cp.float32)
    elevation_map[0] = 1.5
    elevation_map[2] = 1.0

    elevation_map[0, 3, 3] = cp.nan
    elevation_map[2, 3, 3] = 0.0
    elevation_map[0, 0:3, 5:8] = cp.nan
    elevation_map[2, 0:3, 5:8] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = cp.asnumpy(plugin(elevation_map, [], cp.zeros((0, size, size), dtype=cp.float32), []))

    assert np.isclose(result[3, 3], 1.5)
    assert np.isnan(result[0:3, 5:8]).all()


def test_inpainting_does_not_fill_border_touching_invalid_cells():
    elevation_map = _make_elevation_map()
    elevation_map[0, 0, 4] = cp.nan
    elevation_map[2, 0, 4] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = cp.asnumpy(plugin(elevation_map, [], cp.zeros((0, 9, 9), dtype=cp.float32), []))

    assert np.isnan(result[0, 4])
