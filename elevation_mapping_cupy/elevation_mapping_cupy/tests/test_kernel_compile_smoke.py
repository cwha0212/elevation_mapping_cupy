from __future__ import annotations

from pathlib import Path

import numpy as np
import cupy as cp

from elevation_mapping_cupy import ElevationMap, Parameter


def test_cupy_cuda_is_available():
    # Fail loudly: this repo targets CUDA + CuPy.
    n = cp.cuda.runtime.getDeviceCount()
    assert n >= 1


def test_kernels_compile_and_one_update_step_runs():
    # .../elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping_cupy/tests/test_kernel_compile_smoke.py
    # parents[2] = ROS package root (contains config/).
    root = Path(__file__).resolve().parents[2]
    p = Parameter(
        use_chainer=False,
        weight_file=str(root / "config" / "core" / "weights.dat"),
        plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
    )

    # Keep map tiny so the smoke test stays fast.
    p.resolution = 0.2
    p.map_length = 4.0
    p.update()

    emap = ElevationMap(p)

    # A few points on a plane in the sensor frame.
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, -0.5, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [2.0, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    # Should run without exceptions (kernels + traversability torch filter).
    emap.input_pointcloud(pts, ["x", "y", "z"], R, t, 0.0, 0.0)
    emap.update_variance()
    emap.update_time()


def test_extra_point_channels_match_contiguous_xyz_input():
    root = Path(__file__).resolve().parents[2]

    def make_map():
        p = Parameter(
            use_chainer=False,
            weight_file=str(root / "config" / "core" / "weights.dat"),
            plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
        )
        p.resolution = 0.2
        p.map_length = 4.0
        p.enable_visibility_cleanup = False
        p.update()
        return ElevationMap(p)

    xyz = np.array(
        [[1.0, -0.4, 0.0], [1.0, 0.0, 0.1], [1.0, 0.4, 0.2]],
        dtype=np.float32,
    )
    xyzi = np.column_stack((xyz, np.asarray([100.0, 200.0, 300.0], dtype=np.float32)))
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    xyz_map = make_map()
    xyzi_map = make_map()

    xyz_map.input_pointcloud(xyz, ["x", "y", "z"], rotation, translation, 0.0, 0.0)
    xyzi_map.input_pointcloud(
        xyzi,
        ["x", "y", "z", "intensity"],
        rotation,
        translation,
        0.0,
        0.0,
    )

    cp.testing.assert_allclose(xyzi_map.elevation_map[:3], xyz_map.elevation_map[:3])
