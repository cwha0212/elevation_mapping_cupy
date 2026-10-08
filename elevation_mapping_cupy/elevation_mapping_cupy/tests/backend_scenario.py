"""One deterministic mapping scenario, run on whichever backend is active.

Used by test_numpy_backend.py (does it run) and tools/backend_ab.py (does it
match cupy). Synthetic lidar sweeps over a plane with a wall, a step and a
curb, a few map moves, and a camera score image projected in.
"""
from pathlib import Path

import numpy as np


def run_scenario(frames=12, resolution=0.05, map_length=10.0, with_image=True):
    from elevation_mapping_cupy import ElevationMap, Parameter
    from elevation_mapping_cupy.backend import asnumpy

    root = Path(__file__).resolve().parents[2]
    if not (root / "config" / "core" / "weights.dat").exists():
        # installed package: the config lives in the share directory
        from ament_index_python.packages import get_package_share_directory

        root = Path(get_package_share_directory("elevation_mapping_cupy"))
    p = Parameter(
        weight_file=str(root / "config" / "core" / "weights.dat"),
        plugin_config_file=str(root / "config" / "setups" / "haechi" / "plugin_config.yaml"),
    )
    p.resolution = resolution
    p.map_length = map_length
    p.max_ray_length = 5.0
    p.subscriber_cfg = {
        "lidar": {"topic_name": "/points", "data_type": "pointcloud"},
        "front_cam": {"topic_name": "/front_cam/samtp_score", "channels": ["untrav"], "data_type": "image"},
    }
    p.update()
    emap = ElevationMap(p)
    rng = np.random.default_rng(0)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0.0, 0.0, 0.7], dtype=np.float32)   # sensor 0.7 m above ground

    def sweep(n=20000):
        d = rng.uniform(0.3, 8.0, n); a = rng.uniform(-np.pi, np.pi, n)
        x, y = d * np.cos(a), d * np.sin(a)
        z = rng.normal(0.0, 0.01, n)
        z = np.where(x > 3.0, z + 0.5, z)                 # a step up at x = 3
        z = np.where(y > 2.5, z + 0.12, z)                # a curb along y = 2.5
        wall = (np.abs(y + 3.0) < 0.1) & (x > -2) & (x < 2)
        z = np.where(wall, rng.uniform(0.0, 1.5, n), z)   # a wall at y = -3
        pts = np.stack([x, y, z], 1).astype(np.float32)
        pts[:, 2] -= t[2]                                  # sensor frame: z relative to sensor
        return pts

    for i in range(frames):
        if i % 4 == 3:
            t[0] += 0.3
            emap.move_to(t, R)
        emap.input_pointcloud(sweep(), ["x", "y", "z"], R, t, 0.0, 0.0)
        emap.update_variance()
        if i % 3 == 0:
            emap.update_time()
    if with_image:
        H, W = 128, 224
        K = np.array([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]], np.float32)
        # camera looking along +x, 0.6 m up: optical z = world x, optical x = -world y, optical y = -world z
        R_cm = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        t_cm = (-R_cm @ np.array([t[0], 0.0, 0.6], np.float32)).astype(np.float32)
        img = rng.normal(-1.0, 0.3, (H, W)).astype(np.float32)
        img[H // 2:, :] += 3.0
        for _ in range(3):
            emap.input_image([img], ["untrav"], R_cm, t_cm, K, np.zeros((5, 1), np.float32), "plumb_bob", H, W)
    n = emap.cell_n - 2
    out = {}
    buf = np.zeros((n, n), np.float32)
    for layer in ("elevation", "variance", "is_valid", "traversability", "upper_bound", "slope", "step", "roughness", "drivability", "safety", "untrav"):
        emap.copy_layer_rot180(layer, buf)
        out[layer] = buf.copy()
    out["normal_z"] = asnumpy(emap.normal_map[2]).copy()
    return out
