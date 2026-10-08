#!/usr/bin/env python3
"""Steady-state cost per operation on the active backend.
    ELEVATION_BACKEND=numpy taskset -c 4 python3 tools/backend_bench.py [points] [map_length] [resolution]
"""
import sys, time
from pathlib import Path
import numpy as np
from elevation_mapping_cupy import ElevationMap, Parameter
from elevation_mapping_cupy.backend import BACKEND, USE_CUPY

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
L = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
RES = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
root = Path(__file__).resolve().parents[1]
if not (root / "config" / "core" / "weights.dat").exists():
    from ament_index_python.packages import get_package_share_directory
    root = Path(get_package_share_directory("elevation_mapping_cupy"))
p = Parameter(weight_file=str(root / "config/core/weights.dat"), plugin_config_file=str(root / "config/setups/haechi/plugin_config.yaml"))
p.resolution, p.map_length, p.max_ray_length = RES, L, 5.0
p.subscriber_cfg = {"lidar": {"topic_name": "/p", "data_type": "pointcloud"}, "front_cam": {"topic_name": "/i", "channels": ["untrav"], "data_type": "image"}}
p.update(); emap = ElevationMap(p)
rng = np.random.default_rng(0); R = np.eye(3, dtype=np.float32); t = np.array([0, 0, 0.7], np.float32)
def cloud():
    d = rng.uniform(0.3, 8.0, N); a = rng.uniform(-np.pi, np.pi, N); z = rng.normal(0, 0.01, N) + 0.3 * (rng.random(N) < 0.02) - 0.7
    return np.stack([d * np.cos(a), d * np.sin(a), z], 1).astype(np.float32)
H, W = 256, 448; K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]], np.float32)
R_cm = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32); t_cm = (-R_cm @ np.array([0, 0, 0.6], np.float32)).astype(np.float32)
img = rng.normal(-1, 0.5, (H, W)).astype(np.float32); img[H // 2:] += 3
buf = np.zeros((emap.cell_n - 2, emap.cell_n - 2), np.float32)
def sync():
    if USE_CUPY:
        import cupy; cupy.cuda.Stream.null.synchronize()
# warm-up (JIT, caches)
for _ in range(3):
    emap.input_pointcloud(cloud(), ["x", "y", "z"], R, t, 0, 0); emap.update_variance()
    emap.input_image([img], ["untrav"], R_cm, t_cm, K, np.zeros((5, 1), np.float32), "plumb_bob", H, W)
    for l in ("drivability", "safety"): emap.copy_layer_rot180(l, buf)
sync()
def timeit(fn, n=20):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return 1e3 * np.median(ts), 1e3 * np.max(ts)
c = [cloud() for _ in range(20)]; it = iter(c)
print("backend %s  points %d  map %.0f m @ %.2f (%d cells)" % (BACKEND, N, L, RES, emap.cell_n ** 2))
print("  pointcloud fuse (+visibility rays)  med %6.1f ms  max %6.1f" % timeit(lambda: (emap.input_pointcloud(next(it), ["x", "y", "z"], R, t, 0, 0), emap.update_variance())))
p.enable_visibility_cleanup = False; emap.compile_kernels(); it = iter(c)
print("  pointcloud fuse (no visibility)      med %6.1f ms  max %6.1f" % timeit(lambda: emap.input_pointcloud(next(it), ["x", "y", "z"], R, t, 0, 0)))
p.enable_visibility_cleanup = True; emap.compile_kernels()
def chain():
    emap.plugin_manager.reset_layers()
    for l in ("slope", "step", "roughness", "drivability", "safety"): emap.copy_layer_rot180(l, buf)
print("  plugin chain + publish 5 layers       med %6.1f ms  max %6.1f" % timeit(chain))
print("  image projection + fusion             med %6.1f ms  max %6.1f" % timeit(lambda: emap.input_image([img], ["untrav"], R_cm, t_cm, K, np.zeros((5, 1), np.float32), "plumb_bob", H, W)))
print("  move_to 0.3 m                         med %6.1f ms  max %6.1f" % timeit(lambda: emap.move_to(t + np.array([0.3 * rng.random(), 0, 0], np.float32), R), n=5))
