"""stair_step_filter: climbable steps become traversable, everything else keeps the slope/step/roughness verdict.

Synthetic 10 x 10 m map at 0.05 m: a 30-degree staircase (rise 0.165, tread 0.30), a 0.5 m ledge, a 45-degree ramp,
vegetation (random 0-0.15 m), a 0.12 m curb and flat floor, all with 1 cm noise. Needs cupy (GPU), like the other
plugin tests.
"""
import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from elevation_mapping_cupy.plugins.drivability_filter import DrivabilityFilter  # noqa: E402
from elevation_mapping_cupy.plugins.roughness_filter import RoughnessFilter  # noqa: E402
from elevation_mapping_cupy.plugins.slope_filter import SlopeFilter  # noqa: E402
from elevation_mapping_cupy.plugins.stair_step_filter import StairStepFilter  # noqa: E402
from elevation_mapping_cupy.plugins.step_filter import StepFilter  # noqa: E402

N, RES = 200, 0.05
NAMES = ["elevation", "variance", "is_valid", "traversability", "time", "upper_bound", "is_upper_bound"]
LIMITS = [45.0, 0.30, 0.08]


def synthetic():
    rng = np.random.default_rng(0)
    z = np.zeros((N, N), np.float32); X = (np.arange(N) * RES)[None, :].repeat(N, 0)
    z[0:40] = np.floor(np.clip(X[0:40] - 2.0, 0, None) / 0.30) * 0.165          # staircase along x
    z[40:80] = np.where(X[40:80] >= 5.0, 0.5, 0.0)                               # 0.5 m ledge
    z[80:120] = np.clip(X[80:120] - 2.0, 0, None) * np.tan(np.radians(45))      # 45 deg ramp
    z[120:160] = rng.uniform(0, 0.15, (40, N))                                   # vegetation
    z[160:200] = np.where(X[160:200] >= 5.0, 0.12, 0.0)                          # 0.12 m curb
    z += rng.normal(0, 0.01, z.shape).astype(np.float32)
    emap = cp.zeros((7, N, N), cp.float32); emap[0] = cp.asarray(z); emap[2] = 1.0
    return emap


def run(emap, override):
    plugins = {"slope": SlopeFilter(cell_n=N, resolution=RES, window_size=5, min_valid_ratio=0.5),
               "step": StepFilter(cell_n=N, resolution=RES, window_size=5, min_valid_count=10, median_size=3),
               "roughness": RoughnessFilter(cell_n=N, resolution=RES, window_size=5, min_valid_count=6),
               "stairlike": StairStepFilter(cell_n=N, resolution=RES)}
    layers = cp.full((4, N, N), cp.nan, cp.float32); lnames = list(plugins); sem = cp.zeros((0, N, N), cp.float32)
    for i, p in enumerate(plugins.values()):
        layers[i] = p(emap, NAMES, layers, lnames, sem, [])
    kw = dict(override_layer="stairlike", override_value=0.5) if override else {}
    drv = DrivabilityFilter(cell_n=N, resolution=RES, layers=["slope", "step", "roughness"], limits=LIMITS, **kw)(emap, NAMES, layers, lnames, sem, [])
    return layers[3].get(), drv.get()


REGIONS = {"stairs": (slice(5, 35), slice(44, 160)), "ledge_edge": (slice(45, 75), slice(95, 105)), "ramp45": (slice(85, 115), slice(50, 150)),
           "vegetation": (slice(125, 155), slice(20, 180)), "curb_edge": (slice(165, 195), slice(95, 105)), "floor": (slice(165, 195), slice(10, 80))}


def condemned(d, key):
    r, c = REGIONS[key]; v = d[r, c]; return float(np.nanmean(v < 0.4))


def test_stairs_become_traversable_and_nothing_else_opens():
    emap = synthetic()
    sl, d_old = run(emap, False)
    _, d_new = run(emap, True)
    assert condemned(d_old, "stairs") > 0.5, "the plain chain should condemn most of a 30 deg flight (that is the problem)"
    assert condemned(d_new, "stairs") < 0.1, "with the step rule the flight is traversable"
    assert np.nanmean(sl[REGIONS["stairs"]] == 1) > 0.4
    for key in ("ledge_edge", "ramp45", "vegetation", "floor"):
        assert abs(condemned(d_new, key) - condemned(d_old, key)) < 0.02, "%s must not change" % key
    assert condemned(d_new, "ramp45") > 0.95 and condemned(d_new, "ledge_edge") > 0.5
    assert condemned(d_new, "curb_edge") <= condemned(d_old, "curb_edge")
