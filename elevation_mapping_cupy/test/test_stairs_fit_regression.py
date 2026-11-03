"""Regression of the mapping-time stair registration on fixed records.

Inputs: stairs_record.npz files kept outside the map folders (the live pipeline deletes its record after the fit):
  $HAECHI_STAIRS_TESTDATA (default ~/data/haechi_sim/testdata/stairs_records)/<case>.npz
    ds1027_1f_sim        Isaac Sim mapping of the 1F stairs env (6 flights in 2 staircases, risers 0.155-0.21 m)
    ds1027_1f_bag_live   the real 2026-10-27 bag replayed through the same pipeline (6 flights in 2 staircases)
    ds1020_sim           flat campus, no stairs (0 flights: the ramp guard / slope rule must not invent any)
A missing file skips its case (the data is not in the repository).

  pytest -q test/test_stairs_fit_regression.py            (from the package dir; also registered with colcon test)
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from haechi_terrain import stairs_fit as sf  # noqa: E402
from haechi_terrain import stairs_raster  # noqa: E402

DATA = os.path.expanduser(os.environ.get("HAECHI_STAIRS_TESTDATA", "~/data/haechi_sim/testdata/stairs_records"))
CASES = {
    "ds1027_1f_sim": dict(flights=6, staircases=2, per_staircase=3, rise=(0.14, 0.22), tread=(0.25, 0.38), n=(4, 7)),
    "ds1027_1f_bag_live": dict(flights=6, staircases=2, per_staircase=3, rise=(0.14, 0.22), tread=(0.25, 0.38), n=(4, 7)),
    "ds1020_sim": dict(flights=0, staircases=0),
}
_cache = {}


def _fit(name):
    path = os.path.join(DATA, name + ".npz")
    if not os.path.isfile(path):
        pytest.skip("no record %s" % path)
    if name not in _cache:
        d = np.load(path)
        c = sf.cfg(sensor_height=float(d["sensor_height"]))
        P = d["points"].astype(np.float64)
        P = P[sf.clean_points(P, d["frames"].astype(np.int64), d["t_first"], d["t_last"], d["t"], d["lidar"], c, log=lambda *a: None)]
        _cache[name] = sf.fit_run(d["t"], d["lidar"], P, c, log=lambda *a: None)
    return _cache[name]


@pytest.mark.parametrize("name", list(CASES))
def test_registration(name):
    fits, staircases, info = _fit(name)
    exp = CASES[name]
    assert len(fits) == exp["flights"], "flights: %d (expected %d); info %s" % (len(fits), exp["flights"], info)
    assert len(staircases) == exp["staircases"]
    if not fits:
        return
    for k, f in enumerate(fits):
        assert exp["rise"][0] <= f["rise"] <= exp["rise"][1], "flight %d rise %.3f" % (k + 1, f["rise"])
        assert exp["tread"][0] <= f["tread"] <= exp["tread"][1], "flight %d tread %.3f" % (k + 1, f["tread"])
        assert exp["n"][0] <= f["n"] <= exp["n"][1], "flight %d n %d" % (k + 1, f["n"])
        assert f["z_top"] > f["z_bot"] and abs((f["z_top"] - f["z_bot"]) - f["n"] * f["rise"]) < 0.12, "flight %d height vs n*rise" % (k + 1)
        assert f["width"] >= 0.8, "flight %d width %.2f" % (k + 1, f["width"])
    for sc in staircases:
        assert len(sc["flights"]) == exp["per_staircase"], sc
        assert all(b > a for a, b in zip(sc["levels"], sc["levels"][1:])) or all(b < a for a, b in zip(sc["levels"], sc["levels"][1:])), sc


def test_raster_1f_sim():
    """The registered flights rasterise to a plausible segment: every flight footprint is inside, cell count in the band
    the 2026-10-29 bake measured (10475 cells at 5 cm with the walls of the 1F SLAM map cut)."""
    fits, staircases, _ = _fit("ds1027_1f_sim")
    map_yaml = os.path.expanduser("~/data/ds1027_1f_isaac/maps/ds1027_1f/sim/map.yaml")
    grid, x0, y0, n_land, n_wall = stairs_raster.rasterise(fits, staircases, 0.05, map_yaml=map_yaml if os.path.isfile(map_yaml) else "")
    n = int((grid >= 50).sum())
    assert 7000 <= n <= 14000, "stair cells %d" % n
    assert n_land == 4, "landings %d" % n_land                      # 2 staircases x 2 landings between 3 flights
    for f in fits:
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        c = p0 + ax * (f["s0"] + 0.5 * (f["n"] - 1) * f["tread"]) + lat * 0.5 * (f["l_lo"] + f["l_hi"])
        i, j = int((c[1] - y0) / 0.05), int((c[0] - x0) / 0.05)
        assert grid[i, j] >= 50, "flight centre %s not a stair cell" % c
