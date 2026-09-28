"""The stairs detector, against geometry we can state exactly.

Everything here is numpy on synthetic terrain: no ROS, no GPU, no simulator.
That matters because the cases worth guarding are the ones a demo world does
not contain. The filter used to flag a plain 25 degree bank as a climbable
flight, and nothing caught it, because the only slope in the world was 12
degrees and the only stairs were the ones it was written against.

The layers are built the way the real chain builds them: step is a max minus
min over the foot-scale window, slope comes from the local gradient, and
roughness is the residual off the local plane.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

RES = 0.05
GROUND = 0.12


def _load_detector():
    """Import the plugin file without dragging in the ROS package."""
    pkg = "elevation_mapping_cupy.plugins.plugin_manager"
    if pkg not in sys.modules:
        stub = types.ModuleType(pkg)
        stub.PluginBase = object
        for parent in ("elevation_mapping_cupy", "elevation_mapping_cupy.plugins"):
            sys.modules.setdefault(parent, types.ModuleType(parent))
        sys.modules[pkg] = stub
    path = Path(__file__).resolve().parents[1] / (
        "elevation_mapping_cupy/plugins/stairs_filter.py"
    )
    spec = importlib.util.spec_from_file_location("stairs_filter_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DET = _load_detector()


def layers(elev, valid=None):
    """The terrain chain's step, slope and roughness for a height field."""
    if valid is None:
        valid = np.isfinite(elev)
    e = np.where(valid, elev, np.nan)
    filled = np.where(valid, elev, np.nanmedian(e[valid]) if valid.any() else 0.0)

    step = ndi.maximum_filter(filled, size=5) - ndi.minimum_filter(filled, size=5)
    smooth = ndi.uniform_filter(filled, size=3)
    gy, gx = np.gradient(smooth, RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    plane = ndi.uniform_filter(filled, size=5)
    roughness = np.sqrt(
        np.maximum(ndi.uniform_filter((filled - plane) ** 2, size=5), 0.0)
    )
    for a in (step, slope, roughness):
        a[~valid] = np.nan
    return step, slope, roughness


def run(elev, valid=None, robot_rc=None, **params):
    """Detect on a field, with the robot at the map centre by construction."""
    if valid is None:
        valid = np.isfinite(elev)
    if robot_rc is not None:
        # the map is robot-centred, so roll the field to put the robot in
        # the middle rather than pretending the detector can be told
        r, c = robot_rc
        elev = np.roll(np.roll(elev, elev.shape[0] // 2 - r, 0), elev.shape[1] // 2 - c, 1)
        valid = np.roll(np.roll(valid, valid.shape[0] // 2 - r, 0), valid.shape[1] // 2 - c, 1)
    step, slope, roughness = layers(elev, valid)
    e = np.where(valid, elev, 0.0).astype(np.float32)
    return DET.detect_stairs(e, valid, step, slope, roughness, RES, params)


def flight(n_cells=120, riser=0.15, tread=0.40, descending=False, width=None):
    """A flight rising (or dropping) along +x from the middle of the map."""
    elev = np.full((n_cells, n_cells), GROUND, dtype=np.float32)
    start = n_cells // 2
    tread_cells = int(round(tread / RES))
    y0 = 0 if width is None else n_cells // 2 - int(width / RES) // 2
    y1 = n_cells if width is None else y0 + int(width / RES)
    for k in range(1, 5):
        x0 = start + (k - 1) * tread_cells
        x1 = min(x0 + tread_cells, n_cells)
        h = GROUND + (-k if descending else k) * riser
        elev[y0:y1, x0:x1] = h
    if descending:
        elev[y0:y1, start + 4 * tread_cells:] = GROUND - 4 * riser
    return elev


def plane(angle_deg, n_cells=120):
    """A planar bank rising along +x at a given angle."""
    x = np.arange(n_cells) * RES
    rise = np.tan(np.radians(angle_deg)) * x
    return (GROUND + rise[None, :] * np.ones((n_cells, 1))).astype(np.float32)


def frac_flagged(conf, region=None, sign=None):
    a = conf if region is None else conf[region]
    fin = np.isfinite(a)
    if not fin.any():
        return 0.0
    v = a[fin]
    hit = np.abs(v) > 0.5 if sign is None else (v > 0.5 if sign > 0 else v < -0.5)
    return float(hit.sum()) / float(v.size)


# --------------------------------------------------------------------------
# a flight is found, solidly


def test_flight_is_flagged_solid():
    conf = run(flight())
    body = np.zeros_like(conf, dtype=bool)
    body[:, 62:90] = True          # the flight itself, not the ground past it
    assert frac_flagged(conf, body, sign=+1) >= 0.85


def test_flight_includes_the_treads_not_just_the_risers():
    """The whole point: a tread is part of the flight, not a gap in it."""
    elev = flight()
    conf = run(elev)
    # a tread's interior is flat, so its step is ~0 and the old per-cell
    # riser test could never reach it
    tread_col = 60 + int(0.40 / RES) // 2
    column = conf[40:80, tread_col]
    assert np.nanmax(np.abs(column)) > 0.5


# --------------------------------------------------------------------------
# the class of false positive this rewrite exists to remove


@pytest.mark.parametrize("angle", [12, 18, 20, 25, 30, 35, 40, 45])
def test_plain_slopes_are_never_stairs(angle):
    conf = run(plane(angle))
    assert frac_flagged(conf) == 0.0, f"{angle} deg bank flagged as a flight"


def test_rubble_is_not_stairs():
    rng = np.random.default_rng(0)
    elev = GROUND + rng.uniform(-0.10, 0.10, (120, 120)).astype(np.float32)
    assert frac_flagged(run(elev)) == 0.0


def test_curb_is_not_stairs():
    elev = np.full((120, 120), GROUND, dtype=np.float32)
    elev[:, 60:] = GROUND + 0.12
    assert frac_flagged(run(elev)) == 0.0


def test_single_ledge_is_not_stairs():
    elev = np.full((120, 120), GROUND, dtype=np.float32)
    elev[:, 60:] = GROUND + 0.25
    assert frac_flagged(run(elev)) == 0.0


def test_two_tier_terrace_is_not_confirmed():
    """Geometrically a staircase except for tread depth, which is the tell."""
    elev = np.full((160, 160), GROUND, dtype=np.float32)
    deep = int(1.2 / RES)
    elev[:, 80:80 + deep] = GROUND + 0.25
    elev[:, 80 + deep:] = GROUND + 0.50
    conf = run(elev)
    fin = np.isfinite(conf)
    assert not (np.abs(conf[fin]) > 0.7).any()


def test_trench_edge_is_not_stairs():
    elev = np.full((120, 120), GROUND, dtype=np.float32)
    elev[:, 60:66] = GROUND - 0.30
    assert frac_flagged(run(elev)) == 0.0


# --------------------------------------------------------------------------
# direction


def test_descending_flight_comes_back_negative():
    conf = run(flight(descending=True))
    fin = np.isfinite(conf)
    flagged = conf[fin][np.abs(conf[fin]) > 0.5]
    assert flagged.size > 0, "descending flight not detected at all"
    assert (flagged < 0).all(), "a descent was labelled the same as a climb"


# --------------------------------------------------------------------------
# what the map does not know, the detector does not claim


def test_nothing_is_flagged_past_the_frontier():
    elev = flight()
    valid = np.isfinite(elev)
    valid[:, 100:] = False          # the flight self-occludes above here
    conf = run(elev, valid)
    assert np.isnan(conf[:, 100:]).all()


def test_noise_beyond_max_range_is_ignored():
    rng = np.random.default_rng(1)
    elev = GROUND + rng.normal(0, 0.04, (240, 240)).astype(np.float32)
    conf = run(elev, max_range=2.0)
    n = 240
    idx = np.arange(n) - n / 2 + 0.5
    dist = np.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2) * RES
    assert frac_flagged(conf, dist > 2.5) == 0.0


def test_an_obstacle_on_a_tread_is_not_filled_in():
    elev = flight()
    tread_x = 60 + int(0.40 / RES) // 2
    elev[50:60, tread_x - 4:tread_x + 4] = GROUND + 1.2   # someone standing
    conf = run(elev)
    assert frac_flagged(conf, np.s_[52:58, tread_x - 2:tread_x + 2]) == 0.0


def test_a_flight_between_walls_survives():
    """The old veto killed every stairwell; this is the regression guard."""
    elev = flight(width=1.1)
    n = elev.shape[0]
    y0 = n // 2 - int(1.1 / RES) // 2
    y1 = y0 + int(1.1 / RES)
    elev[y0 - 6:y0, 60:] = GROUND + 2.0      # walls either side
    elev[y1:y1 + 6, 60:] = GROUND + 2.0
    conf = run(elev)
    inside = np.zeros_like(conf, dtype=bool)
    inside[y0 + 3:y1 - 3, 62:90] = True
    assert frac_flagged(conf, inside) >= 0.7


# --------------------------------------------------------------------------
# ramps are their own thing, and the two layers do not overlap


def _load_ramp():
    path = Path(__file__).resolve().parents[1] / (
        "elevation_mapping_cupy/plugins/ramp_filter.py"
    )
    src = path.read_text().replace(
        "from elevation_mapping_cupy.plugins.stairs_filter import "
        "_climb_sign, _gpu_modules, _host",
        "_climb_sign, _gpu_modules, _host = DET._climb_sign, DET._gpu_modules, DET._host",
    )
    ns = {"DET": DET, "__name__": "ramp_filter_under_test"}
    exec(compile(src, str(path), "exec"), ns)
    return ns


RAMP = _load_ramp()


def run_ramp(elev, valid=None, **params):
    if valid is None:
        valid = np.isfinite(elev)
    step, slope, roughness = layers(elev, valid)
    e = np.where(valid, elev, 0.0).astype(np.float32)
    return RAMP["detect_ramps"](e, valid, step, slope, roughness, RES, params)


@pytest.mark.parametrize("angle", [12, 20, 25, 30, 40])
def test_a_bank_is_a_ramp(angle):
    conf = run_ramp(plane(angle))
    assert frac_flagged(conf, sign=+1) >= 0.5, f"{angle} deg bank not seen as a ramp"


def test_flat_ground_is_not_a_ramp():
    elev = np.full((120, 120), GROUND, dtype=np.float32)
    assert frac_flagged(run_ramp(elev)) == 0.0


def test_a_staircase_is_not_a_ramp():
    """The layers have to be exclusive or the supervisor gets both answers."""
    assert frac_flagged(run_ramp(flight())) == 0.0


def test_a_ramp_is_not_a_staircase():
    for angle in (12, 25, 35):
        assert frac_flagged(run(plane(angle))) == 0.0


def test_rubble_is_not_a_ramp():
    rng = np.random.default_rng(2)
    elev = GROUND + rng.uniform(-0.10, 0.10, (120, 120)).astype(np.float32)
    assert frac_flagged(run_ramp(elev)) == 0.0


def test_a_descending_bank_is_negative():
    # The robot stands on the flat and the bank falls away ahead of it. A
    # plane running through the robot would be genuinely ambiguous -- uphill
    # one way, downhill the other -- and the detector says so by staying
    # positive, which is the safe reading for ground it is standing on.
    n = 120
    elev = np.full((n, n), GROUND, dtype=np.float32)
    x = (np.arange(n) - n // 2) * RES
    elev += np.where(x > 0, -np.tan(np.radians(25)) * x, 0.0)[None, :].astype(np.float32)
    conf = run_ramp(elev)
    fin = np.isfinite(conf)
    flagged = conf[fin][np.abs(conf[fin]) > 0.5]
    assert flagged.size > 0 and (flagged < 0).all()
