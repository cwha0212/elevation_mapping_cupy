"""Rasterise the stairs registered at mapping time (map_stairs.yaml) into an OccupancyGrid (>= 50 = stair segment).

Used by scripts/stairs_prior_publisher.py (the node that publishes it for terrain_grid_node) and by terrain_map (the bake
of the registered stairs into map_terrain). What a segment is and why walls are cut: see the node's docstring.
"""
import json
import os

import numpy as np
from matplotlib.path import Path as MPath


def load(path):
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        import yaml
        return yaml.safe_load(f)


def s_end(f):
    return f["s0"] + (f["n"] - 1) * f["tread"] + 0.20


def wall_mask(map_yaml, X, Y, margin):
    """True where the SLAM 2D map is occupied within `margin` (m) of the cell (X, Y)."""
    import yaml
    from PIL import Image
    from scipy import ndimage
    my = yaml.safe_load(open(map_yaml))
    img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), my["image"])))[::-1]
    occ = img < 255 * (1.0 - my.get("occupied_thresh", 0.65)) if not my.get("negate", 0) else img > 255 * my.get("occupied_thresh", 0.65)
    k = int(np.ceil(margin / my["resolution"]))
    if k > 0:
        occ = ndimage.binary_dilation(occ, iterations=k)
    ox, oy = my["origin"][:2]; r = my["resolution"]
    i = np.floor((Y - oy) / r).astype(np.int64); j = np.floor((X - ox) / r).astype(np.int64)
    ok = (i >= 0) & (j >= 0) & (i < occ.shape[0]) & (j < occ.shape[1])
    out = np.zeros(X.shape, bool); out[ok] = occ[i[ok], j[ok]]
    return out


def rasterise(flights, staircases, res=0.05, end_margin=1.0, max_landing=2.5, flight_margin=0.3, lateral_margin=0.25,
              map_yaml="", wall_margin=0.15):
    """flight_margin: every flight footprint is extended this far along its axis at both ends. The mapping-time fit places
    the first riser within 1.5-13 cm and the axis within 0-10 deg of the offline reference (1f bag): unextended, the riser
    edge cells it missed stayed lethal and stopped the robot on the A-2/A-3 landing (sim, 2026-10-29)."""
    start_x = {}; end_x = {}
    groups = [[i - 1 for i in sc["flights"]] for sc in staircases] or [[i] for i in range(len(flights))]
    for ids in groups:
        start_x[ids[0]] = end_margin; end_x[ids[-1]] = end_margin
    for k in range(len(flights)):
        start_x[k] = max(start_x.get(k, 0.0), flight_margin); end_x[k] = max(end_x.get(k, 0.0), flight_margin)
    corners = []
    for k, f in enumerate(flights):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        for s in (f["s0"] - start_x.get(k, 0.0), s_end(f) + end_x.get(k, 0.0)):
            for l in (f["l_lo"], f["l_hi"]):
                corners.append(p0 + ax * s + lat * l)
    c = np.array(corners)
    x0, y0 = np.floor(c.min(0) / res) * res - 1.0
    x1, y1 = np.ceil(c.max(0) / res) * res + 1.0
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    yy, xx = np.mgrid[0:H, 0:W]
    X = x0 + (xx + 0.5) * res; Y = y0 + (yy + 0.5) * res
    grid = np.zeros((H, W), np.int8)
    for k, f in enumerate(flights):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        s = (X - p0[0]) * ax[0] + (Y - p0[1]) * ax[1]; l = (X - p0[0]) * lat[0] + (Y - p0[1]) * lat[1]
        grid[(s >= f["s0"] - start_x.get(k, 0.0)) & (s <= s_end(f) + end_x.get(k, 0.0)) & (l >= f["l_lo"] + lateral_margin) & (l <= f["l_hi"] - lateral_margin)] = 100
    P = np.stack([X.ravel(), Y.ravel()], 1); n_land = 0
    land = np.zeros((H, W), bool)                          # landings: part of the staircase core (see wall cut below)
    for ids in groups:
        for i, j in zip(ids, ids[1:]):
            fa, fb = flights[i], flights[j]
            pa, axa, la = np.array(fa["p0"]), np.array(fa["axis"]), np.array(fa["lat"])
            pb, axb, lb = np.array(fb["p0"]), np.array(fb["axis"]), np.array(fb["lat"])
            top = pa + axa * s_end(fa); bot = pb + axb * fb["s0"]
            if np.linalg.norm(bot - top) > max_landing:
                continue
            e = [(pb + lb * fb["l_lo"] - pa) @ la, (pb + lb * fb["l_hi"] - pa) @ la]
            lo, hi = max(fa["l_lo"], min(e)) + lateral_margin, min(fa["l_hi"], max(e)) - lateral_margin
            if hi - lo < 0.5:
                continue
            ta = (top - pa) @ la; tb = (bot - pa) @ la
            poly = [top + la * (lo - ta), top + la * (hi - ta), bot + la * (hi - tb), bot + la * (lo - tb)]
            lm = MPath(np.array(poly)).contains_points(P).reshape(H, W); grid[lm] = 100; land |= lm; n_land += 1
    n_wall = 0
    if map_yaml and os.path.isfile(map_yaml):
        core = land.copy()                                 # flights + landings: the 2D map draws risers/landing edges occupied there
        for f in flights:
            p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
            s = (X - p0[0]) * ax[0] + (Y - p0[1]) * ax[1]; l = (X - p0[0]) * lat[0] + (Y - p0[1]) * lat[1]
            core |= (s >= f["s0"] - 0.1) & (s <= s_end(f) + 0.1) & (l >= f["l_lo"] + lateral_margin) & (l <= f["l_hi"] - lateral_margin)
        w = wall_mask(map_yaml, X, Y, wall_margin) & ~core
        n_wall = int((w & (grid >= 50)).sum()); grid[w] = 0
    return grid, x0, y0, n_land, n_wall
