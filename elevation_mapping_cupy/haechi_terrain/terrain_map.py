"""(haechi_terrain.terrain_map; CLI: scripts/terrain_map_merge.py) Merge the terrain PRIOR into the SLAM 2D map -> map_terrain.pgm/yaml, the elevation stack's 2D map of a map folder.

A mapping run leaves NAVI's artifacts in ~/map_folder/<map>/<submap>/ (map.pcd, map.pgm + map.yaml from the octomap
projection, marker_map.yaml, hybrid: map_global.yaml) and, with `haechi.launch.py prior_record:=true`, the terrain prior
(map_terrain_prior.pgm/yaml + terrain_prior.npz, terrain_prior_recorder). This script puts the two together on the
geometry of map.yaml (origin, resolution, size), so the result is a drop-in Nav2 map file beside the SLAM one:

  map_terrain.png / map_terrain.yaml   mode scale, 0 free .. 89 terrain cost (capped, never a wall) .. 100 SLAM wall,
                                       unknown = transparent (alpha 0). PNG, not PGM: nav2's map_io has no unknown
                                       pixel in scale mode (205 would come back as cost 19); only alpha < 1 is unknown.

Per cell (same rule as terrain_global_node's 'override' merge, prior-only):
  SLAM wall (map.pgm occupied)   -> 100, whatever the terrain says (the terrain never sees a wall face)
  terrain prior judged the cell  -> prior cost (0..cap), replacing SLAM free/unknown
  otherwise                      -> the SLAM value (free 0 / unknown)

map.pgm is never modified (decision 2026-10-21): the robot's default nav2 keeps loading map.yaml, the elevation
variant loads map_terrain.yaml (nav2 `map:=`) or hands map_terrain_prior.yaml to terrain_global_node.

The yaml thresholds are chosen so nav2's map_server (mode scale: occ = (255 - pixel) / 255) returns the cost back:
occupied_thresh 0.995 makes only pixel 0 (SLAM wall) lethal 100, free_thresh 0.005 makes pixel 254 free 0, everything
between comes back as the baked cost (+-1). With the Humble default 0.65 a prior of 89 would read as a wall. Verified
with nav2_map_server (Humble, 2026-10-21): wall -> 100, alpha 0 -> -1, cost 89 -> 89, free -> 0. The global costmap
then needs trinary_costmap false (nav2_params_elevation.yaml). (For comparison, NAVI's own map.yaml with free_thresh
0.25 makes map_server read its 205 unknown pixels as FREE.)

  terrain_map_merge.py <map_dir>                       # map.yaml + map_terrain_prior.yaml in that folder -> map_terrain.*
  terrain_map_merge.py <map_dir> --prior <yaml> --out <stem> --cap 89   # stem -> <stem>.png + <stem>.yaml
  terrain_map_merge.py <map_dir> --map map_slam.yaml   # another SLAM map file in the folder

Runs automatically at the end of terrain_prior_recorder's save when map.yaml already exists in its out_dir, and from
the CLI any time later (the SLAM pgm is written by laserMapping at mapping shutdown, so the recorder may be first).
"""
import os

import numpy as np

from . import config
from .stairs_raster import rasterise

UNKNOWN_PIXEL = 205
OCCUPIED_THRESH = 0.995
FREE_THRESH = 0.005


def _read_map(yaml_path):
    """Nav2 map yaml + pgm -> (cost int16 HxW with row 0 = bottom, -1 unknown, 0 free, 100 wall or 1..99 scaled), info."""
    import yaml
    from PIL import Image
    y = yaml.safe_load(open(yaml_path))
    img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), y["image"])))[::-1]
    transparent = None
    if img.ndim == 3:
        if img.shape[2] == 4:
            transparent = img[..., 3] == 0
        img = img[..., 0]
    if int(y.get("negate", 0)):
        img = 255 - img
    cost = np.full(img.shape, -1, np.int16)
    if str(y.get("mode", "trinary")) == "scale":
        k = (img != UNKNOWN_PIXEL) if transparent is None else ~transparent
        cost[k] = np.round(100.0 * (1.0 - img[k] / 254.0)).astype(np.int16)
    else:
        # pixel classes as laserMapping writes them (0 wall / 205 unknown / 254 free), like terrain_global_node's loader;
        # the yaml thresholds are nav2's reading, and with free_thresh 0.25 nav2 itself reads 205 as free
        cost[img < 50] = 100
        cost[img > 230] = 0
    return cost, y


def _write_map(stem_path, cost, info):
    """cost (-1 unknown / 0..100, row 0 = bottom) -> <stem>.pgm + <stem>.yaml (mode scale, thresholds see module doc)."""
    from PIL import Image
    img = np.full(cost.shape, UNKNOWN_PIXEL, np.uint8)
    k = cost >= 0
    img[k] = np.round(254.0 * (1.0 - np.clip(cost[k], 0, 100) / 100.0)).astype(np.uint8)
    rgba = np.zeros(cost.shape + (4,), np.uint8)
    rgba[..., :3] = img[..., None]
    rgba[..., 3] = np.where(k, 255, 0).astype(np.uint8)       # unknown = transparent (nav2 scale mode)
    Image.fromarray(rgba[::-1], "RGBA").save(stem_path + ".png")
    o = info["origin"]
    with open(stem_path + ".yaml", "w") as f:
        f.write("image: %s\nmode: scale\nresolution: %.4f\norigin:\n- %.4f\n- %.4f\n- %.4f\nnegate: 0\n"
                "occupied_thresh: %.3f\nfree_thresh: %.3f\n" % (os.path.basename(stem_path) + ".png", float(info["resolution"]),
                                                               float(o[0]), float(o[1]), float(o[2]) if len(o) > 2 else 0.0,
                                                               OCCUPIED_THRESH, FREE_THRESH))


def _resample(cost, info_src, shape, info_dst):
    """Nearest-cell resample of a cost grid onto another map geometry; -1 where the source has no cell."""
    H, W = shape
    res_d = float(info_dst["resolution"]); ox_d, oy_d = float(info_dst["origin"][0]), float(info_dst["origin"][1])
    res_s = float(info_src["resolution"]); ox_s, oy_s = float(info_src["origin"][0]), float(info_src["origin"][1])
    rows, cols = np.mgrid[0:H, 0:W]
    X = ox_d + (cols + 0.5) * res_d; Y = oy_d + (rows + 0.5) * res_d
    gi = np.floor((Y - oy_s) / res_s).astype(int); gj = np.floor((X - ox_s) / res_s).astype(int)
    ok = (gi >= 0) & (gi < cost.shape[0]) & (gj >= 0) & (gj < cost.shape[1])
    out = np.full((H, W), -1, np.int16)
    out[ok] = cost[gi[ok], gj[ok]]
    return out


def stairs_lane_cost(stairs_yaml, X, Y, base=8, edge=45):
    """Lane-centre gradient inside the registered flights and their end margins: cost = base at the flight centreline,
    rising linearly to `edge` at the lateral limits. NavFn then runs the global path down the middle of every flight and
    arrives square to the first riser (the A-3 wedge, 2026-10-29: a diagonal entry from the landing). Returns float HxW
    with NaN where no flight applies."""
    import yaml
    d = yaml.safe_load(open(stairs_yaml)); out = np.full(X.shape, np.nan)
    for f in d.get("flights", []):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        s = (X - p0[0]) * ax[0] + (Y - p0[1]) * ax[1]; l = (X - p0[0]) * lat[0] + (Y - p0[1]) * lat[1]
        s1 = f["s0"] + (f["n"] - 1) * f["tread"] + 0.20
        lc = 0.5 * (f["l_lo"] + f["l_hi"]); half = max(0.3, 0.5 * (f["l_hi"] - f["l_lo"]))
        m = (s >= f["s0"] - 1.5) & (s <= s1 + 1.5) & (l >= f["l_lo"]) & (l <= f["l_hi"])
        c = base + (edge - base) * np.clip(np.abs(l - lc) / half, 0, 1)
        out[m] = np.where(np.isnan(out[m]), c[m], np.minimum(out[m], c[m]))
    return out


def stairs_cells(stairs_yaml, map_yaml, shape, info, lateral_margin=0.0):
    """bool HxW on the map geometry: registered stair segments (flights + landings + margins, walls cut) of map_stairs.yaml."""
    import yaml
    d = yaml.safe_load(open(stairs_yaml))
    if not d.get("flights"):
        return np.zeros(shape, bool)
    res = float(info["resolution"])
    grid, x0, y0, _, _ = rasterise(d["flights"], d.get("staircases", []), res, map_yaml=map_yaml, lateral_margin=lateral_margin)
    ox, oy = float(info["origin"][0]), float(info["origin"][1])
    out = np.zeros(shape, bool)
    yy, xx = np.nonzero(grid >= 50)
    i = np.round((y0 + (yy + 0.5) * res - oy) / res - 0.5).astype(int); j = np.round((x0 + (xx + 0.5) * res - ox) / res - 0.5).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < shape[0]) & (j < shape[1]); out[i[ok], j[ok]] = True
    return out


def merge_terrain_map(map_yaml, prior_yaml, out_stem, cap=None, log=print, stairs_yaml="", stairs_cost=None, speckle_cells=None, speckle_prior_max=None,
                      stairs_lateral_margin=None, lane_gradient=True, swath_r=None, lane_edge=None):
    """Permissive global map (2026-10-29): the global planner only needs the certain walls; whether a cell is really
    passable is decided by the live elevation when the robot gets there. So: SLAM occupied clusters up to speckle_cells
    whose terrain prior is at most speckle_prior_max are dropped (projection noise, low clutter), registered stairs are
    written at their full measured width (the local grid keeps its own lateral margin), and the prior stays a cost, never
    a wall (cap)."""
    """Write <out_stem>.pgm/.yaml; returns a dict of counts. Raises on missing inputs.
    stairs_yaml (map_stairs.yaml registered at mapping time): its segments are written at stairs_cost, over the SLAM wall
    (the 2D projection draws flights as occupied) and over the prior (which judges risers non-drivable). Without this the
    staircase was a wall in the global map and the planner only got through where the live terrain happened to override it.
    Numbers left None come from config/setups/haechi/stairs.yaml (terrain_map_merge section)."""
    m = config.load("terrain_map_merge")
    cap = int(m.get("cap", 89)) if cap is None else cap
    stairs_cost = int(m.get("stairs_cost", 8)) if stairs_cost is None else stairs_cost
    lane_edge = int(m.get("lane_edge", 45)) if lane_edge is None else lane_edge
    speckle_cells = int(m.get("speckle_cells", 20)) if speckle_cells is None else speckle_cells
    speckle_prior_max = int(m.get("speckle_prior_max", 60)) if speckle_prior_max is None else speckle_prior_max
    stairs_lateral_margin = float(m.get("lateral_margin", 0.0)) if stairs_lateral_margin is None else stairs_lateral_margin
    swath_r = float(m.get("swath_r", 0.23)) if swath_r is None else swath_r
    slam, info = _read_map(map_yaml)
    prior, pinfo = _read_map(prior_yaml)
    prior[prior >= 100] = -1               # walls copied into a bake are the SLAM map's business
    prior[prior > cap] = cap
    prior = _resample(prior, pinfo, slam.shape, info)
    merged = slam.copy()
    judged = prior >= 0
    merged[judged] = prior[judged]
    wall = slam >= 100
    # 2D-projection speckle (2026-10-29, 1f live map): small occupied clusters on open floor where the terrain prior judged
    # the ground drivable are projection noise, not walls (33 % of the floor-box occupied cells were clusters <= 8 cells,
    # 79 % of those with prior <= 30; real walls/pillars are large clusters with the prior at cap or unjudged). They are
    # dropped; everything else the SLAM map calls a wall stays a wall.
    n_speckle = 0
    if speckle_cells > 0:
        from scipy import ndimage
        lab, nlab = ndimage.label(wall, structure=np.ones((3, 3)))
        if nlab:
            size = np.bincount(lab.ravel()); small = (size[lab] <= speckle_cells) & wall
            drop = small & judged & (prior <= speckle_prior_max)
            n_speckle = int(drop.sum()); wall = wall & ~drop
            merged[drop] = prior[drop]
    # the mapping robot's own swath: SLAM occupied cells within swath_r of the mapping trajectory (stairs_record.npz, same
    # folder) are projection noise / the operator (the robot walked there) -> not walls. On the 1f real map 46 such cells in
    # 20 clusters blocked 12 % of the real route's footprint (B terrace where the robot stood 25 s); without them 2.3 %.
    n_swath = 0
    # mapping_path.npz (written by stairs_fit, stays with the map) or the full record while it still exists
    rec = next((p for p in (os.path.join(os.path.dirname(os.path.abspath(map_yaml)), n) for n in ("mapping_path.npz", "stairs_record.npz")) if os.path.isfile(p)), "")
    if swath_r > 0 and rec:
        try:
            from scipy import ndimage
            L = np.load(rec)["lidar"]; res_ = float(info["resolution"]); ox_, oy_ = float(info["origin"][0]), float(info["origin"][1])
            sw = np.zeros(slam.shape, bool)
            i = np.round((L[:, 1] - oy_) / res_ - 0.5).astype(int); j = np.round((L[:, 0] - ox_) / res_ - 0.5).astype(int)
            ok = (i >= 0) & (j >= 0) & (i < sw.shape[0]) & (j < sw.shape[1]); sw[i[ok], j[ok]] = True
            k = int(np.ceil(swath_r / res_)); yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
            sw = ndimage.binary_dilation(sw, structure=(xx * xx + yy * yy) <= k * k)
            drop = wall & sw; n_swath = int(drop.sum()); wall = wall & ~drop
            merged[drop] = np.where(judged[drop], prior[drop], 0)
            if n_swath:
                log("swath: %d SLAM occupied cells within %.2f m of the mapping trajectory dropped (the robot walked there)" % (n_swath, swath_r))
        except Exception as e:  # noqa: BLE001
            log("swath rule skipped: %s" % e)
    merged[wall] = 100
    n_stairs = 0
    if stairs_yaml and os.path.isfile(stairs_yaml):
        st = stairs_cells(stairs_yaml, map_yaml, slam.shape, info, stairs_lateral_margin)
        n_stairs = int(st.sum()); merged[st] = stairs_cost
        if lane_gradient:
            res_ = float(info["resolution"]); ox_, oy_ = float(info["origin"][0]), float(info["origin"][1])
            yy_, xx_ = np.mgrid[0:slam.shape[0], 0:slam.shape[1]]
            lane = stairs_lane_cost(stairs_yaml, ox_ + (xx_ + 0.5) * res_, oy_ + (yy_ + 0.5) * res_, base=stairs_cost, edge=lane_edge)
            k = st & np.isfinite(lane); merged[k] = np.round(lane[k]).astype(merged.dtype)
            log("stairs: lane-centre gradient %d..%d applied on %d of the stair cells" % (stairs_cost, lane_edge, int(k.sum())))
        log("stairs: %d cells of %s written at cost %d (%d of them were SLAM walls)" % (n_stairs, os.path.basename(stairs_yaml), stairs_cost, int((st & wall).sum())))
    if n_speckle:
        log("speckle: %d SLAM occupied cells in clusters <= %d with prior <= %d dropped" % (n_speckle, speckle_cells, speckle_prior_max))
    _write_map(out_stem, merged, info)
    stats = dict(width=int(slam.shape[1]), height=int(slam.shape[0]), resolution=float(info["resolution"]),
                 slam_wall=int(wall.sum()), slam_free=int((slam == 0).sum()), slam_unknown=int((slam < 0).sum()),
                 terrain_judged=int(judged.sum()), terrain_over_free=int((judged & (slam == 0)).sum()),
                 terrain_over_unknown=int((judged & (slam < 0)).sum()), terrain_cost_cells=int((judged & (prior > 0)).sum()),
                 terrain_at_cap=int((judged & (prior >= cap)).sum()), stairs=n_stairs, out=out_stem + ".yaml")
    log("map_terrain: %dx%d @ %.3f m -> %s | SLAM wall %d free %d unknown %d | terrain judged %d (over free %d, over unknown %d), "
        "cost>0 %d, at cap %d" % (stats["width"], stats["height"], stats["resolution"], stats["out"], stats["slam_wall"], stats["slam_free"],
                                  stats["slam_unknown"], stats["terrain_judged"], stats["terrain_over_free"], stats["terrain_over_unknown"],
                                  stats["terrain_cost_cells"], stats["terrain_at_cap"]))
    return stats
