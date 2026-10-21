#!/usr/bin/env python3
"""Merge the terrain PRIOR into the SLAM 2D map -> map_terrain.pgm/yaml, the elevation stack's 2D map of a map folder.

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
import argparse
import os
import sys

import numpy as np

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


def merge_terrain_map(map_yaml, prior_yaml, out_stem, cap=89, log=print):
    """Write <out_stem>.pgm/.yaml; returns a dict of counts. Raises on missing inputs."""
    slam, info = _read_map(map_yaml)
    prior, pinfo = _read_map(prior_yaml)
    prior[prior >= 100] = -1               # walls copied into a bake are the SLAM map's business
    prior[prior > cap] = cap
    prior = _resample(prior, pinfo, slam.shape, info)
    merged = slam.copy()
    judged = prior >= 0
    merged[judged] = prior[judged]
    wall = slam >= 100
    merged[wall] = 100
    _write_map(out_stem, merged, info)
    stats = dict(width=int(slam.shape[1]), height=int(slam.shape[0]), resolution=float(info["resolution"]),
                 slam_wall=int(wall.sum()), slam_free=int((slam == 0).sum()), slam_unknown=int((slam < 0).sum()),
                 terrain_judged=int(judged.sum()), terrain_over_free=int((judged & (slam == 0)).sum()),
                 terrain_over_unknown=int((judged & (slam < 0)).sum()), terrain_cost_cells=int((judged & (prior > 0)).sum()),
                 terrain_at_cap=int((judged & (prior >= cap)).sum()), out=out_stem + ".yaml")
    log("map_terrain: %dx%d @ %.3f m -> %s | SLAM wall %d free %d unknown %d | terrain judged %d (over free %d, over unknown %d), "
        "cost>0 %d, at cap %d" % (stats["width"], stats["height"], stats["resolution"], stats["out"], stats["slam_wall"], stats["slam_free"],
                                  stats["slam_unknown"], stats["terrain_judged"], stats["terrain_over_free"], stats["terrain_over_unknown"],
                                  stats["terrain_cost_cells"], stats["terrain_at_cap"]))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("map_dir", help="map folder (~/map_folder/<map>/<submap>) holding map.yaml and map_terrain_prior.yaml")
    ap.add_argument("--map", default="map.yaml", help="SLAM 2D map yaml in map_dir (default map.yaml)")
    ap.add_argument("--prior", default="map_terrain_prior.yaml", help="terrain prior yaml, relative to map_dir or absolute")
    ap.add_argument("--out", default="map_terrain", help="output stem in map_dir (default map_terrain -> map_terrain.png/yaml)")
    ap.add_argument("--cap", type=int, default=89, help="max terrain cost (never a wall), default 89")
    a = ap.parse_args(argv)
    d = os.path.abspath(os.path.expanduser(a.map_dir))
    map_yaml = a.map if os.path.isabs(a.map) else os.path.join(d, a.map)
    prior_yaml = a.prior if os.path.isabs(a.prior) else os.path.join(d, a.prior)
    for p in (map_yaml, prior_yaml):
        if not os.path.isfile(p):
            print("terrain_map_merge: missing %s" % p, file=sys.stderr)
            return 2
    merge_terrain_map(map_yaml, prior_yaml, os.path.join(d, a.out), cap=a.cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
