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

from haechi_terrain.terrain_map import merge_terrain_map

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("map_dir", help="map folder (~/map_folder/<map>/<submap>) holding map.yaml and map_terrain_prior.yaml")
    ap.add_argument("--map", default="map.yaml", help="SLAM 2D map yaml in map_dir (default map.yaml)")
    ap.add_argument("--prior", default="map_terrain_prior.yaml", help="terrain prior yaml, relative to map_dir or absolute")
    ap.add_argument("--out", default="map_terrain", help="output stem in map_dir (default map_terrain -> map_terrain.png/yaml)")
    ap.add_argument("--stairs", default=None, help="map_stairs.yaml (default: <map_dir>/map_stairs.yaml if present)")
    ap.add_argument("--stairs_cost", type=int, default=None, help="default: stairs.yaml merge.stairs_cost")
    ap.add_argument("--cap", type=int, default=None, help="max terrain cost (never a wall); default: stairs.yaml merge.cap")
    a = ap.parse_args(argv)
    d = os.path.abspath(os.path.expanduser(a.map_dir))
    map_yaml = a.map if os.path.isabs(a.map) else os.path.join(d, a.map)
    prior_yaml = a.prior if os.path.isabs(a.prior) else os.path.join(d, a.prior)
    for p in (map_yaml, prior_yaml):
        if not os.path.isfile(p):
            print("terrain_map_merge: missing %s" % p, file=sys.stderr)
            return 2
    st = a.stairs if a.stairs is not None else os.path.join(d, "map_stairs.yaml")
    merge_terrain_map(map_yaml, prior_yaml, os.path.join(d, a.out), cap=a.cap, stairs_yaml=st if os.path.isfile(st) else "", stairs_cost=a.stairs_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
