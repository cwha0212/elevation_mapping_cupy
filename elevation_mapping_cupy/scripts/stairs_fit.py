#!/usr/bin/env python3
"""Register the stair flights of a MAPPING run from its stairs_record.npz (CLI of haechi_terrain.stairs_fit).

  stairs_fit.py --record <map_dir>/stairs_record.npz --out <map_dir> [--delete_record]

Writes <out>/map_stairs.yaml + map_stairs.png + stairs_fit.log (+ mapping_path.npz), rebuilds map_terrain.* with the
stairs when map.yaml and map_terrain_prior.yaml are in <out>. Started by stairs_prior_recorder at save time; run it by
hand to refit a kept record. The method is documented in haechi_terrain/stairs_fit.py.
"""
import argparse
import os
import time

import numpy as np

from haechi_terrain.stairs_fit import cfg, clean_points, figure, fit_run, write_yaml
from haechi_terrain.terrain_map import merge_terrain_map

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True, help="stairs_record.npz written by stairs_prior_recorder")
    ap.add_argument("--out", required=True, help="directory for map_stairs.yaml / map_stairs.png")
    ap.add_argument("--sensor_height", type=float, default=None)
    ap.add_argument("--delete_record", action="store_true",
                    help="the record is a temporary file: remove it once map_stairs.yaml/png are written (kept on failure)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # stairs_fit.log is written here, next to map_stairs.yaml, by every run (recorder-started or offline), so the log
    # always belongs to the yaml beside it
    logf = open(os.path.join(a.out, "stairs_fit.log"), "w")

    def log(*m):
        s = " ".join(str(v) for v in m); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log("stairs_fit %s  record %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), os.path.abspath(a.record)))
    d = np.load(a.record)
    c = cfg(sensor_height=a.sensor_height if a.sensor_height is not None else float(d["sensor_height"]))
    t0 = time.time()
    P = d["points"].astype(np.float64)
    if "frames" in d.files:
        P = P[clean_points(P, d["frames"].astype(np.int64), d["t_first"], d["t_last"], d["t"], d["lidar"], c, log=log)]
    fits, scs, info = fit_run(d["t"], d["lidar"], P, c, log=log)
    write_yaml(os.path.join(a.out, "map_stairs.yaml"), fits, scs, dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), source=os.path.abspath(a.record),
                                                                        sensor_height=c.sensor_height, **{k: v for k, v in info.items()}))
    figure(os.path.join(a.out, "map_stairs.png"), fits, scs, d["t"], d["lidar"], P, c)
    log("registered %d flights in %d staircases (walked %d; not registered: %d ramp, %d too little data) -> %s (%.1f s)" % (
        len(fits), len(scs), info.get("flights_walked", 0), info.get("not_registered_ramp", 0), info.get("not_registered_no_data", 0), a.out, time.time() - t0))
    # the mapping bake (terrain_prior_recorder -> map_terrain.*) ran before this fit: redo it with the stairs in
    map_yaml = os.path.join(a.out, "map.yaml"); prior_yaml = os.path.join(a.out, "map_terrain_prior.yaml")
    if fits and os.path.isfile(map_yaml) and os.path.isfile(prior_yaml):
        merge_terrain_map(map_yaml, prior_yaml, os.path.join(a.out, "map_terrain"), stairs_yaml=os.path.join(a.out, "map_stairs.yaml"), log=log)
        log("map_terrain.png/yaml rebuilt with the registered stairs")
    # the mapping path stays beside the map (small): terrain_map's swath rule reads it on a later re-merge
    np.savez_compressed(os.path.join(a.out, "mapping_path.npz"), t=d["t"].astype(np.float32), t0=d["t0"], lidar=d["lidar"].astype(np.float32),
                        sensor_height=d["sensor_height"])
    if a.delete_record:
        os.remove(a.record); log("record removed (temporary file): %s" % a.record)
    logf.close()


if __name__ == "__main__":
    main()
