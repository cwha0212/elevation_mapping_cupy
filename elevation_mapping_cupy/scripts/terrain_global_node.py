#!/usr/bin/env python3
"""Accumulate the rolling terrain grid into a global map for Nav2's GLOBAL costmap.

The elevation map is a 10 m window around the robot, so handing /terrain/local_grid to the global costmap would
forget everything the robot has passed. This node keeps a last-write memory of the terrain grid on the SAME
geometry as the Nav2 map (origin, resolution, size of /map from map_server), merges it with /map (max of the two
costs, so a static obstacle in the loaded map survives and the terrain adds what it judged) and publishes it
transient-local as /terrain/global_grid. The global costmap's static_layer reads that instead of /map; the
robot's map_server / load_map flow stays as it is (every new /map re-seeds the merge).

Parameters
  input_topic   /terrain/local_grid          output_topic  /terrain/global_grid
  map_topic     /map (map_server, transient local; its geometry defines the global grid)
  mode          'last' (last judgement wins, default) | 'max' (never forgets a lethal cell)
  merge         'override' (default): a cell the terrain has judged REPLACES the /map value, so a dirty prior map
                (people, noise, moved objects baked into the SLAM pgm) is cleaned as the robot re-observes it and a
                static obstacle the terrain never saw stays; 'max': old behaviour, /map lethal is permanent
  rate_hz       1.0   publish rate (full-map message; keep low)            unknown_as    -1 keeps -1; set 0 to publish unknown as free
  forget_s      0.0   > 0: a terrain cell not re-seen for this long falls back to the /map value
  prior_file    ''    a terrain PRIOR made at mapping time (map_terrain_prior.yaml + pgm, mode scale, 0..100 = cost; see
                      terrain_map_bake.py): the 2D grid the elevation stack plans on before the robot has re-observed a
                      cell. Merge order per cell: live terrain (judged) > prior > /map. Prior values are capped at
                      prior_cap (89): a baked prior is never a wall, only /map (SLAM walls) and live terrain/scan are.
                      Prior cells >= 100 (the SLAM walls copied into the bake) are dropped and left to /map.
"""
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class TerrainGlobal(Node):
    def __init__(self):
        super().__init__("terrain_global_node")
        self.declare_parameter("input_topic", "/terrain/local_grid")
        self.declare_parameter("output_topic", "/terrain/global_grid")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("mode", "last")
        # 1 Hz: the grid is the full /map (2.6 MB for ds1020 at 0.05 m); 2 Hz overflowed the UDP receive buffer of reliable
        # subscribers sharing the host (RcvbufErrors, 100 s stalls, 2026-10-21) and the global costmap only updates at 1 Hz.
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("forget_s", 0.0)
        self.declare_parameter("lethal_votes", 2)
        # Value published for lethal terrain in the GLOBAL grid. 99 (default): StaticLayer (trinary off) scales it to 251,
        # below the inscribed 253 that NavFn treats as a wall, so the planner routes AROUND terrain at high cost but a
        # narrow lane never seals the robot in (ds1020 start: 1 m lane, GUI run 2026-10-20). The hard constraint stays
        # in the local costmap's terrain layer (true 100) and in the global /scan obstacle layer. 100 = old behaviour.
        self.declare_parameter("lethal_as", 99)
        self.declare_parameter("merge", "override")
        self.declare_parameter("prior_file", "")
        self.declare_parameter("prior_cap", 89)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode, self.forget = str(g("mode")), float(g("forget_s"))
        self.lethal_votes = int(g("lethal_votes")); self.lethal_as = int(g("lethal_as")); self.merge = str(g("merge"))
        self.prior_src = self._load_prior(str(g("prior_file")), int(g("prior_cap")))   # (cost HxW int16, ox, oy, res) or None
        self.prior = None        # prior resampled onto the /map geometry (int8, -1 none)
        self.map = None          # nav map (int8 HxW), geometry
        self.terr = None         # accumulated terrain cost, -1 unknown
        self.seen = None         # last time a terrain value was written
        self.n_in = 0
        tl = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, str(g("output_topic")), tl)
        self.create_subscription(OccupancyGrid, str(g("map_topic")), self.on_map, tl)
        self.create_subscription(OccupancyGrid, str(g("input_topic")), self.on_terrain, 1)
        self.create_timer(1.0 / float(g("rate_hz")), self.publish)
        from std_srvs.srv import Trigger
        self.create_service(Trigger, "/terrain/global_grid/reset", self.on_reset)
        self.get_logger().info("%s + %s -> %s (mode %s)" % (g("input_topic"), g("map_topic"), g("output_topic"), self.mode))

    def _load_prior(self, path, cap):
        if not path:
            return None
        try:
            import yaml
            from PIL import Image
            y = yaml.safe_load(open(path))
            img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(path)), y["image"])))[::-1]   # row 0 = bottom
            known = None
            if img.ndim == 3:                               # map_terrain.png (terrain_map_merge.py): alpha 0 = unknown
                known = (img[..., 3] != 0) if img.shape[2] == 4 else None
                img = img[..., 0]
            if int(y.get("negate", 0)):
                img = 255 - img
            cost = np.full(img.shape, -1, np.int16)
            if str(y.get("mode", "trinary")) == "scale":
                k = (img != 205) if known is None else known
                cost[k] = np.round(100.0 * (1.0 - img[k] / 254.0)).astype(np.int16)
            else:
                cost[img < 50] = 100; cost[img > 230] = 0
            cost[cost >= 100] = -1                      # SLAM walls copied into the bake: /map owns them
            cost[cost > cap] = cap
            o = y["origin"]
            self.get_logger().info("terrain prior %s: %dx%d @ %.3f, %d cost cells (cap %d)" % (path, img.shape[1], img.shape[0], float(y["resolution"]), int((cost > 0).sum()), cap))
            return cost, float(o[0]), float(o[1]), float(y["resolution"])
        except Exception as e:
            self.get_logger().error("terrain prior %s not loaded: %s" % (path, e))
            return None

    def _resample_prior(self, info):
        """Prior -> /map geometry (nearest cell)."""
        cost, px, py, pr = self.prior_src
        rows, cols = np.mgrid[0:info.height, 0:info.width]
        X = info.origin.position.x + (cols + 0.5) * info.resolution; Y = info.origin.position.y + (rows + 0.5) * info.resolution
        gi = ((Y - py) / pr).astype(int); gj = ((X - px) / pr).astype(int)
        ok = (gi >= 0) & (gi < cost.shape[0]) & (gj >= 0) & (gj < cost.shape[1])
        out = np.full((info.height, info.width), -1, np.int8); out[ok] = cost[gi[ok], gj[ok]].astype(np.int8)
        return out

    def on_map(self, m):
        a = np.array(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        if self.map is None or a.shape != self.map.shape or (m.info.origin.position.x, m.info.origin.position.y, m.info.resolution) != self.geom[:3]:
            self.terr = np.full(a.shape, -1, np.int8); self.seen = np.zeros(a.shape, np.float64)
            self.prior = self._resample_prior(m.info) if self.prior_src is not None else None
            self.get_logger().info("global grid geometry from %s: %dx%d @ %.3f origin (%.2f, %.2f)%s" % (
                self.get_parameter("map_topic").value, m.info.width, m.info.height, m.info.resolution, m.info.origin.position.x, m.info.origin.position.y,
                "" if self.prior is None else ", prior cells %d" % int((self.prior >= 0).sum())))
        self.map = a; self.info = m.info; self.frame = m.header.frame_id
        self.geom = (m.info.origin.position.x, m.info.origin.position.y, m.info.resolution, m.info.width, m.info.height)

    def on_terrain(self, t):
        if self.map is None:
            return
        a = np.array(t.data, dtype=np.int8).reshape(t.info.height, t.info.width)
        ox, oy, res, W, H = self.geom
        # terrain cell centres -> global cells; several terrain cells per global cell: worst (max) cost, unknown ignored
        rows, cols = np.mgrid[0:t.info.height, 0:t.info.width]
        X = t.info.origin.position.x + (cols + 0.5) * t.info.resolution; Y = t.info.origin.position.y + (rows + 0.5) * t.info.resolution
        gi = ((Y - oy) / res).astype(int); gj = ((X - ox) / res).astype(int)
        ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W) & (a >= 0)
        if not ok.any():
            return
        flat = gi[ok] * W + gj[ok]
        vals = a[ok].astype(np.int16)
        # Aggregate the 0.05 m terrain cells into the 0.1 m map cell: lethal only when at least `lethal_votes` of the
        # sub-cells are lethal (an isolated lethal speck - one grass blade voxel, one flickering cell - must not become
        # a global wall: with the inscribed band NavFn treats 253 as impassable and a 1 m lane sealed itself at the
        # ds1020 start, GUI run 2026-10-20 17:50); otherwise the max of the non-lethal costs.
        agg = np.full(H * W, -1, np.int16)
        nonleth = vals < 100
        np.maximum.at(agg, flat[nonleth], vals[nonleth])
        nl = np.zeros(H * W, np.int16)
        np.add.at(nl, flat[~nonleth], 1)
        # votes are capped at the number of terrain sub-cells per map cell (a 0.05 m SLAM pgm has ONE: then 1 vote is lethal)
        need = max(1, min(self.lethal_votes, int(round((res / t.info.resolution) ** 2))))
        leth = np.flatnonzero(nl >= need)
        agg[leth] = self.lethal_as
        agg[(nl > 0) & (nl < need) & (agg < 0)] = min(self.lethal_as, 89)   # a lone lethal speck: cost, not a wall
        upd = np.flatnonzero(agg >= 0)
        tv = self.terr.ravel()
        if self.mode == "max":
            tv[upd] = np.maximum(tv[upd], agg[upd]).astype(np.int8)
        else:
            tv[upd] = agg[upd].astype(np.int8)
        self.seen.ravel()[upd] = time.time()
        # Inside the current window the local map is the truth: cells it now reports UNKNOWN lose their memory,
        # otherwise a lethal cell written while the LIO was still initialising (before clear_map) stays forever
        # and walls the start in (GUI run 2026-10-20 17:40: 67 lethal cells within 0.8 m, NavFn found no path).
        # Memory therefore only persists OUTSIDE the 10 m window.
        win = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
        wflat = np.unique(gi[win] * W + gj[win])
        unk = wflat[agg[wflat] < 0]
        tv[unk] = -1
        self.n_in += 1

    def on_reset(self, req, res):
        """Forget the accumulated LIVE terrain (called with the elevation map's clear_map, e.g. after the SLAM init pose); the prior stays."""
        if self.terr is not None:
            self.terr[:] = -1; self.seen[:] = 0.0
        res.success = True; res.message = "terrain memory cleared"
        self.get_logger().info("terrain memory cleared")
        return res

    def publish(self):
        if self.map is None or self.terr is None:
            return
        terr = self.terr
        if self.forget > 0:
            stale = (self.seen > 0) & (time.time() - self.seen > self.forget)
            terr = np.where(stale, -1, terr)
        base = self.map if self.prior is None else np.where(self.prior >= 0, self.prior, self.map)   # prior > /map
        if self.merge == "override":
            out = np.where(terr >= 0, terr, base).astype(np.int8)                                   # live > prior > /map
        else:
            out = np.where(terr >= 0, np.maximum(base, terr), base).astype(np.int8)
        m = OccupancyGrid(); m.header.frame_id = self.frame; m.header.stamp = self.get_clock().now().to_msg()
        m.info = self.info; m.data = out.ravel().tolist()
        self.pub.publish(m)
        if self.n_in and self.n_in % 100 == 0:
            cleared = int(((self.map >= 65) & (terr >= 0) & (terr < 65)).sum())
            self.get_logger().info("global grid: %d terrain updates, lethal %d, judged %d, prior-map occupied cells cleared by terrain %d (merge %s)" % (
                self.n_in, int((out >= self.lethal_as).sum()), int((self.terr >= 0).sum()), cleared, self.merge))


def main():
    rclpy.init()
    n = TerrainGlobal()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):
        pass
    try:
        n.destroy_node(); rclpy.shutdown()
    except BaseException:                       # a second SIGINT from launch lands here; nothing left to do
        pass


if __name__ == "__main__":
    main()
