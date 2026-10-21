#!/usr/bin/env python3
"""Replace Nav2's 2D map with the elevation stack's map_terrain.yaml (LOCALIZATION, only when a terrain prior is given).

The robot's orchestrator loads <map_folder>/<map>/<submap>/map.yaml (the SLAM octomap pgm) into map_server when
localization starts. With a terrain prior this node loads map_terrain.yaml from the same folder instead, so /map (and
every static layer that reads it) carries the SLAM walls plus the terrain cost (terrain_map_merge.py). Started by
haechi.launch.py when terrain_prior is non-empty; without a prior nothing runs and Nav2 keeps map.yaml.

  map2d_yaml       the file to load; empty = <dir of terrain_prior>/map_terrain.yaml
  terrain_prior    the prior yaml (only used to find the folder and to build map_terrain.yaml if it is missing:
                   terrain_map_merge.py on <dir>/map.yaml + the prior)
  load_map_service /map_server/load_map (nav2_msgs/srv/LoadMap)
  watch            true: keep watching /map; when it is replaced by another map (the orchestrator's map.yaml load
                   at localization start, a later load_map) reload ours. A reload is attempted at most once per
                   reload_cooldown_s so two loaders can never fight.
  retry_s          period of the service wait / retry while map_server is not up or the load fails
"""
import importlib.util
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class TerrainMapLoader(Node):
    def __init__(self):
        super().__init__("terrain_map_loader")
        self.declare_parameter("map2d_yaml", "")
        self.declare_parameter("terrain_prior", "")
        self.declare_parameter("load_map_service", "/map_server/load_map")
        self.declare_parameter("watch", True)
        self.declare_parameter("retry_s", 3.0)
        self.declare_parameter("reload_cooldown_s", 5.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        prior = os.path.expanduser(str(g("terrain_prior")).strip())
        yml = os.path.expanduser(str(g("map2d_yaml")).strip()) or (os.path.join(os.path.dirname(prior), "map_terrain.yaml") if prior else "")
        if not yml:
            self.get_logger().error("neither map2d_yaml nor terrain_prior given: nothing to load")
            raise SystemExit(2)
        if not os.path.isfile(yml) and prior:
            self._build(yml, prior)
        if not os.path.isfile(yml):
            self.get_logger().error("%s not found: Nav2 keeps the map it has" % yml)
            raise SystemExit(2)
        self.yml = os.path.abspath(yml)
        self.sig = self._signature(self.yml)        # (width, height, origin x, y, res) of our map, to recognise it on /map
        self.retry = float(g("retry_s")); self.cooldown = float(g("reload_cooldown_s"))
        self.loaded_at = 0.0; self.loads = 0; self.pending = None; self.reload_due = None
        from nav2_msgs.srv import LoadMap
        self.LoadMap = LoadMap
        self.cli = self.create_client(LoadMap, str(g("load_map_service")))
        if bool(g("watch")):
            self.create_subscription(OccupancyGrid, "/map", self.on_map,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(min(self.retry, 1.0), self.tick)
        self.last_try = 0.0
        self.get_logger().info("will load %s into %s%s" % (self.yml, g("load_map_service"), " and keep it there (watch)" if g("watch") else ""))

    # ---- helpers ----
    def _build(self, yml, prior):
        """map_terrain.yaml missing: make it from <dir>/map.yaml + the prior (terrain_map_merge.py next to this script)."""
        d = os.path.dirname(yml); map_yaml = os.path.join(d, "map.yaml")
        if not os.path.isfile(map_yaml) or not os.path.isfile(prior):
            self.get_logger().warn("cannot build %s: need %s and %s" % (yml, map_yaml, prior)); return
        try:
            spec = importlib.util.spec_from_file_location("terrain_map_merge", os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_map_merge.py"))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            mod.merge_terrain_map(map_yaml, prior, os.path.splitext(yml)[0], log=self.get_logger().info)
        except Exception as e:                      # noqa: BLE001
            self.get_logger().error("terrain_map_merge failed: %s" % e)

    @staticmethod
    def _signature(yml):
        import yaml
        from PIL import Image
        y = yaml.safe_load(open(yml))
        w, h = Image.open(os.path.join(os.path.dirname(yml), y["image"])).size
        return (int(w), int(h), round(float(y["origin"][0]), 3), round(float(y["origin"][1]), 3), round(float(y["resolution"]), 4))

    def _is_ours(self, m):
        """Same geometry as our file and the same number of unknown cells as the map we loaded."""
        if not hasattr(self, "unknown_n"):
            return False
        i = m.info
        geom = (int(i.width), int(i.height), round(i.origin.position.x, 3), round(i.origin.position.y, 3), round(i.resolution, 4))
        return geom == self.sig and int(np.count_nonzero(np.array(m.data, np.int8) == -1)) == self.unknown_n

    # ---- flow ----
    def on_map(self, m):
        if not hasattr(self, "unknown_n"):
            return                                   # nothing loaded by us yet; the timer handles the first load
        if self._is_ours(m):
            self.reload_due = None
            return
        if self.reload_due is None:
            # never drop the event: /map is latched, so this is the only notice we get. Reload once the cooldown
            # since our last load has passed (the orchestrator's map.yaml load typically lands right after ours).
            self.reload_due = max(time.time(), self.loaded_at + self.cooldown)
            self.get_logger().info("/map replaced (%dx%d, another map): reloading %s in %.1fs" % (
                m.info.width, m.info.height, os.path.basename(self.yml), self.reload_due - time.time()))

    def tick(self):
        if self.pending is not None:
            if not self.pending.done():
                return
            self._done(self.pending); self.pending = None
            return
        if self.loaded_at > 0.0:
            if self.reload_due is None or time.time() < self.reload_due:
                return                               # loaded and nothing (yet) to redo
            self.reload_due = None
        if time.time() - self.last_try < self.retry:
            return
        self.last_try = time.time()
        if not self.cli.service_is_ready():
            self.get_logger().info("waiting for %s" % self.cli.srv_name, throttle_duration_sec=30.0)
            return
        req = self.LoadMap.Request(); req.map_url = self.yml
        self.pending = self.cli.call_async(req)

    def _done(self, fut):
        try:
            res = fut.result()
        except Exception as e:                      # noqa: BLE001
            self.get_logger().error("load_map call failed: %s (retry in %.0fs)" % (e, self.retry)); return
        if res.result != self.LoadMap.Response.RESULT_SUCCESS:
            self.get_logger().error("load_map %s: result %d (retry in %.0fs)" % (self.yml, res.result, self.retry)); return
        self.loaded_at = time.time(); self.loads += 1
        self.unknown_n = int(np.count_nonzero(np.array(res.map.data, np.int8) == -1))
        self.get_logger().info("Nav2 /map <- %s (%dx%d @ %.3f m, unknown %d; load #%d)" % (
            self.yml, res.map.info.width, res.map.info.height, res.map.info.resolution, self.unknown_n, self.loads))


def main():
    rclpy.init()
    try:
        n = TerrainMapLoader()
    except SystemExit:
        rclpy.shutdown(); return
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
