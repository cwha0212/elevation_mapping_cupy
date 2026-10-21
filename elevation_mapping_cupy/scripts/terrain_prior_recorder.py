#!/usr/bin/env python3
"""Record the terrain analysis during MAPPING and save it as a terrain PRIOR for the elevation navigation stack.

While the robot maps, the rolling terrain grid map (/elevation_mapping_node/elevation_map_terrain, 10 m window) is
accumulated on a global grid that grows as the robot moves. For every cell the recorder keeps the last drivability and
elevation and two counters: how often the cell was observed (n_obs) and how often it was judged non-drivable (n_bad,
drivability < bad_below). On save it writes
  <out_dir>/terrain_prior.npz          x0, y0, res, elevation, drivability, n_obs, n_bad  (same keys as elevation_stitch.py)
  <out_dir>/map_terrain_prior.pgm/yaml mode scale, 0..100 = cost; 205 = unknown   -> terrain_global_node prior_file
The pgm uses the temporal-vote rule of terrain_map_bake.py (council 2026-10-21): a cell needs min_obs observations;
bad in >= vote_frac of its views -> graded cost (free_above 0.6 -> 0, lethal_below 0.2 -> max) capped at cap (89, never a
wall); bad in fewer views -> cost scaled by the vote share, at most minority_cap (60); drivable -> 0. Everything else is
unknown and terrain_global_node falls back to /map there. Lethal is never baked: that is the live terrain's and /scan's job.

Save: Trigger service ~/save, and automatically on shutdown (SIGINT from the orchestrator's mapping stop) when
save_on_shutdown is true. Subscribes best-effort: the 1 MB GridMap at 10 Hz overflows reliable readers on the board.

2D map: when merge_map is true and <out_dir>/map.yaml exists at save time (out_dir = the NAVI map folder, whose map.pgm
laserMapping writes at mapping shutdown), terrain_map_merge.py is run on the saved prior and writes map_terrain.png/yaml
on the map.yaml geometry (SLAM walls 100 + terrain cost capped, never a wall). If map.yaml is not there yet, run
`terrain_map_merge.py <map_dir>` afterwards.
"""
import os
import time

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


class PriorRecorder(Node):
    def __init__(self):
        super().__init__("terrain_prior_recorder")
        self.declare_parameter("topic", "/elevation_mapping_node/elevation_map_terrain")
        self.declare_parameter("out_dir", os.path.expanduser("~/terrain_prior"))
        self.declare_parameter("res", 0.1)
        self.declare_parameter("grow_m", 50.0)          # grid grows by this much whenever a window falls outside
        self.declare_parameter("bad_below", 0.4)        # drivability below this counts as a 'bad' view
        self.declare_parameter("min_obs", 10)
        self.declare_parameter("vote_frac", 0.5)
        self.declare_parameter("free_above", 0.6)
        self.declare_parameter("lethal_below", 0.2)
        self.declare_parameter("cap", 89)
        self.declare_parameter("minority_cap", 60)
        self.declare_parameter("save_on_shutdown", True)
        self.declare_parameter("merge_map", True)          # also write map_terrain.* when <out_dir>/map.yaml exists
        self.declare_parameter("merge_map_yaml", "map.yaml")  # the SLAM 2D map file in out_dir to merge onto
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out_dir, self.res, self.grow = str(g("out_dir")), float(g("res")), float(g("grow_m"))
        self.bad_below, self.min_obs, self.vote = float(g("bad_below")), int(g("min_obs")), float(g("vote_frac"))
        self.free, self.lethal, self.cap, self.mcap = float(g("free_above")), float(g("lethal_below")), int(g("cap")), int(g("minority_cap"))
        self.x0 = self.y0 = None; self.W = self.H = 0
        self.drv = self.elev = self.n_obs = self.n_bad = None
        self.n = 0; self.t0 = time.time(); self.saved = False
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(GridMap, str(g("topic")), self.cb, qos)
        self.create_service(Trigger, "~/save", self.on_save)
        self.get_logger().info("recording %s -> %s (res %.2f m, vote %.0f%% of >= %d views, cap %d)" % (
            g("topic"), self.out_dir, self.res, 100 * self.vote, self.min_obs, self.cap))

    # ---- grid that grows ----
    def _ensure(self, xmin, xmax, ymin, ymax):
        if self.x0 is None:
            self.x0 = np.floor((xmin - self.grow / 2) / self.res) * self.res; self.y0 = np.floor((ymin - self.grow / 2) / self.res) * self.res
            self.W = int((xmax + self.grow / 2 - self.x0) / self.res) + 1; self.H = int((ymax + self.grow / 2 - self.y0) / self.res) + 1
            self.drv = np.full((self.H, self.W), np.nan, np.float32); self.elev = np.full((self.H, self.W), np.nan, np.float32)
            self.n_obs = np.zeros((self.H, self.W), np.uint16); self.n_bad = np.zeros((self.H, self.W), np.uint16)
            return
        px0 = int(np.ceil(max(0.0, self.x0 - xmin + self.grow) / self.res)) if xmin < self.x0 else 0
        py0 = int(np.ceil(max(0.0, self.y0 - ymin + self.grow) / self.res)) if ymin < self.y0 else 0
        px1 = int(np.ceil((xmax + self.grow - (self.x0 + self.W * self.res)) / self.res)) if xmax > self.x0 + self.W * self.res else 0
        py1 = int(np.ceil((ymax + self.grow - (self.y0 + self.H * self.res)) / self.res)) if ymax > self.y0 + self.H * self.res else 0
        if px0 or py0 or px1 or py1:
            pad = ((py0, py1), (px0, px1))
            self.drv = np.pad(self.drv, pad, constant_values=np.nan); self.elev = np.pad(self.elev, pad, constant_values=np.nan)
            self.n_obs = np.pad(self.n_obs, pad); self.n_bad = np.pad(self.n_bad, pad)
            self.x0 -= px0 * self.res; self.y0 -= py0 * self.res; self.H, self.W = self.drv.shape
            self.get_logger().info("grid grown to %dx%d (%.0f x %.0f m)" % (self.W, self.H, self.W * self.res, self.H * self.res))

    def cb(self, m):
        h = m.data[0].layout.dim[0].size; w = m.data[0].layout.dim[1].size; res = m.info.resolution
        cx, cy = m.info.pose.position.x, m.info.pose.position.y
        L = {n: d for n, d in zip(m.layers, m.data)}
        if "drivability" not in L:
            return
        drv = np.array(L["drivability"].data, np.float32).reshape(h, w)
        elev = np.array(L["elevation"].data, np.float32).reshape(h, w) if "elevation" in L else None
        rows, cols = np.mgrid[0:h, 0:w]
        X = cx + (w / 2 - 0.5 - cols) * res; Y = cy + (h / 2 - 0.5 - rows) * res      # grid_map: index 0 = +x/+y corner
        self._ensure(X.min(), X.max(), Y.min(), Y.max())
        gi = ((Y - self.y0) / self.res).astype(int); gj = ((X - self.x0) / self.res).astype(int)
        valid = np.isfinite(drv) & (gi >= 0) & (gi < self.H) & (gj >= 0) & (gj < self.W)
        self.drv[gi[valid], gj[valid]] = drv[valid]
        if elev is not None:
            ev = valid & np.isfinite(elev); self.elev[gi[ev], gj[ev]] = elev[ev]
        self.n_obs[gi[valid], gj[valid]] += 1
        bad = valid & (drv < self.bad_below); self.n_bad[gi[bad], gj[bad]] += 1
        self.n += 1
        if self.n % 100 == 0:
            self.get_logger().info("maps %d, observed cells %d, %.0fs" % (self.n, int((self.n_obs > 0).sum()), time.time() - self.t0))

    # ---- bake + save ----
    def bake(self):
        """cost grid (int16, -1 unknown) from the vote counters: the terrain_map_bake.py 'voted prior' rule."""
        nobs = self.n_obs.astype(np.float32); frac = np.where(nobs > 0, self.n_bad / np.maximum(nobs, 1), 0.0)
        graded = np.clip((self.free - np.nan_to_num(self.drv, nan=1.0)) / (self.free - self.lethal), 0, 1) * 99.0
        cost = np.full(self.drv.shape, -1, np.int16)
        conf = (nobs >= self.min_obs) & np.isfinite(self.drv)
        voted = np.minimum(np.round(graded), self.cap)
        minority = np.minimum(np.round(graded * frac / self.vote), self.mcap)
        cost[conf] = np.where(frac[conf] >= self.vote, voted[conf], minority[conf]).astype(np.int16)
        return cost

    def save(self):
        if self.drv is None:
            self.get_logger().warn("nothing recorded, no prior written"); return False, "nothing recorded"
        os.makedirs(self.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(self.out_dir, "terrain_prior.npz"), x0=self.x0, y0=self.y0, res=self.res,
                            elevation=self.elev, drivability=self.drv, n_obs=self.n_obs, n_bad=self.n_bad)
        cost = self.bake()
        img = np.full(cost.shape, 205, np.uint8); k = cost >= 0
        img[k] = np.round(254 * (1 - cost[k] / 100.0)).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img[::-1]).save(os.path.join(self.out_dir, "map_terrain_prior.pgm"))
        with open(os.path.join(self.out_dir, "map_terrain_prior.yaml"), "w") as f:
            # thresholds as in terrain_map_merge.py: map_server (mode scale) then returns the cost (0.65 would make cost >= 65 a wall)
            f.write("image: map_terrain_prior.pgm\nmode: scale\nresolution: %.3f\norigin:\n- %.4f\n- %.4f\n- 0.0\nnegate: 0\noccupied_thresh: 0.995\nfree_thresh: 0.005\n" % (self.res, self.x0, self.y0))
        msg = "prior saved to %s: %dx%d @ %.2f m, %d maps, %d cells judged, %d with cost (%d at cap)" % (
            self.out_dir, self.W, self.H, self.res, self.n, int(k.sum()), int((cost > 0).sum()), int((cost >= self.cap).sum()))
        self.get_logger().info(msg); self.saved = True
        msg += self._merge_map()
        return True, msg

    def _merge_map(self):
        """map_terrain.png/yaml on the SLAM map geometry (terrain_map_merge.py next to this script), if map.yaml is there."""
        if not self.get_parameter("merge_map").value:
            return ""
        map_yaml = os.path.join(self.out_dir, str(self.get_parameter("merge_map_yaml").value))
        if not os.path.isfile(map_yaml):
            self.get_logger().info("no %s yet: run terrain_map_merge.py %s once the SLAM 2D map is saved" % (map_yaml, self.out_dir))
            return "; map_terrain not written (no %s)" % os.path.basename(map_yaml)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "terrain_map_merge", os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_map_merge.py"))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            mod.merge_terrain_map(map_yaml, os.path.join(self.out_dir, "map_terrain_prior.yaml"),
                                  os.path.join(self.out_dir, "map_terrain"), cap=self.cap, log=self.get_logger().info)
            return "; map_terrain.png/yaml written"
        except Exception as e:                  # noqa: BLE001 - the prior is saved either way
            self.get_logger().error("map_terrain merge failed: %s" % e)
            return "; map_terrain merge failed: %s" % e

    def on_save(self, req, res):
        res.success, res.message = self.save(); return res


def main():
    rclpy.init(); n = PriorRecorder()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):      # rclpy raises ExternalShutdownException on SIGINT
        pass
    finally:
        if n.get_parameter("save_on_shutdown").value:
            try:
                n.save()
            except Exception as e:              # noqa: BLE001
                print("terrain_prior_recorder: save failed: %s" % e)
    try:
        n.destroy_node(); rclpy.shutdown()
    except BaseException:                       # a second SIGINT from launch lands here; nothing left to do
        pass


if __name__ == "__main__":
    main()
