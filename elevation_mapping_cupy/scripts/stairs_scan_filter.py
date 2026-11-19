#!/usr/bin/env python3
"""Drop the laser beams that end on a known stair flight before Nav2 sees them (/scan -> /scan_nav).

/scan is patchwork++'s non-ground cloud flattened to a LaserScan in a band from the ground up to +0.30 m above the
sensor. On a flight the risers ahead and the landing above the robot are 'non-ground' inside that band, so the global
obstacle layer marks the flight lethal and the collision monitor stops the robot at the first riser, whatever the
terrain grid says. This node removes exactly those returns: a beam whose endpoint falls in a stairs cell (the
terrain_grid_node stairs grid: registered stairs = 70) is set to +inf,
dilated by dilate_cells. +inf (not NaN) because the collision monitor ignores out-of-range values and the obstacle
layer (inf_is_valid: true) uses them as clearing rays, so marks left from before the flight was recognised are wiped.
On a stair segment the 2D scan itself is unreliable: its height band is fixed in odom_2d (not relative to the ground
under the robot) and patchwork's ground fit breaks on the steps, so from flight A-1 the whole lower floor (z -0.73, at
the band edge) came through as obstacles and the global planner found no path down (sim D1, 2026-10-29). So while
stairs_reclear reports the robot on a stair segment (/terrain/on_stairs; defer_on_stairs), beams ending on a cell the terrain grid calls traversable
(/terrain/local_grid < free_below) are dropped as well: the elevation map, which judges height relative to the ground,
decides there; walls and other cells it keeps lethal still pass.
Everything else passes untouched; with no fresh stairs grid (stale_s) the scan is republished unchanged, so Nav2
always has /scan_nav.

QoS: /scan is published best-effort (SensorDataQoS) and Nav2 subscribes best-effort; this node uses the same profile
on both sides (a reliable subscriber would never match and the collision monitor would silently ignore the source).

  ros2 run elevation_mapping_cupy stairs_scan_filter.py --ros-args -p scan_in:=/scan -p scan_out:=/scan_nav
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import tf2_ros


class StairsScanFilter(Node):
    def __init__(self):
        super().__init__("stairs_scan_filter")
        self.scan_in = self.declare_parameter("scan_in", "/scan").value
        self.scan_out = self.declare_parameter("scan_out", "/scan_nav").value
        self.stairs_topic = self.declare_parameter("stairs_topic", "/terrain/stairs_grid").value
        self.min_band = int(self.declare_parameter("min_band", 50).value)
        self.dilate = int(self.declare_parameter("dilate_cells", 2).value)
        self.stale_s = float(self.declare_parameter("stale_s", 1.0).value)
        self.enabled = bool(self.declare_parameter("enabled", True).value)
        self.defer = bool(self.declare_parameter("defer_on_stairs", True).value)
        self.free_below = int(self.declare_parameter("free_below", 50).value)
        self.terrain_topic = self.declare_parameter("terrain_topic", "/terrain/local_grid").value
        self.terrain = None
        self.n_defer = 0
        # "robot on a stair segment" is judged once, by stairs_reclear (/terrain/on_stairs, 5 Hz); stale -> not on stairs
        self.on_stairs = (0.0, False)
        self.create_subscription(Bool, self.declare_parameter("on_stairs_topic", "/terrain/on_stairs").value,
                                 lambda m: setattr(self, "on_stairs", (self.now_s(), bool(m.data))), 1)
        self.tf = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.masks = {}                                    # topic -> (stamp_s, mask, ox, oy, res)
        self.create_subscription(OccupancyGrid, self.stairs_topic, lambda m: self.on_grid(m, "local"), 1)
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, self.terrain_topic, self.on_terrain, 1)
        # nav2 /map (walls = 100): while on stairs every beam that does not end on a map wall is dropped - the fixed-height
        # 2D band marks the lower floor / stale terrain when the robot is on a flight (bag test 2026-10-29: 39 obstacle-layer
        # cells on the B terrace, none of them walls)
        self.map = None
        self.create_subscription(OccupancyGrid, self.declare_parameter("map_topic", "/map").value, self.on_map, q)
        self.pub = self.create_publisher(LaserScan, self.scan_out, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_in, self.on_scan, qos_profile_sensor_data)
        self.n = self.n_drop = 0
        self.get_logger().info("%s -> %s: beams ending on stairs (%s >= %d) +%d cells set to +inf (enabled %s)" % (
            self.scan_in, self.scan_out, self.stairs_topic, self.min_band, self.dilate, self.enabled))

    def on_grid(self, m, key):
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        mask = a >= self.min_band
        if self.dilate > 0 and mask.any():
            mask = ndimage.binary_dilation(mask, iterations=self.dilate)
        self.masks[key] = (self.now_s(), mask, m.info.origin.position.x, m.info.origin.position.y, m.info.resolution, m.header.frame_id)

    def on_terrain(self, m):
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        self.terrain = (self.now_s(), (a >= 0) & (a < self.free_below), m.info.origin.position.x, m.info.origin.position.y,
                        m.info.resolution)

    def on_map(self, m):
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        self.map = (a >= 100, m.info.origin.position.x, m.info.origin.position.y, m.info.resolution)

    @staticmethod
    def lookup(mask, ox, oy, res, X, Y):
        j = np.floor((np.atleast_1d(X) - ox) / res).astype(np.int64); i = np.floor((np.atleast_1d(Y) - oy) / res).astype(np.int64)
        ok = (i >= 0) & (j >= 0) & (i < mask.shape[0]) & (j < mask.shape[1])
        hit = np.zeros(len(i), bool); hit[ok] = mask[i[ok], j[ok]]
        return hit

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, s):
        out = s
        live = [v for v in self.masks.values() if self.now_s() - v[0] <= self.stale_s]
        if self.enabled and live:
            frame = live[0][5] or "map"
            # No timeout: the TF listener runs on this single-threaded node, so a blocking wait cannot receive the
            # transform it waits for; it always ran out (+50 ms per scan) and the collision monitor (source_timeout
            # 0.3 s) then dropped /scan_nav in ~95 % of its cycles (Isaac 2026-11-19). The grid frame moves slowly
            # against the scan frame (map vs odom_2d), so the latest TF is a fine fallback.
            try:
                t = self.tf.lookup_transform(frame, s.header.frame_id, rclpy.time.Time.from_msg(s.header.stamp))
            except Exception:
                try:
                    t = self.tf.lookup_transform(frame, s.header.frame_id, rclpy.time.Time())
                except Exception:
                    t = None
            if t is not None:
                r = np.asarray(s.ranges, np.float32)
                ang = s.angle_min + np.arange(len(r), dtype=np.float32) * s.angle_increment
                fin = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
                q = t.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                ex = r * np.cos(ang); ey = r * np.sin(ang)
                X = t.transform.translation.x + math.cos(yaw) * ex - math.sin(yaw) * ey
                Y = t.transform.translation.y + math.sin(yaw) * ex + math.cos(yaw) * ey
                drop = np.zeros(len(r), bool)
                for _, mask, ox, oy, res, _f in live:
                    drop |= fin & self.lookup(mask, ox, oy, res, X, Y)
                t_on, on_stairs = self.on_stairs
                on_stairs = on_stairs and (self.now_s() - t_on <= self.stale_s)
                if self.defer and on_stairs:
                    if self.map is not None:              # keep only beams that end on a map wall
                        wmask, mox, moy, mres = self.map
                        d2 = fin & ~drop & ~self.lookup(wmask, mox, moy, mres, X, Y)
                        self.n_defer += int(d2.sum()); drop |= d2
                    elif self.terrain is not None and self.now_s() - self.terrain[0] <= self.stale_s:
                        _, free, tox, toy, tres = self.terrain
                        d2 = fin & ~drop & self.lookup(free, tox, toy, tres, X, Y)
                        self.n_defer += int(d2.sum()); drop |= d2
                if drop.any():
                    r = r.copy(); r[drop] = np.inf
                    out = LaserScan(header=s.header, angle_min=s.angle_min, angle_max=s.angle_max,
                                    angle_increment=s.angle_increment, time_increment=s.time_increment,
                                    scan_time=s.scan_time, range_min=s.range_min, range_max=s.range_max,
                                    ranges=r.tolist(), intensities=list(s.intensities))
                    self.n_drop += int(drop.sum())
        self.pub.publish(out)
        self.n += 1
        if self.n % 100 == 0:
            self.get_logger().info("%d scans, %d beams dropped so far (%d of them deferred to the terrain grid while on stairs; masks: %s)" % (
                self.n, self.n_drop, self.n_defer, list(self.masks)))


def main():
    rclpy.init()
    n = StairsScanFilter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
