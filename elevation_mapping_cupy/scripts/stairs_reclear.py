#!/usr/bin/env python3
"""Stairs supervisor (stairs:=true only): re-clear the elevation map on a stair segment.

This node never touches Nav2: speed and gait limits on stairs are Nav2's own business (its params), not elevation's
(the runtime controller hold of 2026-10-29 was removed on 2026-11-03).


Measured in the sim (ds1027_1f, D1 MaxPro, 2026-10-29): on the last steps of a climb the upper floor ahead (flat to
+-1.6 cm in the ground truth) carried 45 % cells with safety < 0.4 - heights written earlier from far below, at grazing
angles, while the floor was still above the sensor. The fused map keeps them (variance-weighted), the local costmap
inflates them and MPPI stalls at the top exit. After clear_map the same floor read p50 0.91 and 0 % below 0.4 within a
second. The same happens to the lower floor seen from above on the way down.

So: while the robot stands in the stairs mask (/terrain/stairs_grid >= min_band within radius of it) and its height has
changed by more than dz since the last clear, call /elevation_mapping_cupy/clear_map once; the map refills from the new
vantage point (10 Hz). The stair cells themselves come back from the next scans (and the prior, if one is used).

  ros2 run elevation_mapping_cupy stairs_reclear.py --ros-args -p dz:=0.6
"""
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import tf2_ros


class Reclear(Node):
    def __init__(self):
        super().__init__("stairs_reclear")
        self.dz = float(self.declare_parameter("dz", 0.6).value)
        self.min_band = int(self.declare_parameter("min_band", 50).value)
        self.frame = self.declare_parameter("robot_frame", "lidar_frame").value   # 3D pose (base_link hangs off odom_2d: z is always 0)
        self.service = self.declare_parameter("service", "/elevation_mapping_cupy/clear_map").value
        self.tf = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.radius = float(self.declare_parameter("radius", 0.8).value)
        # arrival on a level (floor, landing, upper floor): height changed >= arrive_dz since the last clear and has been
        # within settle_band for settle_s. The floor below a flight is last seen from the steps above; the dz rule alone
        # left single stale lethal cells at the foot (no 0.6 m change on the last steps) that walled the robot in (sim D1).
        self.arrive_dz = float(self.declare_parameter("arrive_dz", 0.3).value)
        self.settle_s = float(self.declare_parameter("settle_s", 2.0).value)
        self.settle_band = float(self.declare_parameter("settle_band", 0.03).value)
        self.zhist = []
        # settle clear: whenever the robot has been at one height for settle_s near a stair segment and no clear happened
        # since it got there, clear once. The dz rule misses the arrival at the foot (last clear mid-flight, no 0.3 m left)
        # and the lidar does not refresh the floor within ~1.5 m of the body, so the stale lethal band seen from the
        # flight stayed right in front of the robot (16 lethal cells -> 202 inscribed, 2026-10-29).
        self.settled_cleared_z = None; self.last_clear_t = -1e9
        self.grid = None; self.z_ref = None; self.busy = False; self.n = 0
        # the one "robot is on a stair segment" judgement of the stack (stairs_scan_filter reads it instead of judging again)
        self.on_pub = self.create_publisher(Bool, self.declare_parameter("on_stairs_topic", "/terrain/on_stairs").value, 1)
        self.create_subscription(OccupancyGrid, self.declare_parameter("stairs_topic", "/terrain/stairs_grid").value, self.on_grid, 1)
        self.cli = self.create_client(Trigger, self.service)
        self.create_timer(0.2, self.tick)
        self.get_logger().info("clear %s every %.2f m of height change on stairs (dz <= 0: off)" % (self.service, self.dz))

    def on_grid(self, m):
        self.grid = (np.array(m.data, np.int8).reshape(m.info.height, m.info.width) >= self.min_band,
                     m.info.origin.position.x, m.info.origin.position.y, m.info.resolution)

    def tick(self):
        if self.grid is None:
            return
        try:
            p = self.tf.lookup_transform("map", self.frame, rclpy.time.Time()).transform.translation
        except Exception as e:
            self.get_logger().warn("no TF map <- %s: %s" % (self.frame, e), throttle_duration_sec=10.0)
            return
        mask, ox, oy, res = self.grid
        # the cells under the robot are never observed (body blind zone -> -1 in the stairs grid): look for stair cells
        # within `radius` of the robot instead of at its own position
        k = int(np.ceil(self.radius / res))
        i, j = int(np.floor((p.y - oy) / res)), int(np.floor((p.x - ox) / res))
        win = mask[max(0, i - k):max(0, i + k + 1), max(0, j - k):max(0, j + k + 1)]
        on = bool(win.any())
        self.on_pub.publish(Bool(data=on))
        if self.dz <= 0 or self.busy:
            return
        self.get_logger().info("robot (%.2f, %.2f, z %.2f) on stairs %s, z_ref %s" % (
            p.x, p.y, p.z, on, "%.2f" % self.z_ref if self.z_ref is not None else "-"), throttle_duration_sec=5.0)
        if self.z_ref is None or not on:
            self.z_ref = p.z if (self.z_ref is None or not on) else self.z_ref
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.zhist = [(t, z) for t, z in self.zhist if now - t <= self.settle_s] + [(now, p.z)]
        span = now - self.zhist[0][0]; zz = [z for _, z in self.zhist]
        settled = span >= 0.9 * self.settle_s and max(zz) - min(zz) <= self.settle_band
        arrived = settled and abs(p.z - self.z_ref) >= self.arrive_dz
        if settled and self.settled_cleared_z is not None and abs(p.z - self.settled_cleared_z) > 0.3:
            self.settled_cleared_z = None                      # moved to another level: a new settle may clear again
        settle_clear = settled and self.settled_cleared_z is None and now - self.last_clear_t >= self.settle_s
        if (abs(p.z - self.z_ref) >= self.dz or arrived or settle_clear) and self.cli.service_is_ready():
            self.busy = True; self.n += 1
            self.get_logger().info("on stairs, height %+.2f m since last clear%s -> clear_map (#%d)" % (
                p.z - self.z_ref, " (arrived on a level)" if arrived and abs(p.z - self.z_ref) < self.dz else (" (settled near stairs)" if settle_clear else ""), self.n))
            self.z_ref = p.z; self.last_clear_t = now
            if settle_clear:
                self.settled_cleared_z = p.z
            self.cli.call_async(Trigger.Request()).add_done_callback(lambda f: setattr(self, "busy", False))


def main():
    rclpy.init(); n = Reclear()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
