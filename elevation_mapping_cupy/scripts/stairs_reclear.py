#!/usr/bin/env python3
"""Stairs supervisor (stairs:=true only): re-clear the elevation map on a stair segment, and hold the controller to a
straight, slow gait there.

Controller hold (council review 2026-10-29, sim: the body drifted 0.6 m sideways on a 2.2 m flight into the wall
inflation; /speed_limit cannot do this since MPPI scales vx, vy and wz by one ratio): while the robot is on a stair
segment (stairs mask within `radius`, with leave_s hysteresis) the controller's dynamic parameters are set to
    FollowPath.vy_max 0, vx_max stairs_vx, vx_min 0 (no reversing), wz_max stairs_wz, PathAlignCritic.cost_weight x2
and restored to the values read from the controller when the robot leaves. Runtime values only; no params file changes.


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
        # controller hold on stairs
        self.hold = bool(self.declare_parameter("controller_hold", True).value)
        self.ctl_node = self.declare_parameter("controller_node", "/controller_server").value
        self.stairs_vx = float(self.declare_parameter("stairs_vx_max", 0.6).value)   # no cap: 0.3 and 0.5 left the D1 policy wedged on the 0.174 m riser, 0.6 freed it at once (2026-10-29)
        self.stairs_wz = float(self.declare_parameter("stairs_wz_max", 0.5).value)
        # vx_min on stairs: 0 = never reverse (a step backwards goes down a riser). A small negative value lets MPPI back
        # off a riser it is wedged on (the A-3 first riser wedge, 2026-10-29: 15 recoveries, no progress with vx_min 0).
        self.stairs_vx_min = float(self.declare_parameter("stairs_vx_min", -0.15).value)
        self.leave_s = float(self.declare_parameter("leave_s", 1.5).value)
        self.held = False; self.saved = None; self.t_off = None
        if self.hold:
            from rcl_interfaces.srv import GetParameters, SetParameters
            self.cli_get = self.create_client(GetParameters, self.ctl_node + "/get_parameters")
            self.cli_set = self.create_client(SetParameters, self.ctl_node + "/set_parameters")
        self.grid = None; self.z_ref = None; self.busy = False; self.n = 0
        self.create_subscription(OccupancyGrid, self.declare_parameter("stairs_topic", "/terrain/stairs_grid").value, self.on_grid, 1)
        self.cli = self.create_client(Trigger, self.service)
        self.create_timer(0.2, self.tick)
        self.get_logger().info("clear %s every %.2f m of height change on stairs (dz <= 0: off)" % (self.service, self.dz))

    def on_grid(self, m):
        self.grid = (np.array(m.data, np.int8).reshape(m.info.height, m.info.width) >= self.min_band,
                     m.info.origin.position.x, m.info.origin.position.y, m.info.resolution)

    def tick(self):
        if self.dz <= 0 or self.grid is None or self.busy:
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
        if self.hold:
            self.update_hold(on)
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


    HOLD_KEYS = ("FollowPath.vy_max", "FollowPath.vx_max", "FollowPath.vx_min", "FollowPath.wz_max", "FollowPath.PathAlignCritic.cost_weight")

    def update_hold(self, on):
        now = self.get_clock().now().nanoseconds * 1e-9
        if on:
            self.t_off = None
            if not self.held:
                self.engage()
        elif self.held:
            self.t_off = self.t_off or now
            if now - self.t_off >= self.leave_s:
                self.release()

    def _set(self, values, what):
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        from rcl_interfaces.srv import SetParameters
        if not self.cli_set.service_is_ready():
            self.get_logger().warn("%s: %s/set_parameters not available" % (what, self.ctl_node), throttle_duration_sec=5.0); return False
        req = SetParameters.Request()
        for k, v in values.items():
            req.parameters.append(Parameter(name=k, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))))
        self.cli_set.call_async(req).add_done_callback(
            lambda f: self.get_logger().info("%s: %s" % (what, ", ".join("%s=%.2f" % kv for kv in values.items()))))
        return True

    def engage(self):
        from rcl_interfaces.srv import GetParameters
        if self.saved is None:
            if not self.cli_get.service_is_ready():
                self.get_logger().warn("controller hold: %s/get_parameters not available" % self.ctl_node, throttle_duration_sec=5.0); return
            req = GetParameters.Request(); req.names = list(self.HOLD_KEYS)

            def got(f):
                vals = f.result().values
                self.saved = {k: v.double_value for k, v in zip(self.HOLD_KEYS, vals)}
                self.get_logger().info("controller hold: saved %s" % ", ".join("%s=%.2f" % kv for kv in self.saved.items()))
                self.engage()
            self.cli_get.call_async(req).add_done_callback(got); return
        hold = {"FollowPath.vy_max": 0.0, "FollowPath.vx_max": self.stairs_vx, "FollowPath.vx_min": self.stairs_vx_min,
                "FollowPath.wz_max": self.stairs_wz, "FollowPath.PathAlignCritic.cost_weight": 2.0 * self.saved["FollowPath.PathAlignCritic.cost_weight"]}
        if self._set(hold, "on stairs -> controller hold"):
            self.held = True

    def release(self):
        if self.saved and self._set(self.saved, "left stairs -> controller restored"):
            self.held = False; self.t_off = None


def main():
    rclpy.init(); n = Reclear()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
