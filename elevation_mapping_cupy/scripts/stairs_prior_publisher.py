#!/usr/bin/env python3
"""Publish the stairs registered at mapping time (map_stairs.yaml) as an OccupancyGrid for terrain_grid_node.

stairs:=true stairs_map:=<map_dir>/map_stairs.yaml starts this node and points terrain_grid_node's stairs_prior_topic at
it. Cells >= 50 are stair segment cells: terrain_grid_node lifts them to stairs_cost and stairs_scan_filter drops the /scan
beams that end there. A segment is
  - every flight footprint (first riser .. last riser + 0.20 m, l_lo .. l_hi)
  - the landings between consecutive flights of one staircase (gap < max_landing, lateral overlap of the two flights):
    measured in the sim, a lifted flight with an un-lifted landing ahead stopped MPPI (21 % lethal, 25 % unknown there)
  - end_margin beyond the staircase's foot and top exit along its axis: the floor next to the first/last riser carries
    a lethal band ~1 m wide in the terrain grid (0.5 m left a band across the path at the top exit, 2026-10-29)
Walls are never stairs: cells occupied in the SLAM 2D map next to the stairs file (map.yaml, dilated wall_margin) and NOT on
a flight run are cut out of the segment (the 2D map draws the flights themselves and the landing edges as occupied, ~2400
cells on the 1f map, so the cut spares the flight and landing cores; cutting the landing closed landing A completely once
the 0.31 m inscribed inflation was added, 2026-10-29), and the lateral extent is shrunk by lateral_margin on each side. Without this the measured flight
width reached the wall foot, those wall cells were lifted to the stairs cost and the local costmap lost the wall: the
robot hugged the staircase walls (sim, 2026-10-29: 560 of 1445 wall cells within 4 m lifted).
Value 100, map frame, transient-local, re-sent every second. Also reads haechi_sim's stairs.json (same flight schema).

  ros2 run elevation_mapping_cupy stairs_prior_publisher.py --ros-args -p stairs_map:=<map_dir>/map_stairs.yaml
"""
import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from haechi_terrain.stairs_raster import load, rasterise

def main():
    rclpy.init()
    n = Node("stairs_prior_publisher")
    path = os.path.expanduser(n.declare_parameter("stairs_map", "").value)
    topic = n.declare_parameter("topic", "/terrain/stairs_prior").value
    res = float(n.declare_parameter("res", 0.05).value)
    end_margin = float(n.declare_parameter("end_margin", 1.0).value)
    max_landing = float(n.declare_parameter("max_landing", 2.5).value)
    frame = n.declare_parameter("frame_id", "map").value
    if not path or not os.path.isfile(path):
        n.get_logger().error("stairs_map '%s' not found: nothing published" % path); rclpy.shutdown(); return
    d = load(path)
    flights, scs = d.get("flights", []), d.get("staircases", [])
    if not flights:
        n.get_logger().warn("%s has no flights: nothing published" % path); rclpy.spin(n); return
    flight_margin = float(n.declare_parameter("flight_margin", 0.3).value)
    lateral_margin = float(n.declare_parameter("lateral_margin", 0.25).value)
    wall_margin = float(n.declare_parameter("wall_margin", 0.15).value)
    map_yaml = n.declare_parameter("map_yaml", "").value or os.path.join(os.path.dirname(path), "map.yaml")
    grid, x0, y0, n_land, n_wall = rasterise(flights, scs, res, end_margin, max_landing, flight_margin, lateral_margin, map_yaml, wall_margin)
    q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub = n.create_publisher(OccupancyGrid, topic, q)
    m = OccupancyGrid(); m.header.frame_id = frame
    m.info.resolution = res; m.info.width = grid.shape[1]; m.info.height = grid.shape[0]
    m.info.origin.position.x = float(x0); m.info.origin.position.y = float(y0); m.info.origin.orientation.w = 1.0
    m.data = grid.ravel().tolist()

    def tick():
        m.header.stamp = n.get_clock().now().to_msg(); pub.publish(m)
    tick(); n.create_timer(1.0, tick)
    # rviz: the registered flights as step boxes (+ flight number and riser count)
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    ma = MarkerArray(); mid = 0
    for k, f in enumerate(flights):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"]); yaw = float(np.arctan2(ax[1], ax[0]))
        for j in range(f["n"]):
            sa = f["s0"] + j * f["tread"]; sb = sa + (f["tread"] if j < f["n"] - 1 else 0.20); zt = f["z_bot"] + (j + 1) * f["rise"]
            c = p0 + ax * 0.5 * (sa + sb) + lat * 0.5 * (f["l_lo"] + f["l_hi"])
            b_ = Marker(); b_.header.frame_id = frame; b_.ns = "registered steps"; b_.id = mid; mid += 1; b_.type = Marker.CUBE
            b_.pose.position.x, b_.pose.position.y, b_.pose.position.z = float(c[0]), float(c[1]), float(0.5 * (zt + f["z_bot"] - 0.02))
            b_.pose.orientation.z, b_.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
            b_.scale.x, b_.scale.y, b_.scale.z = float(sb - sa), float(f["l_hi"] - f["l_lo"]), float(zt - f["z_bot"] + 0.02)
            b_.color = ColorRGBA(r=0.95, g=0.55, b=0.15, a=0.75); ma.markers.append(b_)
        t_ = Marker(); t_.header.frame_id = frame; t_.ns = "flight labels"; t_.id = k; t_.type = Marker.TEXT_VIEW_FACING
        c = p0 + ax * (f["s0"] + 0.5 * (f["n"] - 1) * f["tread"]) + lat * 0.5 * (f["l_lo"] + f["l_hi"])
        t_.pose.position.x, t_.pose.position.y, t_.pose.position.z = float(c[0]), float(c[1]), float(f["z_top"] + 0.6)
        t_.pose.orientation.w = 1.0; t_.scale.z = 0.3; t_.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t_.text = "flight %d: %d x %.3f m" % (k + 1, f["n"], f["rise"]); ma.markers.append(t_)
    pub_m = n.create_publisher(MarkerArray, "~/markers", q); pub_m.publish(ma)
    n.get_logger().info("stairs prior: %d flights in %d staircases + %d landings from %s -> %s (%dx%d @ %.2f m, %d cells, end margin %.1f m, "
                        "lateral margin %.2f m, %d wall cells cut by %s)" % (
        len(flights), len(scs), n_land, path, topic, grid.shape[1], grid.shape[0], res, int((grid >= 50).sum()), end_margin, lateral_margin,
        n_wall, map_yaml if os.path.isfile(map_yaml) else "(no map.yaml)"))
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
