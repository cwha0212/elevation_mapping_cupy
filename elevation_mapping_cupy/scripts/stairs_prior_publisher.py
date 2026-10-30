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
import json
import os

import numpy as np
import rclpy
from matplotlib.path import Path as MPath
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


def load(path):
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        import yaml
        return yaml.safe_load(f)


def s_end(f):
    return f["s0"] + (f["n"] - 1) * f["tread"] + 0.20


def wall_mask(map_yaml, X, Y, margin):
    """True where the SLAM 2D map is occupied within `margin` (m) of the cell (X, Y)."""
    import yaml
    from PIL import Image
    from scipy import ndimage
    my = yaml.safe_load(open(map_yaml))
    img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), my["image"])))[::-1]
    occ = img < 255 * (1.0 - my.get("occupied_thresh", 0.65)) if not my.get("negate", 0) else img > 255 * my.get("occupied_thresh", 0.65)
    k = int(np.ceil(margin / my["resolution"]))
    if k > 0:
        occ = ndimage.binary_dilation(occ, iterations=k)
    ox, oy = my["origin"][:2]; r = my["resolution"]
    i = np.floor((Y - oy) / r).astype(np.int64); j = np.floor((X - ox) / r).astype(np.int64)
    ok = (i >= 0) & (j >= 0) & (i < occ.shape[0]) & (j < occ.shape[1])
    out = np.zeros(X.shape, bool); out[ok] = occ[i[ok], j[ok]]
    return out


def rasterise(flights, staircases, res=0.05, end_margin=1.0, max_landing=2.5, flight_margin=0.3, lateral_margin=0.25,
              map_yaml="", wall_margin=0.15):
    """flight_margin: every flight footprint is extended this far along its axis at both ends. The mapping-time fit places
    the first riser within 1.5-13 cm and the axis within 0-10 deg of the offline reference (1f bag): unextended, the riser
    edge cells it missed stayed lethal and stopped the robot on the A-2/A-3 landing (sim, 2026-10-29)."""
    start_x = {}; end_x = {}
    groups = [[i - 1 for i in sc["flights"]] for sc in staircases] or [[i] for i in range(len(flights))]
    for ids in groups:
        start_x[ids[0]] = end_margin; end_x[ids[-1]] = end_margin
    for k in range(len(flights)):
        start_x[k] = max(start_x.get(k, 0.0), flight_margin); end_x[k] = max(end_x.get(k, 0.0), flight_margin)
    corners = []
    for k, f in enumerate(flights):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        for s in (f["s0"] - start_x.get(k, 0.0), s_end(f) + end_x.get(k, 0.0)):
            for l in (f["l_lo"], f["l_hi"]):
                corners.append(p0 + ax * s + lat * l)
    c = np.array(corners)
    x0, y0 = np.floor(c.min(0) / res) * res - 1.0
    x1, y1 = np.ceil(c.max(0) / res) * res + 1.0
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    yy, xx = np.mgrid[0:H, 0:W]
    X = x0 + (xx + 0.5) * res; Y = y0 + (yy + 0.5) * res
    grid = np.zeros((H, W), np.int8)
    for k, f in enumerate(flights):
        p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
        s = (X - p0[0]) * ax[0] + (Y - p0[1]) * ax[1]; l = (X - p0[0]) * lat[0] + (Y - p0[1]) * lat[1]
        grid[(s >= f["s0"] - start_x.get(k, 0.0)) & (s <= s_end(f) + end_x.get(k, 0.0)) & (l >= f["l_lo"] + lateral_margin) & (l <= f["l_hi"] - lateral_margin)] = 100
    P = np.stack([X.ravel(), Y.ravel()], 1); n_land = 0
    land = np.zeros((H, W), bool)                          # landings: part of the staircase core (see wall cut below)
    for ids in groups:
        for i, j in zip(ids, ids[1:]):
            fa, fb = flights[i], flights[j]
            pa, axa, la = np.array(fa["p0"]), np.array(fa["axis"]), np.array(fa["lat"])
            pb, axb, lb = np.array(fb["p0"]), np.array(fb["axis"]), np.array(fb["lat"])
            top = pa + axa * s_end(fa); bot = pb + axb * fb["s0"]
            if np.linalg.norm(bot - top) > max_landing:
                continue
            e = [(pb + lb * fb["l_lo"] - pa) @ la, (pb + lb * fb["l_hi"] - pa) @ la]
            lo, hi = max(fa["l_lo"], min(e)) + lateral_margin, min(fa["l_hi"], max(e)) - lateral_margin
            if hi - lo < 0.5:
                continue
            ta = (top - pa) @ la; tb = (bot - pa) @ la
            poly = [top + la * (lo - ta), top + la * (hi - ta), bot + la * (hi - tb), bot + la * (lo - tb)]
            lm = MPath(np.array(poly)).contains_points(P).reshape(H, W); grid[lm] = 100; land |= lm; n_land += 1
    n_wall = 0
    if map_yaml and os.path.isfile(map_yaml):
        core = land.copy()                                 # flights + landings: the 2D map draws risers/landing edges occupied there
        for f in flights:
            p0, ax, lat = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"])
            s = (X - p0[0]) * ax[0] + (Y - p0[1]) * ax[1]; l = (X - p0[0]) * lat[0] + (Y - p0[1]) * lat[1]
            core |= (s >= f["s0"] - 0.1) & (s <= s_end(f) + 0.1) & (l >= f["l_lo"] + lateral_margin) & (l <= f["l_hi"] - lateral_margin)
        w = wall_mask(map_yaml, X, Y, wall_margin) & ~core
        n_wall = int((w & (grid >= 50)).sum()); grid[w] = 0
    return grid, x0, y0, n_land, n_wall


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
