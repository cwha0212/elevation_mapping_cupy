#!/usr/bin/env python3
#
# The driving grid, minus what better evidence overrules.
#
# The octomap cannot un-see. A riser observed from across the pavement goes
# in as an obstacle -- correctly, at the time -- and once the cloud became a
# proper ray cast nothing is asserted behind the first hit, so no free ray
# ever revisits an old mark that sits on or behind structure. Marks from
# before a flight was recognised are permanent; ghosts of things that walked
# away are permanent; the planner's map only ever gets worse.
#
# This node republishes the driving grid with occupancy cleared where one of
# two licences holds, and refuses every clearance where a live hazard vetoes:
#
#   licence 1, memory: the gait core grid -- every cell that was ever
#     CONFIRMED stairs or ramp, undilated, closed only across flag-enclosed
#     gaps. What put a mark there is what the robot walks on in the other
#     gait.
#
#   licence 2, re-observation: ground measured THIS frame as confidently
#     safe with no drop under it. A ghost dies the moment the ground it
#     stood on is seen clearly again. The check reads the WORST of a 0.15 m
#     neighbourhood, not the one cell under the mark: a hazard line one cell
#     wide sits half a cell from perfectly safe ground, and a rounded lookup
#     that lands on the safe neighbour would eat the whole line -- the
#     thinner the hazard, the more surely it dies, which is backwards.
#
#   veto, live hazard: where the terrain right now shows a wall-sized face
#     or a drop within the same neighbourhood, no licence clears. The wall
#     layer carries no exemptions at all, so this holds even inside drop's
#     flag-radius exemption and under the gait channel's closing bulges --
#     which is precisely where a hill's side border lives.
#
import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from scipy import ndimage


class NavGridFuseNode(Node):
    def __init__(self) -> None:
        super().__init__("nav_grid_fuse_node")
        self.nav_topic = self.declare_parameter(
            "nav_topic", "/projected_map"
        ).value
        self.gait_topic = self.declare_parameter(
            "gait_topic", "/gait_core/projected_map"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/projected_map_nav"
        ).value
        # 0 against the core channel: the core is the flags verbatim, and
        # erasure should stop exactly where the evidence stops. (Against the
        # dilated keepout grid this must match its dilation instead; the
        # launch pins the core.)
        self.erode_cells = int(self.declare_parameter("erode_cells", 0).value)

        self.terrain_topic = self.declare_parameter(
            "terrain_topic", "/elevation_mapping_node/elevation_map_terrain"
        ).value
        self.safe_layer = self.declare_parameter("safe_layer", "safety").value
        self.drop_layer = self.declare_parameter("drop_layer", "drop").value
        self.wall_layer = self.declare_parameter("wall_layer", "wall").value
        self.safe_now = float(self.declare_parameter("safe_now", 0.7).value)
        self.drop_max = float(self.declare_parameter("drop_max", 0.08).value)
        self.wall_min = float(self.declare_parameter("wall_min", 0.30).value)
        # A tilted observer cannot re-certify ground. The odometry is
        # planar, so while the robot climbs, every scan is projected through
        # a pose wrong by the grade and the terrain written near it reads
        # smooth and safe where the hill's side border stands -- which both
        # lifts the veto and grants the licence, and the borders vanish
        # exactly while climbing. The map centre IS the robot, so its slope
        # says whether to trust this frame's re-observations at all.
        self.level_slope_deg = float(
            self.declare_parameter("level_slope_deg", 8.0).value
        )

        self.hazard_topic = self.declare_parameter(
            "hazard_topic", "/hazard/projected_map"
        ).value
        self.gait = None
        self.terrain = None
        self.hazard = None
        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, 5)
        self.create_subscription(OccupancyGrid, self.gait_topic, self.on_gait, 5)
        self.create_subscription(
            OccupancyGrid, self.hazard_topic, self.on_hazard, 5
        )
        self.create_subscription(GridMap, self.terrain_topic, self.on_terrain, 5)
        self.create_subscription(OccupancyGrid, self.nav_topic, self.on_nav, 5)
        self.get_logger().info(
            f"'{self.nav_topic}' minus licensed occupancy -> "
            f"'{self.output_topic}' (memory: '{self.gait_topic}', "
            f"relook: safety>={self.safe_now}, veto: wall>={self.wall_min} "
            f"or drop>={self.drop_max})."
        )

    def on_gait(self, msg: OccupancyGrid) -> None:
        self.gait = msg

    def on_hazard(self, msg: OccupancyGrid) -> None:
        self.hazard = msg

    def on_terrain(self, msg: GridMap) -> None:
        self.terrain = msg

    def _terrain_masks(self):
        """(relook-ok, hazard-veto) as world-indexable masks, or None."""
        m = self.terrain
        if m is None:
            return None
        names = list(m.layers)
        if self.safe_layer not in names:
            return None

        def layer(name):
            d = m.data[names.index(name)]
            h, w = d.layout.dim[0].size, d.layout.dim[1].size
            return np.array(d.data, dtype=np.float32).reshape(h, w)

        level = True
        if "slope" in names:
            sl = layer("slope")
            centre = sl[sl.shape[0] // 2, sl.shape[1] // 2]
            level = (not np.isfinite(centre)) or centre < self.level_slope_deg

        safe = layer(self.safe_layer)
        worst_safe = ndimage.minimum_filter(
            np.where(np.isfinite(safe), safe, -1.0), size=3, mode="nearest"
        )
        ok = (worst_safe >= self.safe_now) & level

        veto = np.zeros_like(ok)
        # The veto reaches wider (0.35 m) than the relook gate. Emitted
        # marks land up to the hazard window's half-width PLUS the strike
        # quantisation away from the wall or drop cell that caused them --
        # measured, the south border's marks sat 0.10-0.15 m south of the
        # wall reading. A veto narrower than that offset protects the hazard
        # cell and abandons its own marks. Wider is the safe direction: a
        # veto only ever declines to erase.
        if self.drop_layer in names:
            drop = layer(self.drop_layer)
            worst_drop = ndimage.maximum_filter(
                np.where(np.isfinite(drop), drop, 0.0), size=3, mode="nearest"
            )
            ok &= worst_drop < self.drop_max
            veto |= ndimage.maximum_filter(
                np.where(np.isfinite(drop), drop, 0.0), size=7, mode="nearest"
            ) >= self.drop_max
        if self.wall_layer in names:
            wall = layer(self.wall_layer)
            veto |= ndimage.maximum_filter(
                np.where(np.isfinite(wall), wall, 0.0), size=7, mode="nearest"
            ) >= self.wall_min
        return ok, veto, m.info

    @staticmethod
    def _lookup(mask, info, wx, wy):
        """Sample a robot-centred grid_map layer mask at world points."""
        h, w = mask.shape
        res = info.resolution
        # grid_map convention: row along -Y, column along -X of the centre
        tc = np.round(w / 2 - 0.5 - (wx - info.pose.position.x) / res).astype(int)
        tr = np.round(h / 2 - 0.5 - (wy - info.pose.position.y) / res).astype(int)
        inb = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        out = np.zeros(wx.size, dtype=bool)
        out[inb] = mask[tr[inb], tc[inb]]
        return out

    def on_nav(self, msg: OccupancyGrid) -> None:
        ni = msg.info
        na = np.array(msg.data, dtype=np.int8).reshape(ni.height, ni.width)
        rows, cols = np.nonzero(na > 50)
        if rows.size == 0:
            self.pub.publish(msg)
            return
        wx = ni.origin.position.x + (cols + 0.5) * ni.resolution
        wy = ni.origin.position.y + (rows + 0.5) * ni.resolution

        clear = np.zeros(rows.size, dtype=bool)

        g = self.gait
        if g is not None:
            gi = g.info
            ga = np.array(g.data, dtype=np.int8).reshape(gi.height, gi.width)
            mask = ga > 50
            if self.erode_cells > 0 and mask.any():
                mask = ndimage.binary_erosion(mask, iterations=self.erode_cells)
            if mask.any():
                gc = ((wx - gi.origin.position.x) / gi.resolution).astype(int)
                gr = ((wy - gi.origin.position.y) / gi.resolution).astype(int)
                inb = (gr >= 0) & (gr < gi.height) & (gc >= 0) & (gc < gi.width)
                clear[inb] = mask[gr[inb], gc[inb]]

        masks = self._terrain_masks()
        if masks is not None:
            ok, veto, ti = masks
            clear |= self._lookup(ok, ti, wx, wy)
            # the live veto outranks every licence; outside the terrain
            # window it cannot testify either way and the licences stand
            clear &= ~self._lookup(veto, ti, wx, wy)

        # And the remembered one. The live veto goes blind wherever the
        # terrain is occluded, and the far side of any structure always is:
        # driving on the left erased the right border and driving on the
        # right erased the left, because the world-anchored gait memory
        # kept its licence while the defence lost its witness. A cell that
        # was ever confidently a wall or a drop refuses erasure from memory,
        # occluded or not.
        hz = self.hazard
        if hz is not None:
            hi = hz.info
            ha = np.array(hz.data, dtype=np.int8).reshape(hi.height, hi.width)
            hmask = ha > 50
            if hmask.any():
                hmask = ndimage.binary_dilation(hmask, iterations=2)
                hc2 = ((wx - hi.origin.position.x) / hi.resolution).astype(int)
                hr2 = ((wy - hi.origin.position.y) / hi.resolution).astype(int)
                hin = ((hr2 >= 0) & (hr2 < hi.height)
                       & (hc2 >= 0) & (hc2 < hi.width))
                remembered = np.zeros(rows.size, dtype=bool)
                remembered[hin] = hmask[hr2[hin], hc2[hin]]
                clear &= ~remembered

        if clear.any():
            na[rows[clear], cols[clear]] = 0
            self.get_logger().info(
                f"cleared {int(clear.sum())} licensed marks",
                throttle_duration_sec=10.0,
            )

        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = na.flatten().tolist()
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = NavGridFuseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
