#!/usr/bin/env python3
#
# The driving grid, minus what the gait grid knows better.
#
# The octomap cannot un-see a stair. A riser observed from across the pavement
# goes in as an obstacle -- correctly, at the time -- and once the flight is
# recognised and lifted to passable, no free ray ever passes through the old
# marks to erase them: since the cloud became a proper ray cast, nothing is
# asserted behind the first hit, and the flight IS the first hit. So the marks
# from before recognition are permanent, the corridor over the flight stays
# lethal in the planner's map, and no amount of looking at the stairs fixes
# it, because looking at them was never the problem.
#
# The gait grid is the memory that resolves this. It accumulates every cell
# that was ever confidently stairs or ramp, which is precisely the region
# where an old occupancy mark should not be trusted: the thing that put the
# mark there is the thing the robot can walk on in the other gait. So this
# node republishes the driving grid with gait-marked occupancy cleared to
# free, and the planner reads the fused topic instead.
#
# The gait mask is eroded before it erases anything. The gait channel dilates
# its marks by 0.4 m on purpose (early mode switching), but an eraser must not
# inherit that reach: a stairwell wall sits flush against a real flight, and
# an eraser wider than the flight would eat it. Eroding by the same margin
# the channel added means only the detector's own footprint clears occupancy,
# while the dilated skirt keeps doing its actual job in the gait channel.
#
import numpy as np
import rclpy
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
            "gait_topic", "/stairs/projected_map"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/projected_map_nav"
        ).value
        # Two cells less than the gait channel's dilate_cells (8), and the
        # difference is the whole mechanism. Eroding by the full dilation
        # made the outer 0.4 m of the gait region a place nothing could ever
        # be erased -- and the stale marks live exactly there, because the
        # first riser IS the detection footprint's edge. Measured: 143 of the
        # flight's 221 lethal cells sat in that ring and the corridor stayed
        # shut; they only died once the robot faced the flight and the region
        # grew past them, which read as the eraser being slow. At 6, the
        # eraser reaches 0.1 m past the detector's own footprint, far short
        # of a stairwell wall -- the stairs filter's wall veto already stops
        # the footprint before one.
        self.erode_cells = int(self.declare_parameter("erode_cells", 6).value)

        self.gait = None
        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, 5)
        self.create_subscription(OccupancyGrid, self.gait_topic, self.on_gait, 5)
        self.create_subscription(OccupancyGrid, self.nav_topic, self.on_nav, 5)
        self.get_logger().info(
            f"'{self.nav_topic}' minus occupancy inside '{self.gait_topic}' "
            f"(eroded {self.erode_cells} cells) -> '{self.output_topic}'."
        )

    def on_gait(self, msg: OccupancyGrid) -> None:
        self.gait = msg

    def on_nav(self, msg: OccupancyGrid) -> None:
        if self.gait is None:
            self.pub.publish(msg)
            return

        g = self.gait
        gi, ni = g.info, msg.info
        ga = np.array(g.data, dtype=np.int8).reshape(gi.height, gi.width)
        mask = ga > 50
        if self.erode_cells > 0 and mask.any():
            mask = ndimage.binary_erosion(mask, iterations=self.erode_cells)
        if not mask.any():
            self.pub.publish(msg)
            return

        na = np.array(msg.data, dtype=np.int8).reshape(ni.height, ni.width)
        rows, cols = np.nonzero(na > 50)
        if rows.size:
            # world position of each occupied nav cell, looked up in the gait
            # grid -- the two grids share a frame but not an origin or extent
            wx = ni.origin.position.x + (cols + 0.5) * ni.resolution
            wy = ni.origin.position.y + (rows + 0.5) * ni.resolution
            gc = ((wx - gi.origin.position.x) / gi.resolution).astype(int)
            gr = ((wy - gi.origin.position.y) / gi.resolution).astype(int)
            inb = (gr >= 0) & (gr < gi.height) & (gc >= 0) & (gc < gi.width)
            clear = np.zeros(rows.size, dtype=bool)
            clear[inb] = mask[gr[inb], gc[inb]]
            if clear.any():
                na[rows[clear], cols[clear]] = 0
                self.get_logger().info(
                    f"cleared {int(clear.sum())} stale marks inside the gait "
                    f"region",
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
