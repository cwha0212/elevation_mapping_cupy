#!/usr/bin/env python3
"""Republish a live OccupancyGrid the way a map server would.

Nav2's static layer subscribes to its map topic transient-local, because a
map normally exists before the layer does and arrives exactly once. A live
grid publisher is volatile, and the two never connect: no error, no warning
in the costmap, just a layer that stays empty. This node sits between them.

It is also where the terrain grid's `unknown` is decided upon. The octomap
projection marks ground it has not judged as -1, and a costmap has to be
told whether that means "keep out" or "no information": keeping it at -1
leaves it to the costmap's own track_unknown_space, which is the honest
default and what this passes through unless asked otherwise.
"""
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)


class GridLatchNode(Node):
    def __init__(self) -> None:
        super().__init__("grid_latch_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/terrain/projected_map"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/terrain/map"
        ).value
        # What to do with cells the terrain chain never judged. "keep" passes
        # -1 through; "free" declares them open, which is a decision nobody
        # measured and is here only because some consumers cannot represent
        # unknown at all.
        self.unknown = str(self.declare_parameter("unknown", "keep").value)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, latched)
        self.create_subscription(OccupancyGrid, self.input_topic, self._on_grid, 5)
        self.n = 0
        self.get_logger().info(
            f"'{self.input_topic}' -> '{self.output_topic}' (transient local), "
            f"unknown={self.unknown}"
        )

    def _on_grid(self, msg: OccupancyGrid) -> None:
        if self.unknown == "free":
            a = np.array(msg.data, dtype=np.int8)
            a[a < 0] = 0
            msg.data = a.tolist()
        self.pub.publish(msg)
        self.n += 1
        self.get_logger().info(
            f"relayed {self.n} grids ({msg.info.width}x{msg.info.height})",
            throttle_duration_sec=10.0,
        )


def main() -> None:
    rclpy.init()
    node = GridLatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
