#!/usr/bin/env python3
#
# A world-anchored height map, accumulated from the robot-centric one.
#
# The GPU map is a 12 m window that moves with the robot and forgets what
# leaves it; the octomaps remember, but they remember VERDICTS, with every
# threshold baked in at observation time. This keeps the heights themselves:
# a fixed canvas that each local map update paints its measured patch onto,
# so the terrain the robot has crossed stays queryable as terrain -- for
# look-ahead beyond the window, for post-hoc analysis, and for re-deriving
# any verdict later with different thresholds, which no occupancy grid can do.
#
# Enlarging the robot-centric map instead is the tempting wrong answer: its
# memory is trivial either way (a 40 m window is ~50 MB), but every plugin
# sweeps the whole grid per update and the measured GPU load is 79% at 12 m
# on real lidar -- 11x the cells is not on the table. This node is numpy on
# the CPU, a patch blit per update, and does not touch that budget.
#
# Fusion is newest-wins, on purpose. Averaging across revisits would smear
# odometry drift into permanent double surfaces; overwriting means a revisit
# repaints cleanly and drift shows up only as a transient seam. The stamp
# layer records when each cell was last painted, so a consumer can discount
# ground the robot has not seen in a long time -- which after enough drift
# is exactly the ground not to trust.
#
from array import array

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class GlobalHeightMapNode(Node):
    def __init__(self) -> None:
        super().__init__("global_height_map_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/elevation_mapping_node/elevation_map_raw"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/global_height_map"
        ).value
        self.layer = self.declare_parameter("layer", "elevation").value
        # 0.10, not the local map's 0.05: the canvas serves look-ahead and
        # analysis, not foot placement, and half the resolution is a quarter
        # of the cells to store, publish and render.
        self.resolution = float(self.declare_parameter("resolution", 0.10).value)
        self.extent = float(self.declare_parameter("extent", 80.0).value)
        self.publish_period = float(
            self.declare_parameter("publish_period", 2.0).value
        )

        n = int(round(self.extent / self.resolution))
        self.n = n
        self.height = np.full((n, n), np.nan, dtype=np.float32)
        self.stamp = np.full((n, n), np.nan, dtype=np.float32)
        self.origin = None      # world xy of the canvas centre, set on first map

        # How often the canvas actually repaints. The local map arrives at
        # map rate with a megabyte of layers per message, and just RECEIVING
        # that -- rclpy deserialises every delivery before the callback can
        # decline it -- measured at 45% of a core. So the subscription is
        # raw: bytes are pocketed for free, and one message per period gets
        # deserialised and painted. At 0.4 m/s the window moves 0.4 m
        # between 1 Hz paints of a 12 m window; nothing is lost.
        self.paint_period = float(
            self.declare_parameter("paint_period", 1.0).value
        )
        self._latest_raw = None
        self.pub = self.create_publisher(GridMap, self.output_topic, 1)
        self.create_subscription(
            GridMap, self.input_topic, self.on_raw, 5, raw=True
        )
        self.create_timer(self.paint_period, self.paint_latest)
        self.create_timer(self.publish_period, self.publish)
        self._painted = 0
        self.get_logger().info(
            f"{self.extent:.0f} m canvas at {self.resolution} m from "
            f"'{self.input_topic}' layer '{self.layer}'."
        )

    def on_raw(self, data: bytes) -> None:
        self._latest_raw = data

    def paint_latest(self) -> None:
        data, self._latest_raw = self._latest_raw, None
        if data is None:
            return
        self.on_map(deserialize_message(data, GridMap))

    def on_map(self, msg: GridMap) -> None:
        names = list(msg.layers)
        if self.layer not in names:
            return
        d = msg.data[names.index(self.layer)]
        h, w = d.layout.dim[0].size, d.layout.dim[1].size
        local = np.array(d.data, dtype=np.float32).reshape(h, w)
        res = msg.info.resolution
        cx = msg.info.pose.position.x
        cy = msg.info.pose.position.y
        if self.origin is None:
            # Anchor where mapping starts. A canvas this size is meant to
            # hold the whole site; if the robot ever walks off it, the edge
            # clip below just stops painting rather than wrapping.
            self.origin = (cx, cy)

        rows, cols = np.nonzero(np.isfinite(local))
        if rows.size == 0:
            return
        # grid_map convention: row runs along -Y, column along -X from centre
        wx = cx - (cols - w / 2.0 + 0.5) * res
        wy = cy - (rows - h / 2.0 + 0.5) * res
        gc = np.round((wx - self.origin[0]) / self.resolution
                      + self.n / 2.0 - 0.5).astype(int)
        gr = np.round((wy - self.origin[1]) / self.resolution
                      + self.n / 2.0 - 0.5).astype(int)
        ok = (gr >= 0) & (gr < self.n) & (gc >= 0) & (gc < self.n)
        if not ok.any():
            return
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        # Newest wins. Several local cells land in one canvas cell at half
        # resolution; nothing here needs them reconciled beyond "the last
        # measurement stands".
        self.height[gr[ok], gc[ok]] = local[rows[ok], cols[ok]]
        self.stamp[gr[ok], gc[ok]] = t
        self._painted += int(ok.sum())

    def publish(self) -> None:
        if self.origin is None:
            return
        # Measured, not guessed: serialising 1.28M floats through tolist()
        # every period cost 40% of a core, and most of the time nothing is
        # listening -- this canvas is a reference surface, not a control
        # input. Painting continues regardless; only the marshalling waits
        # for someone to actually want the result.
        if self.pub.get_subscription_count() == 0:
            return
        msg = GridMap()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.length_x = self.extent
        msg.info.length_y = self.extent
        msg.info.pose.position.x = float(self.origin[0])
        msg.info.pose.position.y = float(self.origin[1])
        msg.info.pose.orientation.w = 1.0
        msg.layers = ["elevation", "last_seen"]
        for arr in (self.height, self.stamp):
            a = Float32MultiArray()
            a.layout.dim = [
                MultiArrayDimension(label="column_index", size=self.n,
                                    stride=self.n * self.n),
                MultiArrayDimension(label="row_index", size=self.n,
                                    stride=self.n),
            ]
            # a typed buffer, not a python list: rclpy accepts array('f')
            # directly and skips a million PyFloat allocations per publish
            a.data = array("f", arr.ravel())
            msg.data.append(a)
        self.pub.publish(msg)
        self.get_logger().info(
            f"global canvas: {int(np.isfinite(self.height).sum())} cells known "
            f"({self._painted} paints)",
            throttle_duration_sec=30.0,
        )


def main() -> None:
    rclpy.init()
    node = GlobalHeightMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
