#!/usr/bin/env python3
"""Publish the stair flights as their own occupancy grid.

The main grid deliberately treats a flight as ground: a quadruped rated for
30 cm steps can climb it, and baking it in as an obstacle would leave the
saved map with no stair route at all. But the planner still has to know a
flight when it meets one, because entering it means changing gait first.

So the flags go out a second time on their own channel. Downstream, Nav2's
KeepoutFilter can hold /stairs/projected_map closed until the supervisor
switches the robot over, and open it afterwards; nothing about the main grid
has to change for that.

No free-space fan feeds this one. A staircase does not walk away, so once a
flight has been seen its cells stay marked; there is nothing to erode.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster


class StairsCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("stairs_cloud_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/elevation_mapping_node/elevation_map_terrain"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/stairs/cells"
        ).value
        self.layer = self.declare_parameter("layer", "stairs").value
        self.map_frame = self.declare_parameter("map_frame", "odom").value
        self.cloud_frame = self.declare_parameter(
            "cloud_frame", "stairs_origin"
        ).value

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(GridMap, self.input_topic, self.on_grid_map, 5)
        self._published = 0
        self.get_logger().info(
            f"Publishing '{self.layer}' cells from '{self.input_topic}' "
            f"to '{self.output_topic}'."
        )

    def on_grid_map(self, msg: GridMap) -> None:
        layers = list(msg.layers)
        if self.layer not in layers:
            self.get_logger().warning(
                f"Layer '{self.layer}' not in {layers}.", throttle_duration_sec=5.0
            )
            return

        data = msg.data[layers.index(self.layer)]
        h = data.layout.dim[0].size
        w = data.layout.dim[1].size
        values = np.array(data.data, dtype=np.float32).reshape(h, w)

        res = msg.info.resolution
        # grid_map convention as published: row along -Y, column along -X of
        # the map centre, which is where the cloud frame sits.
        rows, cols = np.nonzero(np.isfinite(values) & (values > 0.5))
        dx = -(cols.astype(np.float32) - w / 2.0 + 0.5) * res
        dy = -(rows.astype(np.float32) - h / 2.0 + 0.5) * res

        stamp = msg.header.stamp
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.cloud_frame
        tf.transform.translation.x = float(msg.info.pose.position.x)
        tf.transform.translation.y = float(msg.info.pose.position.y)
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf)

        self.pub.publish(self._make_cloud(dx, dy, stamp))
        self._published += 1
        self.get_logger().info(
            f"Stair cells this frame: {dx.size} (frames: {self._published})",
            throttle_duration_sec=5.0,
        )

    def _make_cloud(self, dx: np.ndarray, dy: np.ndarray, stamp) -> PointCloud2:
        n = int(dx.size)
        points = np.zeros((n, 3), dtype=np.float32)
        points[:, 0] = dx
        points[:, 1] = dy

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.cloud_frame
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StairsCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
