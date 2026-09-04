#!/usr/bin/env python3
"""Stair cells as an obstacle cloud for the stairs-only octomap.

The main grid must keep calling stairs blocked -- that is the safe default a
planner falls back on -- but "blocked because wall" and "blocked because
stairs" are different facts, and OccupancyGrid has one number for both. So
stairs get their own channel: this node emits the stairs layer's cells as
endpoints, a second octomap instance accumulates them, and /stairs/projected_map
comes out as a grid that annotates the main one. Nav2's KeepoutFilter can hold
it closed until the supervisor switches gait, at which point opening the
stairs is one filter toggle.

No free-space fan here, deliberately. Free space answers "where can I go",
which is the main grid's question; this channel answers "which of the blocked
cells are stairs", and a stair does not stop being one when the robot looks
away -- accumulation without erosion is the behaviour we want from a static
feature.
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
        self.cloud_frame = self.declare_parameter("cloud_frame", "stairs_origin").value

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(GridMap, self.input_topic, self.on_grid_map, 5)
        self._frames = 0

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
        cx, cy = msg.info.pose.position.x, msg.info.pose.position.y

        # grid_map convention: row along -Y, column along -X about the centre.
        rows, cols = np.nonzero(np.isfinite(values) & (values > 0.5))
        dx = -(cols.astype(np.float32) - w / 2.0 + 0.5) * res
        dy = -(rows.astype(np.float32) - h / 2.0 + 0.5) * res

        stamp = msg.header.stamp
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.cloud_frame
        tf.transform.translation.x = float(cx)
        tf.transform.translation.y = float(cy)
        tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf)

        n = int(dx.size)
        points = np.zeros((n, 3), dtype=np.float32)
        points[:, 0] = dx
        points[:, 1] = dy
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.cloud_frame
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name=f, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, f in enumerate(("x", "y", "z"))
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * n
        cloud.is_dense = True
        cloud.data = points.tobytes()
        self.pub.publish(cloud)

        self._frames += 1
        self.get_logger().info(
            f"Stair cells this frame: {n} (frames: {self._frames})",
            throttle_duration_sec=10.0,
        )


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
