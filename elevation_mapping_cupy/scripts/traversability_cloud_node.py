#!/usr/bin/env python3
"""Turn the safety layer into an obstacle cloud for octomap.

Feeding octomap the raw lidar makes it define "obstacle" as a height band in
the odom frame, which cannot tell a ramp from a wall: drive up the ramp and the
surface under the robot reads 98% occupied. The terrain chain already answers
the right question per cell, so this node hands octomap that answer instead of
the geometry it was derived from.

It reads safety rather than drivability. Drivability is geometry alone, and a
roadway is geometrically perfect -- flat, crossable, and exactly where the
robot must not go. Taking the geometric layer here would drop the semantic
verdict before it ever reached the costmap, which is the one place it has to
arrive.

Cells below the threshold become endpoints and everything else emits nothing.
octomap accumulates them with its usual log-odds sensor model, so
/projected_map is a probabilistic 2D costmap whose obstacles are cells the
robot should not enter rather than cells that happen to be tall. Being a
global octree, it also outlives the elevation map's rolling window.

Points go out at z=0 in a robot-centred, rotation-free frame, so every ray is
horizontal and the projection is exactly the traversability decision at any
altitude.

Stair cells are deliberately NOT obstacles here. Baked into the main grid
they would survive into the saved global map as permanently impassable, and
no stair route could ever be planned -- even in the gait that climbs them.
So the march treats stairs as passable ground, the main grid leaves them
free, and the blocking moves to where it can be revoked: the stairs grid,
held closed by Nav2's KeepoutFilter until the supervisor switches gait.
The same cells stay flagged in /stairs/projected_map, so nothing is lost;
what changes is which map carries the veto.

Free space is emitted explicitly, as a synthetic scan. Emitting only obstacle
endpoints left free space to the rays that happened to pass toward an
obstacle: ground the robot had driven straight across stayed unknown forever,
a vanished obstacle could only be erased if some farther obstacle put a ray
through it, and everything behind an object read as a wall of unknown. So the
node marches the safety grid outward along a fan of bearings; a bearing that
hits an unsafe cell contributes an obstacle endpoint there, and a bearing
that stays known-safe for its whole march contributes a beyond-max-range
point, which octomap truncates into a pure free ray. Same-frame obstacle
insertions win over free rays inside octomap, so a free ray crossing a live
obstacle costs nothing; a stale one it erodes, which is the point.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster


class TraversabilityCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("traversability_cloud_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/elevation_mapping_node/elevation_map_terrain"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/traversability/obstacles"
        ).value
        # safety, not drivability: drivability is geometry alone, and a
        # roadway is geometrically fine. Reading the geometric layer here
        # would drop the semantic verdict before it ever reached the
        # costmap, which is the one place it needs to arrive.
        self.layer = self.declare_parameter("layer", "safety").value
        # Below this a cell becomes an octomap obstacle. It is a policy knob,
        # not a physical one: the limits in the plugin config decide what the
        # score means, this decides where to cut. 0.15 m stair risers score
        # 0.25 against a 0.20 m limit, so 0.4 calls them obstacles and 0.2
        # leaves them climbable.
        self.threshold = self.declare_parameter("threshold", 0.4).value
        self.map_frame = self.declare_parameter("map_frame", "odom").value
        self.cloud_frame = self.declare_parameter("cloud_frame", "trav_origin").value
        # The free-space fan. march_range must stay below octomap's
        # sensor_model/max_range so a clear bearing's endpoint lands beyond it
        # and truncates into a pure free ray; far_range is anything past that.
        self.bearings = int(self.declare_parameter("bearings", 720).value)
        self.march_range = float(self.declare_parameter("march_range", 4.4).value)
        self.far_range = float(self.declare_parameter("far_range", 6.0).value)

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(GridMap, self.input_topic, self.on_grid_map, 5)
        self._published = 0
        self.get_logger().info(
            f"Publishing cells with {self.layer} < {self.threshold} from "
            f"'{self.input_topic}' to '{self.output_topic}'."
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
        if "stairs" in layers:
            st = np.array(
                msg.data[layers.index("stairs")].data, dtype=np.float32
            ).reshape(h, w)
            # Conditionally traversable, so not an obstacle for the planner's
            # map; the keepout mask owns the veto. Lifting the value clear of
            # the threshold makes the march walk straight through the flight.
            values = np.where(
                np.isfinite(st) & (st > 0.5), self.threshold + 1.0, values
            )

        res = msg.info.resolution
        cx = msg.info.pose.position.x
        cy = msg.info.pose.position.y

        # March a fan of bearings outward from the map centre over the safety
        # grid (grid_map convention: row along -Y, column along -X). Sampled
        # at the cell size, vectorized over all bearings at once.
        steps = np.arange(res, self.march_range, res, dtype=np.float32)
        theta = np.linspace(0.0, 2 * np.pi, self.bearings, endpoint=False)
        ux, uy = np.cos(theta, dtype=np.float32), np.sin(theta, dtype=np.float32)
        px = ux[:, None] * steps[None, :]              # (B, R) offsets from centre
        py = uy[:, None] * steps[None, :]
        cc = np.clip((w / 2.0 - 0.5 - px / res).round().astype(np.int32), 0, w - 1)
        rr = np.clip((h / 2.0 - 0.5 - py / res).round().astype(np.int32), 0, h - 1)
        sampled = values[rr, cc]                       # (B, R)

        unsafe = np.isfinite(sampled) & (sampled < self.threshold)
        unknown = ~np.isfinite(sampled)
        blocked = unsafe | unknown
        first_block = np.where(blocked.any(axis=1), blocked.argmax(axis=1), steps.size)

        idx = np.arange(self.bearings)
        hit_unsafe = (first_block < steps.size) & unsafe[idx, np.minimum(first_block, steps.size - 1)]
        clear = first_block == steps.size

        # Obstacles: the first unsafe cell along the bearing.
        r_obs = steps[np.minimum(first_block[hit_unsafe], steps.size - 1)]
        dx = ux[hit_unsafe] * r_obs
        dy = uy[hit_unsafe] * r_obs
        # Free space: bearings that stayed known-safe for the whole march emit
        # a beyond-range point; octomap truncates it to a free ray. Bearings
        # stopped by UNKNOWN emit nothing at all -- unknown stays unknown.
        dx = np.concatenate([dx, ux[clear] * self.far_range]).astype(np.float32)
        dy = np.concatenate([dy, uy[clear] * self.far_range]).astype(np.float32)

        stamp = msg.header.stamp
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.cloud_frame
        tf.transform.translation.x = float(cx)
        tf.transform.translation.y = float(cy)
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf)

        self.pub.publish(self._make_cloud(dx, dy, stamp))
        self._published += 1
        self.get_logger().info(
            f"Bearings: {int(hit_unsafe.sum())} obstacle, {int(clear.sum())} free, "
            f"{int(self.bearings - hit_unsafe.sum() - clear.sum())} stopped at unknown "
            f"(frames: {self._published})",
            throttle_duration_sec=5.0,
        )

    def _make_cloud(self, dx: np.ndarray, dy: np.ndarray, stamp) -> PointCloud2:
        n = int(dx.size)
        points = np.zeros((n, 3), dtype=np.float32)
        points[:, 0] = dx
        points[:, 1] = dy
        # z stays 0: the rays are horizontal and the obstacle decision is
        # already altitude-free.

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
    node = TraversabilityCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
