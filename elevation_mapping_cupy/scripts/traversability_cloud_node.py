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

Free space is emitted deliberately rather than left to fall out of the
obstacle rays. Relying on those rays makes open ground a by-product of
whatever obstacle happens to sit behind it, so the better the terrain reads,
the less of it gets cleared -- fix the limits so a staircase stops being an
obstacle and the ground in front of it goes unknown along with it. So the
node also marches a fan of bearings over the layer, and a bearing that stays
above the threshold for its whole march contributes one point past octomap's
max range, which that server truncates into a pure free ray. A bearing
stopped by an unmeasured cell contributes nothing: unknown stays unknown.

The march ignores unmeasured cells within blind_radius. The body filter
removes the sensor's own chassis returns, which leaves a ring of NaN around
the robot; without this every bearing would die in that ring before reaching
anything. The robot is standing on that ground, which is better evidence
than a range return.
"""

import numpy as np
from scipy import ndimage
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
        # A flagged stair flight is conditionally drivable: the risers score
        # under the cut on pure geometry, but a quadruped climbs them once it
        # switches gait. Cells the stairs layer marks are lifted to this
        # score before thresholding -- above the obstacle cut, deliberately
        # short of clean ground.
        self.stairs_score = float(self.declare_parameter("stairs_score", 0.6).value)
        # The flight's first steps never carry the flag: the sustained-climb
        # window straddles the flat ground they rise from, so the gain falls
        # short there. Growing the region onto neighbouring cells whose step
        # is riser-sized picks those up without spilling onto anything else
        # -- a wall steps far higher than this band, open ground far lower.
        self.grow_cells = int(self.declare_parameter("stairs_grow_cells", 8).value)
        # The fan. march_range stays under octomap's sensor_model/max_range so a
        # clear bearing's point lands beyond it and truncates to a free ray;
        # far_range is anything past that. blind_radius is the body filter's
        # own shadow, where unmeasured means "under the robot", not "unknown".
        # An unsafe patch smaller than this is noise, not terrain: a curb,
        # a wall or a person paints dozens of cells, while a mis-projected
        # pixel paints one or two.
        self.min_blob_cells = int(self.declare_parameter("min_blob_cells", 4).value)
        self.bearings = int(self.declare_parameter("bearings", 720).value)
        self.march_range = float(self.declare_parameter("march_range", 5.5).value)
        self.far_range = float(self.declare_parameter("far_range", 9.0).value)
        self.blind_radius = float(self.declare_parameter("blind_radius", 1.0).value)
        self.grow_step_range = [
            float(v) for v in self.declare_parameter(
                "stairs_grow_step_range", [0.09, 0.30]).value
        ]
        self.map_frame = self.declare_parameter("map_frame", "odom").value
        self.cloud_frame = self.declare_parameter("cloud_frame", "trav_origin").value

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
        if "stairs" in layers and self.stairs_score > 0:
            sdata = msg.data[layers.index("stairs")]
            st = np.array(sdata.data, dtype=np.float32).reshape(h, w)
            mask = np.isfinite(st) & (st > 0.5)
            # A flight is a REGION, not a scatter of points. The flag fires on
            # riser cells whose climb window held enough data, which leaves
            # holes wherever the window ran into the map's frontier -- and a
            # single unflagged riser line still spans the full width of the
            # stairs, so after inflation it closes the way up completely.
            # Closing merges the riser stripes into the flight they belong to,
            # filling holes solidifies it, and the final dilation covers the
            # first risers at the entrance (the climb window straddles
            # mid-flight, so those never carry the flag themselves).
            mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), bool))
            mask = ndimage.binary_fill_holes(mask)
            if self.grow_cells > 0 and "step" in layers:
                sd = msg.data[layers.index("step")]
                step_v = np.array(sd.data, dtype=np.float32).reshape(h, w)
                lo, hi = self.grow_step_range
                riser_like = np.isfinite(step_v) & (step_v >= lo) & (step_v <= hi)
                mask |= ndimage.binary_dilation(
                    mask, iterations=self.grow_cells) & riser_like
            # No blanket dilation here. Spreading the flag with no step or
            # validity check paints stairs_score over whatever adjoins the
            # flight -- and a flight ends in a landing edge or, in this
            # world, a 0.60 m cliff, so those cells were being called
            # walkable. Growth has to be earned, which is what the
            # riser-gated pass above does.
            #
            # np.maximum with a NaN left operand returns NaN, so unmeasured
            # cells stay unmeasured through the lift. That is load-bearing:
            # "tidying" this into np.nan_to_num would punch free space into
            # the map wherever the flag overlaps unseen ground.
            values = np.where(mask, np.maximum(values, self.stairs_score), values)

        res = msg.info.resolution
        cx = msg.info.pose.position.x
        cy = msg.info.pose.position.y

        # grid_map convention, as published: row runs along -Y, column along -X
        # about the map centre.
        unsafe_cells = np.isfinite(values) & (values < self.threshold)
        if self.min_blob_cells > 1:
            labels, n_labels = ndimage.label(unsafe_cells)
            if n_labels:
                sizes = ndimage.sum(unsafe_cells, labels, np.arange(1, n_labels + 1))
                small = np.isin(labels, np.nonzero(sizes < self.min_blob_cells)[0] + 1)
                unsafe_cells &= ~small
                values = np.where(small, self.threshold + 0.05, values)
        rows, cols = np.nonzero(unsafe_cells)
        # Straight to the robot-centred frame, so the offsets below are already
        # what the cloud carries.
        dx = -(cols.astype(np.float32) - w / 2.0 + 0.5) * res
        dy = -(rows.astype(np.float32) - h / 2.0 + 0.5) * res

        # Free-space fan.
        if self.bearings > 0 and self.march_range > 0:
            steps = np.arange(res, self.march_range, res, dtype=np.float32)
            theta = np.linspace(0.0, 2 * np.pi, self.bearings, endpoint=False)
            ux = np.cos(theta, dtype=np.float32)
            uy = np.sin(theta, dtype=np.float32)
            mx = ux[:, None] * steps[None, :]
            my = uy[:, None] * steps[None, :]
            mc = np.clip((w / 2.0 - 0.5 - mx / res).round().astype(np.int32), 0, w - 1)
            mr = np.clip((h / 2.0 - 0.5 - my / res).round().astype(np.int32), 0, h - 1)
            sampled = values[mr, mc]
            unsafe_m = np.isfinite(sampled) & (sampled < self.threshold)
            unknown_m = ~np.isfinite(sampled) & (steps[None, :] >= self.blind_radius)
            blocked = unsafe_m | unknown_m
            clear = ~blocked.any(axis=1)
            if clear.any():
                dx = np.concatenate([dx, ux[clear] * self.far_range]).astype(np.float32)
                dy = np.concatenate([dy, uy[clear] * self.far_range]).astype(np.float32)
            self.get_logger().info(
                f"Bearings: {int(clear.sum())} clear of {self.bearings}",
                throttle_duration_sec=5.0,
            )

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
            f"Obstacle cells this frame: {dx.size} (frames: {self._published})",
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
