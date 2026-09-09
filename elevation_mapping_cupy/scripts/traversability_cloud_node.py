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
        # Ascending confirmed only. The layer is signed and graded now, so
        # this is a cut rather than a rule: descent stays out because a
        # costmap cell cannot say "down here, but only in stair gait, only
        # square-on, only from the top" -- that is the supervisor's call, and
        # /stairs/projected_map is where it reads it.
        self.stairs_confidence = float(
            self.declare_parameter("stairs_confidence", 0.7).value
        )
        self.lift_descending = bool(
            self.declare_parameter("lift_descending", False).value
        )

        # An unsafe patch smaller than this is noise, not terrain: a curb, a
        # wall or a person paints dozens of cells, while a mis-projected
        # pixel paints one or two.
        self.min_blob_cells = int(self.declare_parameter("min_blob_cells", 4).value)
        # The fan. march_range stays under octomap's sensor_model/max_range so
        # a clear bearing's point lands beyond it and truncates to a free ray;
        # far_range is anything past that. blind_radius is the body filter's
        # own shadow, where unmeasured means "under the robot", not "unknown".
        self.bearings = int(self.declare_parameter("bearings", 720).value)
        # 3.5 m is where the bearings actually reach from a robot standing
        # still: measured on the live map, 32% of them run clear that far and
        # essentially none reach 4 m, because that is where the ring spacing
        # opens past the support window. Asking for more does not sense
        # further, it just returns nothing at all, and the map fills out past
        # this as the robot moves anyway.
        self.march_range = float(self.declare_parameter("march_range", 3.5).value)
        self.far_range = float(self.declare_parameter("far_range", 9.0).value)
        self.blind_radius = float(self.declare_parameter("blind_radius", 1.0).value)
        # How far either side of a cell the march will look for a measurement
        # before calling it unknown. A 32 beam lidar lays its rings about a
        # metre apart by 3 m out, so at 5 cm most cells are between rings and
        # have never been hit -- measured here, 1.5% of cells are valid at 3 m
        # from a standing robot. Judged cell by cell, every bearing runs into
        # the first gap between rings and stops, which is how a march over a
        # perfectly good surface came back with nothing at all. So a cell
        # counts as measured if anything within this window was, and it counts
        # as unsafe if the worst thing in that window was, which errs the only
        # way it is safe to err.
        #
        # The two questions want different windows and it matters. "Has this
        # been measured" has to reach as far as the rings are apart or free
        # space never opens up; "is there something here" has to stay tight or
        # every obstacle swells by the width of the window and seals gaps the
        # robot fits through. So known-ness reads wide and hazard reads narrow.
        self.support_cells = int(self.declare_parameter("support_cells", 15).value)
        # The mirror image of the stairs lift. A flight is structure the robot
        # can negotiate and gets promoted over what geometry says; an edge is
        # structure it falls off and gets demoted, whatever geometry says. And
        # geometry does say the wrong thing here: a kerb lip goes unmeasured,
        # so step reads 0.007 across a 0.12 m boundary and drivability calls
        # the roadway as walkable as the pavement. The drop layer measures the
        # same edge asymmetrically and reads 0.128.
        self.drop_layer = self.declare_parameter("drop_layer", "drop").value
        self.drop_threshold = float(
            self.declare_parameter("drop_threshold", 0.08).value
        )
        # And the same again for walls. The hill's south border came back
        # with two raw cells after a full pass: drivability's refusal there
        # was not turning into emitted obstacles reliably, and drop cannot
        # help because the face sits inside drop's flag-radius exemption.
        # The wall layer has no exemption, so this demotion guarantees that
        # a wall-sized face is emitted wherever it is measured.
        self.wall_layer = self.declare_parameter("wall_layer", "wall").value
        self.wall_threshold = float(
            self.declare_parameter("wall_threshold", 0.30).value
        )
        self.hazard_cells = int(self.declare_parameter("hazard_cells", 3).value)
        # Drop edges. A bearing that runs out of measured ground close by is
        # telling you something: within this radius the sensor sees all round
        # and anything solid would have stopped the march as an obstacle
        # first, so ground that simply ends is ground that fell away. That is
        # the kerb seen from the pavement -- the roadway below its lip sits in
        # the lip's own shadow, so the step filter has nothing to measure and
        # the boundary reads as open ground running into unknown.
        self.drop_edge_range = float(
            self.declare_parameter("drop_edge_range", 2.5).value
        )
        # A real drop casts a shadow that goes on: from the kerb the whole
        # roadway is hidden, metres of it. A gap in the returns is one or two
        # cells and has measured ground straight after. Requiring the unknown
        # to run this far, with solid ground for this much behind it, is what
        # separates the two -- without it every speckle in the sweep puts an
        # obstacle in the middle of clear pavement.
        self.drop_edge_shadow = float(
            self.declare_parameter("drop_edge_shadow", 0.6).value
        )
        self.drop_edge_support = float(
            self.declare_parameter("drop_edge_support", 0.25).value
        )
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
            conf = np.abs(st) if self.lift_descending else st
            mask = np.isfinite(st) & (conf >= self.stairs_confidence)
            # No morphology here. The layer already arrives as a region the
            # detector stands behind, and growing it a second time in the
            # consumer is how the cliff past the top of a flight came to be
            # painted walkable.
            #
            # np.maximum with a NaN left operand returns NaN, so unmeasured
            # cells stay unmeasured through the lift. That is load-bearing:
            # "tidying" this into np.nan_to_num would punch free space into
            # the map wherever the flag overlaps unseen ground.
            values = np.where(mask, np.maximum(values, self.stairs_score), values)

        if self.drop_layer in layers and self.drop_threshold > 0:
            ddata = msg.data[layers.index(self.drop_layer)]
            dp = np.array(ddata.data, dtype=np.float32).reshape(h, w)
            over = np.isfinite(dp) & (dp >= self.drop_threshold)
            # 0.0 is below any sane threshold, so this refuses the cell
            # outright rather than nudging it. Unmeasured cells keep their
            # NaN: an edge nobody has seen is not an edge yet.
            values = np.where(over, 0.0, values)

        if self.wall_layer in layers and self.wall_threshold > 0:
            wdata = msg.data[layers.index(self.wall_layer)]
            wl = np.array(wdata.data, dtype=np.float32).reshape(h, w)
            over = np.isfinite(wl) & (wl >= self.wall_threshold)
            values = np.where(over, 0.0, values)

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
        def offsets(rows, cols):
            # Straight to the robot-centred frame, so these are already what
            # the cloud carries.
            return (
                -(cols.astype(np.float32) - w / 2.0 + 0.5) * res,
                -(rows.astype(np.float32) - h / 2.0 + 0.5) * res,
            )

        # Every unsafe cell in the map used to go out as an obstacle, wherever
        # it sat. That is not a measurement, it is the contents of a buffer:
        # the map keeps cells the robot drove past an hour ago and cells the
        # camera painted onto a surface it was guessing at, and both went into
        # the grid as though they had just been seen. Worse, a cell in the
        # shadow behind something solid went out too -- an obstacle asserted in
        # the one place the sensor provably could not look.
        #
        # So the cloud is now cast, not dumped. Each bearing marches out and
        # reports the first thing it meets; past that the bearing has nothing
        # to say and says nothing, leaving the shadow to whatever the robot
        # learns when it walks round. This is also what puts the camera in its
        # place: a hazard only reaches the grid where a lidar-measured surface
        # carries it and stands in open line of sight.
        rows, cols = np.zeros(0, np.int64), np.zeros(0, np.int64)
        if not (self.bearings > 0 and self.march_range > 0):
            rows, cols = np.nonzero(unsafe_cells)
        dx, dy = offsets(rows, cols)

        if self.bearings > 0 and self.march_range > 0:
            steps = np.arange(res, self.march_range, res, dtype=np.float32)
            theta = np.linspace(0.0, 2 * np.pi, self.bearings, endpoint=False)
            ux = np.cos(theta, dtype=np.float32)
            uy = np.sin(theta, dtype=np.float32)
            mx = ux[:, None] * steps[None, :]
            my = uy[:, None] * steps[None, :]
            mc = np.clip((w / 2.0 - 0.5 - mx / res).round().astype(np.int32), 0, w - 1)
            mr = np.clip((h / 2.0 - 0.5 - my / res).round().astype(np.int32), 0, h - 1)
            finite = np.isfinite(values)
            worst = ndimage.minimum_filter(
                np.where(finite, values, np.inf),
                size=max(1, self.hazard_cells),
                mode="nearest",
            )
            supported = ndimage.maximum_filter(
                finite, size=max(1, self.support_cells), mode="nearest"
            )
            sam_worst = worst[mr, mc]
            sam_known = supported[mr, mc]
            unsafe_m = sam_known & np.isfinite(sam_worst) & (sam_worst < self.threshold)
            unknown_m = ~sam_known & (steps[None, :] >= self.blind_radius)
            blocked = unsafe_m | unknown_m
            any_blocked = blocked.any(axis=1)
            clear = ~any_blocked
            first = np.where(any_blocked, blocked.argmax(axis=1), steps.size)
            idx = np.arange(self.bearings)
            at = np.minimum(first, steps.size - 1)

            # the first solid cell on each bearing, deduplicated: close in,
            # many bearings land on one cell; at the far end the spacing is a
            # shade under the cell size, so a surface still comes out whole.
            struck = any_blocked & unsafe_m[idx, at]
            n_hit = 0
            if struck.any():
                seen = np.zeros((h, w), dtype=bool)
                seen[mr[idx[struck], at[struck]], mc[idx[struck], at[struck]]] = True
                hr, hc = np.nonzero(seen)
                hx, hy = offsets(hr, hc)
                dx = np.concatenate([dx, hx]).astype(np.float32)
                dy = np.concatenate([dy, hy]).astype(np.float32)
                n_hit = int(seen.sum())

            if clear.any():
                dx = np.concatenate([dx, ux[clear] * self.far_range]).astype(np.float32)
                dy = np.concatenate([dy, uy[clear] * self.far_range]).astype(np.float32)

            n_drop = 0
            shadow = int(round(self.drop_edge_shadow / res))
            support = int(round(self.drop_edge_support / res))
            if self.drop_edge_range > 0 and shadow > 0:
                # bearings that ran out of ground rather than into something,
                # near enough that nothing else explains it, and with room left
                # in the march to see the whole shadow before judging it
                stopped_unknown = any_blocked & unknown_m[idx, at] & ~unsafe_m[idx, at]
                near = (
                    stopped_unknown
                    & (steps[at] <= self.drop_edge_range)
                    & (first >= support)
                    & (first + shadow <= steps.size)
                )
                cand = np.nonzero(near)[0]
                for b in cand:
                    f = first[b]
                    if not unknown_m[b, f:f + shadow].all():
                        continue  # a hole in the returns, not a drop
                    bs, bc = mr[b, f - support:f], mc[b, f - support:f]
                    # measured, and nothing dangerous on it. The hazard window
                    # returns inf where it found no measurement of its own,
                    # which reads as passable and should: support has already
                    # said the ground is known, and the narrow window finding
                    # nothing on it is the definition of clear.
                    back_known = supported[bs, bc]
                    back_worst = worst[bs, bc]
                    # ground has to be measured and passable right up to the
                    # lip, or there is nothing to say the edge is where we
                    # think it is. Passable also exempts a flight or a ramp:
                    # those occlude their own far side, and that frontier is
                    # the one thing all the stairs work exists to keep open.
                    if not (back_known.all() and (back_worst >= self.threshold).all()):
                        continue
                    dx = np.concatenate([dx, [ux[b] * steps[f - 1]]]).astype(np.float32)
                    dy = np.concatenate([dy, [uy[b] * steps[f - 1]]]).astype(np.float32)
                    n_drop += 1

            self.get_logger().info(
                f"Bearings: {int(clear.sum())} clear, {n_hit} struck, "
                f"{n_drop} drop edges, of {self.bearings}",
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
