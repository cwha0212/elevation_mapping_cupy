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
from scipy import ndimage
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
        # One channel, not one per terrain type. Stairs and slopes demand the
        # same gait switch, so a supervisor keying off this grid has no use
        # for knowing which of the two it is standing in front of -- and two
        # grids just mean two subscriptions and an OR the robot has to get
        # right. Layers combine per cell by largest magnitude, which keeps
        # the sign convention (+up, -down) and the grading across the union.
        self.layers = [
            str(v)
            for v in self.declare_parameter("layers", ["stairs", "ramp"]).value
        ]
        self.map_frame = self.declare_parameter("map_frame", "odom").value
        self.cloud_frame = self.declare_parameter(
            "cloud_frame", "stairs_origin"
        ).value

        # This channel closes ground off rather than opening it, so its
        # errors run the other way from the main grid's: marking a stair cell
        # that is not one costs a detour, missing one lets the robot walk on
        # in a gait that cannot handle it. Hence a low cut, both directions,
        # and a margin dilated around what was found.
        # Confirmed only. 0.3 admitted candidate-grade cells, and that was
        # defensible while this grid only closed ground off -- over-marking
        # cost a detour. The fuse eraser changed the stakes: this grid now
        # also authorises DELETING occupancy from the driving map, it never
        # clears, and one candidate-grade misfire while the robot climbed the
        # hill became a permanent licence that erased 25 cells of real
        # wall-adjacent occupancy. A channel with two consumers whose error
        # costs point in opposite directions has to satisfy the stricter one.
        self.min_confidence = float(
            self.declare_parameter("min_confidence", 0.6).value
        )
        # 0.4 m of margin, and it is doing a specific job: the detectors gate
        # on total gain inside a 1.05 m window, so the first half-window of a
        # shallow ramp can never flag itself -- measured, the 11 degree ramp's
        # approach edge came back 17% marked while its interior ran 64% and
        # everything stray was zero. Growing the marks covers the structural
        # blind strip at the foot, and errs by switching gait half a step
        # early, which is the safe direction for this channel by definition.
        self.dilate_cells = int(self.declare_parameter("dilate_cells", 8).value)
        # Nothing here is ever cleared, by design, so one frame of far-field
        # noise would be permanent.
        self.max_range = float(self.declare_parameter("max_range", 4.5).value)

        # A second, undilated cloud for the fuse eraser. The dilation above is
        # margin for early gait switching, and an eraser must not inherit it:
        # grown marks reach past a slope's side edge and erase the occupancy
        # that stands between the robot and a 0.6 m lateral drop. The raw
        # flags already draw the line the eraser needs -- a ramp's side rim
        # cells never flag (the fine window sees the drop), the entrance's
        # riser cells do -- so the core cloud is the flags verbatim.
        self.core_topic = self.declare_parameter(
            "core_topic", "/gait_core/cells"
        ).value
        # The core closes gaps of up to twice this many cells between flag
        # regions before it goes out. A crest between two ramps and the seam
        # of a flight are flat, so they never flag -- yet they are exactly
        # where pre-recognition occupancy accumulates, and without a licence
        # there the eraser can never touch it. Closing only bridges where
        # flags stand on BOTH sides, so a landing's far cliff, which has
        # flags on one side only, stays outside the licence.
        self.core_close_cells = int(
            self.declare_parameter("core_close_cells", 10).value
        )
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 5)
        self.core_pub = self.create_publisher(PointCloud2, self.core_topic, 5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(GridMap, self.input_topic, self.on_grid_map, 5)
        self._published = 0
        self.get_logger().info(
            f"Publishing cells of {self.layers} from '{self.input_topic}' "
            f"to '{self.output_topic}'."
        )

    def on_grid_map(self, msg: GridMap) -> None:
        layers = list(msg.layers)
        present = [name for name in self.layers if name in layers]
        if not present:
            self.get_logger().warning(
                f"None of {self.layers} in {layers}.", throttle_duration_sec=5.0
            )
            return

        values = None
        for name in present:
            data = msg.data[layers.index(name)]
            h = data.layout.dim[0].size
            w = data.layout.dim[1].size
            layer = np.array(data.data, dtype=np.float32).reshape(h, w)
            if values is None:
                values = layer
            else:
                stronger = np.abs(np.nan_to_num(layer)) > np.abs(
                    np.nan_to_num(values)
                )
                values = np.where(stronger, layer, values)

        # grid_map convention as published: row along -Y, column along -X of
        # the map centre, which is where the cloud frame sits.
        res = msg.info.resolution
        flag = np.isfinite(values) & (np.abs(values) >= self.min_confidence)
        if self.max_range > 0:
            ii = np.arange(h, dtype=np.float32) - h / 2.0 + 0.5
            jj = np.arange(w, dtype=np.float32) - w / 2.0 + 0.5
            near = np.hypot(ii[:, None], jj[None, :]) * res <= self.max_range
            flag &= near
        n_up = int((flag & (values > 0)).sum())
        n_down = int((flag & (values < 0)).sum())
        core = flag
        if self.core_close_cells > 0:
            core = ndimage.binary_closing(
                core, iterations=self.core_close_cells
            )
        # Two cells of reach past the flags. The detectors cannot flag the
        # half-window ring at a structure's entrance, and an eraser that
        # stops exactly at the flags leaves that ring's stale marks forever
        # -- measured as the entrance lines returning at the flight and the
        # hill foot, and the planner routing around a hill whose doorway
        # they blocked. Guarding the SIDES is no longer this channel's job:
        # the fuse's wall/drop veto refuses erasure on any real edge, so the
        # reach can afford to be generous where the veto has nothing to say.
        self._core_dilate = getattr(self, "_core_dilate", None)
        if self._core_dilate is None:
            self._core_dilate = int(
                self.declare_parameter("core_dilate_cells", 2).value
            )
        if self._core_dilate > 0 and core.any():
            core = ndimage.binary_dilation(core, iterations=self._core_dilate)
        core_rows, core_cols = np.nonzero(core)
        cdx = -(core_cols.astype(np.float32) - w / 2.0 + 0.5) * res
        cdy = -(core_rows.astype(np.float32) - h / 2.0 + 0.5) * res
        if self.dilate_cells > 0:
            flag = ndimage.binary_dilation(flag, iterations=self.dilate_cells)
        rows, cols = np.nonzero(flag)
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
        self.core_pub.publish(self._make_cloud(cdx, cdy, stamp))
        self._published += 1
        self.get_logger().info(
            f"Gait cells this frame: {dx.size} "
            f"(ascending {n_up}, descending {n_down}; frames: {self._published})",
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
