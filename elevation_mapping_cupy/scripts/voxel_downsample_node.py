#!/usr/bin/env python3
"""Voxel-downsample a cloud before it reaches elevation mapping.

Mapping cost scales with the number of points, and a 0.05 m map cannot record
detail finer than a 0.05 m voxel: past that the extra points are averaged into
the same cells and pay only in GPU time. On the real robot that matters --
three merged lidars measured at 79% GR3D on their own, before SLAM or
segmentation ask for anything.

This sits in front of the mapper rather than inside navi_lidar's own
downsampler, which writes to the topic SLAM reads: thinning the cloud there
would degrade the pose estimate to save time in a consumer that does not need
the density.

Keeps the first point seen in each voxel rather than averaging. Averaging
would move points off the surfaces they were measured on, and a step edge is
exactly where that hurts.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2, PointField


class VoxelDownsampleNode(Node):
    def __init__(self) -> None:
        super().__init__("voxel_downsample_node")
        self.input_topic = self.declare_parameter("input_topic", "/lidar/points").value
        self.output_topic = self.declare_parameter(
            "output_topic", "/lidar/points_downsampled"
        ).value
        self.voxel_size = float(self.declare_parameter("voxel_size", 0.05).value)
        # Points beyond the map's own reach cost time and reach no cell.
        self.max_range = float(self.declare_parameter("max_range", 0.0).value)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 5)
        self.create_subscription(PointCloud2, self.input_topic, self.on_cloud, qos)
        self._in = 0
        self._out = 0
        self._frames = 0
        self.get_logger().info(
            f"Downsampling '{self.input_topic}' to '{self.output_topic}' "
            f"at {self.voxel_size} m"
            + (f", within {self.max_range} m" if self.max_range > 0 else "")
        )

    def on_cloud(self, msg: PointCloud2) -> None:
        pts = self._xyz(msg)
        if pts is None or pts.size == 0:
            self.pub.publish(msg)
            return

        if self.max_range > 0:
            pts = pts[np.linalg.norm(pts, axis=1) <= self.max_range]
            if pts.size == 0:
                return

        keys = np.floor(pts / self.voxel_size).astype(np.int64)
        # One representative per occupied voxel. np.unique on a structured view
        # is the cheap way to say "first of each key" without a Python loop.
        _, keep = np.unique(
            keys.view([("", keys.dtype)] * 3).ravel(), return_index=True
        )
        kept = np.ascontiguousarray(pts[np.sort(keep)], dtype=np.float32)

        self._in += len(pts)
        self._out += len(kept)
        self._frames += 1
        self.get_logger().info(
            f"Downsampled {self._in // max(self._frames, 1)} -> "
            f"{self._out // max(self._frames, 1)} points per frame "
            f"({100.0 * self._out / max(self._in, 1):.0f}% kept)",
            throttle_duration_sec=5.0,
        )
        self.pub.publish(self._make_cloud(kept, msg))

    def _xyz(self, msg: PointCloud2):
        """Read x, y, z out of the raw buffer.

        sensor_msgs_py's numpy reader refuses any cloud whose fields are not
        all one datatype, which rules out every real lidar message -- theirs
        carry intensity, ring and timestamp alongside float32 xyz. Reading the
        three offsets directly costs nothing and does not care.
        """
        offsets = {}
        for field in msg.fields:
            if field.name in ("x", "y", "z"):
                if field.datatype != PointField.FLOAT32 or field.count != 1:
                    self.get_logger().warning(
                        f"Field '{field.name}' is not a single float32; passing "
                        "the cloud through untouched.",
                        throttle_duration_sec=10.0,
                    )
                    return None
                offsets[field.name] = field.offset
        if len(offsets) != 3:
            self.get_logger().warning(
                "Cloud has no float32 xyz; passing it through untouched.",
                throttle_duration_sec=10.0,
            )
            return None

        n = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * msg.point_step)
        raw = raw.reshape(n, msg.point_step)
        cols = [
            raw[:, o:o + 4].copy().view(np.float32).reshape(n)
            for o in (offsets["x"], offsets["y"], offsets["z"])
        ]
        pts = np.stack(cols, axis=1)
        return pts[np.isfinite(pts).all(axis=1)]

    @staticmethod
    def _make_cloud(points: np.ndarray, source: PointCloud2) -> PointCloud2:
        out = PointCloud2()
        out.header = source.header
        out.height = 1
        out.width = int(points.shape[0])
        # Built fresh rather than copied: the output is packed float32 xyz
        # whatever the source carried alongside it.
        out.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z"))
        ]
        out.is_bigendian = False
        out.point_step = 12
        out.row_step = 12 * out.width
        out.is_dense = True
        out.data = points.tobytes()
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoxelDownsampleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
