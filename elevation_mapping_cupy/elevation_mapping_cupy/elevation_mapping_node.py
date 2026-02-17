#!/usr/bin/env python3
import math
import numpy as np
import os
import threading
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional, Tuple, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.serialization import serialize_message, deserialize_message
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros
import tf2_py as tf2

from grid_map_msgs.msg import GridMap
from grid_map_msgs.srv import SetGridMap, ProcessFile
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float32MultiArray
import rosbag2_py

from elevation_mapping_cupy import ElevationMap, Parameter
from elevation_mapping_cupy.elevation_mapping import GridGeometry
from elevation_mapping_cupy.gridmap_utils import encode_layer_to_multiarray, decode_multiarray_to_rows_cols

try:
    import cupy as cp  # optional
except Exception:
    cp = None


# ------------------------------ Math utils ------------------------------


def _quat_to_rot3_f32(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Fast quaternion -> 3x3 rotation matrix, float32.
    Avoids tf_transformations.quaternion_matrix allocation.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    inv = 1.0 / math.sqrt(n)
    x = qx * inv
    y = qy * inv
    z = qz * inv
    w = qw * inv

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = 1.0 - 2.0 * (yy + zz)
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)

    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = 1.0 - 2.0 * (xx + zz)
    R[1, 2] = 2.0 * (yz - wx)

    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _time_msg_to_rclpy_time(stamp_msg) -> rclpy.time.Time:
    return rclpy.time.Time.from_msg(stamp_msg)


# ------------------------------ PointCloud decoding ------------------------------


class _PointCloudXYZDecoder:
    """
    Caches PointCloud2 XYZ layout per stream (sub_key) to reduce per-message overhead.
    Provides fast-paths for common layouts and avoids unnecessary copies when possible.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def decode_xyz_f32(self, msg: PointCloud2, sub_key: str) -> np.ndarray:
        if msg.is_bigendian:
            raise ValueError("PointCloud2 big-endian is not supported.")

        n_points = int(msg.width) * int(msg.height)
        if n_points <= 0:
            return np.empty((0, 3), dtype=np.float32)

        c = self._cache.get(sub_key)
        layout_sig = (msg.point_step, tuple((f.name, f.offset, f.datatype, f.count) for f in msg.fields))

        if c is None or c.get("layout_sig") != layout_sig:
            fields = {f.name: f for f in msg.fields}
            for name in ("x", "y", "z"):
                if name not in fields:
                    raise ValueError(f"PointCloud2 missing required field '{name}'")
                f = fields[name]
                if f.datatype != PointField.FLOAT32 or f.count != 1:
                    raise ValueError(
                        f"PointCloud2 field '{name}' must be FLOAT32 count=1, got datatype={f.datatype} count={f.count}"
                    )

            x_off = int(fields["x"].offset)
            y_off = int(fields["y"].offset)
            z_off = int(fields["z"].offset)
            pt_step = int(msg.point_step)

            common_xyz = (x_off == 0 and y_off == 4 and z_off == 8 and pt_step >= 12)

            c = {
                "layout_sig": layout_sig,
                "x_off": x_off,
                "y_off": y_off,
                "z_off": z_off,
                "pt_step": pt_step,
                "common_xyz": common_xyz,
            }
            self._cache[sub_key] = c

        pt_step = c["pt_step"]
        x_off = c["x_off"]
        y_off = c["y_off"]
        z_off = c["z_off"]

        # Fast path: tightly packed xyz float32
        if c["common_xyz"] and pt_step == 12:
            pts = np.frombuffer(msg.data, dtype=np.float32, count=n_points * 3).reshape((n_points, 3))
            if not msg.is_dense:
                good = np.isfinite(pts).all(axis=1)
                pts = pts[good]
            return pts

        # Semi-fast path: xyz first 3 floats, padded point_step
        if c["common_xyz"] and (pt_step % 4 == 0) and pt_step >= 12:
            cols = pt_step // 4
            raw = np.frombuffer(msg.data, dtype=np.float32, count=n_points * cols).reshape((n_points, cols))
            pts = np.ascontiguousarray(raw[:, 0:3], dtype=np.float32)
            if not msg.is_dense:
                good = np.isfinite(pts).all(axis=1)
                pts = pts[good]
            return pts

        # General path: arbitrary offsets, xyz float32
        dtype = np.dtype(
            {
                "names": ("x", "y", "z"),
                "formats": (np.float32, np.float32, np.float32),
                "offsets": (x_off, y_off, z_off),
                "itemsize": pt_step,
            }
        )
        arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)
        pts = np.empty((n_points, 3), dtype=np.float32)
        pts[:, 0] = arr["x"]
        pts[:, 1] = arr["y"]
        pts[:, 2] = arr["z"]

        if not msg.is_dense:
            good = np.isfinite(pts).all(axis=1)
            pts = pts[good]
        return pts


# ------------------------------ Node ------------------------------


class ElevationMappingNode(Node):
    def __init__(self):
        super().__init__(
            "elevation_mapping_node",
            automatically_declare_parameters_from_overrides=True,
            allow_undeclared_parameters=False,
        )

        # New depth filter params (declared so node runs even if YAML doesn't define them)
        self.declare_parameter("depth_filter_enabled", True)
        self.declare_parameter("depth_filter_axis", "x")      # "x" | "y" | "z"
        self.declare_parameter("depth_filter_max_m", 5.0)     # meters
        self.declare_parameter("depth_filter_use_abs", True) # if True, uses abs(axis)
        self.declare_parameter("depth_filter_gpu", True)     # optional cupy path
        self.declare_parameter("depth_filter_gpu_min_points", 200000)  # avoid GPU overhead on small clouds

        self._cb_fast = ReentrantCallbackGroup()
        self._cb_slow = ReentrantCallbackGroup()
        self._map_lock = threading.Lock()

        # Drop work rather than queue/backlog to keep latency low at 15 Hz
        self._drop_if_busy = True

        self._pcd_decoder = _PointCloudXYZDecoder()
        self._layer_buffers: Dict[str, np.ndarray] = {}

        self._last_stamp_msg = None
        self._last_time = None

        self._cp = cp

        self.root = get_package_share_directory("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        self.param = Parameter(use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file)

        self.initialize_ros()
        self.set_param_values_from_ros()
        self.param.subscriber_cfg = self.my_subscribers

        self.initialize_elevation_mapping()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()
        self.register_services()

    # ------------------ Initialization ------------------

    def initialize_elevation_mapping(self) -> None:
        self.param.update()
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0

        with self._map_lock:
            self._map = ElevationMap(self.param)
            self._cell_inner = int(self._map.cell_n - 2)
            self._map_q: Optional[Quaternion] = None
            self._map_t: Optional[Vector3] = None

        self.get_logger().info(
            f"Initialized map with length: {self._map.map_length}, resolution: {self._map.resolution}, cells: {self._map.cell_n}"
        )

    def initialize_ros(self) -> None:
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_ros_params()

    def get_ros_params(self) -> None:
        self.use_chainer = self.get_parameter("use_chainer").get_parameter_value().bool_value
        self.initialize_frame_id = self.get_parameter("initialize_frame_id").get_parameter_value().string_array_value
        self.initialize_tf_offset = self.get_parameter("initialize_tf_offset").get_parameter_value().double_array_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.corrected_map_frame = self.get_parameter("corrected_map_frame").get_parameter_value().string_value
        self.initialize_method = self.get_parameter("initialize_method").get_parameter_value().string_value
        self.position_lowpass_alpha = self.get_parameter("position_lowpass_alpha").get_parameter_value().double_value
        self.orientation_lowpass_alpha = self.get_parameter("orientation_lowpass_alpha").get_parameter_value().double_value
        self.recordable_fps = self.get_parameter("recordable_fps").get_parameter_value().double_value
        self.update_variance_fps = self.get_parameter("update_variance_fps").get_parameter_value().double_value
        self.time_interval = self.get_parameter("time_interval").get_parameter_value().double_value
        self.update_pose_fps = self.get_parameter("update_pose_fps").get_parameter_value().double_value
        self.initialize_tf_grid_size = self.get_parameter("initialize_tf_grid_size").get_parameter_value().double_value
        self.map_acquire_fps = self.get_parameter("map_acquire_fps").get_parameter_value().double_value
        self.publish_statistics_fps = self.get_parameter("publish_statistics_fps").get_parameter_value().double_value
        self.enable_pointcloud_publishing = self.get_parameter("enable_pointcloud_publishing").get_parameter_value().bool_value
        self.enable_normal_arrow_publishing = self.get_parameter("enable_normal_arrow_publishing").get_parameter_value().bool_value
        self.enable_drift_corrected_TF_publishing = self.get_parameter("enable_drift_corrected_TF_publishing").get_parameter_value().bool_value
        self.use_initializer_at_start = self.get_parameter("use_initializer_at_start").get_parameter_value().bool_value

        # Depth filter config
        self._depth_filter_enabled = self.get_parameter("depth_filter_enabled").get_parameter_value().bool_value
        self._depth_filter_axis = self.get_parameter("depth_filter_axis").get_parameter_value().string_value or "x"
        self._depth_filter_max_m = float(self.get_parameter("depth_filter_max_m").get_parameter_value().double_value)
        self._depth_filter_use_abs = self.get_parameter("depth_filter_use_abs").get_parameter_value().bool_value
        self._depth_filter_gpu = self.get_parameter("depth_filter_gpu").get_parameter_value().bool_value
        self._depth_filter_gpu_min_points = int(
            self.get_parameter("depth_filter_gpu_min_points").get_parameter_value().integer_value
        )

        axis_map = {"x": 0, "y": 1, "z": 2}
        self._depth_axis_idx = axis_map.get(self._depth_filter_axis.lower(), 0)
        if self._depth_filter_axis.lower() not in axis_map:
            self.get_logger().warning(
                f"depth_filter_axis='{self._depth_filter_axis}' invalid; using 'x'."
            )
            self._depth_filter_axis = "x"
            self._depth_axis_idx = 0

        if self._depth_filter_gpu and self._cp is None:
            self.get_logger().warning("depth_filter_gpu=True but cupy is not available; falling back to CPU.")
            self._depth_filter_gpu = False

        subscribers_params = self.get_parameters_by_prefix("subscribers")
        self.my_subscribers = {}
        for param_name, param_value in subscribers_params.items():
            parts = param_name.split(".")
            if len(parts) >= 2:
                sub_key, sub_param = parts[:2]
                self.my_subscribers.setdefault(sub_key, {})[sub_param] = param_value.value

        publishers_params = self.get_parameters_by_prefix("publishers")
        self.my_publishers = {}
        for param_name, param_value in publishers_params.items():
            parts = param_name.split(".")
            if len(parts) >= 2:
                pub_key, pub_param = parts[:2]
                self.my_publishers.setdefault(pub_key, {})[pub_param] = param_value.value

    def set_param_values_from_ros(self):
        self.param.use_chainer = self.use_chainer
        self.param.resolution = self.get_parameter("resolution").get_parameter_value().double_value
        self.param.map_length = self.get_parameter("map_length").get_parameter_value().double_value
        self.param.sensor_noise_factor = self.get_parameter("sensor_noise_factor").get_parameter_value().double_value
        self.param.mahalanobis_thresh = self.get_parameter("mahalanobis_thresh").get_parameter_value().double_value
        self.param.outlier_variance = self.get_parameter("outlier_variance").get_parameter_value().double_value
        self.param.drift_compensation_variance_inlier = self.get_parameter("drift_compensation_variance_inlier").get_parameter_value().double_value
        self.param.checker_layer = self.get_parameter("checker_layer").get_parameter_value().string_value
        self.param.max_drift = self.get_parameter("max_drift").get_parameter_value().double_value
        self.param.drift_compensation_alpha = self.get_parameter("drift_compensation_alpha").get_parameter_value().double_value
        self.param.time_variance = self.get_parameter("time_variance").get_parameter_value().double_value
        self.param.max_variance = self.get_parameter("max_variance").get_parameter_value().double_value
        self.param.initial_variance = self.get_parameter("initial_variance").get_parameter_value().double_value
        self.param.initialized_variance = self.get_parameter("initialized_variance").get_parameter_value().double_value
        self.param.traversability_inlier = self.get_parameter("traversability_inlier").get_parameter_value().double_value
        self.param.dilation_size = self.get_parameter("dilation_size").get_parameter_value().integer_value
        self.param.dilation_size_initialize = self.get_parameter("dilation_size_initialize").get_parameter_value().integer_value
        self.param.wall_num_thresh = self.get_parameter("wall_num_thresh").get_parameter_value().integer_value
        self.param.min_height_drift_cnt = self.get_parameter("min_height_drift_cnt").get_parameter_value().integer_value
        self.param.position_noise_thresh = self.get_parameter("position_noise_thresh").get_parameter_value().double_value
        self.param.orientation_noise_thresh = self.get_parameter("orientation_noise_thresh").get_parameter_value().double_value
        self.param.min_valid_distance = self.get_parameter("min_valid_distance").get_parameter_value().double_value
        self.param.max_height_range = self.get_parameter("max_height_range").get_parameter_value().double_value
        self.param.ramped_height_range_a = self.get_parameter("ramped_height_range_a").get_parameter_value().double_value
        self.param.ramped_height_range_b = self.get_parameter("ramped_height_range_b").get_parameter_value().double_value
        self.param.ramped_height_range_c = self.get_parameter("ramped_height_range_c").get_parameter_value().double_value
        self.param.max_ray_length = self.get_parameter("max_ray_length").get_parameter_value().double_value
        self.param.cleanup_step = self.get_parameter("cleanup_step").get_parameter_value().double_value
        self.param.cleanup_cos_thresh = self.get_parameter("cleanup_cos_thresh").get_parameter_value().double_value
        self.param.safe_thresh = self.get_parameter("safe_thresh").get_parameter_value().double_value
        self.param.safe_min_thresh = self.get_parameter("safe_min_thresh").get_parameter_value().double_value
        self.param.max_unsafe_n = self.get_parameter("max_unsafe_n").get_parameter_value().integer_value
        self.param.overlap_clear_range_xy = self.get_parameter("overlap_clear_range_xy").get_parameter_value().double_value
        self.param.overlap_clear_range_z = self.get_parameter("overlap_clear_range_z").get_parameter_value().double_value
        self.param.enable_edge_sharpen = self.get_parameter("enable_edge_sharpen").get_parameter_value().bool_value
        self.param.enable_visibility_cleanup = self.get_parameter("enable_visibility_cleanup").get_parameter_value().bool_value
        self.param.enable_drift_compensation = self.get_parameter("enable_drift_compensation").get_parameter_value().bool_value
        self.param.enable_overlap_clearance = self.get_parameter("enable_overlap_clearance").get_parameter_value().bool_value
        self.param.use_only_above_for_upper_bound = self.get_parameter("use_only_above_for_upper_bound").get_parameter_value().bool_value

        mask_param = self.get_parameter("masked_replace_service_mask_layer_name").get_parameter_value().string_value
        topic_param = self.get_parameter("save_map_default_topic").get_parameter_value().string_value
        storage_param = self.get_parameter("save_map_storage_id").get_parameter_value().string_value
        service_ns_param = self.get_parameter("service_namespace").get_parameter_value().string_value

        if not mask_param:
            raise ValueError("masked_replace_service_mask_layer_name must be a non-empty string")
        if not topic_param:
            raise ValueError("save_map_default_topic must be a non-empty string")
        if not storage_param:
            raise ValueError("save_map_storage_id must be a non-empty string")

        self.masked_replace_mask_layer_name = mask_param
        self.save_map_default_topic = topic_param
        self.save_map_storage_id = storage_param
        self.service_namespace = self._normalize_namespace(service_ns_param)

    # ------------------ ROS entities ------------------

    def register_subscribers(self) -> None:
        self._pointcloud_subs = {}

        # Low-latency QoS for 15 Hz pointcloud input: drop old frames
        qos_pcd = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        for key, config in self.my_subscribers.items():
            if config.get("data_type") != "pointcloud":
                raise ValueError(
                    f"Unsupported subscriber data_type='{config.get('data_type')}' for '{key}'. "
                    "Supported: pointcloud only."
                )
            if config.get("channels"):
                raise ValueError(f"Subscriber '{key}' sets 'channels', but semantic/rgb channels are not supported.")

            topic_name = config.get("topic_name")
            if not topic_name:
                raise ValueError(f"Subscriber '{key}' is missing required key 'topic_name'.")

            self._pointcloud_subs[key] = self.create_subscription(
                PointCloud2,
                topic_name,
                partial(self.pointcloud_callback, sub_key=key),
                qos_profile=qos_pcd,
                callback_group=self._cb_fast,
            )

    def register_publishers(self) -> None:
        self._publishers_dict = {}
        self._publishers_timers = []

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.get_name()}/{pub_key}"
            self._publishers_dict[pub_key] = self.create_publisher(GridMap, topic_name, 10)

            fps = float(pub_config.get("fps", 1.0) or 0.0)
            if fps <= 0.0:
                continue

            timer = self.create_timer(
                1.0 / fps,
                partial(self.publish_map, key=pub_key),
                callback_group=self._cb_slow,
            )
            self._publishers_timers.append(timer)

    def register_timers(self) -> None:
        pose_hz = float(self.update_pose_fps or 10.0)
        pose_hz = max(1e-3, pose_hz)

        self.time_pose_update = self.create_timer(
            1.0 / pose_hz,
            self.pose_update,
            callback_group=self._cb_fast,
        )
        self.timer_variance = self.create_timer(
            1.0 / max(1e-3, float(self.update_variance_fps or 1.0)),
            self.update_variance,
            callback_group=self._cb_fast,
        )
        self.timer_time = self.create_timer(
            max(1e-6, float(self.time_interval or 0.1)),
            self.update_time,
            callback_group=self._cb_fast,
        )

    def register_services(self) -> None:
        service_masked = self._resolve_service_name("masked_replace")
        service_save = self._resolve_service_name("save_map")
        service_load = self._resolve_service_name("load_map")

        self._srv_masked_replace = self.create_service(
            SetGridMap,
            service_masked,
            self.handle_masked_replace,
            callback_group=self._cb_slow,
        )
        self._srv_save_map = self.create_service(
            ProcessFile,
            service_save,
            self.handle_save_map,
            callback_group=self._cb_slow,
        )
        self._srv_load_map = self.create_service(
            ProcessFile,
            service_load,
            self.handle_load_map,
            callback_group=self._cb_slow,
        )

    # ------------------ Depth filter ------------------

    def _apply_depth_filter(self, pts: np.ndarray) -> np.ndarray:
        """
        Drops points whose chosen axis value exceeds depth_filter_max_m.
        Axis is in the incoming point cloud frame (x/y/z of the PointCloud2 fields).
        """
        if (not self._depth_filter_enabled) or pts.size == 0:
            return pts

        idx = self._depth_axis_idx
        max_m = self._depth_filter_max_m

        # Optional GPU path (note: H2D/D2H copies; beneficial only for very large clouds)
        if (
            self._depth_filter_gpu
            and self._cp is not None
            and int(pts.shape[0]) >= int(self._depth_filter_gpu_min_points)
        ):
            cpts = self._cp.asarray(pts)  # H2D
            if self._depth_filter_use_abs:
                vals = self._cp.abs(cpts[:, idx])
            else:
                vals = cpts[:, idx]
            keep = vals <= max_m
            cpts = cpts[keep]
            return self._cp.asnumpy(cpts)  # D2H

        # CPU path
        if self._depth_filter_use_abs:
            vals = np.abs(pts[:, idx])
        else:
            vals = pts[:, idx]
        keep = vals <= max_m
        if bool(keep.all()):
            return pts
        return pts[keep]

    # ------------------ TF helper ------------------

    def safe_lookup_transform(self, target_frame: str, source_frame: str, time_in: rclpy.time.Time):
        try:
            if self._tf_buffer.can_transform(target_frame, source_frame, time_in, timeout=Duration(seconds=0.0)):
                return self._tf_buffer.lookup_transform(target_frame, source_frame, time_in)
            return self._tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except (
            tf2.LookupException,
            tf2.ConnectivityException,
            tf2.ExtrapolationException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warning(
                f"Transform from '{source_frame}' to '{target_frame}' not available: {e}",
                throttle_duration_sec=5.0,
            )
            return None
        except Exception as e:
            self.get_logger().warning(
                f"Unexpected TF2 error for transform from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=5.0,
            )
            return None

    # ------------------ Publishing ------------------

    def publish_map(self, key: str) -> None:
        if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
            return

        try:
            if self._map_q is None:
                return

            pub_cfg = self.my_publishers.get(key, {})
            layers = list(pub_cfg.get("layers", []) or [])
            basic_layers = list(pub_cfg.get("basic_layers", []) or [])

            stamp_msg = self._last_stamp_msg if self._last_stamp_msg is not None else self.get_clock().now().to_msg()

            center = np.zeros((1, 3), dtype=np.float32)
            self._map.get_center_position(center)
            center = center[0]

            resolution = float(self._map.resolution)
            actual_map_length = float((self._map.cell_n - 2) * self._map.resolution)

            map_t = self._map_t
            map_q = self._map_q

            layer_bufs: List[Tuple[str, np.ndarray]] = []
            for layer in layers:
                buf = self._layer_buffers.get(layer)
                if buf is None or buf.shape != (self._cell_inner, self._cell_inner):
                    buf = np.zeros((self._cell_inner, self._cell_inner), dtype=np.float32)
                    self._layer_buffers[layer] = buf
                self._map.get_map_with_name_ref(layer, buf)
                layer_bufs.append((layer, buf.copy()))
        finally:
            try:
                self._map_lock.release()
            except RuntimeError:
                pass

        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = stamp_msg
        gm.info.resolution = resolution
        gm.info.length_x = actual_map_length
        gm.info.length_y = actual_map_length

        if map_t is not None:
            gm.info.pose.position.x = float(map_t.x)
            gm.info.pose.position.y = float(map_t.y)
            gm.info.pose.position.z = 0.0
        else:
            gm.info.pose.position.x = float(center[0])
            gm.info.pose.position.y = float(center[1])
            gm.info.pose.position.z = 0.0

        gm.info.pose.orientation.x = float(map_q.x)
        gm.info.pose.orientation.y = float(map_q.y)
        gm.info.pose.orientation.z = float(map_q.z)
        gm.info.pose.orientation.w = float(map_q.w)

        gm.layers = []
        gm.basic_layers = basic_layers
        gm.data = []
        gm.outer_start_index = 0
        gm.inner_start_index = 0

        for layer, buf in layer_bufs:
            gm.layers.append(layer)
            gm.data.append(self._numpy_to_multiarray(buf, layout="gridmap_column"))

        pub = self._publishers_dict.get(key)
        if pub is not None:
            pub.publish(gm)

    # ------------------ Services ------------------

    def handle_masked_replace(self, request, response):
        try:
            layer_arrays, geometry = self._grid_map_to_numpy(request.map)
            mask = layer_arrays.pop(self.masked_replace_mask_layer_name, None)
            if not layer_arrays:
                raise ValueError("Provide at least one data layer to update.")

            if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
                raise RuntimeError("Map busy; try again.")

            try:
                self._map.apply_masked_replace(layer_arrays, mask, geometry)
            finally:
                if self._map_lock.locked():
                    self._map_lock.release()

            self._republish_all_once()
            self.get_logger().info(f"masked_replace updated {len(layer_arrays)} layer(s).")
            response.success = True
        except Exception as exc:
            self.get_logger().error(f"masked_replace failed: {exc}")
            response.success = False
        return response

    def handle_save_map(self, request, response):
        try:
            fused_path, raw_path = self._prepare_bag_paths(request.file_path)

            topic_base = request.topic_name or self.save_map_default_topic
            fused_topic = self._resolve_topic_name(topic_base)
            raw_topic = self._resolve_topic_name(f"{topic_base}_raw")

            fused_layer_names = self._collect_fused_layer_names()

            if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
                raise RuntimeError("Map busy; try again.")

            try:
                fused_layers = self._map.export_layers(fused_layer_names)
                raw_layer_names = self._map.list_layers()
                raw_layers = self._map.export_layers(raw_layer_names)
            finally:
                if self._map_lock.locked():
                    self._map_lock.release()

            gm_fused = self._build_grid_map_message(fused_layer_names, fused_layers, self._collect_basic_layers())
            gm_raw = self._build_grid_map_message(raw_layer_names, raw_layers, ["elevation"])

            self._write_grid_map_bag(fused_path, fused_topic, gm_fused)
            self._write_grid_map_bag(raw_path, raw_topic, gm_raw)

            response.success = True
        except Exception as exc:
            self.get_logger().error(f"save_map failed: {exc}")
            response.success = False
        return response

    def handle_load_map(self, request, response):
        try:
            fused_path = Path(request.file_path).expanduser().resolve()
            raw_path = Path(f"{fused_path}_raw")
            if not fused_path.exists():
                raise FileNotFoundError(f"Fused map bag '{fused_path}' does not exist.")
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw map bag '{raw_path}' does not exist.")

            topic_base = request.topic_name or self.save_map_default_topic
            fused_topic = self._resolve_topic_name(topic_base)
            raw_topic = self._resolve_topic_name(f"{topic_base}_raw")

            fused_msg = self._read_latest_grid_map(fused_path, fused_topic)
            raw_msg = self._read_latest_grid_map(raw_path, raw_topic)

            fused_layers, _ = self._grid_map_to_numpy(fused_msg)
            raw_layers, geometry = self._grid_map_to_numpy(raw_msg)

            if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
                raise RuntimeError("Map busy; try again.")

            try:
                self._map.set_full_map(fused_layers, raw_layers, geometry)

                pose_position = raw_msg.info.pose.position
                pose_orientation = raw_msg.info.pose.orientation
                self._map_t = Vector3(x=pose_position.x, y=pose_position.y, z=pose_position.z)
                self._map_q = Quaternion(
                    x=pose_orientation.x,
                    y=pose_orientation.y,
                    z=pose_orientation.z,
                    w=pose_orientation.w,
                )

                self._last_stamp_msg = self.get_clock().now().to_msg()
                self._last_time = rclpy.time.Time()
            finally:
                if self._map_lock.locked():
                    self._map_lock.release()

            self._republish_all_once()
            response.success = True
        except Exception as exc:
            self.get_logger().error(f"load_map failed: {exc}")
            response.success = False
        return response

    # ------------------ Conversions / bag IO ------------------

    def _grid_map_to_numpy(self, grid_map_msg: GridMap):
        if len(grid_map_msg.layers) != len(grid_map_msg.data):
            raise ValueError("Mismatch between GridMap layers and data arrays.")

        arrays: Dict[str, np.ndarray] = {}
        for name, array_msg in zip(grid_map_msg.layers, grid_map_msg.data):
            arrays[name] = decode_multiarray_to_rows_cols(name, array_msg)

        center = np.array(
            [
                grid_map_msg.info.pose.position.x,
                grid_map_msg.info.pose.position.y,
                grid_map_msg.info.pose.position.z,
            ],
            dtype=np.float32,
        )
        orientation = np.array(
            [
                grid_map_msg.info.pose.orientation.x,
                grid_map_msg.info.pose.orientation.y,
                grid_map_msg.info.pose.orientation.z,
                grid_map_msg.info.pose.orientation.w,
            ],
            dtype=np.float32,
        )

        geometry = GridGeometry(
            length_x=grid_map_msg.info.length_x,
            length_y=grid_map_msg.info.length_y,
            resolution=grid_map_msg.info.resolution,
            center=center,
            orientation=orientation,
        )
        return arrays, geometry

    def _collect_fused_layer_names(self) -> List[str]:
        fused: List[str] = []
        for config in self.my_publishers.values():
            fused.extend(config.get("layers", []) or [])
        if not fused:
            fused = ["elevation"]
        ordered: List[str] = []
        for name in fused:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _collect_basic_layers(self) -> List[str]:
        basics: List[str] = []
        for config in self.my_publishers.values():
            basics.extend(config.get("basic_layers", []) or [])
        if not basics:
            basics = ["elevation"]
        ordered: List[str] = []
        for name in basics:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _build_grid_map_message(
        self,
        layer_names: List[str],
        layer_data: Dict[str, np.ndarray],
        basic_layers: List[str],
    ) -> GridMap:
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self._last_stamp_msg if self._last_stamp_msg is not None else self.get_clock().now().to_msg()
        gm.info.resolution = float(self._map.resolution)
        actual_map_length = float((self._map.cell_n - 2) * self._map.resolution)
        gm.info.length_x = actual_map_length
        gm.info.length_y = actual_map_length

        center = np.zeros((1, 3), dtype=np.float32)
        if self._map_lock.acquire(blocking=False):
            try:
                self._map.get_center_position(center)
            finally:
                self._map_lock.release()
        center = center[0]

        gm.info.pose.position.x = float(center[0])
        gm.info.pose.position.y = float(center[1])
        gm.info.pose.position.z = float(center[2])

        if self._map_q is not None:
            gm.info.pose.orientation.x = float(self._map_q.x)
            gm.info.pose.orientation.y = float(self._map_q.y)
            gm.info.pose.orientation.z = float(self._map_q.z)
            gm.info.pose.orientation.w = float(self._map_q.w)
        else:
            gm.info.pose.orientation.w = 1.0

        gm.layers = []
        gm.basic_layers = basic_layers
        gm.data = []
        gm.outer_start_index = 0
        gm.inner_start_index = 0

        for name in layer_names:
            data = layer_data.get(name)
            if data is None:
                continue
            gm.layers.append(name)
            gm.data.append(self._numpy_to_multiarray(data))

        return gm

    def _numpy_to_multiarray(self, data: np.ndarray, layout: str = "gridmap_column") -> Float32MultiArray:
        return encode_layer_to_multiarray(data, layout=layout)

    def _resolve_service_name(self, suffix: str) -> str:
        base = self.service_namespace
        if not base:
            base = f"/{self.get_name()}"
        return f"{base}/{suffix}".replace("//", "/")

    def _resolve_topic_name(self, topic: str) -> str:
        topic = topic.strip("/") or self.save_map_default_topic
        base = self.service_namespace
        if not base:
            base = f"/{self.get_name()}"
        return f"{base}/{topic}".replace("//", "/")

    def _prepare_bag_paths(self, file_path: str):
        if not file_path:
            raise ValueError("file_path must be provided.")
        fused_path = Path(file_path).expanduser().resolve()
        raw_path = Path(f"{fused_path}_raw")
        if fused_path.exists():
            raise FileExistsError(f"Bag path '{fused_path}' already exists.")
        if raw_path.exists():
            raise FileExistsError(f"Bag path '{raw_path}' already exists.")
        fused_path.parent.mkdir(parents=True, exist_ok=True)
        return fused_path, raw_path

    def _make_topic_metadata(self, topic: str) -> rosbag2_py.TopicMetadata:
        msg_type = "grid_map_msgs/msg/GridMap"
        serialization_format = "cdr"
        return rosbag2_py.TopicMetadata(0, topic, msg_type, serialization_format)

    def _write_grid_map_bag(self, path: Path, topic: str, grid_map_msg: GridMap) -> None:
        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id=self.save_map_storage_id)
        converter_options = rosbag2_py.ConverterOptions("", "")
        writer.open(storage_options, converter_options)
        writer.create_topic(self._make_topic_metadata(topic))
        writer.write(topic, serialize_message(grid_map_msg), self.get_clock().now().nanoseconds)

    def _read_latest_grid_map(self, path: Path, topic: str) -> GridMap:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id=self.save_map_storage_id)
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader.open(storage_options, converter_options)

        latest = None
        while reader.has_next():
            current_topic, data, _ = reader.read_next()
            if current_topic != topic:
                continue
            msg = deserialize_message(data, GridMap)
            latest = msg
        if latest is None:
            raise ValueError(f"No messages for topic '{topic}' in bag '{path}'.")
        return latest

    def _republish_all_once(self) -> None:
        if self._map_q is None:
            return
        for key in list(self._publishers_dict.keys()):
            self.publish_map(key)

    def _normalize_namespace(self, value: str) -> str:
        value = value.strip() if value else ""
        if not value:
            return ""
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    # ------------------ High-rate callbacks ------------------

    def pointcloud_callback(self, msg: PointCloud2, sub_key: str) -> None:
        self._last_stamp_msg = msg.header.stamp
        self._last_time = _time_msg_to_rclpy_time(msg.header.stamp)

        pts = self._pcd_decoder.decode_xyz_f32(msg, sub_key=sub_key)
        if pts.size == 0:
            return

        # Depth filter (drops points where axis value > max_m)
        pts = self._apply_depth_filter(pts)
        if pts.size == 0:
            return

        frame_sensor_id = msg.header.frame_id
        if not frame_sensor_id:
            return

        # TF lookup + R/t computation WITHOUT holding the map lock
        if frame_sensor_id == self.map_frame:
            t_np = np.zeros(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
        else:
            tf_time = self._last_time if self._last_time is not None else rclpy.time.Time()
            transform_sensor_to_map = self.safe_lookup_transform(self.map_frame, frame_sensor_id, tf_time)
            if transform_sensor_to_map is None:
                return

            t = transform_sensor_to_map.transform.translation
            q = transform_sensor_to_map.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = _quat_to_rot3_f32(q.x, q.y, q.z, q.w)

        # Acquire lock ONLY for map mutation
        if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
            return
        try:
            self._map.input_pointcloud(pts, ["x", "y", "z"], R, t_np, 0, 0)
            self._pointcloud_process_counter += 1
        finally:
            if self._map_lock.locked():
                self._map_lock.release()

    def pose_update(self) -> None:
        if self._last_time is None or self._last_stamp_msg is None:
            return

        transform = self.safe_lookup_transform(self.map_frame, self.base_frame, self._last_time)
        if transform is None:
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        rot = _quat_to_rot3_f32(q.x, q.y, q.z, q.w)

        if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
            return
        try:
            self._map.move_to(trans, rot)
            self._map_t = Vector3(x=t.x, y=t.y, z=t.z)
            self._map_q = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        finally:
            if self._map_lock.locked():
                self._map_lock.release()

    def update_variance(self) -> None:
        if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
            return
        try:
            self._map.update_variance()
        finally:
            if self._map_lock.locked():
                self._map_lock.release()

    def update_time(self) -> None:
        if self._drop_if_busy and not self._map_lock.acquire(blocking=False):
            return
        try:
            self._map.update_time()
        finally:
            if self._map_lock.locked():
                self._map_lock.release()

    def destroy_node(self) -> None:
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationMappingNode()

    # Multi-thread executor to reduce latency (map access protected by lock)
    num_threads = max(2, min(4, (os.cpu_count() or 2)))
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
