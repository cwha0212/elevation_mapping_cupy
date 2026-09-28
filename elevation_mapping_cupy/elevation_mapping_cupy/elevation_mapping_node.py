#!/usr/bin/env python3
import math
import message_filters
import numpy as np
import os
from functools import partial
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from elevation_map_msgs.msg import ChannelInfo
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf_transformations import quaternion_matrix
import tf2_ros
import tf2_py as tf2
from rclpy.duration import Duration
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from std_srvs.srv import Trigger
from elevation_mapping_cupy import ElevationMap, Parameter
from elevation_mapping_cupy.gridmap_utils import encode_layer_to_multiarray, encode_rot180_as_gridmap_column

PDC_DATATYPE = {
    "1": np.int8,
    "2": np.uint8,
    "3": np.int16,
    "4": np.uint16,
    "5": np.int32,
    "6": np.uint32,
    "7": np.float32,
    "8": np.float64,
}

def _pointcloud2_xyz_f32(msg: PointCloud2) -> np.ndarray:
    """
    Convert a PointCloud2 into an (N,3) float32 numpy array for fields (x,y,z).

    Supported (fail-loudly):
      - little-endian clouds
      - fields x,y,z present and FLOAT32

    This intentionally does not support arbitrary field layouts or RGB/semantic channels.
    """
    if msg.is_bigendian:
        raise ValueError("PointCloud2 big-endian is not supported.")

    want = {"x", "y", "z"}
    fields = {f.name: f for f in msg.fields}
    missing = want.difference(fields.keys())
    if missing:
        raise ValueError(f"PointCloud2 is missing required fields: {sorted(missing)}")

    for name in ("x", "y", "z"):
        f = fields[name]
        if f.datatype != PointField.FLOAT32 or f.count != 1:
            raise ValueError(
                f"PointCloud2 field '{name}' must be FLOAT32 count=1, got datatype={f.datatype} count={f.count}"
            )

    dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (np.float32, np.float32, np.float32),
            "offsets": (fields["x"].offset, fields["y"].offset, fields["z"].offset),
            "itemsize": msg.point_step,
        }
    )
    if msg.height == 0 or msg.width == 0:
        return np.empty((0, 3), dtype=np.float32)
    packed_row_size = msg.point_step * msg.width
    if msg.row_step < packed_row_size:
        raise ValueError(
            f"PointCloud2 row_step={msg.row_step} is smaller than the packed row size {packed_row_size}."
        )
    required_bytes = msg.row_step * msg.height
    if len(msg.data) < required_bytes:
        raise ValueError(
            f"PointCloud2 data has {len(msg.data)} bytes, expected at least {required_bytes}."
        )

    rows = np.ndarray(
        shape=(msg.height, msg.width),
        dtype=dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    )
    pts = np.stack((rows["x"], rows["y"], rows["z"]), axis=-1).reshape(-1, 3)

    if not msg.is_dense:
        good = np.isfinite(pts).all(axis=1)
        pts = pts[good]
    return pts

class ElevationMappingNode(Node):
    def __init__(self):
        super().__init__(
            'elevation_mapping_node',
            automatically_declare_parameters_from_overrides=True,
            allow_undeclared_parameters=False
        )

        self.root = get_package_share_directory("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        # Initialize parameters with some defaults
        self.param = Parameter(
            weight_file=weight_file,
            plugin_config_file=plugin_config_file
        )

        # Read ROS parameters (including YAML)
        self.initialize_ros()
        self.set_param_values_from_ros()

        # Overwrite subscriber_cfg from loaded YAML
        self.param.subscriber_cfg = self.my_subscribers

        self.initialize_elevation_mapping()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()
        self.register_services()
        self._last_t = None

    def initialize_elevation_mapping(self) -> None:
        self.param.update()
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        # Pinned so the per-layer device-to-host copy is a straight DMA.
        import cupyx
        self._map_data = cupyx.zeros_pinned(
            (self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32
        )
        self.get_logger().info(f"Initialized map with length: {self._map.map_length}, resolution: {self._map.resolution}, cells: {self._map.cell_n}")

        self._map_q = None
        self._map_t = None

    def initialize_ros(self) -> None:
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_ros_params()

    def get_ros_params(self) -> None:
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.corrected_map_frame = self.get_parameter('corrected_map_frame').get_parameter_value().string_value
        self.update_variance_fps = self.get_parameter('update_variance_fps').get_parameter_value().double_value
        self.time_interval = self.get_parameter('time_interval').get_parameter_value().double_value
        self.update_pose_fps = self.get_parameter('update_pose_fps').get_parameter_value().double_value
        # Self-body cut on the incoming cloud, in the cloud's own frame: points
        # inside the box [min, max] are the robot and are dropped before
        # fusion; dedup_voxel keeps one point per voxel of that size (0 = off).
        # Empty (or min == max) disables the box. This replaces the separate
        # body-cut node: one parse of the cloud instead of parse, filter,
        # serialize, parse.
        for name, default in (("body_filter_min", [0.0, 0.0, 0.0]), ("body_filter_max", [0.0, 0.0, 0.0]),
                              ("dedup_voxel", 0.0)):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        lo = np.array([float(v) for v in self.get_parameter('body_filter_min').value], dtype=np.float32)
        hi = np.array([float(v) for v in self.get_parameter('body_filter_max').value], dtype=np.float32)
        self._body_box = (lo, hi) if lo.shape == (3,) and hi.shape == (3,) and np.any(hi > lo) else None
        self._dedup_voxel = float(self.get_parameter('dedup_voxel').value)
        if not self.has_parameter('cupy_memory_pool_trim_interval_s'):
            self.declare_parameter('cupy_memory_pool_trim_interval_s', 0.0)
        self.cupy_memory_pool_trim_interval_s = float(
            self.get_parameter('cupy_memory_pool_trim_interval_s').value
        )
        subscribers_params = self.get_parameters_by_prefix('subscribers')
        self.my_subscribers = {}
        for param_name, param_value in subscribers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                sub_key, sub_param = parts[:2]
                if sub_key not in self.my_subscribers:
                    self.my_subscribers[sub_key] = {}
                self.my_subscribers[sub_key][sub_param] = param_value.value
        publishers_params = self.get_parameters_by_prefix('publishers')
        self.my_publishers = {}
        for param_name, param_value in publishers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                pub_key, pub_param = parts[:2]
                if pub_key not in self.my_publishers:
                    self.my_publishers[pub_key] = {}
                self.my_publishers[pub_key][pub_param] = param_value.value


    def set_param_values_from_ros(self):
        # Assign to self.param so it won't use defaults. This is research code: crash loudly if
        # a required parameter is missing or mistyped.
        if self.has_parameter("plugin_config_file"):
            # One file, or a list of files merged in order (the gait chain
            # rides on top of the navigation chain that way).
            plugin_config_file = self.get_parameter("plugin_config_file").value
            assert plugin_config_file
            if not isinstance(plugin_config_file, str):
                plugin_config_file = [str(v) for v in plugin_config_file]
            self.param.plugin_config_file = plugin_config_file
        if self.has_parameter("weight_file"):
            weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
            assert weight_file
            self.param.weight_file = weight_file
        self.param.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.param.map_length = self.get_parameter('map_length').get_parameter_value().double_value
        self.param.sensor_noise_factor = self.get_parameter('sensor_noise_factor').get_parameter_value().double_value
        self.param.mahalanobis_thresh = self.get_parameter('mahalanobis_thresh').get_parameter_value().double_value
        self.param.outlier_variance = self.get_parameter('outlier_variance').get_parameter_value().double_value
        self.param.drift_compensation_variance_inlier = self.get_parameter(
            'drift_compensation_variance_inlier'
        ).get_parameter_value().double_value
        self.param.max_drift = self.get_parameter('max_drift').get_parameter_value().double_value
        self.param.drift_compensation_alpha = self.get_parameter(
            'drift_compensation_alpha'
        ).get_parameter_value().double_value
        self.param.time_variance = self.get_parameter('time_variance').get_parameter_value().double_value
        self.param.max_variance = self.get_parameter('max_variance').get_parameter_value().double_value
        self.param.initial_variance = self.get_parameter('initial_variance').get_parameter_value().double_value
        self.param.traversability_inlier = self.get_parameter(
            'traversability_inlier'
        ).get_parameter_value().double_value
        self.param.dilation_size = self.get_parameter('dilation_size').get_parameter_value().integer_value
        self.param.wall_num_thresh = self.get_parameter('wall_num_thresh').get_parameter_value().integer_value
        self.param.min_height_drift_cnt = self.get_parameter(
            'min_height_drift_cnt'
        ).get_parameter_value().integer_value
        self.param.position_noise_thresh = self.get_parameter(
            'position_noise_thresh'
        ).get_parameter_value().double_value
        self.param.orientation_noise_thresh = self.get_parameter(
            'orientation_noise_thresh'
        ).get_parameter_value().double_value
        self.param.min_valid_distance = self.get_parameter(
            'min_valid_distance'
        ).get_parameter_value().double_value
        self.param.max_height_range = self.get_parameter(
            'max_height_range'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_a = self.get_parameter(
            'ramped_height_range_a'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_b = self.get_parameter(
            'ramped_height_range_b'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_c = self.get_parameter(
            'ramped_height_range_c'
        ).get_parameter_value().double_value
        self.param.max_ray_length = self.get_parameter('max_ray_length').get_parameter_value().double_value
        self.param.cleanup_step = self.get_parameter('cleanup_step').get_parameter_value().double_value
        self.param.cleanup_cos_thresh = self.get_parameter(
            'cleanup_cos_thresh'
        ).get_parameter_value().double_value
        self.param.overlap_clear_range_xy = self.get_parameter(
            'overlap_clear_range_xy'
        ).get_parameter_value().double_value
        self.param.overlap_clear_range_z = self.get_parameter(
            'overlap_clear_range_z'
        ).get_parameter_value().double_value
        self.param.enable_edge_sharpen = self.get_parameter(
            'enable_edge_sharpen'
        ).get_parameter_value().bool_value
        self.param.enable_visibility_cleanup = self.get_parameter(
            'enable_visibility_cleanup'
        ).get_parameter_value().bool_value
        self.param.enable_drift_compensation = self.get_parameter(
            'enable_drift_compensation'
        ).get_parameter_value().bool_value
        self.param.enable_overlap_clearance = self.get_parameter(
            'enable_overlap_clearance'
        ).get_parameter_value().bool_value
        self.param.use_only_above_for_upper_bound = self.get_parameter(
            'use_only_above_for_upper_bound'
        ).get_parameter_value().bool_value

        service_ns_param = self.get_parameter('service_namespace').get_parameter_value().string_value
        if not service_ns_param:
            raise ValueError("service_namespace must be a non-empty string")
        self.service_namespace = self._normalize_namespace(service_ns_param)

    def register_subscribers(self) -> None:
        self._pointcloud_subs = {}
        self._image_syncs = {}
        self._image_filter_subs = {}
        self._channel_info_subs = {}
        self._image_channels = {}

        if any(config.get("data_type") == "image" for config in self.my_subscribers.values()):
            self.cv_bridge = CvBridge()

        for key, config in self.my_subscribers.items():
            data_type = config.get("data_type")
            if data_type == "image":
                topic_name = config.get("topic_name")
                camera_info_topic_name = config.get(
                    "camera_info_topic_name",
                    config.get("topic_name_camera_info"),
                )
                if not topic_name:
                    raise ValueError(f"Image subscriber '{key}' is missing required key 'topic_name'.")
                if not camera_info_topic_name:
                    raise ValueError(
                        f"Image subscriber '{key}' is missing required key 'camera_info_topic_name'."
                    )

                camera_sub = message_filters.Subscriber(self, Image, topic_name)
                camera_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic_name)
                image_sync = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub],
                    queue_size=10,
                    slop=0.5,
                )
                image_sync.registerCallback(partial(self.image_callback, sub_key=key))
                self._image_filter_subs[key] = [camera_sub, camera_info_sub]
                self._image_syncs[key] = image_sync

                channel_info_topic_name = config.get("channel_info_topic_name")
                if channel_info_topic_name:
                    self._channel_info_subs[key] = self.create_subscription(
                        ChannelInfo,
                        channel_info_topic_name,
                        partial(self.channel_info_callback, sub_key=key),
                        10,
                    )
                continue

            if data_type != "pointcloud":
                raise ValueError(
                    f"Unsupported subscriber data_type='{data_type}' for '{key}'. "
                    "Supported: pointcloud and image."
                )

            topic_name = config.get("topic_name")
            if not topic_name:
                raise ValueError(f"Subscriber '{key}' is missing required key 'topic_name'.")

            # Use sensor data QoS (BEST_EFFORT) for point clouds
            qos_profile = QoSPresetProfiles.get_from_short_key("sensor_data")
            self._pointcloud_subs[key] = self.create_subscription(
                PointCloud2,
                topic_name,
                partial(self.pointcloud_callback, sub_key=key),
                qos_profile,
            )

    def channel_info_callback(self, msg: ChannelInfo, sub_key: str) -> None:
        self._image_channels[sub_key] = list(msg.channels)

    def resolve_image_channels(self, sub_key: str) -> List[str]:
        configured_channels = self.param.subscriber_cfg[sub_key].get("channels", [])
        if configured_channels:
            return configured_channels

        live_channels = self._image_channels.get(sub_key, [])
        if live_channels:
            return live_channels

        self.get_logger().warning(
            (
                f"Image subscriber '{sub_key}' has no resolved channels yet. "
                "Configure 'channels' or wait for ChannelInfo."
            ),
            throttle_duration_sec=5.0,
        )
        return []

    def register_publishers(self) -> None:
        self._publishers_dict = {}
        self._publishers_timers = []

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.get_name()}/{pub_key}"
            publisher = self.create_publisher(GridMap, topic_name, 10)
            self._publishers_dict[pub_key] = publisher

            fps = pub_config.get("fps", 1.0)
            timer = self.create_timer(
                1.0 / fps,
                partial(self.publish_map, key=pub_key)
            )
            self._publishers_timers.append(timer)

    def register_timers(self) -> None:
        self.time_pose_update = self.create_timer(
            1.0 / self.update_pose_fps,
            self.pose_update
        )
        self.timer_variance = self.create_timer(
            1.0 / self.update_variance_fps,
            self.update_variance
        )
        self.timer_time = self.create_timer(
            self.time_interval,
            self.update_time
        )
        self.timer_cupy_memory_pool = None
        if self.cupy_memory_pool_trim_interval_s > 0.0:
            self.timer_cupy_memory_pool = self.create_timer(
                self.cupy_memory_pool_trim_interval_s,
                self.trim_cupy_memory_pool
            )

    def register_services(self) -> None:
        service_clear = self._resolve_service_name('clear_map')
        self._srv_clear_map = self.create_service(
            Trigger,
            service_clear,
            self.handle_clear_map
        )

    def publish_map(self, key: str) -> None:
        if self._map_q is None:
            return
        publisher = self._publishers_dict[key]
        if publisher.get_subscription_count() == 0:
            return
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self._last_t if self._last_t is not None else self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        actual_map_length = (self._map.cell_n - 2) * self._map.resolution
        gm.info.length_x = actual_map_length
        gm.info.length_y = actual_map_length
        if self._map_t is not None:
            gm.info.pose.position.x = self._map_t.x
            gm.info.pose.position.y = self._map_t.y
            # grid_map_ros (and our RViz usage) treats GridMap as a horizontal 2.5D surface and ignores pose.z and
            # pose.orientation. Foxglove's GridMap renderer *does* apply them, which can make the map appear tilted
            # and shifted in Z when we embed the robot pose here. Keep pose.x/y as the map center in `map_frame`,
            # but publish a neutral pose for visualization sanity.
            gm.info.pose.position.z = 0.0
        else:
            # Only before the first pose update; afterwards _map_t is the truth
            # and this device-to-host sync is not paid.
            center = self._get_map_center()
            gm.info.pose.position.x = float(center[0])
            gm.info.pose.position.y = float(center[1])
            gm.info.pose.position.z = 0.0

        gm.info.pose.orientation.x = 0.0
        gm.info.pose.orientation.y = 0.0
        gm.info.pose.orientation.z = 0.0
        gm.info.pose.orientation.w = 1.0
        gm.layers = []
        gm.basic_layers = self.my_publishers[key]["basic_layers"]

        for layer in self.my_publishers[key].get("layers", []):
            # A layer named in the config but absent from the map used to take
            # the whole node down mid-publish. Warn and carry the rest: losing
            # one layer is a configuration problem, losing the mapper is an
            # outage.
            if not self._map.exists_layer(layer):
                self.get_logger().warning(
                    f"Publisher '{key}' asks for layer '{layer}', which the map "
                    f"does not have. Skipping it.",
                    throttle_duration_sec=5.0,
                )
                continue
            gm.layers.append(layer)
            self._map.copy_layer_rot180(layer, self._map_data)
            gm.data.append(encode_rot180_as_gridmap_column(self._map_data))

        gm.outer_start_index = 0
        gm.inner_start_index = 0
        publisher.publish(gm)

    def handle_clear_map(self, request, response):
        del request
        try:
            self._map.clear()
            self._last_t = self.get_clock().now().to_msg()
            for key in self._publishers_dict.keys():
                self.publish_map(key)
            response.success = True
            response.message = "Elevation map cleared."
            self.get_logger().info("clear_map: reset elevation map to empty state.")
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"clear_map failed: {exc}")
        return response

    def _resolve_service_name(self, suffix: str) -> str:
        base = self.service_namespace
        if not base:
            base = f"/{self.get_name()}"
        return f"{base}/{suffix}".replace('//', '/')

    def _get_map_center(self) -> np.ndarray:
        center = np.zeros((1, 3), dtype=np.float32)
        self._map.get_center_position(center)
        return center[0]

    def _normalize_namespace(self, value: str) -> str:
        value = value.strip() if value else ''
        if not value:
            return ''
        if not value.startswith('/'):
            value = f'/{value}'
        return value.rstrip('/')

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time
            )
        except tf2_ros.ExtrapolationException:
            # Time is in the future/past, try with latest available
            try:
                return self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time()
                )
            # NOTE: The second lookup can also throw ExtrapolationException (e.g., TF buffer not populated yet,
            # or timestamps are discontinuous during sim resets). If we don't catch it here the whole node dies.
            except (
                tf2.LookupException,
                tf2.ConnectivityException,
                tf2.ExtrapolationException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.get_logger().warning(
                    f"Transform from '{source_frame}' to '{target_frame}' not available: {e}",
                    throttle_duration_sec=5.0
                )
                return None
        except tf2.LookupException as e:
            # Frame doesn't exist
            self.get_logger().warning(
                f"Frame '{target_frame}' or '{source_frame}' does not exist: {e}",
                throttle_duration_sec=5.0
            )
            return None
        except tf2.ConnectivityException as e:
            # No transform path between frames
            self.get_logger().warning(
                f"No transform path from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=5.0
            )
            return None
        except Exception as e:
            # Catch any other unexpected TF2 errors
            self.get_logger().warning(
                f"Unexpected TF2 error for transform from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=5.0
            )
            return None

    def image_callback(self, camera_msg: Image, camera_info_msg: CameraInfo, sub_key: str) -> None:
        self._last_t = camera_msg.header.stamp

        frame_sensor_id = camera_msg.header.frame_id
        if not frame_sensor_id:
            raise ValueError("Image header.frame_id is empty.")

        semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")
        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, idx] for idx in range(semantic_img.shape[2])]
        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(camera_info_msg.d, dtype=np.float32).reshape(-1, 1)

        if frame_sensor_id == self.map_frame:
            t_np = np.zeros(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
        else:
            # input_image projects map cells through K @ [R|t], so it needs
            # map->camera. tf2's lookup_transform(target, source) returns
            # source->target, so the camera frame is the target here -- the
            # opposite order from pointcloud_callback, which really does want
            # sensor->map to carry its points across.
            transform_map_to_camera = self.safe_lookup_transform(
                frame_sensor_id,
                self.map_frame,
                camera_msg.header.stamp,
            )
            if transform_map_to_camera is None:
                return
            t = transform_map_to_camera.transform.translation
            q = transform_map_to_camera.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

        channels = self.resolve_image_channels(sub_key)
        if not channels:
            return

        self._map.input_image(
            semantic_img,
            channels,
            R,
            t_np,
            K,
            D,
            camera_info_msg.distortion_model,
            camera_info_msg.height,
            camera_info_msg.width,
        )
        self._image_process_counter += 1
        # TF lookups that fail drop the frame and warn only every 5 s, so a
        # camera can go half-ignored without anything obvious in the log.
        self.get_logger().info(
            f"Projected image frames from '{sub_key}': {self._image_process_counter}",
            throttle_duration_sec=5.0,
        )

    def _cut_body(self, pts: np.ndarray) -> np.ndarray:
        """Drop the robot's own returns, then thin to one point per voxel.

        The same sequence the body-cut node ran: finite points only, box
        test, then a first-of-each-voxel dedup with the same key and order.
        """
        pts = pts[np.isfinite(pts).all(axis=1)]
        lo, hi = self._body_box
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        dropped = int(inside.sum())
        pts = pts[~inside]
        if dropped:
            self.get_logger().info(
                f"Body filter dropped {dropped} self returns.", throttle_duration_sec=10.0)
        if pts.shape[0] and self._dedup_voxel > 0.0:
            keys = np.floor(pts / self._dedup_voxel).astype(np.int64)
            _, keep = np.unique(keys.view([("", keys.dtype)] * 3).ravel(), return_index=True)
            pts = np.ascontiguousarray(pts[np.sort(keep)], dtype=np.float32)
        return pts

    def pointcloud_callback(self, msg: PointCloud2, sub_key: str) -> None:
        pts = _pointcloud2_xyz_f32(msg)
        if self._body_box is not None:
            pts = self._cut_body(pts)
        if pts.size == 0:
            return
        self._last_t = msg.header.stamp

        frame_sensor_id = msg.header.frame_id
        if not frame_sensor_id:
            raise ValueError("PointCloud2 header.frame_id is empty.")

        if frame_sensor_id == self.map_frame:
            t_np = np.zeros(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
        else:
            transform_sensor_to_map = self.safe_lookup_transform(
                self.map_frame,
                frame_sensor_id,
                msg.header.stamp,
            )
            if transform_sensor_to_map is None:
                # Transform not available yet.
                return
            t = transform_sensor_to_map.transform.translation
            q = transform_sensor_to_map.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

        self._map.input_pointcloud(pts, ["x", "y", "z"], R, t_np, 0, 0)
        self._pointcloud_process_counter += 1
        self.get_logger().info(
            f"Fused point clouds from '{sub_key}': {self._pointcloud_process_counter}",
            throttle_duration_sec=5.0,
        )

    def pose_update(self) -> None:
        if self._last_t is None:
            return
        transform = self.safe_lookup_transform(
            self.map_frame,
            self.base_frame,
            self._last_t
        )
        if transform is None:
            # Transform not available, skip pose update
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        self._map.move_to(trans, rot)
        self._map_t = t
        self._map_q = q

    def update_variance(self) -> None:
        self._map.update_variance()

    def update_time(self) -> None:
        self._map.update_time()

    def trim_cupy_memory_pool(self) -> None:
        self._map.trim_memory_pool()

    def destroy_node(self) -> None:
        super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationMappingNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # launch_testing / signal handlers can already have shut down the context.
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
