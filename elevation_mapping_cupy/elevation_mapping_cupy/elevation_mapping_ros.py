#!/usr/bin/env python3
import numpy as np
import os
import threading
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory
import ros2_numpy as rnp
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from tf_transformations import quaternion_matrix
import tf2_ros
import message_filters
from cv_bridge import CvBridge
from rclpy.duration import Duration
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from rclpy.serialization import serialize_message
from elevation_mapping_cupy import ElevationMap, Parameter

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

class ElevationMappingNode(Node):
    def __init__(self):
        super().__init__(
            'elevation_mapping_node',
            automatically_declare_parameters_from_overrides=True,
            allow_undeclared_parameters=True,
            parameter_overrides=[
                rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )
        # Create callback groups to separate timer execution from pointcloud processing
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.pointcloud_callback_group = ReentrantCallbackGroup()
        self.root = get_package_share_directory("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        # Initialize parameters with some defaults
        self.param = Parameter(
            use_chainer=False,
            weight_file=weight_file,
            plugin_config_file=plugin_config_file
        )

        # Read ROS parameters (including YAML)
        self.initialize_ros()
        self.set_param_values_from_ros()

        # Overwrite subscriber_cfg from loaded YAML
        self.param.subscriber_cfg = self.my_subscribers

        self.initialize_elevation_mapping()
        self.get_logger().info("\n=== REGISTERING ROS INTERFACES ===")
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()
        self._last_t = None
        self.get_logger().info("\n=== ELEVATION MAPPING NODE READY ===")
        self.get_logger().info(f"Waiting for pointclouds on configured topics...")
        self.get_logger().info(f"Total timers created: {len(self._publishers_timers) + 3}")  # publishers + pose/variance/time

    def initialize_elevation_mapping(self) -> None:
        self.get_logger().info("\n=== INITIALIZING ELEVATION MAPPING ===")
        self.param.update()
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        self._map_data = np.zeros(
            (self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32
        )
        self.get_logger().info(f"Map initialized: length={self._map.map_length}m, resolution={self._map.resolution}m, cells={self._map.cell_n}")

        self._map_q = None
        self._map_t = None
        self._lock = threading.Lock()  # Thread safety for shared state
        self.get_logger().info("_map_q and _map_t initialized to None (will be set on first pose update)")

    def initialize_ros(self) -> None:
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_ros_params()

    def get_ros_params(self) -> None:
        self.use_chainer = self.get_parameter('use_chainer').get_parameter_value().bool_value
        self.weight_file = self.get_parameter('weight_file').get_parameter_value().string_value
        self.plugin_config_file = self.get_parameter('plugin_config_file').get_parameter_value().string_value
        self.initialize_frame_id = self.get_parameter('initialize_frame_id').get_parameter_value().string_value
        self.initialize_tf_offset = self.get_parameter('initialize_tf_offset').get_parameter_value().double_array_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.corrected_map_frame = self.get_parameter('corrected_map_frame').get_parameter_value().string_value
        self.initialize_method = self.get_parameter('initialize_method').get_parameter_value().string_value
        self.position_lowpass_alpha = self.get_parameter('position_lowpass_alpha').get_parameter_value().double_value
        self.orientation_lowpass_alpha = self.get_parameter('orientation_lowpass_alpha').get_parameter_value().double_value
        self.recordable_fps = self.get_parameter('recordable_fps').get_parameter_value().double_value
        self.update_variance_fps = self.get_parameter('update_variance_fps').get_parameter_value().double_value
        self.time_interval = self.get_parameter('time_interval').get_parameter_value().double_value
        self.update_pose_fps = self.get_parameter('update_pose_fps').get_parameter_value().double_value
        self.initialize_tf_grid_size = self.get_parameter('initialize_tf_grid_size').get_parameter_value().double_value
        self.map_acquire_fps = self.get_parameter('map_acquire_fps').get_parameter_value().double_value
        self.publish_statistics_fps = self.get_parameter('publish_statistics_fps').get_parameter_value().double_value
        self.enable_pointcloud_publishing = self.get_parameter('enable_pointcloud_publishing').get_parameter_value().bool_value
        self.enable_normal_arrow_publishing = self.get_parameter('enable_normal_arrow_publishing').get_parameter_value().bool_value
        self.enable_drift_corrected_TF_publishing = self.get_parameter('enable_drift_corrected_TF_publishing').get_parameter_value().bool_value
        self.use_initializer_at_start = self.get_parameter('use_initializer_at_start').get_parameter_value().bool_value
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
        # Assign to self.param so it won't use defaults
        # Use try/except so missing params won't cause errors
        try: self.param.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        except: pass
        try: self.param.map_length = self.get_parameter('map_length').get_parameter_value().double_value
        except: pass
        try: self.param.sensor_noise_factor = self.get_parameter('sensor_noise_factor').get_parameter_value().double_value
        except: pass
        try: self.param.mahalanobis_thresh = self.get_parameter('mahalanobis_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.outlier_variance = self.get_parameter('outlier_variance').get_parameter_value().double_value
        except: pass
        try: 
            # The YAML has 'drift_compensation_variance_inler', but our param is 'drift_compensation_variance_inlier'
            self.param.drift_compensation_variance_inlier = self.get_parameter('drift_compensation_variance_inler').get_parameter_value().double_value
        except: pass
        try: self.param.max_drift = self.get_parameter('max_drift').get_parameter_value().double_value
        except: pass
        try: self.param.drift_compensation_alpha = self.get_parameter('drift_compensation_alpha').get_parameter_value().double_value
        except: pass
        try: self.param.time_variance = self.get_parameter('time_variance').get_parameter_value().double_value
        except: pass
        try: self.param.max_variance = self.get_parameter('max_variance').get_parameter_value().double_value
        except: pass
        try: self.param.initial_variance = self.get_parameter('initial_variance').get_parameter_value().double_value
        except: pass
        try: self.param.traversability_inlier = self.get_parameter('traversability_inlier').get_parameter_value().double_value
        except: pass
        try: self.param.dilation_size = self.get_parameter('dilation_size').get_parameter_value().integer_value
        except: pass
        try: self.param.wall_num_thresh = self.get_parameter('wall_num_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.min_height_drift_cnt = self.get_parameter('min_height_drift_cnt').get_parameter_value().double_value
        except: pass
        try: self.param.position_noise_thresh = self.get_parameter('position_noise_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.orientation_noise_thresh = self.get_parameter('orientation_noise_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.min_valid_distance = self.get_parameter('min_valid_distance').get_parameter_value().double_value
        except: pass
        try: self.param.max_height_range = self.get_parameter('max_height_range').get_parameter_value().double_value
        except: pass
        try: self.param.ramped_height_range_a = self.get_parameter('ramped_height_range_a').get_parameter_value().double_value
        except: pass
        try: self.param.ramped_height_range_b = self.get_parameter('ramped_height_range_b').get_parameter_value().double_value
        except: pass
        try: self.param.ramped_height_range_c = self.get_parameter('ramped_height_range_c').get_parameter_value().double_value
        except: pass
        try: self.param.max_ray_length = self.get_parameter('max_ray_length').get_parameter_value().double_value
        except: pass
        try: self.param.cleanup_step = self.get_parameter('cleanup_step').get_parameter_value().double_value
        except: pass
        try: self.param.cleanup_cos_thresh = self.get_parameter('cleanup_cos_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.safe_thresh = self.get_parameter('safe_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.safe_min_thresh = self.get_parameter('safe_min_thresh').get_parameter_value().double_value
        except: pass
        try: self.param.max_unsafe_n = self.get_parameter('max_unsafe_n').get_parameter_value().integer_value
        except: pass
        try: self.param.overlap_clear_range_xy = self.get_parameter('overlap_clear_range_xy').get_parameter_value().double_value
        except: pass
        try: self.param.overlap_clear_range_z = self.get_parameter('overlap_clear_range_z').get_parameter_value().double_value
        except: pass
        try: self.param.enable_edge_sharpen = self.get_parameter('enable_edge_sharpen').get_parameter_value().bool_value
        except: pass
        try: self.param.enable_visibility_cleanup = self.get_parameter('enable_visibility_cleanup').get_parameter_value().bool_value
        except: pass
        try: self.param.enable_drift_compensation = self.get_parameter('enable_drift_compensation').get_parameter_value().bool_value
        except: pass
        try: self.param.enable_overlap_clearance = self.get_parameter('enable_overlap_clearance').get_parameter_value().bool_value
        except: pass
        try: self.param.use_only_above_for_upper_bound = self.get_parameter('use_only_above_for_upper_bound').get_parameter_value().bool_value
        except: pass

    def register_subscribers(self) -> None:
        self.get_logger().info("Registering subscribers...")
        if any(config.get("data_type") == "image" for config in self.my_subscribers.values()):
            self.cv_bridge = CvBridge()

        pointcloud_subs = {}
        image_subs = {}

        for key, config in self.my_subscribers.items():
            data_type = config.get("data_type")
            self.get_logger().info(f"  - Subscriber '{key}': type={data_type}, topic={config.get('topic_name', 'N/A')}")
            if data_type == "image":
                topic_name_camera = config.get("topic_name_camera", "/camera/image")
                topic_name_camera_info = config.get("topic_name_camera_info", "/camera/camera_info")
                camera_sub = message_filters.Subscriber(
                    self,
                    Image,
                    topic_name_camera
                )
                camera_info_sub = message_filters.Subscriber(
                    self,
                    CameraInfo,
                    topic_name_camera_info
                )
                image_sync = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub], queue_size=10, slop=0.5
                )
                image_sync.registerCallback(partial(self.image_callback, sub_key=key))
                image_subs[key] = image_sync
            elif data_type == "pointcloud":
                topic_name = config.get("topic_name", "/pointcloud")
                qos_profile = rclpy.qos.QoSProfile(
                    depth=10,
                    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST
                )
                subscription = self.create_subscription(
                    PointCloud2,
                    topic_name,
                    partial(self.pointcloud_callback, sub_key=key),
                    qos_profile,
                    callback_group=self.pointcloud_callback_group
                )
                pointcloud_subs[key] = subscription

    def register_publishers(self) -> None:
        self.get_logger().info(f"Registering publishers for: {list(self.my_publishers.keys())}")
        self._publishers_dict = {}
        self._publishers_timers = []

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.get_name()}/{pub_key}"
            self.get_logger().info(f"Creating publisher for {pub_key} on topic {topic_name}")
            publisher = self.create_publisher(GridMap, topic_name, 10)
            self._publishers_dict[pub_key] = publisher

            fps = pub_config.get("fps", 1.0)
            period = 1.0 / fps
            self.get_logger().info(f"Creating timer for {pub_key} with period {period}s (fps={fps})")
            timer = self.create_timer(
                period,
                partial(self.publish_map, key=pub_key),
                callback_group=self.timer_callback_group
            )
            self._publishers_timers.append(timer)
            self.get_logger().info(f"  - Timer created for {pub_key}: {timer}")
        self.get_logger().info(f"Publisher registration complete! Created {len(self._publishers_timers)} timers")

    def register_timers(self) -> None:
        self.get_logger().info("Registering timers...")
        self.get_logger().info(f"Creating pose update timer with period 0.1s")
        self.time_pose_update = self.create_timer(
            0.1,
            self.pose_update,
            callback_group=self.timer_callback_group
        )
        self.get_logger().info(f"  - Pose update timer created: {self.time_pose_update}")
        
        self.get_logger().info(f"Creating variance timer with period {1.0 / self.update_variance_fps}s")
        self.timer_variance = self.create_timer(
            1.0 / self.update_variance_fps,
            self.update_variance,
            callback_group=self.timer_callback_group
        )
        self.get_logger().info(f"  - Variance update timer created: {self.timer_variance}")
        
        self.get_logger().info(f"Creating time timer with period {self.time_interval}s")
        self.timer_time = self.create_timer(
            self.time_interval,
            self.update_time,
            callback_group=self.timer_callback_group
        )
        self.get_logger().info(f"  - Time update timer created: {self.timer_time}")
        self.get_logger().info("Timer registration complete!")

    def publish_map(self, key: str) -> None:
        self.get_logger().info(f"publish_map timer fired for key: {key}")
        with self._lock:
            if self._map_q is None:
                self.get_logger().warn(f"publish_map: _map_q is None, skipping")
                return
            if self._map_t is None:
                self.get_logger().warn(f"publish_map: _map_t is None, skipping")
                return
            # Copy values to avoid holding lock too long
            map_t = self._map_t
            map_q = self._map_q
        self.get_logger().info(f"Publishing map for key: {key}")
        
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        gm.info.length_x = self._map.map_length
        gm.info.length_y = self._map.map_length
        gm.info.pose.position.x = map_t.x
        gm.info.pose.position.y = map_t.y
        gm.info.pose.position.z = 0.0
        gm.info.pose.orientation.w = 1.0
        gm.info.pose.orientation.x = 0.0
        gm.info.pose.orientation.y = 0.0
        gm.info.pose.orientation.z = 0.0
        gm.layers = []
        gm.basic_layers = self.my_publishers[key]["basic_layers"]

        for layer in self.my_publishers[key].get("layers", []):
            gm.layers.append(layer)
            self._map.get_map_with_name_ref(layer, self._map_data)
            arr = Float32MultiArray()
            arr.layout = MAL()
            N = self._map_data.shape[0]
            arr.layout.dim.append(MAD(label="column_index", size=N, stride=int(N * N)))
            arr.layout.dim.append(MAD(label="row_index", size=N, stride=N))
            arr.data = self._map_data.T.flatten().tolist()
            gm.data.append(arr)

        gm.outer_start_index = 0
        gm.inner_start_index = 0
        self._publishers_dict[key].publish(gm)
        self.get_logger().info(f"Published GridMap for {key} with {len(gm.layers)} layers")

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            self.get_logger().info(f"TF lookup: {source_frame} -> {target_frame}")
            result = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time
            )
            self.get_logger().info(f"TF lookup successful")
            return result
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF ExtrapolationException, trying with current time: {e}")
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def image_callback(self, camera_msg: Image, camera_info_msg: CameraInfo, sub_key: str) -> None:
        self._last_t = camera_msg.header.stamp
        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")
        except:
            return
        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, k] for k in range(semantic_img.shape[2])]
        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(camera_info_msg.d, dtype=np.float32).reshape(-1, 1)

        transform_camera_to_map = self.safe_lookup_transform(
            self.map_frame,
            camera_msg.header.frame_id,
            camera_msg.header.stamp
        )
        t = transform_camera_to_map.transform.translation
        q = transform_camera_to_map.transform.rotation
        t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        self._map.input_image(
            sub_key, semantic_img, R, t_np, K, D,
            camera_info_msg.height, camera_info_msg.width
        )
        self._image_process_counter += 1

    def pointcloud_callback(self, msg: PointCloud2, sub_key: str) -> None:
        self._last_t = msg.header.stamp
        
        channels = ["x", "y", "z"] + self.param.subscriber_cfg[sub_key].get("channels", [])
        
        # Try multiple parsing methods
        points_xyz = None
        
        # Method 1: Try ros2_numpy
        try:
            points = rnp.numpify(msg)
            self.get_logger().info(f"ros2_numpy result type: {type(points)}")
            
            # Check if it's a structured array or a dict
            if isinstance(points, dict):
                self.get_logger().info(f"ros2_numpy returned dict with keys: {points.keys()}")
                # If it's a dict, try to extract x, y, z directly
                if 'x' in points and 'y' in points and 'z' in points:
                    x = points['x']
                    y = points['y'] 
                    z = points['z']
                    points_xyz = np.stack([x, y, z], axis=-1)
                    self.get_logger().info(f"Extracted from dict, shape: {points_xyz.shape}")
            elif hasattr(points, 'dtype'):
                self.get_logger().info(f"ros2_numpy parsed array, dtype: {points.dtype}, shape: {points.shape}")
                self.get_logger().info(f"Field names in parsed array: {points.dtype.names}")
                
                # Try to access xyz data in different ways
                if 'xyz' in points.dtype.names:
                    points_xyz = points['xyz']
                    self.get_logger().info(f"Found 'xyz' field, shape: {points_xyz.shape}")
                elif 'x' in points.dtype.names and 'y' in points.dtype.names and 'z' in points.dtype.names:
                    points_xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
                    self.get_logger().info(f"Stacked x,y,z fields, shape: {points_xyz.shape}")
                else:
                    self.get_logger().warn(f"Could not find xyz data in dtype fields")
        except Exception as e:
            self.get_logger().error(f"ros2_numpy failed: {e}")
            
        # Method 2: Try point_cloud2 with generator
        if points_xyz is None:
            try:
                self.get_logger().info("Trying point_cloud2.read_points generator method...")
                points_list = []
                for p in point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True):
                    points_list.append([p[0], p[1], p[2]])
                    if len(points_list) >= 100:  # Sample first 100 to test
                        break
                
                if points_list:
                    # Process all points if test succeeded
                    points_list = []
                    for p in point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True):
                        points_list.append([p[0], p[1], p[2]])
                    points_xyz = np.array(points_list, dtype=np.float32)
                    self.get_logger().info(f"point_cloud2 parsed {len(points_list)} points, shape: {points_xyz.shape}")
                else:
                    self.get_logger().warn("point_cloud2.read_points returned no points")
            except Exception as e:
                self.get_logger().error(f"point_cloud2 generator method failed: {e}")
                
        # Method 3: Manual parsing as last resort
        if points_xyz is None:
            try:
                self.get_logger().info("Trying manual parsing...")
                # Parse the binary data directly
                import struct
                points_list = []
                for i in range(0, len(msg.data), msg.point_step):
                    x = struct.unpack('f', msg.data[i:i+4])[0]
                    y = struct.unpack('f', msg.data[i+4:i+8])[0]
                    z = struct.unpack('f', msg.data[i+8:i+12])[0]
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        points_list.append([x, y, z])
                
                if points_list:
                    points_xyz = np.array(points_list, dtype=np.float32)
                    self.get_logger().info(f"Manual parsing succeeded: {len(points_list)} points, shape: {points_xyz.shape}")
            except Exception as e:
                self.get_logger().error(f"Manual parsing failed: {e}")
        
        if points_xyz is None or points_xyz.size == 0:
            self.get_logger().error(f"Failed to extract pointcloud data for {sub_key}")
            return
            
        self.get_logger().info(f"Successfully extracted {points_xyz.shape[0]} points")
        
        # TF lookup
        frame_sensor_id = msg.header.frame_id
        self.get_logger().info(f"Looking up transform: {frame_sensor_id} -> {self.map_frame}")
        transform_sensor_to_map = self.safe_lookup_transform(
            self.map_frame,
            frame_sensor_id,
            msg.header.stamp
        )
        if transform_sensor_to_map is None:
            self.get_logger().error(f"Transform lookup failed!")
            return
            
        t = transform_sensor_to_map.transform.translation
        q = transform_sensor_to_map.transform.rotation
        t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        
        self.get_logger().info(f"Transform: t=({t.x:.2f}, {t.y:.2f}, {t.z:.2f})")
        self.get_logger().info(f"About to input pointcloud to map...")
        
        try:
            self._map.input_pointcloud(points_xyz, channels, R, t_np, 0, 0)
            self._pointcloud_process_counter += 1
            # Log every 100th pointcloud to reduce spam
            if self._pointcloud_process_counter % 100 == 0:
                self.get_logger().info(f"Processed {self._pointcloud_process_counter} pointclouds")
        except Exception as e:
            self.get_logger().error(f"Failed to input pointcloud to map: {e}")

    def pose_update(self) -> None:
        self.get_logger().info("pose_update timer fired")
        if self._last_t is None:
            self.get_logger().warn("pose_update: _last_t is None, skipping")
            return
        
        transform = self.safe_lookup_transform(
            self.map_frame,
            self.base_frame,
            self._last_t
        )
        if transform is None:
            self.get_logger().warn("pose_update: transform lookup failed")
            return
        
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        
        self.get_logger().info(f"pose_update: moving map to ({t.x:.2f}, {t.y:.2f}, {t.z:.2f})")
        self._map.move_to(trans, rot)
        
        with self._lock:
            self._map_t = t
            self._map_q = q
            self.get_logger().info("pose_update: _map_t and _map_q updated")

    def update_variance(self) -> None:
        self.get_logger().info("update_variance timer fired")
        self._map.update_variance()

    def update_time(self) -> None:
        self.get_logger().info("update_time timer fired")
        self._map.update_time()

    def destroy_node(self) -> None:
        super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationMappingNode()
    # Use MultiThreadedExecutor with more threads to handle both pointcloud processing and timers
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
