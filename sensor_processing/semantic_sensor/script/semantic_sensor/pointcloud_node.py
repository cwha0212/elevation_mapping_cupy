import rclpy
from rclpy.node import Node
import sys
import numpy as np
try:
    import cupy as cp
except ImportError:
    import numpy as cp # Fallback if cupy not present, though loop might be slow

# np.float = np.float64 # Removed as it might cause issues and is a workaround for old libraries

import matplotlib.pyplot as plt
# from skimage.io import imshow # Optional

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

from semantic_sensor.pointcloud_parameters import PointcloudParameter
from semantic_sensor.networks import resolve_model
from semantic_sensor.utils import decode_max
from sklearn.decomposition import PCA

# Helper to create PointCloud2 from structured numpy array
def msgify(msg_type, numpy_array):
    assert msg_type == PointCloud2
    msg = PointCloud2()
    msg.height = 1
    msg.width = numpy_array.shape[0]
    # msg.header will be set later
    msg.fields = []
    offset = 0
    for name, format_type in numpy_array.dtype.descr:
        pf = PointField()
        pf.name = name
        if format_type == '<f4':
            pf.datatype = PointField.FLOAT32
            pf.count = 1
            pf.offset = offset
            offset += 4
        elif format_type == '<u4': # Assuming color is uint32
            pf.datatype = PointField.UINT32
            pf.count = 1
            pf.offset = offset
            offset += 4
        else:
            # Fallback or error
            pf.datatype = PointField.FLOAT32
            pf.count = 1
            pf.offset = offset
            offset += 4
        msg.fields.append(pf)
    
    msg.is_bigendian = False
    msg.point_step = offset
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = numpy_array.tobytes()
    return msg

class PointcloudNode(Node):
    def __init__(self, sensor_name):
        """Get parameter from server, initialize variables and semantics, register publishers and subscribers.

        Args:
            sensor_name (str): Name of the sensor in the ros param server.
        """
        super().__init__('semantic_pointcloud_node')
        
        # Enable parameter declarations from yaml
        self.get_node_options().automatically_declare_parameters_from_overrides(True)
        
        self.param: PointcloudParameter = PointcloudParameter()
        self.param.feature_config.input_size = [80, 160]
        
        # Load parameters from ROS2 params
        # The yaml structure is:
        # front_cam_pointcloud:
        #   channels: ...
        #   ...
        # So variables are `front_cam_pointcloud.channels`
        # We need to read these into the PointcloudParameter struct.
        
        self.param.sensor_name = sensor_name
        
        # Helper to safely get parameter
        def get_p(name, default):
            try:
                # declare_parameter if not exists?
                # If we rely on automatic declaration from overrides (yaml), we use get_parameter
                # But if it's not in yaml, we might want a default.
                if not self.has_parameter(name):
                    self.declare_parameter(name, default)
                return self.get_parameter(name).value
            except Exception as e:
                self.get_logger().warn(f"Could not load param {name}, using default: {e}")
                return default

        prefix = sensor_name # e.g. "front_cam_pointcloud"
        
        self.param.topic_name = get_p(f"{prefix}.topic_name", self.param.topic_name)
        self.param.channels = get_p(f"{prefix}.channels", self.param.channels)
        self.param.fusion = get_p(f"{prefix}.fusion", self.param.fusion)
        self.param.semantic_segmentation = get_p(f"{prefix}.semantic_segmentation", self.param.semantic_segmentation)
        self.param.publish_segmentation_image = get_p(f"{prefix}.publish_segmentation_image", self.param.publish_segmentation_image)
        self.param.segmentation_image_topic = get_p(f"{prefix}.segmentation_image_topic", self.param.segmentation_image_topic)
        self.param.segmentation_model = get_p(f"{prefix}.segmentation_model", self.param.segmentation_model)
        self.param.show_label_legend = get_p(f"{prefix}.show_label_legend", self.param.show_label_legend)
        
        self.param.cam_info_topic = get_p(f"{prefix}.cam_info_topic", self.param.cam_info_topic)
        self.param.image_topic = get_p(f"{prefix}.image_topic", self.param.image_topic)
        self.param.depth_topic = get_p(f"{prefix}.depth_topic", self.param.depth_topic)
        self.param.cam_frame = get_p(f"{prefix}.cam_frame", self.param.cam_frame)
        self.param.confidence = get_p(f"{prefix}.confidence", self.param.confidence)
        self.param.confidence_topic = get_p(f"{prefix}.confidence_topic", self.param.confidence_topic)
        self.param.confidence_threshold = get_p(f"{prefix}.confidence_threshold", self.param.confidence_threshold)
        
        self.param.feature_extractor = get_p(f"{prefix}.feature_extractor", self.param.feature_extractor)
        self.param.publish_feature_image = get_p(f"{prefix}.publish_feature_image", self.param.publish_feature_image)
        
        # Feature config nested? 
        # FeatureExtractorParameter fields in config?
        # The yaml doesn't show feature config params for pointcloud, but might be there.
        # We'll skip deep nested feature config for now unless needed.
        
        self.get_logger().info("--------------Pointcloud Parameters-------------------")
        self.get_logger().info(f"Sensor: {self.param.sensor_name}")
        self.get_logger().info(f"Topic: {self.param.topic_name}")
        self.get_logger().info(f"Channels: {self.param.channels}")
        self.get_logger().info("--------------End of Parameters-----------------------")
        
        self.semseg_color_map = None
        # setup custom dtype
        self.create_custom_dtype()
        # setup semantics
        self.feature_extractor = None
        self.semantic_model = None
        self.segmentation_channels = None
        self.feature_channels = None
        self.initialize_semantics()

        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None
        self.feat_img = None

    def initialize_semantics(self):
        """Resolve the feature and segmentation mode and create segmentation_channel and feature_channels."""
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model, self.param)
            self.segmentation_channels = {}
            for i, (chan, fusion) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fusion in ["class_bayesian", "class_average", "class_max"]:
                    self.segmentation_channels[chan] = fusion
            assert len(self.segmentation_channels.keys()) > 0
        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)
            self.feature_channels = {}
            for i, (chan, fusion) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fusion in ["average"]:
                    self.feature_channels[chan] = fusion
            assert len(self.feature_channels.keys()) > 0

    def register_sub_pub(self):
        """Register publishers and subscribers."""
        # subscribers
        self.create_subscription(CameraInfo, self.param.cam_info_topic, self.cam_info_callback, 10)
        
        rgb_sub = message_filters.Subscriber(self, Image, self.param.image_topic)
        depth_sub = message_filters.Subscriber(self, Image, self.param.depth_topic)
        if self.param.confidence:
            confidence_sub = message_filters.Subscriber(self, Image, self.param.confidence_topic)
            ts = message_filters.ApproximateTimeSynchronizer(
                [depth_sub, rgb_sub, confidence_sub], queue_size=10, slop=0.5,
            )
        else:
            ts = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.image_callback)

        self.pcl_pub = self.create_publisher(PointCloud2, self.param.topic_name, 2)
        
        # publishers
        if self.param.semantic_segmentation:
            if self.param.publish_segmentation_image:
                self.seg_pub = self.create_publisher(Image, self.param.segmentation_image_topic, 2)
            if "class_max" in self.param.fusion:
                self.labels = self.semantic_model["model"].get_classes()
            else:
                self.labels = list(self.segmentation_channels.keys())
            self.semseg_color_map = self.color_map(len(self.labels))
            if self.param.show_label_legend:
                pass # Visualization blocked
        if self.param.feature_extractor:
            # todo
            if True:
                self.feat_pub = self.create_publisher(Image, self.param.feature_config.feature_image_topic, 2)

    def color_map(self, N=256, normalized=False):
        """Create a color map for the class labels."""
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N + 1, 3), dtype=dtype)
        for i in range(N + 1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])
        cmap[1] = np.array([188, 63, 59])
        cmap[2] = np.array([81, 113, 162])
        cmap[3] = np.array([136, 49, 132])
        cmap = cmap / 255 if normalized else cmap
        return cmap[1:]

    def create_custom_dtype(self):
        """Generate a new dtype according to the channels in the params."""
        self.dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
        ]
        for chan, fus in zip(self.param.channels, self.param.fusion):
            self.dtype.append((chan, np.float32))
        print(self.dtype)

    def cam_info_callback(self, msg):
        """Subscribe to the camera infos to get projection matrix and header."""
        # ROS2 CameraInfo uses k (array of 9)
        a = cp.asarray(msg.k)
        self.P = cp.resize(a, (3, 3)) # Intrinsic matrix K
        
        # The logic in original code uses 3x4 P matrix for projection?
        # msg.P is 3x4 projection matrix. msg.K is 3x3 intrinsic.
        # Original: self.P = cp.resize(a, (3, 4)) where a was msg.P
        # And in another place it used K. Let's check original.
        # Original: a = cp.asarray(msg.P); self.P = cp.resize(a, (3, 4))
        # Let's restore that logic for msg.P.
        
        a_p = cp.asarray(msg.p)
        self.P = cp.resize(a_p, (3, 4))
        
        self.height = msg.height
        self.width = msg.width
        self.header = msg.header

    def image_callback(self, depth_msg, rgb_msg=None, confidence_msg=None):
        confidence = None
        image = None
        if self.P is None:
            return
        if rgb_msg is not None:
            image = cp.asarray(self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8"))
        depth = cp.asarray(self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        if confidence_msg is not None:
            confidence = cp.asarray(self.cv_bridge.imgmsg_to_cv2(confidence_msg, desired_encoding="passthrough"))

        pcl = self.create_pcl_from_image(image, depth, confidence)
        self.publish_pointcloud(pcl, depth_msg.header)

        if self.param.publish_segmentation_image:
            self.publish_segmentation_image(self.prediction_img)
        if self.param.publish_feature_image and self.param.feature_extractor:
            self.publish_feature_image(self.feat_img)

    def create_pcl_from_image(self, image, depth, confidence):
        u, v = self.get_coordinates(depth, confidence)

        # create pointcloud
        world_x = (u.astype(np.float32) - self.P[0, 2]) * depth[v, u] / self.P[0, 0]
        world_y = (v.astype(np.float32) - self.P[1, 2]) * depth[v, u] / self.P[1, 1]
        world_z = depth[v, u]
        points = np.zeros(world_x.shape, dtype=self.dtype)
        points["x"] = cp.asnumpy(world_x)
        points["y"] = cp.asnumpy(world_y)
        points["z"] = cp.asnumpy(world_z)
        self.process_image(image, u, v, points)
        return points

    def get_coordinates(self, depth, confidence):
        pos = cp.where(depth > 0, 1, 0)
        low = cp.where(depth < 8, 1, 0)
        if confidence is not None:
            conf = cp.where(confidence >= self.param.confidence_threshold, 1, 0)
        else:
            conf = cp.ones(pos.shape)
        fin = cp.isfinite(depth)
        temp = cp.maximum(cp.rint(fin + pos + conf + low - 2.6), 0)
        mask = cp.nonzero(temp)
        u = mask[1]
        v = mask[0]
        return u, v

    def process_image(self, image, u, v, points):
        if "color" in self.param.fusion:
            valid_rgb = image[v, u].get()
            r = np.asarray(valid_rgb[:, 0], dtype=np.uint32)
            g = np.asarray(valid_rgb[:, 1], dtype=np.uint32)
            b = np.asarray(valid_rgb[:, 2], dtype=np.uint32)
            rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
            rgb_arr.dtype = np.float32
            position = self.param.fusion.index("color")
            points[self.param.channels[position]] = rgb_arr

        if self.segmentation_channels is not None:
            self.perform_segmentation(image, points, u, v)
        if self.feature_channels is not None:
            self.extract_features(image, points, u, v)

    def perform_segmentation(self, image, points, u, v):
        prediction = self.semantic_model["model"](image)
        values = prediction[:, v.get(), u.get()].get()
        for it, channel in enumerate(self.semantic_model["model"].actual_channels):
            points[channel] = values[it]
        if self.param.publish_segmentation_image and self.param.semantic_segmentation:
            self.prediction_img = prediction

    def extract_features(self, image, points, u, v):
        prediction = self.feature_extractor["model"](image)
        values = prediction[:, v.get(), u.get()].cpu().detach().numpy()
        for it, channel in enumerate(self.feature_channels.keys()):
            points[channel] = values[it]
        if False and self.param.feature_extractor:
            self.feat_img = prediction

    def publish_segmentation_image(self, probabilities):
        if self.param.semantic_segmentation:
            colors = cp.asarray(self.semseg_color_map)
            assert colors.ndim == 2 and colors.shape[1] == 3
        if self.P is None:
            return
        prob = cp.zeros((len(self.labels),) + probabilities.shape[1:])
        if "class_max" in self.param.fusion:
            it = 0
            for iit, (chan, fuse) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fuse in ["class_max"]:
                    temp = probabilities[it]
                    temp_p, temp_i = decode_max(temp)
                    temp_i.choose(prob)
                    c = cp.mgrid[0 : temp_i.shape[0], 0 : temp_i.shape[1]]
                    prob[temp_i, c[0], c[1]] = temp_p
                    it += 1
                elif fuse in ["class_bayesian", "class_average"]:
                    if chan in self.semantic_model["model"].segmentation_channels:
                        prob[self.semantic_model["model"].segmentation_channels[chan]] = probabilities[it]
                        it += 1
            img = cp.argmax(prob, axis=0)

        else:
            img = cp.argmax(probabilities, axis=0)
        img = colors[img].astype(cp.uint8)  # N x H x W x 3
        img = img.get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header.frame_id = self.header.frame_id
        self.seg_pub.publish(seg_msg)

    def publish_feature_image(self, features):
        data = np.reshape(features.cpu().detach().numpy(), (features.shape[0], -1)).T
        n_components = 3
        try:
            pca = PCA(n_components=n_components).fit(data)
            pca_descriptors = pca.transform(data)
            img_pca = pca_descriptors.reshape(features.shape[1], features.shape[2], n_components)
            comp = img_pca
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            comp_img = (comp - comp_min) / (comp_max - comp_min)
            comp_img = (comp_img * 255).astype(np.uint8)
            feat_msg = self.cv_bridge.cv2_to_imgmsg(comp_img, encoding="passthrough")
            feat_msg.header.frame_id = self.header.frame_id
            self.feat_pub.publish(feat_msg)
        except Exception as e:
            self.get_logger().warn(f"PCA visualization error: {e}")

    def publish_pointcloud(self, pcl, header):
        # Using custom msgify
        pc2 = msgify(PointCloud2, pcl)
        pc2.header = header
        self.pcl_pub.publish(pc2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sensor_name = sys.argv[1]
    else:
        sensor_name = "camera" # Default
    rclpy.init()
    node = PointcloudNode(sensor_name)
    rclpy.spin(node)
    rclpy.shutdown()
