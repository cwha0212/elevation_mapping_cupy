#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sys

import numpy as np
try:
    import cupy as cp
except ImportError:
    import numpy as cp
import cv2

# np.float = np.float64  # temp fix removed
np.bool = np.bool_

import matplotlib.pyplot as plt
# from skimage.io import imshow

from sensor_msgs.msg import Image, CameraInfo, CompressedImage, PointCloud2
from cv_bridge import CvBridge

from semantic_sensor.image_parameters import ImageParameter
from semantic_sensor.networks import resolve_model
from simple_parsing.helpers import Serializable
from sklearn.decomposition import PCA

from elevation_map_msgs.msg import ChannelInfo

class SemanticSegmentationNode(Node):
    def __init__(self, sensor_name):
        """Get parameter from server, initialize variables and semantics, register publishers and subscribers.

        Args:
            sensor_name (str): Name of the sensor in the ros param server.
        """
        super().__init__('semantic_segmentation_node')
        
        # Automatic declaration
        self.get_node_options().automatically_declare_parameters_from_overrides(True)

        self.param: ImageParameter = ImageParameter()
        self.param.feature_config.input_size = [80, 160]
        
        self.param.sensor_name = sensor_name
        
        # Helper to safely get parameter
        def get_p(name, default):
            try:
                if not self.has_parameter(name):
                    self.declare_parameter(name, default)
                return self.get_parameter(name).value
            except Exception as e:
                self.get_logger().warn(f"Could not load param {name}, using default: {e}")
                return default

        # Logic for ImageParameter loading
        # Based on sensor_parameter.yaml structure:
        # front_cam_image: ...
        prefix = sensor_name

        self.param.publish_topic = get_p(f"{prefix}.publish_topic", self.param.publish_topic)
        self.param.publish_image_topic = get_p(f"{prefix}.publish_image_topic", self.param.publish_image_topic)
        self.param.publish_camera_info_topic = get_p(f"{prefix}.publish_camera_info_topic", self.param.publish_camera_info_topic)
        self.param.publish_fusion_info_topic = get_p(f"{prefix}.publish_fusion_info_topic", self.param.publish_fusion_info_topic)
        
        self.param.channels = get_p(f"{prefix}.channels", self.param.channels)
        self.param.fusion_methods = get_p(f"{prefix}.fusion_methods", self.param.fusion_methods)
        
        self.param.semantic_segmentation = get_p(f"{prefix}.semantic_segmentation", self.param.semantic_segmentation)
        self.param.feature_extractor = get_p(f"{prefix}.feature_extractor", self.param.feature_extractor)
        self.param.segmentation_model = get_p(f"{prefix}.segmentation_model", self.param.segmentation_model)
        self.param.show_label_legend = get_p(f"{prefix}.show_label_legend", self.param.show_label_legend)
        
        self.param.image_topic = get_p(f"{prefix}.image_topic", self.param.image_topic)
        self.param.camera_info_topic = get_p(f"{prefix}.camera_info_topic", self.param.camera_info_topic)
        self.param.resize = get_p(f"{prefix}.resize", 1.0) # default 1.0 if not set, though was None in code?

        self.get_logger().info("--------------Pointcloud Parameters-------------------")
        self.get_logger().info(f"Sensor: {sensor_name}")
        self.get_logger().info(f"Topic: {self.param.publish_topic}")
        self.get_logger().info("--------------End of Parameters-----------------------")
        self.semseg_color_map = None
        # setup custom dtype
        # setup semantics
        self.feature_extractor = None
        self.semantic_model = None
        self.initialize_semantics()

        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None

    def initialize_semantics(self):
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model, self.param)

        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)

    def register_sub_pub(self):
        """Register publishers and subscribers."""

        node_name = self.get_name()
        # subscribers
        if self.param.camera_info_topic is not None and self.param.resize is not None:
            self.create_subscription(CameraInfo, self.param.camera_info_topic, self.image_info_callback, 10)
            self.feat_im_info_pub = self.create_publisher(
                CameraInfo,
                node_name + "/" + self.param.camera_info_topic + "_resized", 2
            )

        if "compressed" in self.param.image_topic:
            self.compressed = True
            self.subscriber = self.create_subscription(
                CompressedImage, self.param.image_topic, self.image_callback, 2
            )
        else:
            self.compressed = False
            self.create_subscription(Image, self.param.image_topic, self.image_callback, 10)

        # publishers
        if self.param.semantic_segmentation:
            self.seg_pub = self.create_publisher(Image, node_name + "/" + self.param.publish_topic, 2)
            self.seg_im_pub = self.create_publisher(Image, node_name + "/" + self.param.publish_image_topic, 2)
            self.semseg_color_map = self.color_map(len(self.param.channels))
            if self.param.show_label_legend:
                pass # self.color_map_viz()
        if self.param.feature_extractor:
            self.feature_pub = self.create_publisher(Image, node_name + "/" + self.param.feature_topic, 2)
            self.feat_im_pub = self.create_publisher(Image, node_name + "/" + self.param.feat_image_topic, 2)
            self.feat_channel_info_pub = self.create_publisher(
                ChannelInfo,
                node_name + "/" + self.param.feat_channel_info_topic, 2
            )

        self.channel_info_pub = self.create_publisher(
            ChannelInfo,
            node_name + "/" + self.param.channel_info_topic, 2
        )

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
        cmap[1] = np.array([81, 113, 162])
        cmap[2] = np.array([81, 113, 162])
        cmap[3] = np.array([188, 63, 59])
        cmap = cmap / 255 if normalized else cmap
        return cmap[1:]

    def image_info_callback(self, msg):
        """Callback for camera info."""
        self.P = np.array(msg.k).reshape(3, 3) 
        # Matches logic in pointcloud_node
        self.P = np.array(msg.p).reshape(3, 4)
        
        self.height = int(self.param.resize * msg.height)
        self.width = int(self.param.resize * msg.width)
        self.info = msg
        self.info.height = self.height
        self.info.width = self.width
        
        # Scaling P
        self.P[:2, :] = self.P[:2, :] * self.param.resize
        self.info.k = self.P[:3, :3].flatten().tolist()
        self.info.p = self.P.flatten().tolist()

    def image_callback(self, rgb_msg):
        if self.param.camera_info_topic is not None and self.P is None:
             return
             
        if self.compressed:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg)
            if self.param.resize is not None:
                image = cv2.resize(image, dsize=(self.width, self.height))
            image = cp.asarray(image)
        else:
            image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            if self.param.resize is not None:
                image = cv2.resize(image, dsize=(self.width, self.height))
            image = cp.asarray(image)
        self.header = rgb_msg.header
        self.process_image(image)

        if self.param.semantic_segmentation:
            self.publish_segmentation()
            self.publish_segmentation_image()
            self.publish_channel_info([f"sem_{c}" for c in self.param.channels], self.channel_info_pub)
        if self.param.feature_extractor:
            self.publish_feature()
            self.publish_feature_image(self.features)
            self.publish_channel_info([f"feat_{i}" for i in range(self.features.shape[0])], self.feat_channel_info_pub)
        if self.param.resize is not None:
            self.pub_info()

    def pub_info(self):
        self.feat_im_info_pub.publish(self.info)

    def publish_channel_info(self, channels, pub):
        """Publish fusion info."""
        info = ChannelInfo()
        info.header = self.header
        info.channels = channels
        pub.publish(info)

    def process_image(self, image):
        if self.param.semantic_segmentation:
            self.sem_seg = self.semantic_model["model"](image)

        if self.param.feature_extractor:
            self.features = self.feature_extractor["model"](image)

    def publish_segmentation(self):
        probabilities = self.sem_seg
        img = probabilities.get()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        seg_msg.header.frame_id = self.header.frame_id
        seg_msg.header.stamp = self.header.stamp
        self.seg_pub.publish(seg_msg)

    def publish_feature(self):
        features = self.features
        img = features.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        feature_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        feature_msg.header.frame_id = self.header.frame_id
        feature_msg.header.stamp = self.header.stamp
        self.feature_pub.publish(feature_msg)

    def publish_segmentation_image(self):
        colors = None
        probabilities = self.sem_seg
        if self.param.semantic_segmentation:
            colors = cp.asarray(self.semseg_color_map)
            # assert colors.ndim == 2 and colors.shape[1] == 3

        img = cp.argmax(probabilities, axis=0)
        img = colors[img].astype(cp.uint8)  # N x H x W x 3
        img = img.get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header.frame_id = self.header.frame_id
        seg_msg.header.stamp = self.header.stamp
        self.seg_im_pub.publish(seg_msg)

    def publish_feature_image(self, features):
        data = np.reshape(features.cpu().detach().numpy(), (features.shape[0], -1)).T
        n_components = 3
        try:
            pca = PCA(n_components=n_components).fit(data)
            pca_descriptors = pca.transform(data)
            img_pca = pca_descriptors.reshape(features.shape[1], features.shape[2], n_components)
            comp = img_pca  # [:, :, -3:]
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            comp_img = (comp - comp_min) / (comp_max - comp_min)
            comp_img = (comp_img * 255).astype(np.uint8)
            feat_msg = self.cv_bridge.cv2_to_imgmsg(comp_img, encoding="passthrough")
            feat_msg.header.frame_id = self.header.frame_id
            self.feat_im_pub.publish(feat_msg)
        except Exception as e:
            self.get_logger().warn(f"PCA error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sensor_name = sys.argv[1]
    else:
        sensor_name = "camera"
    rclpy.init()
    node = SemanticSegmentationNode(sensor_name)
    rclpy.spin(node)
    rclpy.shutdown()
