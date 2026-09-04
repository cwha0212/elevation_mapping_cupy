#!/usr/bin/env python3

import copy

import cupy as cp
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from elevation_map_msgs.msg import ChannelInfo
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from semantic_sensor.config_io import load_sensor_config
from semantic_sensor.image_parameters import ImageParameter
from semantic_sensor.networks import resolve_model


class SemanticImageNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "semantic_image_node",
            automatically_declare_parameters_from_overrides=True,
        )

        sensor_name = self._require_string_parameter("sensor_name")
        config_path = self._optional_string_parameter("config_path")
        sensor_config = load_sensor_config(sensor_name, config_path or None)

        self.param = ImageParameter.from_dict(sensor_config)
        self.param.feature_config.input_size = [80, 160]
        self.get_logger().info(f"Loaded semantic image config for '{sensor_name}'.")

        self.cv_bridge = CvBridge()
        self.header = None
        self.info = None
        self.semseg_color_map = None
        self.camera_info_output_topic = self._camera_info_output_topic()

        # Cameras that publish no CameraInfo can carry their calibration in the
        # sensor config instead. Seeding self.info here reuses the resize logic
        # in image_info_callback, so nothing downstream changes.
        self._static_info = self._build_static_camera_info()
        if self._static_info is not None:
            self.image_info_callback(self._static_info)
            self.get_logger().info(
                f"Using static camera intrinsics from config for '{sensor_name}' "
                f"(no subscription to '{self.param.camera_info_topic}')."
            )

        self.feature_extractor = None
        self.semantic_model = None
        self.initialize_semantics()
        self.register_sub_pub()

    def _require_string_parameter(self, name: str) -> str:
        if not self.has_parameter(name):
            raise ValueError(f"Required parameter '{name}' was not provided.")
        value = self.get_parameter(name).get_parameter_value().string_value
        if not value:
            raise ValueError(f"Required parameter '{name}' must not be empty.")
        return value

    def _optional_string_parameter(self, name: str) -> str:
        if not self.has_parameter(name):
            return ""
        return self.get_parameter(name).get_parameter_value().string_value

    def initialize_semantics(self) -> None:
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model, self.param)
        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)

    def register_sub_pub(self) -> None:
        if self._static_info is None:
            self.create_subscription(CameraInfo, self.param.camera_info_topic, self.image_info_callback, 2)

        if "compressed" in self.param.image_topic:
            self.compressed = True
            self.create_subscription(CompressedImage, self.param.image_topic, self.image_callback, 2)
        else:
            self.compressed = False
            self.create_subscription(Image, self.param.image_topic, self.image_callback, 2)

        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_output_topic, 2)

        if self.param.semantic_segmentation:
            self.seg_pub = self.create_publisher(Image, self.param.publish_topic, 2)
            self.seg_im_pub = self.create_publisher(Image, self.param.publish_image_topic, 2)
            self.channel_info_pub = self.create_publisher(ChannelInfo, self.param.channel_info_topic, 2)
            self.semseg_color_map = self.color_map(len(self.param.channels))

        if self.param.feature_extractor:
            self.feature_pub = self.create_publisher(Image, self.param.feature_topic, 2)
            self.feat_im_pub = self.create_publisher(Image, self.param.feat_image_topic, 2)
            self.feat_channel_info_pub = self.create_publisher(ChannelInfo, self.param.feat_channel_info_topic, 2)

    def _shift_stamp(self, header):
        """Apply the calibrated camera-to-reference clock offset."""
        if not self.param.time_offset_s:
            return header
        shifted = copy.deepcopy(header)
        ns = (
            shifted.stamp.sec * 1_000_000_000
            + shifted.stamp.nanosec
            + int(round(self.param.time_offset_s * 1e9))
        )
        shifted.stamp.sec, shifted.stamp.nanosec = divmod(ns, 1_000_000_000)
        return shifted

    def _build_static_camera_info(self) -> CameraInfo | None:
        """Build a CameraInfo from the configured intrinsics, or None to use the topic."""
        ci = self.param.camera_intrinsics
        if not ci.k:
            return None
        if len(ci.k) != 9:
            raise ValueError(f"camera_intrinsics.k must hold 9 values, got {len(ci.k)}.")
        if ci.p and len(ci.p) != 12:
            raise ValueError(f"camera_intrinsics.p must hold 12 values, got {len(ci.p)}.")
        if not ci.width or not ci.height:
            raise ValueError("camera_intrinsics.width and .height must be set.")

        msg = CameraInfo()
        msg.width = int(ci.width)
        msg.height = int(ci.height)
        msg.distortion_model = ci.distortion_model
        msg.k = [float(v) for v in ci.k]
        msg.d = [float(v) for v in ci.d] if ci.d else [0.0] * 5
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if ci.p:
            msg.p = [float(v) for v in ci.p]
        else:
            # image_info_callback derives the resized k from p, so p must be filled.
            k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            msg.p = np.hstack([k, np.zeros((3, 1))]).reshape(-1).tolist()
        return msg

    def _camera_info_output_topic(self) -> str:
        if not self.param.camera_info_topic:
            raise ValueError("camera_info_topic must be configured for semantic image sensors.")
        if self.param.resize is not None:
            return f"{self.param.camera_info_topic.lstrip('/')}_resized"
        return self.param.publish_camera_info_topic

    def color_map(self, n: int = 256, normalized: bool = False) -> np.ndarray:
        def bitget(byteval: int, idx: int) -> bool:
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((n + 1, 3), dtype=dtype)
        for i in range(n + 1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        # Hand-picked colours for the first few entries, applied only as far as
        # the palette actually goes: writing all three unconditionally made any
        # sensor with fewer than three channels fail at startup, which is every
        # sensor that names one or two classes.
        for idx, colour in enumerate(
            ([81, 113, 162], [81, 113, 162], [188, 63, 59]), start=1
        ):
            if idx < len(cmap):
                cmap[idx] = np.array(colour)
        return cmap[1:] / 255 if normalized else cmap[1:]

    def image_info_callback(self, msg: CameraInfo) -> None:
        info = copy.deepcopy(msg)
        if self.param.resize is not None:
            info.height = int(round(self.param.resize * msg.height))
            info.width = int(round(self.param.resize * msg.width))

            p = np.array(msg.p, dtype=np.float32).reshape(3, 4)
            p[:2, :3] *= self.param.resize
            info.k = p[:3, :3].reshape(-1).tolist()
            info.p = p.reshape(-1).tolist()
        self.info = info

    def image_callback(self, rgb_msg: Image | CompressedImage) -> None:
        if self.info is None:
            return

        if self.compressed:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        else:
            image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        if self.param.resize is not None:
            image = cv2.resize(image, dsize=(self.info.width, self.info.height))

        self.header = self._shift_stamp(rgb_msg.header)
        image_cp = cp.asarray(image)
        self.process_image(image_cp)

        if self.param.semantic_segmentation:
            self.publish_segmentation()
            self.publish_segmentation_image()
            self.publish_channel_info(self.param.channels, self.channel_info_pub)

        if self.param.feature_extractor:
            self.publish_feature()
            self.publish_feature_image(self.features)
            feat_channels = [f"feat_{i}" for i in range(self.features.shape[0])]
            self.publish_channel_info(feat_channels, self.feat_channel_info_pub)

        info_msg = copy.deepcopy(self.info)
        info_msg.header = self.header
        self.camera_info_pub.publish(info_msg)

    def publish_channel_info(self, channels: list[str], publisher) -> None:
        info = ChannelInfo()
        info.header = self.header
        info.channels = channels
        publisher.publish(info)

    def process_image(self, image: cp.ndarray) -> None:
        if self.param.semantic_segmentation:
            self.sem_seg = self.semantic_model["model"](image)
        if self.param.feature_extractor:
            self.features = self.feature_extractor["model"](image)

    def publish_segmentation(self) -> None:
        img = self.sem_seg.get()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        seg_msg.header = self.header
        self.seg_pub.publish(seg_msg)

    def publish_feature(self) -> None:
        img = self.features.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        feature_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        feature_msg.header = self.header
        self.feature_pub.publish(feature_msg)

    def publish_segmentation_image(self) -> None:
        colors = cp.asarray(self.semseg_color_map)
        img = cp.argmax(self.sem_seg, axis=0)
        img = colors[img].astype(cp.uint8).get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header = self.header
        self.seg_im_pub.publish(seg_msg)

    def publish_feature_image(self, features) -> None:
        from sklearn.decomposition import PCA

        data = np.reshape(features.cpu().detach().numpy(), (features.shape[0], -1)).T
        pca = PCA(n_components=3).fit(data)
        pca_descriptors = pca.transform(data)
        img_pca = pca_descriptors.reshape(features.shape[1], features.shape[2], 3)
        comp_min = img_pca.min(axis=(0, 1))
        comp_max = img_pca.max(axis=(0, 1))
        comp_img = (img_pca - comp_min) / (comp_max - comp_min)
        comp_img = (comp_img * 255).astype(np.uint8)
        feat_msg = self.cv_bridge.cv2_to_imgmsg(comp_img, encoding="passthrough")
        feat_msg.header = self.header
        self.feat_im_pub.publish(feat_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticImageNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
