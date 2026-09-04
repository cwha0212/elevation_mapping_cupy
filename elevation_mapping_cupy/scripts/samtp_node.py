#!/usr/bin/env python3
"""SAM-TP traversability scores as a semantic channel for elevation mapping.

SAM-TP (GeNIE's perception core: SAM2-tiny with the prompt encoder replaced by
learned embeddings) turns one camera image into a continuous per-pixel
traversability logit. That is a different judgement from a class label -- not
"this is road", but "this can be driven on" -- and it was trained on
in-the-wild navigation footage, which is why it survives surfaces that break a
Cityscapes classifier.

This node runs the TensorRT engine and publishes ``1 - sigmoid(logit)`` as a
single float channel named ``untrav``: high means hard or impossible ground.
elevation_mapping ingests it through the same image path as any semantic
channel, accumulates it on the map, and semantic_safety_filter treats it as
one more hazard. Nothing downstream knows a network changed.

The engine file is machine-specific (TensorRT serializes for one GPU and one
version) and therefore lives outside the repo. Rebuild with:

    /usr/src/tensorrt/bin/trtexec --onnx=samtp_512.onnx --fp16 \\
        --saveEngine=samtp_512_fp16.engine
"""

import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from elevation_map_msgs.msg import ChannelInfo
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class SamTPNode(Node):
    def __init__(self) -> None:
        super().__init__("samtp_node")
        self.engine_path = self.declare_parameter(
            "engine_path", os.path.expanduser("~/samtp/samtp_512_fp16.engine")
        ).value
        self.image_topic = self.declare_parameter("image_topic", "image").value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "camera_info"
        ).value
        # The score is meant for a 0.05 m map, so publishing it at full camera
        # resolution buys nothing; half is plenty and a quarter is usually fine.
        self.output_scale = float(self.declare_parameter("output_scale", 0.5).value)
        # Inference costs ~21 ms; the map accumulates, so a few Hz suffices and
        # the GPU is shared with the mapper. 0 disables the limit.
        self.max_rate = float(self.declare_parameter("max_rate", 4.0).value)

        self._load_engine()

        self.bridge = CvBridge()
        self.info = None
        self._last_stamp_ns = 0
        self._frames = 0

        self.score_pub = self.create_publisher(Image, "samtp_score", 2)
        self.info_pub = self.create_publisher(CameraInfo, "samtp_camera_info", 2)
        self.channel_pub = self.create_publisher(ChannelInfo, "samtp_channel_info", 2)

        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_info, 2
        )
        if "compressed" in self.image_topic:
            self.create_subscription(
                CompressedImage, self.image_topic, self._on_image, 2
            )
        else:
            self.create_subscription(Image, self.image_topic, self._on_image, 2)

        self.get_logger().info(
            f"SAM-TP on '{self.image_topic}' -> samtp_score (untrav), "
            f"engine {self.engine_path}"
        )

    def _load_engine(self) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"SAM-TP engine not found: {self.engine_path}. Engines are "
                "machine-specific; build one with trtexec from samtp_512.onnx."
            )
        logger = trt.Logger(trt.Logger.ERROR)
        with open(self.engine_path, "rb") as handle:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(
                f"Engine failed to deserialize: {self.engine_path}. It was "
                "likely built on another machine or TensorRT version; rebuild."
            )
        self.ctx = self.engine.create_execution_context()
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        self.side = int(self.engine.get_tensor_shape(in_name)[-1])
        self.in_t = torch.empty(
            (1, 3, self.side, self.side), dtype=torch.float32, device="cuda"
        )
        self.out_t = torch.empty(
            tuple(self.engine.get_tensor_shape(out_name)),
            dtype=torch.float32, device="cuda",
        )
        self.ctx.set_tensor_address(in_name, self.in_t.data_ptr())
        self.ctx.set_tensor_address(out_name, self.out_t.data_ptr())
        self._mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_image(self, msg) -> None:
        if self.info is None:
            self.get_logger().warning(
                "No CameraInfo yet; dropping frames.", throttle_duration_sec=5.0
            )
            return
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if self.max_rate > 0 and stamp_ns - self._last_stamp_ns < 1e9 / self.max_rate:
            return
        self._last_stamp_ns = stamp_ns

        torch = self._torch
        if isinstance(msg, CompressedImage):
            rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
        else:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        h, w = rgb.shape[:2]

        # Resize on the GPU: the CPU decode is already the pipeline's slow part.
        x = torch.from_numpy(np.ascontiguousarray(rgb)).cuda()
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(self.side, self.side), mode="bilinear", align_corners=False
        )
        self.in_t.copy_((x[0] - self._mean) / self._std)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        out_h = max(1, int(round(h * self.output_scale)))
        out_w = max(1, int(round(w * self.output_scale)))
        # The logit map corresponds to the square-stretched input, so resizing
        # it to the image's own aspect undoes the stretch and the original
        # intrinsics apply again (scaled below).
        untrav = 1.0 - torch.sigmoid(
            torch.nn.functional.interpolate(
                self.out_t, size=(out_h, out_w), mode="bilinear", align_corners=False
            )
        )[0, 0]
        score = untrav.cpu().numpy().astype(np.float32)

        out = self.bridge.cv2_to_imgmsg(score, encoding="32FC1")
        out.header = msg.header
        self.score_pub.publish(out)

        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = out_w, out_h
        sx = out_w / float(self.info.width)
        sy = out_h / float(self.info.height)
        k = np.array(self.info.k, dtype=np.float64).reshape(3, 3)
        k[0, :] *= sx
        k[1, :] *= sy
        info.k = k.reshape(-1).tolist()
        p = np.array(self.info.p, dtype=np.float64).reshape(3, 4)
        p[0, :] *= sx
        p[1, :] *= sy
        info.p = p.reshape(-1).tolist()
        info.d = list(self.info.d)
        info.distortion_model = self.info.distortion_model
        info.r = list(self.info.r)
        self.info_pub.publish(info)

        channels = ChannelInfo()
        channels.header = msg.header
        channels.channels = ["untrav"]
        self.channel_pub.publish(channels)

        self._frames += 1
        self.get_logger().info(
            f"SAM-TP frames: {self._frames}", throttle_duration_sec=10.0
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SamTPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
