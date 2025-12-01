#!/usr/bin/env python3
"""
Utility for issuing masked_replace requests to elevation_mapping_cupy.
Useful to test the masked_replace service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap
from grid_map_msgs.srv import SetGridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension


def positive_float(value: str) -> float:
    val = float(value)
    if val <= 0.0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return val


def non_negative_float(value: str) -> float:
    val = float(value)
    if val < 0.0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a masked_replace patch to elevation_mapping_cupy.")
    parser.add_argument("--service", default="/elevation_mapping_cupy/masked_replace", help="Service name to call.")
    parser.add_argument("--mask-layer", default="mask", help="Name of the mask layer expected by the node.")
    parser.add_argument("--frame", default="odom", help="Frame ID used in the GridMap header.")
    parser.add_argument("--center-x", type=float, default=0.0, help="Patch center X coordinate (meters).")
    parser.add_argument("--center-y", type=float, default=0.0, help="Patch center Y coordinate (meters).")
    parser.add_argument("--center-z", type=float, default=0.0, help="Patch center Z coordinate (meters).")
    parser.add_argument("--size-x", type=positive_float, default=1.0, help="Patch length in X (meters).")
    parser.add_argument("--size-y", type=positive_float, default=1.0, help="Patch length in Y (meters).")
    parser.add_argument("--resolution", type=positive_float, default=0.1, help="Grid resolution (meters per cell).")
    parser.add_argument("--elevation", type=float, default=0.1, help="Elevation value to set (meters).")
    parser.add_argument("--variance", type=non_negative_float, default=0.05, help="Variance value to set.")
    parser.add_argument("--mask-value", type=float, default=1.0, help="Mask value applied over the patch (NaN keeps cells untouched).")
    parser.add_argument("--valid-layer", dest="valid_layer", action="store_true", help="Also set the 'is_valid' layer to 1.0 in the patch region (default).")
    parser.add_argument("--no-valid-layer", dest="valid_layer", action="store_false", help="Do not modify the 'is_valid' layer.")
    parser.add_argument(
        "--invalidate-first",
        dest="invalidate_first",
        action="store_true",
        help="Emulate the LiDAR update flow by first invalidating cells (set is_valid=0) before writing new values.",
    )
    parser.add_argument(
        "--no-invalidate-first",
        dest="invalidate_first",
        action="store_false",
        help="Write the new data directly without a preceding invalidation pass.",
    )
    parser.set_defaults(valid_layer=True)
    parser.set_defaults(invalidate_first=False)
    return parser


@dataclass
class PatchConfig:
    center_x: float
    center_y: float
    center_z: float
    length_x: float
    length_y: float
    resolution: float
    frame_id: str
    mask_layer: str
    elevation: float
    variance: float
    mask_value: float
    add_valid_layer: bool
    invalidate_first: bool

    @property
    def shape(self) -> Dict[str, int]:
        cols = max(1, ceil(self.length_x / self.resolution))
        rows = max(1, ceil(self.length_y / self.resolution))
        return {"rows": rows, "cols": cols}

    @property
    def actual_length_x(self) -> float:
        return self.shape["cols"] * self.resolution

    @property
    def actual_length_y(self) -> float:
        return self.shape["rows"] * self.resolution


class MaskedReplaceClient(Node):
    def __init__(self, service_name: str, config: PatchConfig):
        super().__init__("masked_replace_client")
        self._config = config
        self._client = self.create_client(SetGridMap, service_name)
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service {service_name} ...")

    def send_request(self):
        cfg = self._config
        if cfg.invalidate_first and not cfg.add_valid_layer:
            self.get_logger().warning(
                "--invalidate-first requested but --no-valid-layer is set; falling back to single-pass update."
            )
        if cfg.invalidate_first and cfg.add_valid_layer:
            self._call_stage("invalidate", self._build_validity_message(value=0.0))
            self._call_stage("update-while-invalid", self._build_data_message(valid_value=None))
            self._call_stage("revalidate", self._build_validity_message(value=1.0))
        else:
            valid_value = 1.0 if cfg.add_valid_layer else None
            self._call_stage("update", self._build_data_message(valid_value=valid_value))

    def _call_stage(self, label: str, grid_map: GridMap):
        self.get_logger().info(f"Issuing masked_replace stage '{label}'.")
        req = SetGridMap.Request()
        req.map = grid_map
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("masked_replace request completed.")
        else:
            self.get_logger().error(f"masked_replace call failed: {future.exception()}")

    def _base_grid_map(self) -> GridMap:
        cfg = self._config
        gm = GridMap()
        gm.header.frame_id = cfg.frame_id
        gm.header.stamp = self.get_clock().now().to_msg()
        gm.info.resolution = cfg.resolution
        gm.info.length_x = cfg.actual_length_x
        gm.info.length_y = cfg.actual_length_y
        gm.info.pose.position.x = cfg.center_x
        gm.info.pose.position.y = cfg.center_y
        gm.info.pose.position.z = cfg.center_z
        gm.info.pose.orientation.w = 1.0
        gm.basic_layers = ["elevation"]
        return gm

    def _mask_array(self, force_value: Optional[float] = None) -> np.ndarray:
        cfg = self._config
        rows = cfg.shape["rows"]
        cols = cfg.shape["cols"]
        mask_value = cfg.mask_value if force_value is None else force_value
        if np.isnan(mask_value):
            mask_value = 1.0
        return np.full((rows, cols), mask_value, dtype=np.float32)

    def _build_validity_message(self, value: float) -> GridMap:
        gm = self._base_grid_map()
        mask = self._mask_array()
        rows, cols = mask.shape
        gm.layers = [self._config.mask_layer, "is_valid"]
        arrays = {
            self._config.mask_layer: mask,
            "is_valid": np.full((rows, cols), value, dtype=np.float32),
        }
        for layer in gm.layers:
            gm.data.append(self._numpy_to_multiarray(arrays[layer]))
        return gm

    def _build_data_message(self, valid_value: Optional[float]) -> GridMap:
        gm = self._base_grid_map()
        mask = self._mask_array()
        rows, cols = mask.shape
        gm.layers = [self._config.mask_layer, "elevation", "variance"]
        arrays = {
            self._config.mask_layer: mask,
            "elevation": np.full((rows, cols), self._config.elevation, dtype=np.float32),
            "variance": np.full((rows, cols), self._config.variance, dtype=np.float32),
        }
        if valid_value is not None:
            gm.layers.append("is_valid")
            arrays["is_valid"] = np.full((rows, cols), valid_value, dtype=np.float32)
        for layer in gm.layers:
            gm.data.append(self._numpy_to_multiarray(arrays[layer]))
        return gm

    @staticmethod
    def _numpy_to_multiarray(array: np.ndarray) -> Float32MultiArray:
        msg = Float32MultiArray()
        layout = MultiArrayLayout()
        rows, cols = array.shape
        layout.dim.append(MultiArrayDimension(label="column_index", size=cols, stride=rows * cols))
        layout.dim.append(MultiArrayDimension(label="row_index", size=rows, stride=rows))
        msg.layout = layout
        msg.data = array.flatten(order="F").tolist()
        return msg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = PatchConfig(
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        length_x=args.size_x,
        length_y=args.size_y,
        resolution=args.resolution,
        frame_id=args.frame,
        mask_layer=args.mask_layer,
        elevation=args.elevation,
        variance=args.variance,
        mask_value=args.mask_value,
        add_valid_layer=args.valid_layer,
        invalidate_first=args.invalidate_first,
    )

    rclpy.init()
    node = MaskedReplaceClient(args.service, cfg)
    node.send_request()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
