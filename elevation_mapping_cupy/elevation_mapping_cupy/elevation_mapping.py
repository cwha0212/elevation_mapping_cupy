#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import math
import os
import threading
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union, Optional

import numpy as np

from elevation_mapping_cupy.traversability_filter import get_filter_torch
from elevation_mapping_cupy.parameter import Parameter

from elevation_mapping_cupy.kernels import (
    add_points_kernel,
)

from elevation_mapping_cupy.kernels import error_counting_kernel
from elevation_mapping_cupy.kernels import finalize_map_kernel
from elevation_mapping_cupy.kernels import dilation_filter_kernel
from elevation_mapping_cupy.kernels import normal_filter_kernel
from elevation_mapping_cupy.kernels import image_to_map_correspondence_kernel

from elevation_mapping_cupy.plugins.plugin_manager import PluginManager
from elevation_mapping_cupy.semantic_map import SemanticMap

import cupy as cp

xp = cp
pool = cp.get_default_memory_pool()
cp.cuda.set_allocator(pool.malloc)




class ElevationMap:
    """Core elevation mapping class."""

    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param
        self.data_type = self.param.data_type
        self.resolution = param.resolution
        self.center = xp.array([0, 0, 0], dtype=self.data_type)
        self.base_rotation = xp.eye(3, dtype=self.data_type)
        self.map_length = param.map_length
        self.cell_n = param.cell_n

        self.map_lock = threading.Lock()
        self.elevation_map = xp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.layer_names = [
            "elevation",
            "variance",
            "is_valid",
            "traversability",
            "time",
            "upper_bound",
            "is_upper_bound",
        ]

        # buffers
        self.traversability_buffer = xp.full((self.cell_n, self.cell_n), xp.nan, dtype=self.data_type)
        # One device buffer the publish path rotates each layer into, so the
        # host copy is a single contiguous transfer (see copy_layer_rot180).
        self._publish_buf = cp.empty((self.cell_n - 2, self.cell_n - 2), dtype=cp.float32)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n), dtype=self.data_type)
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        # overlap clearance
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        cell_range = np.clip(cell_range, 0, self.cell_n)
        self.cell_min = self.cell_n // 2 - cell_range // 2
        self.cell_max = self.cell_n // 2 + cell_range // 2

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        self.compile_kernels()
        self.compile_image_kernels()

        # No shell substitutions in research code: param.weight_file is expected to be a real path.
        param.load_weights(param.weight_file)

        self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)

        # Semantic layers fed by image and pointcloud channels.
        self.semantic_map = SemanticMap(param)
        self.semantic_map.initialize_fusion()
        # Semantic layers are otherwise created by the first message carrying
        # the channel, so a publisher firing before any camera frame asks for a
        # layer that does not exist yet. Declare the configured ones up front:
        # they read NaN until something fills them, which is the truth anyway.
        for config in param.subscriber_cfg.values():
            for channel in config.get("channels", []) or []:
                self.semantic_map.add_layer(channel)

        # Plugins
        self.plugin_manager = PluginManager(cell_n=self.cell_n, resolution=self.resolution)
        self.plugin_manager.load_plugin_settings(param.plugin_config_file)


    def clear(self):
        """Reset all the layers of the elevation & the semantic map."""
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
            self.semantic_map.clear()
            self.plugin_manager.reset_layers()

        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_center_position(self, position):
        """Return the position of the map center.

        Args:
            position (numpy.ndarray):

        """
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        """Shift the map along all three axes according to the input.

        Args:
            delta_position (numpy.ndarray):
        """
        # Shift map using delta position.
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position, R):
        """Shift the map to an absolute position and update the rotation of the robot.

        Args:
            position (numpy.ndarray):
            R (cupy._core.core.ndarray):
        """
        # Shift map to the center of robot.
        self.base_rotation = xp.asarray(R, dtype=self.data_type)
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y-axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, delta_pixel):
        """Shift the map along the horizontal axes according to the input.

        Args:
            delta_pixel (cupy._core.core.ndarray): Shift in [x, y] order (world coordinates).
                x corresponds to columns (axis 2), y corresponds to rows (axis 1).

        Note:
            The map array has shape (layers, height, width) = (layers, rows, cols).
            In row-major convention: axis 1 = rows = Y, axis 2 = cols = X.
            cp.roll with axis=(1, 2) expects [row_shift, col_shift] = [y_shift, x_shift].
            Since delta_pixel is [x, y], we swap to [y, x] for correct axis mapping.
        """
        # Swap [x, y] to [y, x] to match axis=(1, 2) = (rows=Y, cols=X)
        shift_value = cp.array([delta_pixel[1], delta_pixel[0]], dtype=cp.int32)
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)
            self.semantic_map.shift_map_xy(shift_value)
            # Plugin layers are computed on-demand; invalidate cache when shifting.
            self.plugin_manager.reset_layers()

    def shift_map_z(self, delta_z):
        """Shift the relevant layers along the vertical axis.

        Args:
            delta_z (cupy._core.core.ndarray):
        """
        with self.map_lock:
            # elevation
            self.elevation_map[0] += delta_z
            # upper bound
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        """Compile all kernels belonging to the elevation map."""

        self.new_map = cp.zeros(
            (self.elevation_map.shape[0], self.cell_n, self.cell_n),
            dtype=self.data_type,
        )
        self.error = cp.zeros(1, dtype=cp.float32)
        self.error_cnt = cp.zeros(1, dtype=cp.float32)
        self.map_snapshot = cp.empty_like(self.elevation_map)
        self.visibility_map = cp.empty(
            (3, self.cell_n, self.cell_n),
            dtype=self.data_type,
        )
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)

        self.add_points_kernel = add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
        )
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.drift_compensation_variance_inlier,
            self.param.traversability_inlier,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
        )
        self.finalize_map_kernel = finalize_map_kernel(
            self.cell_n,
            self.cell_n,
            self.param.max_variance,
            self.initial_variance,
            self.param.outlier_variance,
        )

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

    def compile_image_kernels(self):
        """Allocate the correspondence buffers and compile the projection kernel.

        Only when a subscriber actually feeds images: the buffers are cell_n^2
        each and the kernel costs a JIT compile at startup.
        """
        for config in self.param.subscriber_cfg.values():
            if config.get("data_type") == "image":
                self.valid_correspondence = cp.zeros((self.cell_n, self.cell_n), dtype=cp.bool_)
                self.uv_correspondence = cp.zeros(
                    (2, self.cell_n, self.cell_n), dtype=cp.float32
                )
                self.image_to_map_correspondence_kernel = image_to_map_correspondence_kernel(
                    resolution=self.resolution,
                    width=self.cell_n,
                    height=self.cell_n,
                    tolerance_z_collision=0.10,
                )
                break

    def shift_translation_to_map_center(self, t):
        """Deduct the map center to get the translation of a point w.r.t. the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        """
        t -= self.center

    def update_map_with_kernel(self, points_all, channels, R, t, position_noise, orientation_noise):
        """Update map with new measurement.

        Args:
            points_all (cupy._core.core.ndarray):
            channels (List[str]):
            R (cupy._core.core.ndarray):
            t (cupy._core.core.ndarray):
            position_noise (float):
            orientation_noise (float):
        """
        points = cp.ascontiguousarray(points_all[:, :3])

        with self.map_lock:
            self.new_map.fill(0.0)
            self.error.fill(0.0)
            self.error_cnt.fill(0.0)
            self.map_snapshot[...] = self.elevation_map
            self.visibility_map.fill(0.0)
            self.visibility_map[2].fill(cp.inf)
            t = t.copy()
            t[2] -= self.center[2]
            self.error_counting_kernel(
                self.map_snapshot,
                points,
                self.center[:1],
                self.center[1:2],
                R,
                t,
                self.new_map,
                self.error,
                self.error_cnt,
                size=(points.shape[0]),
            )
            if (
                self.param.enable_drift_compensation
                and self.error_cnt > self.param.min_height_drift_cnt
                and (
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
                )
            ):
                self.mean_error = self.error / self.error_cnt
                self.additive_mean_error += self.mean_error
                if np.abs(self.mean_error) < self.param.max_drift:
                    correction = self.mean_error * self.param.drift_compensation_alpha
                    self.map_snapshot[0] += correction
            self.add_points_kernel(
                self.center[:1],
                self.center[1:2],
                R,
                t,
                self.map_snapshot,
                self.normal_map,
                points,
                self.new_map,
                self.visibility_map,
                size=(points.shape[0]),
            )

            self.finalize_map_kernel(
                self.map_snapshot,
                self.new_map,
                self.visibility_map,
                self.elevation_map,
                size=(self.cell_n * self.cell_n),
            )

            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)

            self.traversability_input *= 0.0
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],
                self.traversability_input,
                self.traversability_mask_dummy,
                size=(self.cell_n * self.cell_n),
            )

            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )
            self.plugin_manager.reset_layers()

        self.update_normal(self.traversability_input)

    def clear_overlap_map(self, t):
        """Clear overlapping areas around the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        """

        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z
        near_map = self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max]
        valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
        near_map[1] = cp.where(valid_idx, near_map[1], self.initial_variance)
        near_map[2] = cp.where(valid_idx, near_map[2], 0.0)
        valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
        near_map[6] = cp.where(valid_idx, near_map[6], 0.0)
        self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max] = near_map

    def update_variance(self):
        """Adds the time variacne to the valid cells."""
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    def update_time(self):
        """adds the time interval to the time layer."""
        self.elevation_map[4] += self.param.time_interval

    def input_pointcloud(
        self,
        raw_points: cp._core.core.ndarray,
        channels: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        position_noise: float,
        orientation_noise: float,
    ):
        """Input the point cloud and fuse the new measurements to update the elevation map.

        Args:
            raw_points (cupy._core.core.ndarray):
            channels (List[str]):
            R  (cupy._core.core.ndarray):
            t (cupy._core.core.ndarray):
            position_noise (float):
            orientation_noise (float):

        Returns:
            None:
        """
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        additional_channels = channels[3:]
        self.update_map_with_kernel(
            raw_points,
            additional_channels,
            cp.asarray(R, dtype=self.data_type),
            cp.asarray(t, dtype=self.data_type),
            position_noise,
            orientation_noise,
        )

    def input_image(
        self,
        image: List[cp._core.core.ndarray],
        channels: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        K: cp._core.core.ndarray,
        D: cp._core.core.ndarray,
        distortion_model: str,
        image_height: int,
        image_width: int,
    ):
        """Project image channels into the map using the camera calibration and pose."""

        image = np.stack(image, axis=0)
        if len(image.shape) == 2:
            image = image[None]

        # Build the projection on the host. These are 3x3 and 3x4 matrices, and
        # running them through cupy pulls in cuBLAS just to create its handle --
        # which fails with CUBLAS_STATUS_ALLOC_FAILED on Jetson once the
        # simulator and RViz already hold CUDA contexts, even with most of the
        # memory free. numpy is also simply faster at this size.
        K_h = np.asarray(K, dtype=np.float32).reshape(3, 3)
        R_h = np.asarray(R, dtype=np.float32).reshape(3, 3)
        t_h = np.asarray(t, dtype=np.float32).reshape(3)
        P_h = K_h @ np.concatenate([R_h, t_h[:, None]], axis=1)
        t_cam_map = -R_h.T @ t_h - cp.asnumpy(self.center)

        image = cp.asarray(image, dtype=self.data_type)
        K = cp.asarray(K, dtype=self.data_type)
        D = cp.asarray(D, dtype=self.data_type)
        image_height = cp.float32(image_height)
        image_width = cp.float32(image_width)

        if len(D) < 4:
            D = cp.zeros(5, dtype=self.data_type)
        elif len(D) == 4:
            D = cp.concatenate([D, cp.zeros(1, dtype=self.data_type)])
        else:
            D = D[:5]

        if distortion_model == "radtan":
            pass
        elif distortion_model in {"equidistant", "plumb_bob"}:
            D *= 0
        else:
            D *= 0

        P = cp.asarray(P_h, dtype=np.float32)
        # Camera cell for the kernel's occlusion walk, in the same Row=Y, Col=X
        # order the kernel indexes with: the row follows world Y, the column X.
        x1 = cp.uint32((self.cell_n / 2) + (t_cam_map[1] / self.resolution))
        y1 = cp.uint32((self.cell_n / 2) + (t_cam_map[0] / self.resolution))
        z1 = cp.float32(t_cam_map[2])

        self.uv_correspondence *= 0
        self.valid_correspondence[:, :] = False

        with self.map_lock:
            self.image_to_map_correspondence_kernel(
                self.elevation_map,
                x1,
                y1,
                z1,
                P.reshape(-1),
                K.reshape(-1),
                D.reshape(-1),
                image_height,
                image_width,
                self.center,
                self.uv_correspondence,
                self.valid_correspondence,
                size=int(self.cell_n * self.cell_n),
            )
            self.semantic_map.update_layers_image(
                image,
                channels,
                self.uv_correspondence,
                self.valid_correspondence,
                image_height,
                image_width,
            )

    def update_normal(self, dilated_map):
        """Clear the normal map and then apply the normal kernel with dilated map as input.

        Args:
            dilated_map (cupy._core.core.ndarray):
        """
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(
                dilated_map,
                self.elevation_map[2],
                self.normal_map,
                size=(self.cell_n * self.cell_n),
            )

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False, xp=cp):
        """Process the input_map according to the fill_nan and add_z flags.

        Args:
            input_map (cupy._core.core.ndarray):
            fill_nan (bool):
            add_z (bool):
            xp (module):

        Returns:
            cupy._core.core.ndarray:
        """
        # Nothing here writes into the input, so no copy: fill_nan and add_z
        # both allocate their own result, and the plain case is a view.
        m = input_map
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]

    def get_elevation(self):
        """Get the elevation layer.

        Returns:
            elevation layer

        """
        return self.process_map_for_publish(self.elevation_map[0], fill_nan=True, add_z=True)

    def get_variance(self):
        """Get the variance layer.

        Returns:
            variance layer
        """
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        """Get the traversability layer.

        Returns:
            traversability layer
        """
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5,
            self.elevation_map[3],
            cp.nan,
        )
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        """Get the time layer.

        Returns:
            time layer
        """
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        """Get the upper bound layer.

        Returns:
            upper_bound: upper bound layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5),
                self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        """Get the is upper bound layer.

        Returns:
            is_upper_bound: layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5),
                self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def trim_memory_pool(self):
        """Release cached CuPy allocator blocks that are not currently in use."""
        pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def exists_layer(self, name):
        """Check if the layer exists in elevation map or in the semantic map.

        Args:
            name (str): Layer name

        Returns:
            bool: Indicates if layer exists.
        """
        if name in self.layer_names:
            return True
        elif name in self.semantic_map.layer_names:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    def _layer_for_publish(self, name):
        """Resolve a layer to a (cell_n-2, cell_n-2) device view or array.

        Called with map_lock held. Plugin layers are computed on demand
        through the manager's generation cache.
        """
        if name == "elevation":
            return self.get_elevation()
        if name == "variance":
            return self.get_variance()
        if name == "is_valid":
            return self.elevation_map[2, 1:-1, 1:-1]
        if name == "traversability":
            return self.get_traversability()
        if name == "time":
            return self.get_time()
        if name == "upper_bound":
            return self.get_upper_bound()
        if name == "is_upper_bound":
            return self.get_is_upper_bound()
        if name == "normal_x":
            return self.normal_map[0, 1:-1, 1:-1]
        if name == "normal_y":
            return self.normal_map[1, 1:-1, 1:-1]
        if name == "normal_z":
            return self.normal_map[2, 1:-1, 1:-1]
        if name in self.semantic_map.layer_names:
            return self.semantic_map.get_map_with_name(name)
        if name in self.plugin_manager.layer_names:
            self.plugin_manager.update_with_name(
                name,
                self.elevation_map,
                self.layer_names,
                semantic_map=self.semantic_map.semantic_map,
                semantic_params=self.semantic_map.layer_names,
                rotation=self.base_rotation,
                elements_to_shift=self.semantic_map.elements_to_shift,
            )
            m = self.plugin_manager.get_map_with_name(name)
            p = self.plugin_manager.get_param_with_name(name)
            return self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer, xp=cp)
        raise KeyError(f"Layer '{name}' is not in the map.")

    def copy_layer_rot180(self, name, host_out):
        """Copy a layer to the host as the bytes a GridMap column layout wants.

        The wire format (see gridmap_utils.encode_rot180_as_gridmap_column) is
        the C order of the layer rotated by 180 degrees: grid_map's buffer runs
        Row -> -X, Col -> -Y while this map keeps Row = Y, Col = X, and the
        transpose that converts between them cancels against the column-major
        packing of the message. One strided read into a contiguous device
        buffer, then one synchronous transfer into host_out (pinned or not).

        Args:
            name (str): layer name.
            host_out (numpy.ndarray): (cell_n-2, cell_n-2) float32, C order.
        """
        with self.map_lock:
            m = self._layer_for_publish(name)
            cp.copyto(self._publish_buf, m[::-1, ::-1])
            self._publish_buf.get(out=host_out)

    def get_map_with_name_ref(self, name, data):
        """Load a layer in grid_map buffer order (rows -> -X, cols -> -Y) into data.

        Kept for tests and external callers; the node publishes through
        copy_layer_rot180, which produces the same bytes without the transpose.
        """
        tmp = np.empty_like(data)
        self.copy_layer_rot180(name, tmp)
        data[...] = tmp.T

    def _transform_to_grid_map_coordinate_convention(self, m):
        """Transform the map to the grid_map coordinate convention.

        elevation_mapping_cupy uses Row=Y, Col=X (see kernels/custom_kernels.py:35)
        grid_map uses Row→-X, Col→-Y (see grid_map_core/src/GridMapMath.cpp:64-67
        transformBufferOrderToMapFrame returns {-index[0], -index[1]})
        Required transformation:
           1. Transpose: swap axes so Row=X, Col=Y (matching grid_map's axis assignment)
           2. Flip axis 0: so increasing row → decreasing X (matching grid_map's -X)
           3. Flip axis 1: so increasing col → decreasing Y (matching grid_map's -Y)

        This is equivalent to: rot90(m.T, k=2) or flip(flip(m.T, 0), 1)

        Args:
            m (cupy._core.core.ndarray):

        Returns:
            cupy._core.core.ndarray:
        """
        m = m.T
        m = xp.flip(m, 0)
        m = xp.flip(m, 1)
        return m
