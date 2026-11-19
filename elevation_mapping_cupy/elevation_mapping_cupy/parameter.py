#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Parameter:
    """Parameters of the elevation map. The node fills them from ROS parameters
    (config/core/core_param.yaml + config/setups/haechi/haechi.yaml); the values
    here are only the fallbacks for keys the node does not set."""
    resolution: float = 0.04  # resolution in m.
    subscriber_cfg: dict = field(
        default_factory=lambda: {
            "front_cam": {
                "channels": ["rgb", "person"],
                "topic_name": "/elevation_mapping/pointcloud_semantic",
                "data_type": "pointcloud",
            }
        }
    )  # configuration for the subscriber
    fusion_algorithms: list = field(
        default_factory=lambda: ["image_exponential"]
    )  # image fusion algorithms (SAM-TP score -> exponential); the map takes xyz-only pointclouds
    image_channel_fusions: dict = field(default_factory=lambda: {"default": "exponential"})  # fusion for image channels
    data_type: str = np.float32  # data type for the map

    map_length: float = 8.0  # map's size in m.
    sensor_noise_factor: float = 0.05  # point's noise is sensor_noise_factor*z^2 (z is distance from sensor).
    mahalanobis_thresh: float = 2.0  # points outside this distance is outlier.
    outlier_variance: float = 0.01  # if point is outlier, add this value to the cell.
    drift_compensation_variance_inlier: float = 0.1  # cells under this value is used for drift compensation.
    time_variance: float = 0.01  # add this value when update_variance is called.
    time_interval: float = 0.1  # Time layer is updated with this interval.

    max_variance: float = 1.0  # maximum variance for each cell.
    dilation_size: int = 2  # dilation filter size before the normal filter.
    drift_compensation_alpha: float = 1.0  # drift compensation alpha for smoother update of drift compensation.

    traversability_inlier: float = 0.1  # cells with higher traversability are used for drift compensation.
    wall_num_thresh: int = 100  # if there are more points than this value, only higher points than the current height are used to make the wall more sharp.
    min_height_drift_cnt: int = 100  # drift compensation only happens if the valid cells are more than this number.

    max_ray_length: float = 2.0  # maximum length for ray tracing.
    cleanup_step: float = 0.01  # substitute this value from validity layer at visibility cleanup.
    cleanup_cos_thresh: float = 0.5  # substitute this value from validity layer at visibility cleanup.
    min_valid_distance: float = 0.3  # points with shorter distance will be filtered out.
    max_height_range: float = 1.0  # points higher than this value from sensor will be filtered out to disable ceiling.
    ramped_height_range_a: float = 0.3  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_b: float = 1.0  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_c: float = 0.2  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.


    max_drift: float = 0.10  # maximum drift for the compensation

    overlap_clear_range_xy: float = 4.0  # xy range [m] for clearing overlapped area. this defines the valid area for overlap clearance. (used for multi floor setting)
    overlap_clear_range_z: float = 2.0  # z range [m] for clearing overlapped area. cells outside this range will be cleared. (used for multi floor setting)

    enable_edge_sharpen: bool = True  # enable edge sharpening
    enable_drift_compensation: bool = True  # enable drift compensation
    enable_visibility_cleanup: bool = True  # enable visibility cleanup
    enable_overlap_clearance: bool = True  # enable overlap clearance
    use_only_above_for_upper_bound: bool = True  # use only above for upper bound
    position_noise_thresh: float = 0.1  # if the position change is bigger than this value, the drift compensation happens.
    orientation_noise_thresh: float = 0.1  # if the orientation change is bigger than this value, the drift compensation happens.

    plugin_config_file: str = ""  # plugin chain yaml (set by the node from the launch)

    initial_variance: float = 10.0  # initial variance for each cell.

    # # not configurable params
    true_map_length: float = None  # true length of the map
    cell_n: int = None  # number of cells in the map
    true_cell_n: int = None  # true number of cells in the map

    def update(self):
        """
        Update the parameters related to the map size and resolution.
        """
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.true_cell_n = round(self.map_length / self.resolution)
        self.true_map_length = self.true_cell_n * self.resolution

