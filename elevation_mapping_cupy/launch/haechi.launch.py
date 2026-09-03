"""Elevation mapping for haechi: 3 merged LiDARs plus a front RGB camera.

    ros2 launch elevation_mapping_cupy haechi.launch.py

Geometry comes from /points/merged_deskewed (frame `lidar_frame`), semantics
from the front camera. Bring the geometry up on its own first --
`use_semantics:=false` -- because the camera labels are projected onto the
elevation surface, so a wrong surface moves the labels with it.

Camera extrinsic below is copied from ~/haechi_data/calib/haechi_calibration.yaml
(2026-09-03), section `camera.extrinsic`, which is T_lidarframe_camera. Its
rotation maps camera x/y/z onto lidar_frame -Y / -Z / +X, i.e. right/down/front:
the optical convention, so it publishes directly against the camera's own
`camera_color_optical_frame`. Intrinsics from the same file live in
semantic_sensor/config/haechi.yaml; re-running the calibration means updating
both.
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# T_lidarframe_camera. Degrees here, radians at the call site -- the same
# convention navi_lidar's robot profiles use.
CAMERA_TRANSLATION = (0.683797, -0.063316, -0.023919)
CAMERA_RPY_DEG = (-86.6693, -3.3177, -87.4774)
CAMERA_PARENT_FRAME = "lidar_frame"
CAMERA_CHILD_FRAME = "camera_color_optical_frame"


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    core_param_path = os.path.join(share_dir, "config", "core", "core_param.yaml")
    robot_param_path = os.path.join(share_dir, "config", "setups", "haechi", "haechi.yaml")
    for path in (core_param_path, robot_param_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} does not exist")

    semantic_config_path = os.path.join(
        get_package_share_directory("semantic_sensor"), "config", "haechi.yaml"
    )

    use_semantics = LaunchConfiguration("use_semantics")
    use_sim_time = LaunchConfiguration("use_sim_time")

    x, y, z = CAMERA_TRANSLATION
    roll, pitch, yaw = (math.radians(v) for v in CAMERA_RPY_DEG)

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_frame_to_camera",
        output="screen",
        condition=IfCondition(use_semantics),
        arguments=[
            "--x", str(x),
            "--y", str(y),
            "--z", str(z),
            "--roll", str(roll),
            "--pitch", str(pitch),
            "--yaw", str(yaw),
            "--frame-id", CAMERA_PARENT_FRAME,
            "--child-frame-id", CAMERA_CHILD_FRAME,
        ],
    )

    # Namespaced so the republished camera_info lands on
    # /front_cam/camera_info_resized, which the elevation mapping config expects.
    semantic_node = Node(
        package="semantic_sensor",
        executable="image_node",
        namespace="front_cam",
        name="semantic_image_node",
        output="screen",
        condition=IfCondition(use_semantics),
        parameters=[
            {
                "sensor_name": "haechi_front_cam",
                "config_path": semantic_config_path,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[core_param_path, robot_param_path, {"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_semantics",
                default_value="true",
                description="Run the front camera semantic branch. Set false to bring "
                "up LiDAR-only geometry first.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            camera_tf,
            semantic_node,
            elevation_mapping_node,
        ]
    )
