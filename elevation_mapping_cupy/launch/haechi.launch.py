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
`camera_color_optical_frame`. Intrinsics from the same file ride in as
samtp_node parameters below; re-running the calibration means updating both.
SAM-TP replaced the class segmentation outright -- the classifier's verdict
flipped run to run in simulation, while the traversability score held.
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
    # Navigation terrain chain (slope/step/roughness/drivability), not the
    # digging chain the core config would load by default.
    plugin_config_path = os.path.join(share_dir, "config", "setups", "haechi", "plugin_config.yaml")
    for path in (core_param_path, robot_param_path, plugin_config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} does not exist")

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

    # SAM-TP replaces the class segmentation. The camera publishes no
    # CameraInfo, so the calibration file's intrinsics ride in as parameters,
    # and the 19.554 ms camera-to-reference clock offset is applied to the
    # republished stamps the same way the old node applied it.
    samtp_node = Node(
        package="elevation_mapping_cupy",
        executable="samtp_node.py",
        namespace="front_cam",
        name="samtp_node",
        output="screen",
        condition=IfCondition(use_semantics),
        parameters=[
            {
                "image_topic": "/camera/image_raw/compressed",
                "engine_path": LaunchConfiguration("samtp_engine"),
                "camera_size": [1280, 720],
                "camera_k": [632.05027422, 0.0, 626.09047259,
                             0.0, 633.89951429, 343.05708742,
                             0.0, 0.0, 1.0],
                "time_offset_s": 0.019554,
                "output_scale": 0.5,
                "max_rate": 4.0,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            core_param_path,
            robot_param_path,
            {"use_sim_time": use_sim_time, "plugin_config_file": plugin_config_path},
        ],
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
            DeclareLaunchArgument(
                "samtp_engine",
                default_value=os.path.expanduser("~/samtp/samtp_512_fp16.engine"),
                description="TensorRT engine (machine-specific, not in the repo).",
            ),
            camera_tf,
            samtp_node,
            elevation_mapping_node,
        ]
    )
