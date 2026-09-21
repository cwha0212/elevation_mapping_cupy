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

# haechi_data/calib/haechi_calibration.yaml, `camera:` -- the robot publishes
# no CameraInfo, so these travel with the launch instead. Row-major K, the
# resolution it was calibrated at, and t_reference = t_camera + offset.
CAMERA_K = [632.05027422, 0.0, 626.09047259,
            0.0, 633.89951429, 343.05708742,
            0.0, 0.0, 1.0]
CAMERA_SIZE = [1280, 720]
CAMERA_TIME_OFFSET_S = 0.019554


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

    # SAM-TP, the same model the simulated path runs. This branch used to be
    # semantic_sensor's Cityscapes classifier and was left behind when SAM-TP
    # replaced it: the real robot has been running with no camera verdict at
    # all, which is why `safety` here has been nothing but the geometry.
    #
    # haechi publishes no CameraInfo, so the calibration is injected -- K and
    # size straight from haechi_calibration.yaml, and the camera-to-reference
    # time offset with them.
    semantic_node = Node(
        package="elevation_mapping_cupy",
        executable="samtp_node.py",
        namespace="front_cam",
        name="samtp_node",
        output="screen",
        condition=IfCondition(use_semantics),
        parameters=[
            {
                "engine_path": LaunchConfiguration("samtp_engine"),
                # Straight off the camera's own topic. SAM-TP decodes JPEG
                # itself, so the decode hop that used to sit here is gone --
                # and with it a QoS mismatch that silently starved the whole
                # branch: image_transport's republish subscribes reliably,
                # the camera publishes best effort, and nothing crossed.
                "image_topic": "/camera/image_raw/compressed",
                "camera_k": CAMERA_K,
                "camera_size": CAMERA_SIZE,
                "time_offset_s": CAMERA_TIME_OFFSET_S,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # The same thinning the simulated robot has had all along, which the real
    # one was missing: the merged cloud went into the mapper raw, at full
    # density and full range, and the terrain chain measured noise instead of
    # ground past about 3 m -- slope median 33 degrees at 3-5 m against 7
    # degrees underfoot, 80% of those cells called undrivable. Range is the
    # cure: a grazing return at 8 m lands a whole smear of cells on one
    # reading, and the map is 10 m wide anyway.
    #
    # The self box is stated in `lidar_frame` (origin = left MID360, axes
    # robot FLU): the body reaches forward to the nose sensor at x 0.69 and
    # sits mostly to -y of the left mount. Its floor stops 0.2 m above the
    # measured ground (-0.70 m here) so the filter eats the chassis and not
    # the ground under it. Provisional until the body is measured -- the
    # robot profile's own sensor_height is still a TODO copied from guard.
    downsample = Node(
        package="elevation_mapping_cupy",
        executable="voxel_downsample_node.py",
        name="merged_downsample",
        output="screen",
        parameters=[{
            "input_topic": "/points/merged_deskewed",
            "output_topic": "/points/merged_deskewed_ds",
            "voxel_size": 0.05,
            "max_range": 8.0,
            "use_sim_time": use_sim_time,
            "self_filter_min": [-1.00, -0.60, -0.50],
            "self_filter_max": [0.85, 0.20, 0.30],
        }],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            core_param_path,
            robot_param_path,
            {"use_sim_time": use_sim_time, "plugin_config_file": plugin_config_path,
             # Last wins, so this overrides the config. The map is allocated
             # once at startup -- setting map_length on a running node does
             # nothing at all, quietly, which cost one experiment already.
             "map_length": LaunchConfiguration("map_length")},
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
                "map_length",
                default_value="10.0",
                description="Side of the square elevation map, metres. Bigger "
                "reaches further and costs cells: 20 m at 0.05 is four times "
                "the map.",
            ),
            DeclareLaunchArgument(
                "samtp_engine",
                default_value=os.path.expanduser("~/samtp/samtp_512_fp16.engine"),
                description="TensorRT engine for SAM-TP. Machine specific, so "
                "it lives outside the repo and is rebuilt per device with "
                "trtexec --onnx=... --fp16.",
            ),
            camera_tf,
            semantic_node,
            downsample,
            elevation_mapping_node,
        ]
    )
