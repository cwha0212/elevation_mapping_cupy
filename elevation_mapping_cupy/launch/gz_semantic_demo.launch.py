"""Semantic projection demo on Gazebo Fortress.

    ros2 launch elevation_mapping_cupy gz_semantic_demo.launch.py
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Runs the full camera path end to end for the first time: Gazebo camera ->
semantic_sensor segmentation -> elevation mapping projection -> class layers
on the map. That is haechi's path; the only piece swapped out is where the
intrinsics come from, since Gazebo publishes a CameraInfo and haechi does not.

The world puts people and a car at known positions, because the model shipped
with the repo speaks COCO_WITH_VOC and knows those. Roadway and sidewalk, the
classes this is ultimately for, are not in its vocabulary -- swapping in a
Cityscapes model changes the channel lists and nothing else here.

Geometry stays with the lidar. The camera never contributes a surface; it
contributes labels that land on the surface the lidar built.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

LIDAR_FRAME = "robot/base_link/lidar"
LIDAR_XYZ = (0.0, 0.0, 0.25)

# Optical convention, as the projection requires: x right, y down, z forward,
# composed with the camera's 5 degree downward tilt.
COLOR_FRAME = "robot/base_link/color"
COLOR_XYZ = (0.25, 0.0, 0.28)
COLOR_QUAT_XYZW = (-0.521341815, 0.521341815, -0.477705675, 0.477705675)

# L4T ships only Mesa in the glvnd vendor directory, so ogre2 renders nothing
# unless the loader is pointed at NVIDIA's own vendor file.
TEGRA_EGL_VENDOR = "/usr/lib/aarch64-linux-gnu/tegra-egl/nvidia.json"


def _render_env(headless: bool):
    """EGL vendor override, and only where it is needed.

    Headless, ogre2 goes through EGL, and L4T ships only Mesa in the glvnd
    vendor directory: it never reaches nvidia-drm and the sensors render
    nothing at all.

    With a display ogre2 uses GLX instead, and forcing the EGL vendor there
    wedges startup -- the server never loads the world and never publishes a
    clock, so every node comes up, subscribes, and waits forever on input that
    is not coming.
    """
    if headless and os.path.exists(TEGRA_EGL_VENDOR):
        return {"__EGL_VENDOR_LIBRARY_FILENAMES": TEGRA_EGL_VENDOR}
    return {}


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    core_param_path = os.path.join(share_dir, "config", "core", "core_param.yaml")
    setup_param_path = os.path.join(
        share_dir, "config", "setups", "semantic_demo", "semantic_demo.yaml"
    )
    world_path = os.path.join(share_dir, "gazebo", "worlds", "semantic_demo.sdf")
    # The navigation terrain chain, shared with the terrain demo rather than
    # copied: the limits have to stay identical for the layers to mean the
    # same thing in both.
    plugin_config_path = os.path.join(
        share_dir, "config", "setups", "gz_demo", "plugin_config.yaml"
    )
    for path in (core_param_path, setup_param_path, plugin_config_path, world_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    semantic_config_path = os.path.join(
        get_package_share_directory("semantic_sensor"), "config", "gz_demo.yaml"
    )

    gui = LaunchConfiguration("gui")
    launch_rviz = LaunchConfiguration("launch_rviz")

    gz_server = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-s", "-v", "2", world_path],
        output="screen", additional_env=_render_env(headless=True),
        condition=UnlessCondition(gui),
    )
    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "2", world_path],
        output="screen", additional_env=_render_env(headless=False),
        condition=IfCondition(gui),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/lidar/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/color_cam@sensor_msgs/msg/Image[ignition.msgs.Image",
            # A plain camera sensor puts its info on the bare /camera_info.
            "/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/model/robot/odom3d@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            "/model/robot/pose3d@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/model/robot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
        ],
        remappings=[
            ("/model/robot/pose3d", "/tf"),
            ("/model/robot/odom3d", "/odom"),
            ("/model/robot/cmd_vel", "/cmd_vel"),
            ("/color_cam", "/color_cam/image"),
            # image_node resolves its relative camera_info topic inside its
            # own namespace, and republishes the resized one beside it, which
            # is where the elevation mapping config looks. Land it there.
            ("/camera_info", "/front_cam/camera_info"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    lidar_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="base_link_to_lidar", output="screen",
        arguments=[
            "--x", str(LIDAR_XYZ[0]), "--y", str(LIDAR_XYZ[1]), "--z", str(LIDAR_XYZ[2]),
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "base_link", "--child-frame-id", LIDAR_FRAME,
        ],
        parameters=[{"use_sim_time": True}],
    )

    color_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="base_link_to_color", output="screen",
        arguments=[
            "--x", str(COLOR_XYZ[0]), "--y", str(COLOR_XYZ[1]), "--z", str(COLOR_XYZ[2]),
            "--qx", str(COLOR_QUAT_XYZW[0]), "--qy", str(COLOR_QUAT_XYZW[1]),
            "--qz", str(COLOR_QUAT_XYZW[2]), "--qw", str(COLOR_QUAT_XYZW[3]),
            "--frame-id", "base_link", "--child-frame-id", COLOR_FRAME,
        ],
        parameters=[{"use_sim_time": True}],
    )

    # Namespaced so its outputs land on /front_cam/..., which is what the
    # elevation mapping config subscribes to.
    semantic_node = Node(
        package="semantic_sensor", executable="image_node",
        namespace="front_cam", name="semantic_image_node", output="screen",
        parameters=[{
            "sensor_name": "gz_front_cam",
            "config_path": semantic_config_path,
            "use_sim_time": True,
        }],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy", executable="elevation_mapping_node.py",
        name="elevation_mapping_node", output="screen",
        parameters=[
            core_param_path,
            setup_param_path,
            {"use_sim_time": True, "plugin_config_file": plugin_config_path},
        ],
    )

    # RViz can show the segmentation too, but its Image displays live in dock
    # panels that a config without a saved window layout tends to bury. This
    # opens in its own window, so the segmentation is visible without hunting
    # for a panel.
    image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="segmentation_view",
        arguments=["/front_cam/semantic_image_debug"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("image_view")),
        parameters=[{"use_sim_time": True}],
    )

    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen", condition=IfCondition(launch_rviz),
        parameters=[{"use_sim_time": True}],
    )

    octomap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share_dir, "launch", "gz_octomap.launch.py")
        ),
        launch_arguments={"source": LaunchConfiguration("octomap_source")}.items(),
        condition=IfCondition(LaunchConfiguration("octomap")),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "octomap",
            default_value="true",
            description="Build the octomap 2D grid from drivability. Needs "
            "octomap_server2: source ~/dependencies/octomap_ws/install/setup.bash.",
        ),
        DeclareLaunchArgument(
            "octomap_source",
            default_value="traversability",
            description="traversability (drivability layer) or lidar (height band).",
        ),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument(
            "image_view",
            default_value="true",
            description="Open the segmentation output in its own window.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([share_dir, "rviz", "semantic_demo.rviz"]),
        ),
        gz_server, gz_gui, bridge, lidar_tf, color_tf,
        semantic_node, elevation_mapping_node, rviz_node, image_view, octomap,
    ])
