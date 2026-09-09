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
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

LIDAR_FRAME = "robot/base_link/lidar"
LIDAR_XYZ = (0.66, -0.07, 0.0)

# Straight from haechi_data/calib/haechi_calibration.yaml, which states every
# extrinsic as T_lidarframe_sensor. The camera's rotation there is already the
# optical convention the projection wants (x right, y down, z forward), and the
# world file reproduces the same camera-to-lidar offset, so what the sim
# projects and what the robot projects are the same geometry.
COLOR_FRAME = "robot/base_link/color"
COLOR_XYZ = (0.675, -0.063, 0.135)
COLOR_QUAT_XYZW = (-0.51018308, 0.45905178, -0.51700551, 0.51155795)

# L4T ships only Mesa in the glvnd vendor directory, so ogre2 renders nothing
# unless the loader is pointed at NVIDIA's own vendor file.
TEGRA_EGL_VENDOR = "/usr/lib/aarch64-linux-gnu/tegra-egl/nvidia.json"


def _world_with_textures(world_path: str, share_dir: str) -> str:
    """Write the world out with absolute texture paths.

    Gazebo resolves a mesh's textures relative to the mesh, but an albedo_map
    written straight into a world file is looked up against the resource path
    and the install layout does not put it where that search reaches -- the
    surfaces then load untextured, with one error line and no other sign.
    Absolute paths avoid the search entirely.
    """
    textures = os.path.join(share_dir, "gazebo", "materials", "textures")
    with open(world_path) as handle:
        body = handle.read().replace("@TEXTURES@", textures)
    out = os.path.join(tempfile.gettempdir(), "semantic_demo_resolved.sdf")
    with open(out, "w") as handle:
        handle.write(body)
    return out


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

    resolved_world = _world_with_textures(world_path, share_dir)
    gz_server = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-s", "-v", "2", resolved_world],
        output="screen", additional_env=_render_env(headless=True),
        condition=UnlessCondition(gui),
    )
    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "2", resolved_world],
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
            "/lidar_left/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/lidar_right/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
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

    lidar_left_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="base_link_to_lidar_left",
        arguments=[
            "--x", "-0.023", "--y", "0.034", "--z", "0.189",
            "--roll", "-0.015359", "--pitch", "-0.604931", "--yaw", "-1.551598",
            "--frame-id", "base_link",
            "--child-frame-id", "robot/base_link/lidar_left",
        ],
        parameters=[{"use_sim_time": True}],
    )

    lidar_right_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="base_link_to_lidar_right",
        arguments=[
            "--x", "-0.024", "--y", "-0.18", "--z", "0.187",
            "--roll", "0.000000", "--pitch", "-0.558505", "--yaw", "1.570796",
            "--frame-id", "base_link",
            "--child-frame-id", "robot/base_link/lidar_right",
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
        package="elevation_mapping_cupy", executable="samtp_node.py",
        namespace="front_cam", name="samtp_node", output="screen",
        parameters=[{
            "engine_path": LaunchConfiguration("samtp_engine"),
            "image_topic": "/color_cam/image",
            "camera_info_topic": "/front_cam/camera_info",
            "use_sim_time": True,
        }],
    )

    # The map is 0.05 m, so points finer than that are averaged into cells
    # they already share. Thinning here rather than in navi_lidar keeps SLAM's
    # own input at full density.
    downsample = Node(
        package="elevation_mapping_cupy",
        executable="voxel_downsample_node.py",
        name="lidar_downsample",
        output="screen",
        parameters=[{
            "input_topic": "/lidar/points",
            "output_topic": "/lidar/points_downsampled",
            "voxel_size": 0.05,
            "max_range": 8.0,
            "use_sim_time": True,
            "self_filter_min": [-1.40, -0.32, -0.46],
            "self_filter_max": [0.16, 0.46, 0.22],
        }],
    )

    # Same body box for both side sensors, stated in base coordinates:
    # they sit on the deck tilted 33 degrees down, so the chassis, wheels
    # and each other are all in view, and a sensor-frame box would need
    # hand-rotating per mount.
    lidar_left_downsample = Node(
        package="elevation_mapping_cupy",
        executable="voxel_downsample_node.py",
        name="lidar_left_downsample",
        output="screen",
        parameters=[{
            "input_topic": "/lidar_left/points",
            "output_topic": "/lidar_left/points_downsampled",
            "voxel_size": 0.05,
            "max_range": 8.0,
            "use_sim_time": True,
            "sensor_mount_pose": [-0.023, 0.034, 0.189, -0.015359, -0.604931, -1.551598],
            "self_filter_min": [-0.80, -0.42, -0.50],
            "self_filter_max": [0.80, 0.42, 0.30],
        }],
    )

    # Same body box for both side sensors, stated in base coordinates:
    # they sit on the deck tilted 33 degrees down, so the chassis, wheels
    # and each other are all in view, and a sensor-frame box would need
    # hand-rotating per mount.
    lidar_right_downsample = Node(
        package="elevation_mapping_cupy",
        executable="voxel_downsample_node.py",
        name="lidar_right_downsample",
        output="screen",
        parameters=[{
            "input_topic": "/lidar_right/points",
            "output_topic": "/lidar_right/points_downsampled",
            "voxel_size": 0.05,
            "max_range": 8.0,
            "use_sim_time": True,
            "sensor_mount_pose": [-0.024, -0.18, 0.187, 0.000000, -0.558505, 1.570796],
            "self_filter_min": [-0.80, -0.42, -0.50],
            "self_filter_max": [0.80, 0.42, 0.30],
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

    # The GridMap display resolves its topic when it is created and does
    # not come back to it, so an RViz that starts before the mapper shows
    # an empty world for the rest of the session. Everything else in the
    # config recovers on its own; these do not.
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
            "samtp_engine",
            default_value=os.path.expanduser("~/samtp/samtp_512_fp16.engine"),
            description="TensorRT engine for SAM-TP (machine specific, not in the repo).",
        ),
        DeclareLaunchArgument(
            "image_view",
            default_value="false",
            description="Open the segmentation output in its own window.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([share_dir, "rviz", "semantic_demo.rviz"]),
        ),
        gz_server, gz_gui, bridge, lidar_tf, color_tf,
        lidar_left_tf, lidar_right_tf,
        downsample, lidar_left_downsample, lidar_right_downsample, semantic_node, elevation_mapping_node, image_view, octomap,
        TimerAction(period=20.0, actions=[rviz_node]),
    ])
