"""Elevation mapping demo on Gazebo Fortress.

    ros2 launch elevation_mapping_cupy gz_fortress_demo.launch.py
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
        '{linear: {x: 0.3}, angular: {z: 0.1}}'

Replaces turtlesim_init.launch.py, which cannot run any more: it targets Gazebo
Classic, and Classic 11 went EOL, taking gazebo_ros_pkgs and turtlebot3_gazebo
out of the apt repositories with it. No TurtleBot3 world was ever ported to
Fortress either, so gazebo/worlds/elevation_demo.sdf defines its own robot and
terrain instead.

Requires gz-fortress (apt) and ros_gz_sim built from source -- ros_gz_sim has
no Humble binary, unlike ros_gz_bridge which does. Source that overlay before
launching.

RViz is where the map shows up; the Gazebo window just shows the world. Both
are on by default -- pass gui:=false to drop Gazebo's window, launch_rviz:=false
to drop RViz, or set both false to run fully headless over ssh.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# Sensor poses relative to base_link, copied from the world file. Fortress
# stamps sensor messages with the scoped name <model>/<link>/<sensor>, so those
# are the frames the transforms have to land on for the TF lookups to resolve.
#
# The lidar keeps the body convention its points come in: x forward, z up.
LIDAR_FRAME = "robot/base_link/lidar"
LIDAR_XYZ = (0.0, 0.0, 0.25)

# The colour camera sits apart from the depth sensor on purpose, mirroring how
# haechi's camera is mounted well ahead of the lidar origin.
COLOR_FRAME = "robot/base_link/color"
COLOR_XYZ = (0.25, 0.0, 0.28)
# The projection multiplies by K, so the frame it resolves has to be the
# optical one: x right, y down, z forward. A plain 20 degree pitch would be the
# body convention and puts everything in front of the robot behind the camera.
# This is the same convention haechi's calibration already uses. Quaternion
# rather than rpy because it is the composition of the tilt with body->optical.
COLOR_QUAT_XYZW = (-0.579234890, 0.579234890, -0.405569897, 0.405569897)

# On L4T the glvnd vendor directory ships only Mesa, so ogre2 loads the Mesa
# EGL, fails to reach nvidia-drm, and the camera renders nothing -- silently,
# since the sensor simply never publishes. NVIDIA's vendor file lives outside
# that directory, so point the loader at it directly.
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
    setup_param_path = os.path.join(share_dir, "config", "setups", "gz_demo", "gz_demo.yaml")
    # The navigation terrain chain, not the digging chain the core config loads.
    plugin_config_path = os.path.join(share_dir, "config", "setups", "gz_demo", "plugin_config.yaml")
    world_path = os.path.join(share_dir, "gazebo", "worlds", "elevation_demo.sdf")
    for path in (core_param_path, setup_param_path, plugin_config_path, world_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    gui = LaunchConfiguration("gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    # ign gazebo is driven directly rather than through ros_gz_sim's launch
    # file, so the demo only needs ros_gz_sim's binaries on the path.
    gz_server = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-s", "-v", "2", world_path],
        output="screen",
        additional_env=_render_env(headless=True),
        condition=UnlessCondition(gui),
    )
    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "2", world_path],
        output="screen",
        additional_env=_render_env(headless=False),
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
            # A plain camera sensor puts its info on the bare /camera_info,
            # not under <topic>/camera_info the way rgbd_camera does.
            "/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            # Full 3D pose from OdometryPublisher; the planar DiffDrive
            # odometry stays unbridged on purpose (see the world file).
            "/model/robot/odom3d@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            "/model/robot/pose3d@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/model/robot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
        ],
        remappings=[
            ("/model/robot/pose3d", "/tf"),
            ("/model/robot/odom3d", "/odom"),
            ("/model/robot/cmd_vel", "/cmd_vel"),
            ("/color_cam", "/color_cam/image"),
            ("/camera_info", "/color_cam/camera_info"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    # DiffDrive only publishes odom->base_link; the sensor frames are ours to
    # declare. The lidar is mounted level, and its points arrive in its own
    # body-convention frame, so this is a pure translation.
    lidar_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_lidar",
        output="screen",
        arguments=[
            "--x", str(LIDAR_XYZ[0]),
            "--y", str(LIDAR_XYZ[1]),
            "--z", str(LIDAR_XYZ[2]),
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", LIDAR_FRAME,
        ],
        parameters=[{"use_sim_time": True}],
    )

    color_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_color",
        output="screen",
        arguments=[
            "--x", str(COLOR_XYZ[0]),
            "--y", str(COLOR_XYZ[1]),
            "--z", str(COLOR_XYZ[2]),
            "--qx", str(COLOR_QUAT_XYZW[0]),
            "--qy", str(COLOR_QUAT_XYZW[1]),
            "--qz", str(COLOR_QUAT_XYZW[2]),
            "--qw", str(COLOR_QUAT_XYZW[3]),
            "--frame-id", "base_link",
            "--child-frame-id", COLOR_FRAME,
        ],
        parameters=[{"use_sim_time": True}],
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
        }],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            core_param_path,
            setup_param_path,
            {"use_sim_time": True, "plugin_config_file": plugin_config_path},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(launch_rviz),
        parameters=[{"use_sim_time": True}],
    )

    # gz_octomap.launch.py brings no simulator, no mapping and no RViz of its
    # own, so on its own it looks like nothing happened. Pull it in from here
    # instead, where everything it needs is already running.
    octomap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share_dir, "launch", "gz_octomap.launch.py")
        ),
        launch_arguments={"source": LaunchConfiguration("octomap_source")}.items(),
        condition=IfCondition(LaunchConfiguration("octomap")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "octomap",
                default_value="false",
                description="Also build the octomap 2D grid. Needs octomap_server2: "
                "source ~/dependencies/octomap_ws/install/setup.bash first.",
            ),
            DeclareLaunchArgument(
                "octomap_source",
                default_value="traversability",
                description="traversability (drivability layer) or lidar (height band).",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Show the Gazebo window. Set false over ssh.",
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Show the map in RViz. This is the demo's actual output; "
                "the Gazebo window only shows the world being driven through.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                # Not synthetic_demo.rviz: that one is fixed to the map frame and
                # the /elevation_map topic, so against this demo it draws nothing.
                default_value=PathJoinSubstitution([share_dir, "rviz", "gz_demo.rviz"]),
            ),
            gz_server,
            gz_gui,
            bridge,
            lidar_tf,
            color_tf,
            downsample,
            elevation_mapping_node,
            rviz_node,
            octomap,
        ]
    )
