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
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# Where the RGBD sensor's own pose sits relative to base_link, copied from the
# sensor <pose> in the world file. Fortress stamps sensor messages with the
# scoped name <model>/<link>/<sensor>, so that is the frame the transform has
# to land on for elevation mapping's TF lookup to resolve.
CAMERA_FRAME = "robot/base_link/depth"
CAMERA_XYZ = (0.20, 0.0, 0.12)
CAMERA_PITCH = 0.3491  # 20 degrees down

# On L4T the glvnd vendor directory ships only Mesa, so ogre2 loads the Mesa
# EGL, fails to reach nvidia-drm, and the camera renders nothing -- silently,
# since the sensor simply never publishes. NVIDIA's vendor file lives outside
# that directory, so point the loader at it directly.
TEGRA_EGL_VENDOR = "/usr/lib/aarch64-linux-gnu/tegra-egl/nvidia.json"


def _render_env():
    if os.path.exists(TEGRA_EGL_VENDOR):
        return {"__EGL_VENDOR_LIBRARY_FILENAMES": TEGRA_EGL_VENDOR}
    return {}


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    core_param_path = os.path.join(share_dir, "config", "core", "core_param.yaml")
    setup_param_path = os.path.join(share_dir, "config", "setups", "gz_demo", "gz_demo.yaml")
    world_path = os.path.join(share_dir, "gazebo", "worlds", "elevation_demo.sdf")
    for path in (core_param_path, setup_param_path, world_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    gui = LaunchConfiguration("gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    # ign gazebo is driven directly rather than through ros_gz_sim's launch
    # file, so the demo only needs ros_gz_sim's binaries on the path.
    render_env = _render_env()
    gz_server = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-s", "-v", "2", world_path],
        output="screen",
        additional_env=render_env,
        condition=UnlessCondition(gui),
    )
    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "2", world_path],
        output="screen",
        additional_env=render_env,
        condition=IfCondition(gui),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/model/robot/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            "/model/robot/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/model/robot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
        ],
        remappings=[
            ("/model/robot/tf", "/tf"),
            ("/model/robot/odometry", "/odom"),
            ("/model/robot/cmd_vel", "/cmd_vel"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    # DiffDrive only publishes odom->base_link; the sensor frame is ours to
    # declare. Fortress point clouds come out in the sensor's own frame, so the
    # rotation here is the mount tilt only.
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_camera",
        output="screen",
        arguments=[
            "--x", str(CAMERA_XYZ[0]),
            "--y", str(CAMERA_XYZ[1]),
            "--z", str(CAMERA_XYZ[2]),
            "--roll", "0",
            "--pitch", str(CAMERA_PITCH),
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", CAMERA_FRAME,
        ],
        parameters=[{"use_sim_time": True}],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[core_param_path, setup_param_path, {"use_sim_time": True}],
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

    return LaunchDescription(
        [
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
            camera_tf,
            elevation_mapping_node,
            rviz_node,
        ]
    )
