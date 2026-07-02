import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_name = "elevation_mapping_cupy"
    share_dir = get_package_share_directory(package_name)

    core_param_path = os.path.join(share_dir, "config", "core", "core_param.yaml")
    g1_param_path = os.path.join(share_dir, "config", "setups", "g1", "g1_parameters.yaml")
    g1_sensor_param_path = os.path.join(share_dir, "config", "setups", "g1", "g1_sensor_parameter.yaml")
    g1_plugin_config_path = os.path.join(share_dir, "config", "setups", "g1", "g1_plugin_config.yaml")

    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Whether to launch RViz2",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="",
        description="Optional RViz config file path",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )

    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    elevation_mapping_node = Node(
        package=package_name,
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            core_param_path,
            g1_param_path,
            g1_sensor_param_path,
            {
                "plugin_config_file": g1_plugin_config_path,
                "use_sim_time": use_sim_time,
            },
        ],
    )

    mid360_to_d435_broadcaster = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="mid360_to_d435_broadcaster",
        arguments=[
            "--x", "0.05734",
            "--y", "0.01750",
            "--z", "0.01369",
            "--yaw", "0",
            "--pitch", "0.79064",
            "--roll", "0",
            "--frame-id", "body",
            "--child-frame-id", "camera_link",
        ],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        launch_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        elevation_mapping_node,
        mid360_to_d435_broadcaster,
        rviz_node,
    ])
