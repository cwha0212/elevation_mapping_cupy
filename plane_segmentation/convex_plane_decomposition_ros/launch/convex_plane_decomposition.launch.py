import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('convex_plane_decomposition_ros')
    
    parameter_file = LaunchConfiguration('parameter_file')
    node_parameter_file = LaunchConfiguration('node_parameter_file')
    
    default_parameter_file = os.path.join(pkg_share, 'config', 'parameters.yaml')
    default_node_parameter_file = os.path.join(pkg_share, 'config', 'node.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'parameter_file',
            default_value=default_parameter_file,
            description='Path to the parameter file'
        ),
        DeclareLaunchArgument(
            'node_parameter_file',
            default_value=default_node_parameter_file,
            description='Path to the node parameter file'
        ),
        Node(
            package='convex_plane_decomposition_ros',
            executable='convex_plane_decomposition_ros_node',
            name='convex_plane_decomposition_ros',
            output='screen',
            parameters=[parameter_file, node_parameter_file]
        )
    ])
