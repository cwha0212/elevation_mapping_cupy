import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('semantic_sensor')
    config_file = os.path.join(pkg_share, 'config', 'sensor_parameter.yaml')
    
    return LaunchDescription([
        Node(
            package='semantic_sensor',
            executable='pointcloud_node',
            name='semantic_pointcloud',
            output='screen',
            arguments=['front_cam'],
            parameters=[config_file]
        )
    ])
