import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('elevation_mapping_cupy')
    
    config_core = os.path.join(pkg_share, 'config', 'core', 'core_param.yaml')
    config_example = os.path.join(pkg_share, 'config', 'core', 'example_setup.yaml')
    
    return LaunchDescription([
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping',
            output='screen',
            parameters=[config_core, config_example]
        )
    ])
