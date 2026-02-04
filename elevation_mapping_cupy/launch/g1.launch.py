import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('elevation_mapping_cupy')
    
    config_g1 = os.path.join(pkg_share, 'config', 'setups', 'g1', 'g1_parameters.yaml')
    config_g1_sensor = os.path.join(pkg_share, 'config', 'setups', 'g1', 'g1_sensor_parameter.yaml')
    
    return LaunchDescription([
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping',
            output='screen',
            parameters=[config_g1, config_g1_sensor]
        )
    ])
