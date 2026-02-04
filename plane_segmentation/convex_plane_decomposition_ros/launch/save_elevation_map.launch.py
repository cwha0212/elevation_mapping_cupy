import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('convex_plane_decomposition_ros')
    
    return LaunchDescription([
        Node(
            package='convex_plane_decomposition_ros',
            executable='convex_plane_decomposition_ros_save_elevationmap', # Updated executable name
            name='save_elevation_map',
            output='screen',
            parameters=[{
                'frequency': 0.1,
                'elevation_topic': '/elevation_mapping/elevation_map_raw',
                'height_layer': 'elevation',
                'imageName': os.path.join(pkg_share, 'data', 'elevationMap')
            }]
        )
    ])
