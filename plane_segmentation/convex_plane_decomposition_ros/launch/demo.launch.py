import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('convex_plane_decomposition_ros')
    
    datafile = LaunchConfiguration('datafile')
    max_height = LaunchConfiguration('max_height')
    
    datafile_arg = DeclareLaunchArgument(
        'datafile',
        default_value='terrain.png',
        description='Name of the terrain image file'
    )
    
    max_height_arg = DeclareLaunchArgument(
        'max_height',
        default_value='1.0',
        description='Maximum height of the terrain'
    )

    # Note: Validating grid_map_demos executables in ROS2 environment is hard without running.
    # We assume standard naming.
    
    image_publisher = Node(
        package='grid_map_demos',
        executable='image_publisher', # Assuming executable name
        name='image_publisher',
        output='screen',
        parameters=[{
            'image_path': [os.path.join(pkg_share, 'data'), '/', datafile],
            'topic': '/image'
        }]
    )

    image_to_gridmap_demo = Node(
        package='grid_map_demos',
        executable='image_to_gridmap_demo', 
        name='image_to_gridmap_demo',
        output='screen',
        parameters=[{
            'image_topic': '/image',
            'min_height': 0.0,
            'max_height': max_height,
            'resolution': 0.04
        }]
    )

    add_noise_node = Node(
        package='convex_plane_decomposition_ros',
        executable='convex_plane_decomposition_ros_add_noise',
        name='convex_plane_decomposition_ros_add_noise',
        output='screen',
        parameters=[{
            'noiseGauss': 0.01,
            'noiseUniform': 0.01,
            'outlier_percentage': 5.0,
            'blur': False,
            'frequency': 30.0,
            'elevation_topic_in': '/image_to_gridmap_demo/grid_map',
            'elevation_topic_out': '/elevation_mapping/elevation_map_raw',
            'height_layer': 'elevation',
            'imageName': os.path.join(pkg_share, 'data', 'elevationMap') # This might need full path resolution
        }]
    )
    
    # Static transform
    # args="0.0 0.0 0.0 0.0 0.0 0.0 map odom"
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map2odom_broadcaster',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'map', 'odom']
    )
    
    # Include convex_plane_decomposition.launch.py
    convex_plane_decomposition_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'convex_plane_decomposition.launch.py')
        ),
        launch_arguments={
            'node_parameter_file': os.path.join(pkg_share, 'config', 'demo_node.yaml')
        }.items()
    )

    approximation_demo_node = Node(
        package='convex_plane_decomposition_ros',
        executable='convex_plane_decomposition_ros_approximation_demo_node',
        name='convex_plane_decomposition_ros_approximation_demo_node',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'config_demo.rviz')],
        # respawn=True # ROS2 respawn parameter
    )

    return LaunchDescription([
        datafile_arg,
        max_height_arg,
        image_publisher,
        image_to_gridmap_demo,
        add_noise_node,
        static_tf,
        convex_plane_decomposition_launch,
        approximation_demo_node,
        rviz_node
    ])
