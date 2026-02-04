from setuptools import setup
import os

package_name = 'semantic_sensor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={'': 'script'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            ['launch/semantic_image.launch.py', 'launch/semantic_pointcloud.launch.py']), 
        (os.path.join('share', package_name, 'config'), ['config/sensor_parameter.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gian Erni',
    maintainer_email='gerni@ethz.ch',
    description='The semantic_sensor package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_node = semantic_sensor.pointcloud_node:main',
            'image_node = semantic_sensor.image_node:main',
        ],
    },
)
