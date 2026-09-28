from setuptools import setup, find_packages

package_name = 'elevation_mapping_cupy'

setup(
    name=package_name,
    version='2.2.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    install_requires=['setuptools'],
    zip_safe=True,
    author='Takahiro Miki, Gian Erni',
    maintainer='Lorenzo Terenzi',
    maintainer_email='lorenzoterenzi96@gmail.com',
    description='Elevation mapping on GPU',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'elevation_mapping_node.py = elevation_mapping_cupy.elevation_mapping_node:main',
        ],
    },
)
