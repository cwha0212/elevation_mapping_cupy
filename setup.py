from setuptools import setup, find_packages

setup(
    name="elevation_mapping_cupy",
    version="0.1.0",
    description="Elevation mapping with cupy acceleration",
    author="Jonas Frey",
    author_email="jonfrey@ethz.ch",
    packages=find_packages(where="elevation_mapping_cupy"),
    package_dir={"": "elevation_mapping_cupy"},
    install_requires=[
        "numpy",
        "scipy",
        "ruamel.yaml",
        "opencv-python",
        "simple-parsing",
        "scikit-image",
        "matplotlib",
        "catkin-tools",
        "networkx==3.0",
        "shapely",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "elevation_mapping_cupy": [
            "*",
        ],
    },
)

# Original versions used
# shapely==1.7.1
# scipy==1.7
# scikit-image==0.19

# We installed for Python 3.11
#  + catkin-pkg==1.0.0
#  + catkin-tools==0.9.5
#  + docstring-parser==0.17.0
#  + docutils==0.21.2
#  + elevation-mapping-cupy==0.1.0 (from file:///home/jonfrey/git/elevation_mapping_cupy)
#  + lazy-loader==0.4
#  + networkx==3.0
#  + osrf-pycommon==2.0.2
#  + ruamel-yaml==0.18.14
#  + ruamel-yaml-clib==0.2.12
#  + scikit-image==0.25.2
#  + shapely==2.1.1
#  + simple-parsing==0.1.7
#  + tifffile==2025.6.11
