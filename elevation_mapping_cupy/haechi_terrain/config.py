"""The stairs pipeline's numbers live in ONE file: config/setups/haechi/stairs.yaml (installed to the package share).

ROS nodes get their sections as a parameter file from haechi.launch.py; the plain-python parts (stairs_fit, terrain_map)
read theirs here. HAECHI_STAIRS_YAML overrides the file (tests, experiments)."""
import os


def path():
    p = os.environ.get("HAECHI_STAIRS_YAML")
    if p:
        return os.path.expanduser(p)
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("elevation_mapping_cupy"), "config", "setups", "haechi", "stairs.yaml")
    except Exception:  # noqa: BLE001 - no ROS environment: the source tree copy
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "setups", "haechi", "stairs.yaml")


def load(section=None):
    """dict of the whole file, or of one section's ros__parameters ({} when the file or section is missing).
    Sections are shaped like ROS nodes (name: {ros__parameters: {...}}) so the same file is a valid --params-file."""
    import yaml
    try:
        with open(path()) as f:
            d = yaml.safe_load(f) or {}
    except OSError:
        d = {}
    if not section:
        return d
    sec = d.get(section) or {}
    return sec.get("ros__parameters", sec) or {}
