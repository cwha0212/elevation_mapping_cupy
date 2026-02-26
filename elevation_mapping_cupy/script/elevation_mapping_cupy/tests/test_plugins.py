import pytest
from elevation_mapping_cupy import semantic_map, parameter
import cupy as cp
from elevation_mapping_cupy.plugins.plugin_manager import PluginManager
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parents[2]
CORE_CONFIG_DIR = PACKAGE_ROOT / "config" / "core"
WEIGHT_FILE = CORE_CONFIG_DIR / "weights.dat"
PLUGIN_PATH = TEST_DIR / "plugin_config.yaml"


@pytest.fixture()
def semmap_ex(add_lay, fusion_alg):
    p = parameter.Parameter(
        use_chainer=False, weight_file=str(WEIGHT_FILE), plugin_config_file=str(PLUGIN_PATH),
    )
    p.subscriber_cfg["front_cam"]["channels"] = add_lay
    p.subscriber_cfg["front_cam"]["fusion"] = fusion_alg
    p.update()
    e = semantic_map.SemanticMap(p)
    e.initialize_fusion()
    return e


@pytest.mark.parametrize(
    "add_lay, fusion_alg,channels",
    [
        (
            ["grass", "tree", "fence", "person"],
            ["class_average", "class_average", "class_average", "class_average"],
            ["grass"],
        ),
        (["grass", "tree"], ["class_average", "class_average"], ["grass"]),
        (["grass", "tree"], ["class_average", "class_max"], ["tree"]),
        (["max1", "max2"], ["class_max", "class_max"], ["max1", "max2"]),
    ],
)
def test_plugin_manager(semmap_ex, channels):
    manager = PluginManager(202)
    manager.load_plugin_settings(str(PLUGIN_PATH))
    elevation_map = cp.zeros((7, 202, 202)).astype(cp.float32)
    rotation = cp.eye(3, dtype=cp.float32)
    layer_names = [
        "elevation",
        "variance",
        "is_valid",
        "traversability",
        "time",
        "upper_bound",
        "is_upper_bound",
    ]
    elevation_map[0] = cp.random.randn(202, 202)
    elevation_map[2] = cp.abs(cp.random.randn(202, 202))
    manager.update_with_name("min_filter", elevation_map, layer_names)
    manager.update_with_name("smooth_filter", elevation_map, layer_names)
    manager.update_with_name(
        "sem_fil",
        elevation_map,
        layer_names,
        semmap_ex.semantic_map,
        semmap_ex.layer_names,
        rotation,
        semmap_ex.elements_to_shift,
    )
    manager.update_with_name(
        "sem_traversability",
        elevation_map,
        layer_names,
        semmap_ex.semantic_map,
        semmap_ex.layer_names,
        rotation,
        semmap_ex.elements_to_shift,
    )
    manager.get_map_with_name("smooth")
    for lay in manager.get_layer_names():
        manager.update_with_name(
            lay,
            elevation_map,
            layer_names,
            semmap_ex.semantic_map,
            semmap_ex.layer_names,
            rotation,
            semmap_ex.elements_to_shift,
        )
        manager.get_map_with_name(lay)
