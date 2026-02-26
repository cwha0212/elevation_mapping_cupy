import cupy as cp
import pytest
from pathlib import Path

from elevation_mapping_cupy import parameter, semantic_map


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parents[2]
CORE_CONFIG_DIR = PACKAGE_ROOT / "config" / "core"
WEIGHT_FILE = CORE_CONFIG_DIR / "weights.dat"
PLUGIN_CONFIG_FILE = CORE_CONFIG_DIR / "plugin_config.yaml"


@pytest.fixture()
def semmap_ex(sem_lay, fusion_alg):
    p = parameter.Parameter(
        use_chainer=False,
        weight_file=str(WEIGHT_FILE),
        plugin_config_file=str(PLUGIN_CONFIG_FILE),
    )
    # Explicitly map test channels to the requested fusion modes.
    for layer, fusion in zip(sem_lay, fusion_alg):
        p.pointcloud_channel_fusions[layer] = fusion
    p.update()
    return semantic_map.SemanticMap(p)


@pytest.mark.parametrize(
    "sem_lay,fusion_alg,channels",
    [
        (["feat_0", "feat_1"], ["average", "average"], ["feat_0"]),
        (["feat_0", "feat_1"], ["average", "average"], []),
        (["feat_0", "feat_1", "rgb"], ["average", "average", "color"], ["rgb", "feat_0"]),
        (["feat_0", "feat_1", "rgb"], ["class_bayesian", "class_max", "color"], ["rgb", "feat_0", "feat_1"]),
        (["max1", "max2", "rgb"], ["class_max", "class_max", "color"], ["rgb", "max1", "max2"]),
    ],
)
def test_get_fusion_current_api(semmap_ex, channels):
    process_channels, fusion = semmap_ex.get_fusion(
        channels=channels,
        channel_fusions=semmap_ex.param.pointcloud_channel_fusions,
        layer_specs=semmap_ex.layer_specs_points,
    )
    assert len(process_channels) == len(fusion)
    assert all(isinstance(item, str) for item in fusion)


@pytest.mark.parametrize(
    "sem_lay,fusion_alg,channels,target_fusion",
    [
        (["feat_0", "feat_1", "rgb"], ["average", "average", "color"], ["rgb", "feat_0"], "color"),
        (["max1", "max2", "rgb"], ["class_max", "class_max", "color"], ["rgb", "max1", "max2"], "class_max"),
    ],
)
def test_get_indices_fusion_current_api(semmap_ex, channels, target_fusion):
    process_channels, _ = semmap_ex.get_fusion(
        channels=channels,
        channel_fusions=semmap_ex.param.pointcloud_channel_fusions,
        layer_specs=semmap_ex.layer_specs_points,
    )
    for channel in process_channels:
        if channel not in semmap_ex.layer_names:
            semmap_ex.add_layer(channel)

    pcl_indices, layer_indices = semmap_ex.get_indices_fusion(
        pcl_channels=process_channels,
        fusion_alg=target_fusion,
        layer_specs=semmap_ex.layer_specs_points,
    )
    assert pcl_indices.dtype == cp.int32
    assert layer_indices.dtype == cp.int32
