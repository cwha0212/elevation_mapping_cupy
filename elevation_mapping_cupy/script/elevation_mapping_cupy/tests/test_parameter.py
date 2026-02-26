import pytest
from elevation_mapping_cupy.parameter import Parameter
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parents[2]
CORE_CONFIG_DIR = PACKAGE_ROOT / "config" / "core"
WEIGHT_FILE = CORE_CONFIG_DIR / "weights.dat"
PLUGIN_CONFIG_FILE = CORE_CONFIG_DIR / "plugin_config.yaml"


def test_parameter():
    param = Parameter(
        use_chainer=False,
        weight_file=str(WEIGHT_FILE),
        plugin_config_file=str(PLUGIN_CONFIG_FILE),
    )
    res = param.resolution
    param.set_value("resolution", 0.1)
    param.get_types()
    param.get_names()
    param.update()
    assert param.resolution == param.get_value("resolution")
    param.load_weights(param.weight_file)
