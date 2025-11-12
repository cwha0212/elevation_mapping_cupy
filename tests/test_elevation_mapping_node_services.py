import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.modules.setdefault("cupy", mock.MagicMock())

if "grid_map_msgs" not in sys.modules:
    grid_map_module = ModuleType("grid_map_msgs")
    grid_map_msg_module = ModuleType("grid_map_msgs.msg")
    grid_map_srv_module = ModuleType("grid_map_msgs.srv")

    class _DummyGridMap:
        pass

    class _DummySetGridMap:
        pass

    class _DummyProcessFile:
        pass

    grid_map_msg_module.GridMap = _DummyGridMap
    grid_map_srv_module.SetGridMap = _DummySetGridMap
    grid_map_srv_module.ProcessFile = _DummyProcessFile

    grid_map_module.msg = grid_map_msg_module
    grid_map_module.srv = grid_map_srv_module

    sys.modules["grid_map_msgs"] = grid_map_module
    sys.modules["grid_map_msgs.msg"] = grid_map_msg_module
    sys.modules["grid_map_msgs.srv"] = grid_map_srv_module

from grid_map_msgs.msg import GridMap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "elevation_mapping_cupy"))

from elevation_mapping_cupy import elevation_mapping_node as node_mod


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class DummyClock:
    def now(self):
        return SimpleNamespace(nanoseconds=123)


def bind_method(method, instance):
    """Bind a class method to a lightweight instance for unit testing."""
    return method.__get__(instance, node_mod.ElevationMappingNode)


def make_dummy_node_for_masked_replace():
    dummy = SimpleNamespace()
    dummy.masked_replace_mask_layer_name = "mask"
    dummy.logger = DummyLogger()
    dummy.get_logger = lambda: dummy.logger
    dummy._map = SimpleNamespace()
    dummy._map.apply_masked_replace = mock.Mock()
    dummy._republish_all_once = mock.Mock()
    return dummy


def test_handle_masked_replace_sets_success_flag():
    dummy = make_dummy_node_for_masked_replace()

    size = 2
    data_layers = {
        "mask": np.ones((size, size), dtype=np.float32),
        "elevation": np.full((size, size), 1.0, dtype=np.float32),
    }
    mask_layer = data_layers["mask"]
    geometry = SimpleNamespace()
    dummy._grid_map_to_numpy = mock.Mock(return_value=(data_layers, geometry))

    request = SimpleNamespace(map=GridMap())
    response = SimpleNamespace(success=False)

    handler = bind_method(node_mod.ElevationMappingNode.handle_masked_replace, dummy)
    handler(request, response)

    assert response.success is True
    dummy._map.apply_masked_replace.assert_called_once()
    call_args = dummy._map.apply_masked_replace.call_args
    layers_arg, mask_arg, geometry_arg = call_args.args
    assert "mask" not in layers_arg
    assert mask_arg is mask_layer
    assert geometry_arg is geometry
    dummy._republish_all_once.assert_called_once()


def test_handle_masked_replace_failure_sets_success_false():
    dummy = make_dummy_node_for_masked_replace()
    dummy._grid_map_to_numpy = mock.Mock(side_effect=RuntimeError("boom"))

    request = SimpleNamespace(map=GridMap())
    response = SimpleNamespace(success=True)

    handler = bind_method(node_mod.ElevationMappingNode.handle_masked_replace, dummy)
    handler(request, response)

    assert response.success is False


def test_write_grid_map_bag_uses_correct_topic_metadata(tmp_path, monkeypatch):
    dummy = SimpleNamespace()
    dummy.save_map_storage_id = "mcap"
    dummy._clock = DummyClock()
    dummy.get_clock = lambda: dummy._clock

    storage_mock = mock.Mock(name="StorageOptions")
    converter_mock = mock.Mock(name="ConverterOptions")
    metadata_mock = mock.Mock(name="TopicMetadataReturn")
    writer_mock = mock.Mock(name="SequentialWriter")

    storage_cls = mock.Mock(return_value=storage_mock)
    converter_cls = mock.Mock(return_value=converter_mock)
    call_sequence = {"count": 0}

    def topic_metadata_side_effect(*args, **kwargs):
        call_sequence["count"] += 1
        if call_sequence["count"] == 1:
            assert args == (topic, "grid_map_msgs/msg/GridMap", "cdr", "")
            raise TypeError("legacy signature only accepts name first")
        assert args == (0, topic, "grid_map_msgs/msg/GridMap", "cdr")
        return metadata_mock

    metadata_cls = mock.Mock(side_effect=topic_metadata_side_effect)
    writer_cls = mock.Mock(return_value=writer_mock)

    monkeypatch.setattr(node_mod.rosbag2_py, "StorageOptions", storage_cls)
    monkeypatch.setattr(node_mod.rosbag2_py, "ConverterOptions", converter_cls)
    monkeypatch.setattr(node_mod.rosbag2_py, "TopicMetadata", metadata_cls)
    monkeypatch.setattr(node_mod.rosbag2_py, "SequentialWriter", writer_cls)
    monkeypatch.setattr(node_mod, "serialize_message", lambda msg: b"serialized")

    path = tmp_path / "map_bag"
    topic = "/test_topic"
    grid_map_msg = GridMap()

    writer = bind_method(node_mod.ElevationMappingNode._write_grid_map_bag, dummy)
    dummy._make_topic_metadata = bind_method(node_mod.ElevationMappingNode._make_topic_metadata, dummy)
    writer(path, topic, grid_map_msg)

    storage_cls.assert_called_once_with(uri=str(path), storage_id="mcap")
    converter_cls.assert_called_once_with("", "")
    assert metadata_cls.call_count == 2
    assert metadata_cls.call_args_list[-1].args == (0, topic, "grid_map_msgs/msg/GridMap", "cdr")
    writer_mock.create_topic.assert_called_once_with(metadata_mock)
    writer_mock.write.assert_called_once()
