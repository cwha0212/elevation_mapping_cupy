#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from elevation_mapping_cupy.parameter import Parameter
import cupy as cp
import numpy as np
from typing import List, Dict
import re


from elevation_mapping_cupy.fusion.fusion_manager import FusionManager

xp = cp


class SemanticMap:
    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param

        self.layer_specs_points = {}
        self.layer_specs_image = {}
        self.layer_names = []
        self.unique_fusion = []
        self.unique_data = []
        self.elements_to_shift = {}

        self.unique_fusion = self.param.fusion_algorithms

        self.amount_layer_names = len(self.layer_names)

        self.semantic_map = xp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n), dtype=param.data_type,
        )
        self.new_map = xp.zeros((self.amount_layer_names, self.param.cell_n, self.param.cell_n), param.data_type,)
        # which layers should be reset to zero at each update, per default everyone,
        # if a layer should not be reset, it is defined in compile_kernels function
        self.delete_new_layers = cp.ones(self.new_map.shape[0], cp.bool_)
        self.fusion_manager = FusionManager(self.param)

    def clear(self):
        """Clear the semantic map."""
        self.semantic_map *= 0.0

    def initialize_fusion(self):
        """Register the image fusion algorithms (the only ones this map uses)."""
        for fusion in self.unique_fusion:
            self.fusion_manager.register_plugin(fusion)

    def add_layer(self, name):
        """
        Add a new layer to the semantic map.

        Args:
            name (str): The name of the new layer.
        """
        if name not in self.layer_names:
            self.layer_names.append(name)
            self.semantic_map = cp.append(
                self.semantic_map,
                cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=self.param.data_type),
                axis=0,
            )
            self.new_map = cp.append(
                self.new_map, cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=self.param.data_type), axis=0,
            )
            self.delete_new_layers = cp.append(self.delete_new_layers, cp.array([1], dtype=cp.bool_))

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y-axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, shift_value):
        """Shift the map along x,y-axis according to shift values.

        Args:
            shift_value:
        """
        self.semantic_map = cp.roll(self.semantic_map, shift_value, axis=(1, 2))
        self.pad_value(self.semantic_map, shift_value, value=0.0)
        self.new_map = cp.roll(self.new_map, shift_value, axis=(1, 2))
        self.pad_value(self.new_map, shift_value, value=0.0)
        # In place: cp.roll returns a new array, and rebinding the loop
        # variable left the dict holding the unshifted one, so nothing
        # registered here ever actually moved with the map.
        for el in self.elements_to_shift.values():
            el[...] = cp.roll(el, shift_value, axis=(1, 2))
            self.pad_value(el, shift_value, value=0.0)

    def get_fusion(
        self, channels: List[str], channel_fusions: Dict[str, str], layer_specs: Dict[str, str]
    ) -> List[str]:
        """Get all fusion algorithms that need to be applied to a specific pointcloud.

        Args:
            channels (List[str]):
        """
        fusion_list = []
        process_channels = []
        for channel in channels:
            if channel not in layer_specs:
                # If the channel is not in the layer_specs, we use the default fusion algorithm
                matched_fusion = self.get_matching_fusion(channel, channel_fusions)
                if matched_fusion is None:
                    if "default" in channel_fusions:
                        default_fusion = channel_fusions["default"]
                        print(
                            f"[WARNING] Layer {channel} not found in layer_specs. Using {default_fusion} algorithm as default."
                        )
                        layer_specs[channel] = default_fusion
                    # If there's no default fusion algorithm, we skip this channel
                    else:
                        print(
                            f"[WARNING] Layer {channel} not found in layer_specs ({layer_specs}) and no default fusion is configured. Skipping."
                        )
                        continue
                else:
                    layer_specs[channel] = matched_fusion
            x = layer_specs[channel]
            fusion_list.append(x)
            process_channels.append(channel)
        return process_channels, fusion_list

    def get_matching_fusion(self, channel: str, fusion_algs: Dict[str, str]):
        """ Use regular expression to check if the fusion algorithm matches the channel name."""
        for fusion_alg, alg_value in fusion_algs.items():
            if re.match(f"^{fusion_alg}$", channel):
                return alg_value
        return None

    def update_layers_image(
        self,
        # sub_key: str,
        image: cp._core.core.ndarray,
        channels: List[str],
        # fusion_methods: List[str],
        uv_correspondence: cp._core.core.ndarray,
        valid_correspondence: cp._core.core.ndarray,
        image_height: cp._core.core.ndarray,
        image_width: cp._core.core.ndarray,
    ):
        """Update the semantic map with the new image.

        Args:
            sub_key:
            image:
            uv_correspondence:
            valid_correspondence:
            image_height:
            image_width:
        """

        process_channels, fusion_methods = self.get_fusion(
            channels, self.param.image_channel_fusions, self.layer_specs_image
        )
        self.new_map[self.delete_new_layers] = 0.0
        for j, (fusion, channel) in enumerate(zip(fusion_methods, process_channels)):
            if channel not in self.layer_names:
                print(f"Layer {channel} not found, adding it to the semantic map")
                self.add_layer(channel)
            sem_map_idx = self.get_index(channel)

            if sem_map_idx == -1:
                print(f"Layer {channel} not found!")
                return

            # update the layers with the fusion algorithm
            self.fusion_manager.execute_image_plugin(
                fusion,
                cp.uint64(sem_map_idx),
                image,
                j,
                uv_correspondence,
                valid_correspondence,
                image_height,
                image_width,
                self.semantic_map,
                self.new_map,
            )

    def get_map_with_name(self, name):
        """Return the map with the given name.

        Args:
            name: layer name

        Returns:
            cp.array: map
        """
        # If the layer is a color layer, return the rgb map
        if name in self.layer_specs_points and self.layer_specs_points[name] == "color":
            m = self.get_rgb(name)
            return m
        elif name in self.layer_specs_image and self.layer_specs_image[name] == "color":
            m = self.get_rgb(name)
            return m
        else:
            m = self.get_semantic(name)
            return m

    def get_rgb(self, name):
        """Return the rgb map with the given name.

        Args:
            name:

        Returns:
            cp.array: rgb map
        """
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        c = c.astype(np.float32)
        return c

    def get_semantic(self, name):
        """Return the semantic map layer with the given name.

        Args:
            name(str): layer name

        Returns:
            cp.array: semantic map layer
        """
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        return c

    def process_map_for_publish(self, input_map):
        """Remove padding.

        Args:
            input_map(cp.array): map layer

        Returns:
            cp.array: map layer without padding
        """
        m = input_map.copy()
        return m[1:-1, 1:-1]

    def get_index(self, name):
        """Return the index of the layer with the given name.

        Args:
            name(str):

        Returns:
            int: index
        """
        if name not in self.layer_names:
            return -1
        else:
            return self.layer_names.index(name)
