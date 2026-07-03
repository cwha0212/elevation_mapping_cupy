#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from typing import List

import cupy as cp
import cupyx.scipy.ndimage as ndimage

from .plugin_manager import PluginBase


class MinSmoothFilter(PluginBase):
    """
    ROS1 G1-compatible min-smooth filter.

    Applies two median filters followed by a minimum filter to the selected
    input layer. This mirrors the behavior from the ROS1 G1 branch.
    """

    def __init__(self, cell_n: int = 100, input_layer_name: str = "elevation", size: int = 3, **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.size = size

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            h = elevation_map[idx]
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            h = plugin_layers[idx]
        else:
            print(f"layer name {self.input_layer_name} was not found. Using elevation layer.")
            h = elevation_map[0]

        hs1 = ndimage.median_filter(h, size=3)
        hs1 = ndimage.median_filter(hs1, size=3)
        hs1 = ndimage.minimum_filter(hs1, size=self.size)
        return hs1
