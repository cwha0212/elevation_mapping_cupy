#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# The learned traversability filter: three dilated 3x3 convolutions, |.|,
# a 1x1 mix and exp(-x). torch on the cupy backend, scipy on numpy. Both
# return a (cell_n-6, cell_n-6) float32 map shaped (1, 1, h, w).
#
import numpy as np

from elevation_mapping_cupy.backend import USE_CUPY


def get_filter_torch(*args, **kwargs):
    import cupy as cp
    import torch
    import torch.nn as nn

    class TraversabilityFilter(nn.Module):
        def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
            self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
            self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
            self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

            # Set weights.
            self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
            self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
            self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
            self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

        def __call__(self, elevation_cupy):
            # Convert cupy tensor to pytorch.
            elevation_cupy = elevation_cupy.astype(cp.float32, copy=False)
            elevation = torch.as_tensor(elevation_cupy, device=self.conv1.weight.device)

            with torch.no_grad():
                out1 = self.conv1(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out2 = self.conv2(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out3 = self.conv3(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))

                out1 = out1[:, :, 2:-2, 2:-2]
                out2 = out2[:, :, 1:-1, 1:-1]
                out = torch.cat((out1, out2, out3), dim=1)
                # out = F.concat((out1, out2, out3), axis=1)
                out = self.conv_out(out.abs())
                out = torch.exp(-out)
                out_cupy = cp.asarray(out)

            return out_cupy

    traversability_filter = TraversabilityFilter(*args, **kwargs).cuda().eval()
    return traversability_filter


class TraversabilityFilterNumpy:
    """Same network with scipy correlations (cross-correlation, like torch)."""

    def __init__(self, w1, w2, w3, w_out, **kwargs):
        from scipy.ndimage import correlate

        self._correlate = correlate
        self.w = [np.asarray(w, dtype=np.float32) for w in (w1, w2, w3)]
        self.w_out = np.asarray(w_out, dtype=np.float32).reshape(-1)  # (12,)
        self.dilations = (1, 2, 3)

    def __call__(self, elevation):
        e = np.asarray(elevation, dtype=np.float32)
        h, w = e.shape
        feats = []
        for wk, d in zip(self.w, self.dilations):
            # dilate the 3x3 taps into a (2d+1)x(2d+1) kernel with zeros between
            for o in range(wk.shape[0]):
                k = np.zeros((2 * d + 1, 2 * d + 1), np.float32)
                k[::d, ::d] = wk[o, 0]
                full = self._correlate(e, k, mode="constant", cval=0.0)
                feats.append(np.abs(full[3:-3, 3:-3]))  # common valid crop of the three branches
        out = np.zeros((h - 6, w - 6), np.float32)
        for c, f in zip(self.w_out, feats):
            out += c * f
        return np.exp(-out).reshape(1, 1, h - 6, w - 6)


def get_filter(*args, **kwargs):
    if USE_CUPY:
        return get_filter_torch(*args, **kwargs)
    return TraversabilityFilterNumpy(*args, **kwargs)
