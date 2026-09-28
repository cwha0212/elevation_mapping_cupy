#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp


def get_filter_torch(*args, **kwargs):
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
