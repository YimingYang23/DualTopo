# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: gru. No MMCV dependency."""

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvGRU(nn.Module):
    def __init__(self, out_channels=256, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.convz = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.convr = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.convq = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.ln = nn.LayerNorm(out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, h, x):
        if h.dim() == 3:
            h = h.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1)
        q = self.convq(new_x)
        out = ((1 - z) * h + z * q).squeeze(0)
        out = self.ln(out.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        return out
