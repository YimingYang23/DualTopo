# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""DualTopo inference layers: backbone. No MMCV dependency."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """ResNet50 backbone that returns multi-scale features.

    Args:
        out_indices: Tuple of stage indices (0-based) to return features from.
                     Default (1,2,3) returns C3, C4, C5.
        frozen_stages: Number of stages to freeze (1 = freeze stem + stage1).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, out_indices=(1, 2, 3), frozen_stages=1, pretrained=True):
        super().__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList(
            [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        )
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode:
            for i in range(self.frozen_stages):
                self.stages[i].eval()
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class FPN(nn.Module):
    """Feature Pyramid Network.

    Args:
        in_channels: List of input channel sizes from backbone.
        out_channels: Output channel size for all levels.
        num_outs: Number of output feature maps.
        start_level: Start level index of backbone features to use.
        add_extra_convs: Whether and how to add extra convolutions.
        relu_before_extra_convs: Whether to apply relu before extra convs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.num_ins = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.num_ins):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - (self.num_ins - self.start_level)
        self.extra_convs = nn.ModuleList()
        if extra_levels > 0 and add_extra_convs:
            for i in range(extra_levels):
                extra_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: List of feature maps from backbone, ordered from low to high resolution.
        Returns:
            List of FPN output feature maps.
        """
        used_inputs = inputs[self.start_level :]
        laterals = [
            lateral_conv(used_inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        if self.add_extra_convs:
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    outs.append(extra_conv(outs[-1]))
                elif self.relu_before_extra_convs:
                    outs.append(extra_conv(F.relu(outs[-1])))
                else:
                    outs.append(extra_conv(outs[-1]))
        return outs
