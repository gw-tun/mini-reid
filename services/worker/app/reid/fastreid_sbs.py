"""FastReID's "SBS" person network, inference only: ResNet50-IBN-a with
non-local blocks, GeM pooling and a BNNeck.

Derived from FastReID v0.1.1 (https://github.com/JDAI-CV/fast-reid), Apache-2.0
(licenses/fast-reid-LICENSE.txt), by the FastReID authors. Source files:

    fastreid/modeling/backbones/resnet.py     ResNet, Bottleneck, non-local placement
    fastreid/layers/batch_norm.py             IBN
    fastreid/layers/non_local.py              Non_local
    fastreid/layers/pooling.py                GeneralizedMeanPoolingP
    fastreid/modeling/heads/bnneck_head.py    BNneckHead (its evaluation path)
    fastreid/modeling/meta_arch/baseline.py   pixel normalisation

CHANGED FROM UPSTREAM (Apache-2.0 §4b): inference only. Removed the config
system, registries, training losses and classifier, SE blocks, BasicBlock,
ghost/sync/frozen batch norm and the pretrained-weight download; fixed to the
one configuration configs/MSMT17/sbs_R50-ibn.yml builds; the four per-stage
non-local loops are written once (`_with_non_local`). Layer names are
unchanged, so the official checkpoint loads with strict=True, and so is the
arithmetic -- checked against the upstream modules on the same input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IBN(nn.Module):
    """Instance norm on the first half of the channels, batch norm on the rest:
    instance norm removes lighting/colour cast, which differs between cameras."""

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.half = int(planes / 2)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        first, second = torch.split(x, self.half, 1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, with_ibn: bool, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class NonLocal(nn.Module):
    """Lets every position of the feature map look at every other one (a person's
    shoes and shoulders inform each other). FastReID uses ONE inner channel
    (`reduc_ratio // reduc_ratio`); the checkpoint's shapes say so too."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.inter_channels = 1
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
                               nn.BatchNorm2d(in_channels))
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(n, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        y = torch.matmul(f / f.size(-1), g_x)
        y = y.permute(0, 2, 1).contiguous().view(n, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


def _with_non_local(stage: nn.Sequential, blocks: nn.ModuleList, x):
    """Run a stage, with one non-local block after each of its last len(blocks) blocks."""
    first = len(stage) - len(blocks)
    for i, block in enumerate(stage):
        x = block(x)
        if i >= first:
            x = blocks[i - first](x)
    return x


class ResNet50IbnNL(nn.Module):
    """FastReID `build_resnet_backbone` for depth 50x, WITH_IBN, WITH_NL, LAST_STRIDE 1."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1, with_ibn=True)
        self.layer2 = self._make_layer(128, 4, stride=2, with_ibn=True)
        self.layer3 = self._make_layer(256, 6, stride=2, with_ibn=True)
        self.layer4 = self._make_layer(512, 3, stride=1, with_ibn=False)  # last stride 1
        # Non-local blocks per stage: 0, 2, 3, 0 (upstream `nl_layers_per_stage` for 50x).
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(2)])
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(3)])

    def _make_layer(self, planes: int, blocks: int, stride: int, with_ibn: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, with_ibn, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes, with_ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = _with_non_local(self.layer2, self.NL_2, x)
        x = _with_non_local(self.layer3, self.NL_3, x)
        return self.layer4(x)


class GeneralizedMeanPooling(nn.Module):
    """GeM: between average pooling (p = 1) and max pooling (p -> inf); p is learned."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class BNNeckHead(nn.Module):
    """Pool, then batch norm ("BNNeck"); evaluation returns the normalised feature
    (NECK_FEAT "after"). The training classifier is not built."""

    def __init__(self, in_feat: int = 2048) -> None:
        super().__init__()
        self.pool_layer = GeneralizedMeanPooling()
        self.bnneck = nn.BatchNorm2d(in_feat)

    def forward(self, features):
        return self.bnneck(self.pool_layer(features))[..., 0, 0]


class SBSPersonNet(nn.Module):
    """RGB images on a 0-255 scale, (N, 3, 384, 128) -> (N, 2048) features."""

    def __init__(self) -> None:
        super().__init__()
        # Filled from the checkpoint: ImageNet's mean and std on the 0-255 scale.
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.backbone = ResNet50IbnNL()
        self.heads = BNNeckHead()

    def forward(self, images):
        return self.heads(self.backbone((images - self.pixel_mean) / self.pixel_std))
