# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import CBAM, Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "SPDConv",
    "AConv",
    "ADown",
    "AFPNFuse2",
    "AFFFuse2",
    "BiFPNAdd2",
    "Attention",
    "ASFF2",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "CARAFEUpsample",
    "GatedConcat",
    "C2fCIB",
    "C2fPSA",
    "PConvFasterC3k2",
    "C3Ghost",
    "C3k2",
    "D3Block",
    "D3C2f",
    "C3k2PKI",
    "C3k2_CrossConvLite",
    "C3k2_DDFM",
    "C3k2_DCBLite",
    "C3k2_DSConv",
    "C3k2_EAConvLite",
    "C3k2_FADC",
    "C3k2_FDConv",
    "C3k2_EMA",
    "C3k2_ODConv",
    "C3k2_FocalModulationLite",
    "C3k2_GCResidual",
    "C3k2_CAAResidual",
    "C3k2_SEResidual",
    "C3k2_InceptionNeXtLite",
    "C3k2_MCA",
    "C3k2_MobileOneLite",
    "C3k2_HorNetResidual",
    "C3k2_MogaResidual",
    "C3k2_GhostV2DFC",
    "C3k2_MSBlock",
    "C3k2_MSBlockLite",
    "C3k2_RFAConv",
    "C3k2_RepHELANLite",
    "C3k2_RepHMSLite",
    "C3k2_RepMixerLite",
    "C3k2_SCConv",
    "C3k2_SFSConv",
    "C3k2_SCSALite",
    "C3k2_SHSA",
    "C3k2_StarBlockLite",
    "CSPStageLite",
    "FeaturePyramidSharedConv",
    "FeaturePyramidSharedConvLite",
    "HyperACEFullPADNeck",
    "NeckFeatureSelect",
    "SFEWPFPNNeck",
    "SPAFPNC3k2Neck",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "CoordAtt",
    "DDFMLite",
    "DASI2",
    "DLU",
    "ECALayer",
    "SimAM",
    "TripletAttention",
    "TripletAttentionGate",
    "ZPool",
    "DSConv",
    "DWRLite",
    "DySample",
    "SAPAUpsample",
    "ELA",
    "EUCB",
    "FusionFactor",
    "FADCLite",
    "FDConv",
    "GAM",
    "GCBlock",
    "GRNBlock",
    "MobileViTv2SSA",
    "GCConv",
    "GhostBottleneck",
    "GhostBottleneckV2DFC",
    "GhostModuleV2DFC",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "LSKBlock",
    "LineRefineLite",
    "MFAM",
    "GnConvLite",
    "MogaBlockLite",
    "MSBlock",
    "HierarchicalMSBlockLite",
    "ODConv",
    "Proto",
    "RFAConv",
    "RepC3",
    "RepDown",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResCBAM",
    "ResNetLayer",
    "LDownLite",
    "RFPNLiteFuse",
    "SCDown",
    "SEAM",
    "SNIFuse2",
    "StripEnhance",
    "SFSConv",
    "FreqFusionLite",
    "TorchVision",
    "WTConvDown",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CSPStageLite(nn.Module):
    """Lightweight CSP stage for neck fusion with limited feature distribution shift."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSPStageLite.

        This stage keeps one branch as a shallow identity-like path and refines the other branch with
        lightweight bottlenecks, then fuses both branches with a 1x1 projection.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.blocks = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse preserved and refined CSP branches."""
        return self.cv3(torch.cat((self.cv1(x), self.blocks(self.cv2(x))), 1))


class FeaturePyramidSharedConvLite(nn.Module):
    """Lightweight shared-convolution block for a single FPN/PAN fusion level."""

    def __init__(self, c1: int, c2: int, n: int = 2, e: float = 1.0):
        """Initialize a shared depthwise-pointwise refinement block."""
        super().__init__()
        c_ = int(c2 * e)
        self.reduce = Conv(c1, c_, 1, 1) if c1 != c_ else nn.Identity()
        self.shared = nn.Sequential(DWConv(c_, c_, 3, 1), Conv(c_, c_, 1, 1))
        self.expand = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
        self.n = max(int(n), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply repeated refinement with shared parameters."""
        y = self.reduce(x)
        for _ in range(self.n):
            y = y + self.shared(y)
        return self.expand(y)


class FeaturePyramidSharedConv(nn.Module):
    """Paper-style FPSC using one recursively shared kernel at multiple dilation rates."""

    def __init__(self, c1, c2, e=0.25, dilations=(1, 3, 5)):
        """Reduce channels, build a shared-convolution pyramid, then fuse it."""
        super().__init__()
        c_ = max(int(c2 * e), 8)
        self.reduce = Conv(c1, c_, 1, 1)
        self.shared = nn.Conv2d(c_, c_, 3, stride=1, padding=1, bias=False)
        self.dilations = tuple(int(d) for d in dilations)
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("FPSC dilations must be positive integers")
        self.norms = nn.ModuleList(nn.BatchNorm2d(c_) for _ in self.dilations)
        self.act = nn.SiLU(inplace=True)
        self.fuse = Conv(c_ * (len(self.dilations) + 1), c2, 1, 1)

    def forward(self, x):
        """Apply the same 3x3 weights recursively with dilation rates 1, 3 and 5."""
        y = self.reduce(x)
        features = [y]
        for dilation, norm in zip(self.dilations, self.norms):
            y = F.conv2d(
                y,
                self.shared.weight,
                stride=1,
                padding=dilation,
                dilation=dilation,
            )
            y = self.act(norm(y))
            features.append(y)
        return self.fuse(torch.cat(features, dim=1))



class GatedConcat(nn.Module):
    """Concatenate feature branches after lightweight channel gating."""

    def __init__(self, c1: list[int], dimension: int = 1, reduction: int = 16):
        """Initialize branch-wise channel gates."""
        super().__init__()
        self.d = dimension
        self.gates = nn.ModuleList()
        for c in c1:
            hidden = max(c // reduction, 8)
            self.gates.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(c, hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, c, 1, bias=True),
                    nn.Sigmoid(),
                )
            )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Gate each input branch, then concatenate."""
        return torch.cat([x * gate(x) for x, gate in zip(xs, self.gates)], self.d)


class CARAFEUpsample(nn.Module):
    """Content-aware reassembly upsampling from CARAFE."""

    def __init__(self, c1: int, scale: int = 2, encoder_kernel: int = 3, up_kernel: int = 5, compressed_channels: int = 64):
        """Initialize CARAFE upsampling."""
        super().__init__()
        self.scale = scale
        self.up_kernel = up_kernel
        cm = min(compressed_channels, c1)
        self.comp = Conv(c1, cm, 1, 1)
        self.enc = nn.Conv2d(cm, (scale * up_kernel) ** 2, encoder_kernel, 1, encoder_kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-aware feature reassembly."""
        b, c, h, w = x.shape
        mask = self.enc(self.comp(x))
        mask = F.pixel_shuffle(mask, self.scale)
        mask = F.softmax(mask.view(b, self.up_kernel * self.up_kernel, h * self.scale, w * self.scale), dim=1)
        x_up = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        pad = self.up_kernel // 2
        patches = F.unfold(F.pad(x_up, (pad, pad, pad, pad)), self.up_kernel)
        patches = patches.view(b, c, self.up_kernel * self.up_kernel, h * self.scale, w * self.scale)
        return (patches * mask.unsqueeze(1)).sum(dim=2)


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck block for lightweight neck processing."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, expansion: float = 1.0):
        """Initialize MBConvBlock."""
        super().__init__()
        hidden = max(int(c1 * expansion), 8)
        self.use_res = shortcut and c1 == c2
        self.conv = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            Conv(hidden, c2, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverted bottleneck with optional residual."""
        y = self.conv(x)
        return x + y if self.use_res else y


class C2fMBConv(C2f):
    """C2f block using lightweight MBConv bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, expansion: float = 1.0, e: float = 0.5):
        """Initialize C2fMBConv."""
        super().__init__(c1, c2, n, shortcut, 1, e)
        self.m = nn.ModuleList(MBConvBlock(self.c, self.c, shortcut, expansion) for _ in range(n))


class FusedMBConvBlock(nn.Module):
    """EfficientNetV2 fused inverted bottleneck block for neck processing."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, expansion: float = 1.0):
        """Initialize FusedMBConvBlock."""
        super().__init__()
        hidden = max(int(c1 * expansion), 8)
        self.use_res = shortcut and c1 == c2
        self.conv = nn.Sequential(
            Conv(c1, hidden, 3, 1),
            Conv(hidden, c2, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused inverted bottleneck with optional residual."""
        y = self.conv(x)
        return x + y if self.use_res else y





class C2fFusedMBConv(C2f):
    """C2f block using EfficientNetV2 fused MBConv bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, expansion: float = 1.0, e: float = 0.5):
        """Initialize C2fFusedMBConv."""
        super().__init__(c1, c2, n, shortcut, 1, e)
        self.m = nn.ModuleList(FusedMBConvBlock(self.c, self.c, shortcut, expansion) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class GhostModuleV2DFC(nn.Module):
    """GhostNetV2 module with decoupled fully-connected spatial attention.

    The attention path follows the official GhostNetV2 implementation: it is
    evaluated at half resolution and uses depthwise 1x5 followed by 5x1
    convolutions to obtain a hardware-friendly long-range spatial gate.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        ratio: int = 2,
        dw_k: int = 3,
        act: bool = True,
        attention: bool = False,
    ):
        """Initialize a GhostNetV2 feature generator."""
        super().__init__()
        if ratio < 2:
            raise ValueError(f"GhostModuleV2DFC ratio must be at least 2, got {ratio}")
        primary_channels = math.ceil(c2 / ratio)
        cheap_channels = primary_channels * (ratio - 1)
        self.c2 = c2
        self.attention = attention
        self.primary = Conv(c1, primary_channels, k, s, act=nn.ReLU(inplace=True) if act else False)
        self.cheap = Conv(
            primary_channels,
            cheap_channels,
            dw_k,
            1,
            g=primary_channels,
            act=nn.ReLU(inplace=True) if act else False,
        )
        if attention:
            self.dfc = nn.Sequential(
                Conv(c1, c2, k, s, act=False),
                Conv(c2, c2, (1, 5), 1, g=c2, act=False),
                Conv(c2, c2, (5, 1), 1, g=c2, act=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate intrinsic and cheap features, optionally gated by DFC attention."""
        primary = self.primary(x)
        out = torch.cat((primary, self.cheap(primary)), dim=1)[:, : self.c2]
        if not self.attention:
            return out
        gate_input = F.avg_pool2d(x, kernel_size=2, stride=2)
        gate = torch.sigmoid(self.dfc(gate_input))
        gate = F.interpolate(gate, size=out.shape[-2:], mode="nearest")
        return out * gate


class GhostBottleneckV2DFC(nn.Module):
    """Stride-one GhostNetV2 bottleneck adapted for CSP neck feature refinement."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, expansion: float = 2.0):
        """Initialize GhostNetV2 expansion, DFC attention, and projection paths."""
        super().__init__()
        hidden = max(c2, int(c2 * expansion))
        self.ghost1 = GhostModuleV2DFC(c1, hidden, act=True, attention=True)
        self.ghost2 = GhostModuleV2DFC(hidden, c2, act=False, attention=False)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GhostNetV2 bottleneck with an optional residual connection."""
        y = self.ghost2(self.ghost1(x))
        return x + y if self.shortcut else y


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class FADCLite(nn.Module):
    """Frequency-adaptive dilated convolution for lightweight spatial feature enhancement."""

    def __init__(self, c: int, reduction: int = 4):
        """Initialize FADCLite.

        Args:
            c (int): Input and output channels.
            reduction (int): Channel reduction ratio for the frequency gate.
        """
        super().__init__()
        hidden = max(c // reduction, 8)
        self.pre = Conv(c, c, 1, 1)
        self.dw1 = nn.Conv2d(c, c, 3, 1, 1, dilation=1, groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, 3, 1, 2, dilation=2, groups=c, bias=False)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 3, dilation=3, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c * 3, 1, bias=True),
        )
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-guided adaptive fusion of multi-dilation depthwise convolutions."""
        y = self.pre(x)
        low = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        freq = (y - low).abs()
        b, c, h, w = y.shape
        weights = self.gate(freq).view(b, 3, c, 1, 1).softmax(dim=1)
        branches = torch.stack((self.dw1(y), self.dw2(y), self.dw3(y)), dim=1)
        y = (branches * weights).sum(dim=1)
        return self.fuse(self.bn(y))


class Bottleneck_FADC(nn.Module):
    """Bottleneck using FADCLite as the spatial mixing operator."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_FADC with Bottleneck-compatible arguments."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.fadc = FADCLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FADC bottleneck with optional shortcut connection."""
        y = self.cv2(self.fadc(self.cv1(x)))
        return x + y if self.add else y


class FDConv(nn.Module):
    """Frequency dynamic convolution with adaptive multi-dilation spatial mixing."""

    def __init__(self, c: int, reduction: int = 4):
        """Initialize FDConv.

        Args:
            c (int): Input and output channels.
            reduction (int): Channel reduction ratio for dynamic branch weighting.
        """
        super().__init__()
        hidden = max(c // reduction, 16)
        self.pre = Conv(c, c, 1, 1)
        self.dw1 = nn.Conv2d(c, c, 3, 1, 1, dilation=1, groups=c, bias=False)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 3, dilation=3, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 3, 1, 5, dilation=5, groups=c, bias=False)
        self.high_refine = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c * 4, 1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-guided dynamic fusion of local, large-context, and high-frequency branches."""
        y = self.pre(x)
        low = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
        high = y - low
        desc = torch.cat((y, high.abs()), 1)
        b, c, _, _ = y.shape
        weights = self.branch_gate(desc).view(b, 4, c, 1, 1).softmax(dim=1)
        branches = torch.stack((self.dw1(y), self.dw3(y), self.dw5(y), self.high_refine(high)), dim=1)
        y = (branches * weights).sum(dim=1)
        y = y * self.channel_gate(y)
        return self.fuse(y)


class Bottleneck_FDConv(nn.Module):
    """Bottleneck using FDConv as the spatial-frequency dynamic operator."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_FDConv with Bottleneck-compatible arguments."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.fdconv = FDConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FDConv bottleneck with optional shortcut connection."""
        y = self.cv2(self.fdconv(self.cv1(x)))
        return x + y if self.add else y


class SFSConv(nn.Module):
    """Spatial-frequency selection convolution for degraded and low-contrast features."""

    def __init__(self, c: int, reduction: int = 4):
        """Initialize SFSConv.

        Args:
            c (int): Input and output channels.
            reduction (int): Channel reduction ratio for spatial-frequency selection.
        """
        super().__init__()
        hidden = max(c // reduction, 16)
        self.pre = Conv(c, c, 1, 1)
        self.local = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.large = nn.Sequential(
            nn.Conv2d(c, c, (1, 7), 1, (0, 3), groups=c, bias=False),
            nn.Conv2d(c, c, (7, 1), 1, (3, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.freq = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.selector = nn.Sequential(
            Conv(c * 3, hidden, 1, 1),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Selectively fuse local spatial, large-strip spatial, and high-frequency responses."""
        y = self.pre(x)
        low = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        high = y - low
        local = self.local(y)
        large = self.large(y)
        freq = self.freq(high)
        weights = torch.softmax(self.selector(torch.cat((local, large, freq), 1)), dim=1)
        y = local * weights[:, 0:1] + large * weights[:, 1:2] + freq * weights[:, 2:3]
        y = y * self.channel_gate(y)
        return self.fuse(y)


class Bottleneck_SFSConv(nn.Module):
    """Bottleneck using SFSConv as the spatial-frequency selection operator."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_SFSConv with Bottleneck-compatible arguments."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.sfsconv = SFSConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SFSConv bottleneck with optional shortcut connection."""
        y = self.cv2(self.sfsconv(self.cv1(x)))
        return x + y if self.add else y


def _valid_group_count(channels: int, max_groups: int = 16) -> int:
    """Return the largest group count up to max_groups that divides channels."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ELA(nn.Module):
    """Efficient local attention with separate height and width context encoding."""

    def __init__(self, c1: int, c2: int | None = None, k: int = 7):
        """Initialize ELA.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to c1.
            k (int): 1D local context kernel size.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        groups = _valid_group_count(c2)
        pad = k // 2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.conv_h = nn.Conv1d(c2, c2, k, padding=pad, groups=c2, bias=False)
        self.conv_w = nn.Conv1d(c2, c2, k, padding=pad, groups=c2, bias=False)
        self.gn_h = nn.GroupNorm(groups, c2)
        self.gn_w = nn.GroupNorm(groups, c2)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply directional local attention to suppress background responses."""
        x = self.proj(x)
        attn_h = self.act(self.gn_h(self.conv_h(x.mean(dim=3)))).unsqueeze(3)
        attn_w = self.act(self.gn_w(self.conv_w(x.mean(dim=2)))).unsqueeze(2)
        return x * attn_h * attn_w


class EUCB(nn.Module):
    """Efficient up-convolution block for enhanced neck upsampling."""

    def __init__(self, c1: int, c2: int, k: int = 3, scale: int = 2):
        """Initialize EUCB.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Depthwise convolution kernel size.
            scale (int): Upsampling scale factor.
        """
        super().__init__()
        self.scale = scale
        self.dw = nn.Sequential(
            nn.Conv2d(c1, c1, k, 1, autopad(k), groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.pw = Conv(c1, c2, 1, 1)
        self.attn = ELA(c2, c2, k=7)

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int = 4) -> torch.Tensor:
        """Shuffle channels when possible to mix depthwise groups."""
        b, c, h, w = x.shape
        if c % groups != 0:
            return x
        x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and refine features for top-down neck fusion."""
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = self._channel_shuffle(self.dw(x))
        return self.attn(self.pw(x))


class ODConv(nn.Module):
    """Omni-dimensional dynamic convolution style block with kernel, channel, and filter gates."""

    def __init__(self, c: int, kernel_num: int = 4, reduction: int = 4):
        """Initialize ODConv.

        Args:
            c (int): Input and output channels.
            kernel_num (int): Number of parallel depthwise kernel branches.
            reduction (int): Channel reduction ratio for attention generation.
        """
        super().__init__()
        hidden = max(c // reduction, 16)
        self.pre = Conv(c, c, 1, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1 + i, dilation=1 + i, groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
            for i in range(kernel_num)
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(nn.Conv2d(hidden, c, 1, bias=True), nn.Sigmoid())
        self.filter_gate = nn.Sequential(nn.Conv2d(hidden, c, 1, bias=True), nn.Sigmoid())
        self.kernel_gate = nn.Conv2d(hidden, kernel_num, 1, bias=True)
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic channel, filter, and kernel-branch modulation."""
        y = self.pre(x)
        context = self.context(y)
        y = y * self.channel_gate(context)
        kernel_weight = torch.softmax(self.kernel_gate(context), dim=1)
        branches = torch.stack([branch(y) for branch in self.branches], dim=1)
        y = (branches * kernel_weight.unsqueeze(2)).sum(dim=1)
        y = y * self.filter_gate(context)
        return self.fuse(y)


class Bottleneck_ODConv(nn.Module):
    """Bottleneck using ODConv as the dynamic spatial operator."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_ODConv with Bottleneck-compatible arguments."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.odconv = ODConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ODConv bottleneck with optional shortcut connection."""
        y = self.cv2(self.odconv(self.cv1(x)))
        return x + y if self.add else y


class PartialConv(nn.Module):
    """Partial convolution that applies spatial convolution to only part of the channels."""

    def __init__(self, c1: int, n_div: int = 4, k: int = 3):
        """Initialize PartialConv.

        Args:
            c1 (int): Input and output channels.
            n_div (int): Channel division factor. 4 means convolving one quarter of channels.
            k (int): Spatial kernel size.
        """
        super().__init__()
        self.dim_conv = max(c1 // n_div, 1)
        self.dim_untouched = c1 - self.dim_conv
        self.partial_conv = Conv(self.dim_conv, self.dim_conv, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial convolution to a channel subset and keep the rest unchanged."""
        if self.dim_untouched == 0:
            return self.partial_conv(x)
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv(x1), x2), 1)


class WTConvDown(nn.Module):
    """Haar wavelet downsampling with lightweight frequency fusion for neck bottom-up paths."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        """Initialize WTConvDown.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size after wavelet decomposition.
            s (int): Downsampling stride. Only stride 2 uses wavelet decomposition.
        """
        super().__init__()
        self.stride = s
        c_freq = c1 * 4 if s == 2 else c1
        self.freq_dw = nn.Sequential(
            nn.Conv2d(c_freq, c_freq, k, 1, autopad(k), groups=c_freq, bias=False),
            nn.BatchNorm2d(c_freq),
            nn.SiLU(inplace=True),
        )
        self.fuse = Conv(c_freq, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Haar low/high-frequency decomposition before convolution."""
        if self.stride != 2:
            return self.fuse(self.freq_dw(x))
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return self.fuse(self.freq_dw(torch.cat((ll, lh, hl, hh), 1)))


class Bottleneck_PConv(nn.Module):
    """Bottleneck that replaces full spatial convolutions with a partial-convolution branch."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, n_div: int = 4
    ):
        """Initialize a PConv bottleneck."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.pconv = PartialConv(c_, n_div=n_div, k=3)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PConv bottleneck with optional shortcut."""
        y = self.cv2(self.pconv(self.cv1(x)))
        return x + y if self.add else y


class SRULite(nn.Module):
    """Spatial reconstruction unit from SCConv, simplified for lightweight neck use."""

    def __init__(self, c1: int, groups: int = 16, gate_threshold: float = 0.5):
        """Initialize SRULite."""
        super().__init__()
        groups = min(groups, c1)
        while c1 % groups != 0:
            groups -= 1
        self.gn = nn.GroupNorm(groups, c1)
        self.gate_threshold = float(gate_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct spatially informative and less-informative responses."""
        gamma = self.gn.weight / (self.gn.weight.sum() + 1e-6)
        gate = torch.sigmoid(self.gn(x) * gamma.view(1, -1, 1, 1))
        info = gate >= self.gate_threshold
        keep = info * x
        drop = (~info) * x
        keep1, keep2 = torch.chunk(keep, 2, dim=1)
        drop1, drop2 = torch.chunk(drop, 2, dim=1)
        return torch.cat((keep1 + drop2, keep2 + drop1), dim=1)


class CRULite(nn.Module):
    """Channel reconstruction unit from SCConv with reduced grouped convolution."""

    def __init__(self, c1: int, alpha: float = 0.5, squeeze: int = 2, groups: int = 2):
        """Initialize CRULite."""
        super().__init__()
        upper = max(1, int(c1 * alpha))
        lower = c1 - upper
        if lower <= 0:
            upper, lower = c1 // 2, c1 - c1 // 2
        up_s = max(1, upper // squeeze)
        low_s = max(1, lower // squeeze)
        groups = min(groups, up_s)
        while up_s % groups != 0:
            groups -= 1

        self.upper = upper
        self.lower = lower
        self.squeeze_up = nn.Conv2d(upper, up_s, 1, bias=False)
        self.squeeze_low = nn.Conv2d(lower, low_s, 1, bias=False)
        self.group_conv = nn.Conv2d(up_s, c1, 3, 1, 1, groups=groups, bias=False)
        self.point_up = nn.Conv2d(up_s, c1, 1, bias=False)
        self.point_low = nn.Conv2d(low_s, c1 - low_s, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct channel responses and suppress redundancy."""
        up, low = torch.split(x, [self.upper, self.lower], dim=1)
        up = self.squeeze_up(up)
        low = self.squeeze_low(low)
        y1 = self.group_conv(up) + self.point_up(up)
        y2 = torch.cat((self.point_low(low), low), dim=1)
        y = torch.cat((y1, y2), dim=1)
        y = y * F.softmax(self.pool(y), dim=1)
        y1, y2 = torch.chunk(y, 2, dim=1)
        return y1 + y2


class SCConvLite(nn.Module):
    """Spatial and channel reconstruction convolution for redundant neck features."""

    def __init__(self, c1: int, groups: int = 16, gate_threshold: float = 0.5, alpha: float = 0.5, squeeze: int = 2):
        """Initialize SCConvLite."""
        super().__init__()
        self.sru = SRULite(c1, groups=groups, gate_threshold=gate_threshold)
        self.cru = CRULite(c1, alpha=alpha, squeeze=squeeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial and channel reconstruction."""
        return self.cru(self.sru(x))


class Bottleneck_SCConv(nn.Module):
    """Bottleneck that uses SCConvLite for redundancy-aware neck feature refinement."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, groups: int = 16
    ):
        """Initialize Bottleneck_SCConv."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.scconv = SCConvLite(c_, groups=groups)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SCConv bottleneck with optional shortcut."""
        y = self.cv2(self.scconv(self.cv1(x)))
        return x + y if self.add else y


class InceptionDWConvLite(nn.Module):
    """InceptionNeXt-style split depthwise convolution for lightweight spatial mixing."""

    def __init__(self, c1: int, square_k: int = 3, band_k: int = 11, branch_ratio: float = 0.125):
        """Initialize InceptionDWConvLite."""
        super().__init__()
        gc = max(1, int(c1 * branch_ratio))
        gc = min(gc, c1 // 4) if c1 >= 4 else 1
        self.split_indexes = (c1 - 3 * gc, gc, gc, gc)
        self.dw_square = nn.Conv2d(gc, gc, square_k, 1, square_k // 2, groups=gc, bias=False)
        self.dw_horizontal = nn.Conv2d(gc, gc, (1, band_k), 1, (0, band_k // 2), groups=gc, bias=False)
        self.dw_vertical = nn.Conv2d(gc, gc, (band_k, 1), 1, (band_k // 2, 0), groups=gc, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply split depthwise square, horizontal, and vertical convolutions."""
        x_id, x_square, x_h, x_w = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat((x_id, self.dw_square(x_square), self.dw_horizontal(x_h), self.dw_vertical(x_w)), dim=1)
        return self.act(self.bn(x))


class Bottleneck_InceptionNeXtLite(nn.Module):
    """Bottleneck using InceptionNeXt-lite spatial mixing instead of full 3x3 convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_InceptionNeXtLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.mix = InceptionDWConvLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply InceptionNeXt-lite bottleneck with optional shortcut."""
        y = self.cv2(self.mix(self.cv1(x)))
        return x + y if self.add else y


class StarBlockLite(nn.Module):
    """StarNet-style lightweight spatial block with element-wise feature interaction."""

    def __init__(self, c1: int):
        """Initialize StarBlockLite."""
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.f1 = nn.Conv2d(c1, c1, 1, bias=False)
        self.f2 = nn.Conv2d(c1, c1, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply StarNet-style multiplicative interaction."""
        y = self.dw(x)
        return self.proj(self.act(self.f1(y)) * self.f2(y))


class Bottleneck_StarBlockLite(nn.Module):
    """Bottleneck using StarBlock-lite for lightweight nonlinear spatial interaction."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_StarBlockLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.star = StarBlockLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply StarBlock-lite bottleneck with optional shortcut."""
        y = self.cv2(self.star(self.cv1(x)))
        return x + y if self.add else y


class MobileOneBlockLite(nn.Module):
    """MobileOne-style lightweight re-parameterizable local convolution block."""

    def __init__(self, c1: int, branches: int = 2):
        """Initialize MobileOneBlockLite."""
        super().__init__()
        self.dw_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )
            for _ in range(branches)
        )
        self.dw_1x1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1, 0, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.identity = nn.BatchNorm2d(c1)
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-branch local convolution and pointwise mixing."""
        y = self.identity(x) + self.dw_1x1(x)
        for branch in self.dw_branches:
            y = y + branch(x)
        return self.act(self.pw(self.act(y)))


class Bottleneck_MobileOneLite(nn.Module):
    """Bottleneck using MobileOne-lite local convolution for P4 neck refinement."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_MobileOneLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.mobileone = MobileOneBlockLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MobileOne-lite bottleneck with optional shortcut."""
        y = self.cv2(self.mobileone(self.cv1(x)))
        return x + y if self.add else y


class RepMixerLite(nn.Module):
    """FastViT RepMixer-style local token mixing without attention or global pooling."""

    def __init__(self, c1: int, k: int = 3):
        """Initialize RepMixerLite."""
        super().__init__()
        self.norm = nn.BatchNorm2d(c1)
        self.mixer = nn.Sequential(
            nn.Conv2d(c1, c1, k, 1, k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.scale = nn.Parameter(torch.zeros(1, c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bounded residual local token mixing."""
        y = self.norm(x)
        return x + self.scale * (self.mixer(y) - y)


class Bottleneck_RepMixerLite(nn.Module):
    """Bottleneck using RepMixer-lite local token mixing for P4 neck refinement."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_RepMixerLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.mixer = RepMixerLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RepMixer-lite bottleneck with optional shortcut."""
        y = self.cv2(self.mixer(self.cv1(x)))
        return x + y if self.add else y


class RepHELANLite(nn.Module):
    """RepHELAN-inspired heterogeneous local convolution for lightweight neck refinement."""

    def __init__(self, c1: int):
        """Initialize RepHELANLite with small and medium depthwise branches."""
        super().__init__()
        self.dw3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(c1, c1, 5, 1, 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.identity = nn.BatchNorm2d(c1)
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix heterogeneous local kernels with a lightweight projection."""
        y = self.dw3(x) + self.dw5(x) + self.identity(x)
        return self.act(self.proj(self.act(y)))


class Bottleneck_RepHELANLite(nn.Module):
    """CSP bottleneck using RepHELAN-lite heterogeneous convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_RepHELANLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.rephelan = RepHELANLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RepHELAN-lite bottleneck with optional shortcut."""
        y = self.cv2(self.rephelan(self.cv1(x)))
        return x + y if self.add else y


class RepHMSLite(nn.Module):
    """RepHMS-inspired directional multi-scale convolution for elongated structures."""

    def __init__(self, c1: int):
        """Initialize RepHMSLite with square, horizontal, and vertical depthwise branches."""
        super().__init__()
        self.dw3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.dwh = nn.Sequential(
            nn.Conv2d(c1, c1, (1, 5), 1, (0, 2), groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.dwv = nn.Sequential(
            nn.Conv2d(c1, c1, (5, 1), 1, (2, 0), groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.identity = nn.BatchNorm2d(c1)
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix square and directional kernels for line-like neck features."""
        y = self.dw3(x) + self.dwh(x) + self.dwv(x) + self.identity(x)
        return self.act(self.proj(self.act(y)))


class Bottleneck_RepHMSLite(nn.Module):
    """CSP bottleneck using RepHMS-lite directional multi-scale convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_RepHMSLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.rephms = RepHMSLite(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RepHMS-lite bottleneck with optional shortcut."""
        y = self.cv2(self.rephms(self.cv1(x)))
        return x + y if self.add else y


class RFAConv(nn.Module):
    """Receptive-field attention convolution.

    RFAConv assigns adaptive weights to each position inside a convolutional
    receptive field, strengthening local structure modeling for elongated and
    weak-texture targets.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int = 3, stride: int = 1):
        """Initialize RFAConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            kernel_size (int): Receptive field size.
            stride (int): Stride used when sampling the original feature map.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(c1, c1 * kernel_size**2, 1, groups=c1, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(
                c1,
                c1 * kernel_size**2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=c1,
                bias=False,
            ),
            nn.BatchNorm2d(c1 * kernel_size**2),
            nn.SiLU(inplace=True),
        )
        self.conv = Conv(c1, c2, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply receptive-field attention convolution."""
        b, c, _, _ = x.shape
        k = self.kernel_size
        weight = self.get_weight(x)
        _, _, h, w = weight.shape
        weight = weight.view(b, c, k * k, h, w).softmax(dim=2)
        feature = self.generate_feature(x).view(b, c, k * k, h, w)
        feature = feature * weight
        feature = feature.view(b, c, k, k, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * k, w * k)
        return self.conv(feature)


class DSConv(nn.Module):
    """Dynamic snake convolution for local elongated-structure modeling.

    This implementation samples horizontal and vertical snake-like receptive
    fields with dynamic offsets, then fuses them with a standard 3x3 branch.
    It keeps the feature map size unchanged for use inside YOLO bottlenecks.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int = 3, extend_scope: float = 1.0):
        """Initialize DSConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            kernel_size (int): Number of sampling points along each snake axis.
            extend_scope (float): Offset range multiplier.
        """
        super().__init__()
        assert kernel_size % 2 == 1, "DSConv kernel_size must be odd"
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.offset = nn.Sequential(
            nn.Conv2d(c1, 2 * kernel_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * kernel_size),
            nn.Tanh(),
        )
        self.conv = Conv(c1, c2, 3, 1)
        self.conv_h = Conv(c1 * kernel_size, c2, 1, 1)
        self.conv_v = Conv(c1 * kernel_size, c2, 1, 1)
        self.fuse = Conv(c2, c2, 1, 1)

    def _snake_offset(self, offset: torch.Tensor) -> torch.Tensor:
        """Accumulate offsets from the center point to preserve snake continuity."""
        center = self.kernel_size // 2
        left = torch.flip(torch.cumsum(torch.flip(offset[:, :center], dims=[1]), dim=1), dims=[1])
        middle = torch.zeros_like(offset[:, center : center + 1])
        right = torch.cumsum(offset[:, center + 1 :], dim=1)
        snake = torch.cat((left, middle, right), dim=1)
        return snake * self.extend_scope

    @staticmethod
    def _normalize_grid(x: torch.Tensor, y: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert absolute coordinates to the normalized grid_sample range."""
        x = 2.0 * x / (w - 1) - 1.0 if w > 1 else torch.zeros_like(x)
        y = 2.0 * y / (h - 1) - 1.0 if h > 1 else torch.zeros_like(y)
        return torch.stack((x, y), dim=-1)

    def _sample_axis(self, x: torch.Tensor, snake_offset: torch.Tensor, axis: str) -> torch.Tensor:
        """Sample a horizontal or vertical snake receptive field."""
        _, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        y_base, x_base = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        center = self.kernel_size // 2
        features = []
        for i in range(self.kernel_size):
            delta = i - center
            x_ref = x_base.unsqueeze(0).expand_as(snake_offset[:, i])
            y_ref = y_base.unsqueeze(0).expand_as(snake_offset[:, i])
            if axis == "h":
                grid = self._normalize_grid(x_ref + delta, y_ref + snake_offset[:, i], h, w)
            else:
                grid = self._normalize_grid(x_ref + snake_offset[:, i], y_ref + delta, h, w)
            features.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True))
        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic snake convolution."""
        offset_h, offset_v = self.offset(x).chunk(2, dim=1)
        h_feat = self.conv_h(self._sample_axis(x, self._snake_offset(offset_h), "h"))
        v_feat = self.conv_v(self._sample_axis(x, self._snake_offset(offset_v), "v"))
        return self.fuse(self.conv(x) + h_feat + v_feat)


class DDFMLite(nn.Module):
    """Lightweight directional detail feature module for weak edges and thin objects."""

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 4):
        """Initialize DDFM-lite.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
            reduction (int): Channel reduction ratio for the detail gate.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(c2 // reduction, 8)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.local = Conv(c2, c2, 3, 1)
        self.h_detail = nn.Sequential(
            nn.Conv2d(c2, c2, (1, 3), 1, (0, 1), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.v_detail = nn.Sequential(
            nn.Conv2d(c2, c2, (3, 1), 1, (1, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance local features with gated horizontal and vertical detail branches."""
        y = self.proj(x)
        local = self.local(y)
        detail = self.h_detail(local) + self.v_detail(local)
        return self.fuse(local + detail * self.gate(detail))


class SPDConv(nn.Module):
    """Space-to-depth convolution for detail-preserving stride-2 downsampling."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        """Initialize SPDConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size after space-to-depth rearrangement.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange neighboring pixels into channels, then convolve."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class StripEnhance(nn.Module):
    """Residual strip feature enhancement for thin and elongated objects."""

    def __init__(self, c1: int, k: int = 7, reduction: int = 4, init_alpha: float = 0.0):
        """Initialize StripEnhance.

        Args:
            c1 (int): Input and output channels.
            k (int): Strip kernel size for horizontal and vertical depthwise branches.
            reduction (int): Channel reduction ratio.
            init_alpha (float): Initial residual scale. Zero starts as an identity mapping.
        """
        super().__init__()
        hidden = max(c1 // reduction, 16)
        pad = k // 2
        self.reduce = Conv(c1, hidden, 1, 1)
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.h_strip = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, k), 1, (0, pad), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.v_strip = nn.Sequential(
            nn.Conv2d(hidden, hidden, (k, 1), 1, (pad, 0), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.expand = Conv(hidden, c1, 1, 1, act=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance line-like structures while preserving the original feature path."""
        y = self.reduce(x)
        y = self.local(y) + self.h_strip(y) + self.v_strip(y)
        return x + self.alpha * self.expand(y)


class DWRLite(nn.Module):
    """Lightweight dilation-wise residual block for local context enhancement."""

    def __init__(self, c1: int, reduction: int = 4, init_alpha: float = 0.0, max_scale: float = 0.1):
        """Initialize DWRLite.

        Args:
            c1 (int): Input and output channels.
            reduction (int): Channel reduction ratio for the residual branch.
            init_alpha (float): Initial residual scale parameter.
            max_scale (float): Maximum residual strength after tanh scaling.
        """
        super().__init__()
        hidden = max(c1 // reduction, 16)
        self.reduce = Conv(c1, hidden, 1, 1)
        self.dw1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, dilation=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 3, dilation=3, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.expand = Conv(hidden * 3, c1, 1, 1, act=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.max_scale = float(max_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bounded residual multi-dilation enhancement."""
        y = self.reduce(x)
        y = self.expand(torch.cat((self.dw1(y), self.dw2(y), self.dw3(y)), 1))
        return x + self.max_scale * torch.tanh(self.alpha) * y


class DASI2(nn.Module):
    """Two-input dimension-aware selective integration for lightweight neck fusion."""

    def __init__(self, c1: list[int], c2: int, chunks: int = 4, refine: bool = False):
        """Initialize DASI2.

        Args:
            c1 (list[int]): Input channels of [top-down, lateral] features.
            c2 (int): Output channels after selective fusion.
            chunks (int): Number of channel chunks used for selective integration.
            refine (bool): Whether to use an extra 1x1 refinement after fusion.
        """
        super().__init__()
        if not isinstance(c1, (list, tuple)) or len(c1) != 2:
            raise ValueError("DASI2 expects exactly two input feature maps: [top-down, lateral].")
        self.chunks = max(1, min(int(chunks), c2))
        while c2 % self.chunks != 0:
            self.chunks -= 1

        self.semantic_proj = nn.Identity() if c1[0] == c2 else Conv(c1[0], c2, 1, 1, act=False)
        self.detail_proj = nn.Identity() if c1[1] == c2 else Conv(c1[1], c2, 1, 1, act=False)
        self.refine = Conv(c2, c2, 1, 1, act=False) if refine else nn.Identity()
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse top-down semantic and lateral detail features with chunk-wise gates."""
        semantic, detail = x
        semantic = self.semantic_proj(semantic)
        detail = self.detail_proj(detail)

        semantic_chunks = torch.chunk(semantic, self.chunks, dim=1)
        detail_chunks = torch.chunk(detail, self.chunks, dim=1)
        fused = []
        for sem, det in zip(semantic_chunks, detail_chunks):
            gate = torch.sigmoid(det)
            fused.append(gate * det + (1.0 - gate) * sem)

        return self.act(self.bn(self.refine(torch.cat(fused, dim=1)) + detail))


class FusionFactor(nn.Module):
    """Learnable scalar gate for controlling top-down feature fusion strength."""

    def __init__(self, c1: int, init: float = 0.75):
        """Initialize FusionFactor.

        Args:
            c1 (int): Input and output channels.
            init (float): Initial gate value in (0, 1).
        """
        super().__init__()
        init_tensor = torch.tensor(float(init)).clamp(1e-4, 1.0 - 1e-4)
        self.logit = nn.Parameter(torch.logit(init_tensor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the incoming feature map with a bounded learnable factor."""
        return x * torch.sigmoid(self.logit)


class LineRefineLite(nn.Module):
    """Identity-preserving P4 refinement for thin line-like objects in the neck."""

    def __init__(self, c1: int, k: int = 9, reduction: int = 8, max_scale: float = 0.2, init_alpha: float = 0.0):
        """Initialize LineRefineLite.

        Args:
            c1 (int): Input and output channels.
            k (int): Strip kernel size for horizontal and vertical depthwise branches.
            reduction (int): Channel reduction ratio for the residual branch.
            max_scale (float): Maximum residual scale after tanh bounding.
            init_alpha (float): Initial residual scale. Zero starts as an identity mapping.
        """
        super().__init__()
        hidden = max(c1 // reduction, 16)
        pad = k // 2
        self.reduce = Conv(c1, hidden, 1, 1)
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.h_line = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, k), 1, (0, pad), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.v_line = nn.Sequential(
            nn.Conv2d(hidden, hidden, (k, 1), 1, (pad, 0), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        self.expand = Conv(hidden, c1, 1, 1, act=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.max_scale = float(max_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bounded residual line refinement while preserving the original feature."""
        y = self.reduce(x)
        y = self.local(y) + self.h_line(y) + self.v_line(y)
        y = self.expand(y * self.gate(y))
        return x + self.max_scale * torch.tanh(self.alpha) * y


class DySample(nn.Module):
    """Dynamic upsampling by learning sampling offsets for neck feature alignment."""

    def __init__(self, c1: int, scale: int = 2, groups: int = 4, dyscope: bool = False):
        """Initialize DySample.

        Args:
            c1 (int): Input and output channels.
            scale (int): Upsampling scale factor.
            groups (int): Sampling groups.
            dyscope (bool): Whether to use a dynamic scope gate for offsets.
        """
        super().__init__()
        self.scale = scale
        self.groups = min(groups, c1)
        while c1 % self.groups != 0:
            self.groups -= 1
        self.offset = nn.Conv2d(c1, 2 * self.groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)
        self.scope = nn.Conv2d(c1, 2 * self.groups * scale * scale, 1) if dyscope else None
        if self.scope is not None:
            nn.init.constant_(self.scope.weight, 0)
            nn.init.constant_(self.scope.bias, 0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        """Create initial regular sampling offsets within each upsampled cell."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        yy, xx = torch.meshgrid(h, h, indexing="ij")
        pos = torch.stack((xx, yy), 0).reshape(2, -1, 1, 1).repeat(1, self.groups, 1, 1)
        return pos.reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample input features with learned offsets."""
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        yy, xx = torch.meshgrid(coords_h, coords_w, indexing="ij")
        coords = torch.stack((xx, yy), 0).unsqueeze(0).unsqueeze(2)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w).permute(0, 2, 3, 4, 1).contiguous()
        coords = coords.flatten(0, 1)
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic sampling upsampling."""
        if self.scope is None:
            offset = self.offset(x) * 0.25 + self.init_pos
        else:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        return self._sample(x, offset)


class SAPAUpsample(nn.Module):
    """SAPA-G similarity-aware guided upsampling for an adjacent feature-pyramid pair.

    The first input is the high-resolution encoder/lateral feature and the second
    input is the low-resolution decoder feature to be upsampled. This follows the
    gated-query SAPA formulation while providing a portable PyTorch fallback for
    the paper's optional custom CUDA operators.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int],
        scale: int = 2,
        kernel_size: int = 5,
        embedding_dim: int = 32,
        q_mode: str = "gate",
    ):
        """Initialize encoder-guided similarity-aware point affiliation."""
        super().__init__()
        if len(channels) != 2:
            raise ValueError("SAPAUpsample expects [encoder_channels, decoder_channels]")
        encoder_channels, decoder_channels = (int(c) for c in channels)
        if scale < 1:
            raise ValueError(f"SAPA scale must be positive, got {scale}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"SAPA kernel_size must be a positive odd number, got {kernel_size}")

        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        self.embedding_dim = max(int(embedding_dim), 8)
        self.q_mode = str(q_mode).lower()
        if self.q_mode not in {"basic", "gate"}:
            raise ValueError(f"SAPA q_mode must be 'basic' or 'gate', got {q_mode}")

        self.norm_encoder = nn.LayerNorm(encoder_channels)
        self.norm_decoder = nn.LayerNorm(decoder_channels)
        if self.q_mode == "gate":
            self.q_encoder = nn.Linear(encoder_channels, self.embedding_dim, bias=True)
            self.q_decoder = nn.Linear(decoder_channels, self.embedding_dim, bias=True)
            self.query_gate = nn.Linear(decoder_channels, 1, bias=True)
        else:
            self.q = nn.Linear(encoder_channels, self.embedding_dim, bias=True)
        self.k = nn.Linear(decoder_channels, self.embedding_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        """Use the initialization from the official SAPA implementation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _nearest_channels_last(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Nearest-neighbor resize an NHWC tensor and keep channels last."""
        return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="nearest").permute(0, 2, 3, 1)

    def _attention_pytorch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply SAPA point affiliation using portable PyTorch tensor operators."""
        b, high_h, high_w, _ = q.shape
        _, low_h, low_w, channels = v.shape
        expected_size = (low_h * self.scale, low_w * self.scale)
        if (high_h, high_w) != expected_size:
            raise ValueError(
                f"SAPA encoder size {(high_h, high_w)} must equal decoder size {(low_h, low_w)} "
                f"times scale={self.scale}"
            )

        q_subpixels = F.pixel_unshuffle(q.permute(0, 3, 1, 2), self.scale).reshape(
            b, self.embedding_dim, self.scale**2, low_h, low_w
        )
        key_patches = F.unfold(
            k.permute(0, 3, 1, 2),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).reshape(b, self.embedding_dim, self.kernel_size**2, low_h, low_w)
        logits = torch.einsum("bdphw,bdkhw->bpkhw", q_subpixels, key_patches)
        attention = logits.softmax(dim=2)

        value_patches = F.unfold(
            v.permute(0, 3, 1, 2),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).reshape(b, channels, self.kernel_size**2, low_h, low_w)
        output = torch.einsum("bpkhw,bckhw->bcphw", attention, value_patches)
        return F.pixel_shuffle(output.reshape(b, channels * self.scale**2, low_h, low_w), self.scale)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Upsample the decoder feature using the high-resolution encoder feature as guidance."""
        encoder, decoder = x
        encoder_nhwc = encoder.permute(0, 2, 3, 1).contiguous()
        decoder_nhwc = decoder.permute(0, 2, 3, 1).contiguous()
        encoder_norm = self.norm_encoder(encoder_nhwc)
        decoder_norm = self.norm_decoder(decoder_nhwc)

        if self.q_mode == "gate":
            target_size = encoder.shape[-2:]
            gate = self._nearest_channels_last(self.query_gate(decoder_norm).sigmoid(), target_size)
            decoder_high = self._nearest_channels_last(decoder_nhwc, target_size)
            q = gate * self.q_encoder(encoder_norm) + (1.0 - gate) * self.q_decoder(decoder_high)
        else:
            q = self.q(encoder_norm)
        k = self.k(decoder_norm)
        return self._attention_pytorch(q, k, decoder_nhwc)


class MSBlock(nn.Module):
    """Multi-scale depthwise convolution block inspired by YOLO-MS."""

    def __init__(self, c1: int, c2: int | None = None, kernels: tuple[int, ...] = (3, 5, 7)):
        """Initialize MSBlock.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
            kernels (tuple[int, ...]): Depthwise kernel sizes for multi-scale branches.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        splits = [c2 // len(kernels) for _ in kernels]
        splits[-1] += c2 - sum(splits)
        self.splits = splits
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
            for c, k in zip(splits, kernels)
        )
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and fuse multi-scale local features."""
        x = self.proj(x)
        y = [branch(part) for branch, part in zip(self.branches, torch.split(x, self.splits, dim=1))]
        return self.fuse(torch.cat(y, dim=1)) + x


class _MSBlockLiteLayer(nn.Module):
    """Light inverted depthwise layer used inside a hierarchical MSBlock branch."""

    def __init__(self, channels: int, kernel_size: int, expansion: float = 2.0):
        """Initialize an expansion, depthwise filtering, and projection sequence."""
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.expand = Conv(channels, hidden, 1, 1)
        self.depthwise = Conv(hidden, hidden, kernel_size, 1, g=hidden)
        self.project = Conv(hidden, channels, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight spatial refinement with a residual connection."""
        return x + self.project(self.depthwise(self.expand(x)))


class HierarchicalMSBlockLite(nn.Module):
    """Lite MSBlock with the hierarchical branch communication used by YOLO-MS.

    Channels are split into equal groups. Starting from the second group, each
    branch first receives the preceding branch output and then applies its own
    spatial kernel. This differs from the older parallel-only ``MSBlock`` in
    this project while retaining a compact single-layer branch design.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        kernels: tuple[int, ...] = (1, 3, 5),
        expansion: float = 2.0,
        shortcut: bool = True,
    ):
        """Initialize hierarchical multi-scale branches and feature projections."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        if len(kernels) < 2:
            raise ValueError("HierarchicalMSBlockLite requires at least two kernel branches")
        branch_channels = math.ceil(c2 / len(kernels))
        inner_channels = branch_channels * len(kernels)
        self.branch_channels = branch_channels
        self.in_proj = Conv(c1, inner_channels, 1, 1)
        self.branches = nn.ModuleList(
            nn.Identity() if kernel_size == 1 else _MSBlockLiteLayer(branch_channels, kernel_size, expansion)
            for kernel_size in kernels
        )
        self.out_proj = Conv(inner_channels, c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate features progressively across increasing receptive-field branches."""
        parts = torch.split(self.in_proj(x), self.branch_channels, dim=1)
        outputs: list[torch.Tensor] = []
        for i, (part, branch) in enumerate(zip(parts, self.branches)):
            if i:
                part = part + outputs[-1]
            outputs.append(branch(part))
        y = self.out_proj(torch.cat(outputs, dim=1))
        return x + y if self.shortcut else y


class GCBlock(nn.Module):
    """Global context block for lightweight context-guided feature recalibration."""

    def __init__(self, c1: int, c2: int | None = None, ratio: float = 0.25):
        """Initialize GCBlock.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
            ratio (float): Bottleneck ratio for context transform.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(int(c2 * ratio), 8)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.conv_mask = nn.Conv2d(c2, 1, 1)
        self.channel_add = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=False),
        )

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling to obtain global context."""
        b, c, h, w = x.shape
        input_x = x.view(b, c, h * w).unsqueeze(1)
        context_mask = self.conv_mask(x).view(b, 1, h * w)
        context_mask = torch.softmax(context_mask, dim=2).unsqueeze(-1)
        return torch.matmul(input_x, context_mask).view(b, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recalibrate features with global context while preserving residual content."""
        x = self.proj(x)
        return x + self.channel_add(self.spatial_pool(x))


class GRNBlock(nn.Module):
    """ConvNeXt V2 Global Response Normalization for NCHW backbone features."""

    def __init__(self, c1: int, c2: int | None = None, eps: float = 1e-6):
        """Initialize a channel-preserving GRN block with an optional input projection."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spatial response magnitudes relative to the channel mean."""
        x = self.proj(x)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class MobileViTv2SSA(nn.Module):
    """Residual MobileViTv2 separable self-attention adapted to NCHW backbone features."""

    def __init__(self, c1: int, c2: int | None = None):
        """Initialize linear-complexity spatial attention with identity-safe residual scaling."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.qkv_proj = nn.Conv2d(c2, 1 + 2 * c2, 1, bias=True)
        self.out_proj = nn.Conv2d(c2, c2, 1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a global key context and modulate every spatial value token with it."""
        x = self.proj(x)
        query, key, value = torch.split(self.qkv_proj(x), (1, x.shape[1], x.shape[1]), dim=1)
        context_scores = torch.softmax(query.flatten(2), dim=-1).view_as(query)
        context_vector = (key * context_scores).sum(dim=(2, 3), keepdim=True)
        attended = self.out_proj(F.relu(value, inplace=False) * context_vector)
        return x + self.gamma * attended


class MCAContext(nn.Module):
    """Moment channel attention using global feature statistics for recalibration."""

    def __init__(self, c1: int, reduction: int = 16):
        """Initialize MCAContext."""
        super().__init__()
        hidden = max(8, c1 // reduction)
        self.excite = nn.Sequential(
            nn.Conv2d(c1 * 3, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recalibrate channels using mean, variance, and third central moment."""
        mean = x.mean((2, 3), keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean((2, 3), keepdim=True)
        skew = centered.pow(3).mean((2, 3), keepdim=True)
        moments = torch.cat((mean, var, skew), dim=1)
        return x * self.excite(moments)


class GCConv(nn.Module):
    """Convolution followed by GCBlock for context-guided downsampling or feature projection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
        ratio: float = 0.25,
    ):
        """Initialize GCConv with Conv-compatible arguments."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.gc = GCBlock(c2, c2, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution then global-context recalibration."""
        return self.gc(self.conv(x))


class Bottleneck_RFAConv(nn.Module):
    """Bottleneck that replaces the second 3x3 convolution with RFAConv."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_RFAConv."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = RFAConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RFAConv bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DSConv(nn.Module):
    """Bottleneck that replaces the second 3x3 convolution with DSConv."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_DSConv."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DSConv(c_, c2, 3, 1.0)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DSConv bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DDFM(nn.Module):
    """Bottleneck that replaces the second 3x3 convolution with DDFM-lite."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize DDFM-lite bottleneck."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DDFMLite(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DDFM-lite bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_MSBlock(nn.Module):
    """Bottleneck that replaces the second 3x3 convolution with MSBlock."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize MSBlock bottleneck."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = MSBlock(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MSBlock bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DCBLite(nn.Module):
    """Depthwise inverted residual detail block for lightweight local feature extraction."""

    def __init__(self, c1: int, c2: int | None = None, expansion: float = 2.0, k: int = 3):
        """Initialize DCBLite with MobileNet-style expansion and depthwise filtering."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(c1, int(c1 * expansion))
        self.block = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            nn.Conv2d(hidden, hidden, k, 1, autopad(k), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            Conv(hidden, c2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight inverted residual detail extraction."""
        return self.block(x)


class EAConvLite(nn.Module):
    """Efficient attention convolution with depthwise local mixing and channel-spatial gates."""

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 8):
        """Initialize EAConvLite."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(8, c2 // reduction)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient local attention while preserving a residual path."""
        x = self.proj(x)
        y = self.dw(x)
        y = y * self.channel_gate(y) * self.spatial_gate(y)
        return x + self.fuse(y)


class CrossConvLite(nn.Module):
    """Lightweight cross-shaped depthwise convolution for directional feature enhancement."""

    def __init__(self, c1: int, c2: int | None = None, k: int = 5):
        """Initialize CrossConvLite with horizontal and vertical depthwise branches."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.horizontal = nn.Sequential(
            nn.Conv2d(c2, c2, (1, k), 1, (0, k // 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(c2, c2, (k, 1), 1, (k // 2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix horizontal and vertical directional cues with a residual path."""
        x = self.proj(x)
        return x + self.fuse(self.horizontal(x) + self.vertical(x))


class Bottleneck_DCBLite(nn.Module):
    """CSP bottleneck using DCBLite for lightweight weak-texture detail extraction."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_DCBLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCBLite(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCBLite bottleneck with optional shortcut."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class Bottleneck_EAConvLite(nn.Module):
    """CSP bottleneck using EAConvLite for confidence-friendly local attention."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_EAConvLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = EAConvLite(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EAConvLite bottleneck with optional shortcut."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class Bottleneck_CrossConvLite(nn.Module):
    """CSP bottleneck using CrossConvLite for elongated directional structures."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize Bottleneck_CrossConvLite."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = CrossConvLite(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CrossConvLite bottleneck with optional shortcut."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class D3Block(nn.Module):
    """YOLO-ULM D3 block with cascaded depthwise and large-kernel convolutions."""

    def __init__(self, c: int, shortcut: bool = True, k: int = 7):
        """Initialize a D3 block.

        The spatial path follows the paper's three-depthwise-convolution design:
        pointwise + 3x3 DWConv, large-kernel DWConv, then pointwise + 3x3
        DWConv. The residual is enabled when requested.
        """
        super().__init__()
        if k < 3 or k % 2 == 0:
            raise ValueError(f"D3Block kernel size must be odd and >= 3, got {k}")
        self.pw1 = Conv(c, c, 1, 1)
        self.dw1 = Conv(c, c, 3, 1, g=c)
        self.dw_large = Conv(c, c, k, 1, g=c)
        self.pw2 = Conv(c, c, 1, 1)
        self.dw2 = Conv(c, c, 3, 1, g=c)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cascaded local-global-local feature extraction."""
        y = self.dw2(self.pw2(self.dw_large(self.dw1(self.pw1(x)))))
        return x + y if self.add else y


class D3C2f(C2f):
    """C2f feature aggregation using YOLO-ULM D3 blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        k: int = 7,
        e: float = 0.5,
    ):
        """Initialize D3C2f with configurable large-kernel size."""
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=1, e=e)
        self.m = nn.ModuleList(D3Block(self.c, shortcut=shortcut, k=k) for _ in range(n))




class ContextAnchorAttention(nn.Module):
    """Light context-anchor attention inspired by recent poly-kernel neck designs."""

    def __init__(self, c: int, k: int = 11):
        super().__init__()
        hidden = max(c // 4, 16)
        pad = k // 2
        self.pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.reduce = Conv(c, hidden, 1, 1)
        self.h_conv = nn.Conv2d(hidden, hidden, (1, k), 1, (0, pad), groups=hidden, bias=False)
        self.v_conv = nn.Conv2d(hidden, hidden, (k, 1), 1, (pad, 0), groups=hidden, bias=False)
        self.expand = nn.Conv2d(hidden, c, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.reduce(y)
        y = self.h_conv(y)
        y = self.v_conv(y)
        return self.act(self.expand(y))


class PolyKernelInceptionBlock(nn.Module):
    """Parallel depthwise poly-kernel block for multi-scale neck refinement."""

    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.shortcut = shortcut
        self.pre = Conv(c, c, 1, 1)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw7 = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c * 3)
        self.fuse = Conv(c * 3, c, 1, 1)
        self.attn = ContextAnchorAttention(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = torch.cat((self.dw3(y), self.dw5(y), self.dw7(y)), 1)
        y = self.fuse(self.bn(y))
        y = y * self.attn(y)
        return x + y if self.shortcut else y


class C3k2PKI(C3k2):
    """C3k2 neck variant with lightweight poly-kernel inception blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(PolyKernelInceptionBlock(self.c, shortcut) for _ in range(n))

class PConvFasterC3k2(C3k2):
    """C3k2 variant using partial-convolution bottlenecks for lightweight neck fusion."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        n_div: int = 4,
    ):
        """Initialize PConvFasterC3k2 with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_PConv(self.c, self.c, shortcut, g, e=1.0, n_div=n_div)
            for _ in range(n)
        )


class C3k2_SCConv(C3k2):
    """C3k2 variant using SCConvLite bottlenecks for redundancy-aware neck fusion."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        groups: int = 16,
    ):
        """Initialize C3k2_SCConv with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_SCConv(self.c, self.c, shortcut, g, e=1.0, groups=groups)
            for _ in range(n)
        )


class C3k2_InceptionNeXtLite(C3k2):
    """C3k2 variant using InceptionNeXt-lite bottlenecks for bottom-up neck spatial mixing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_InceptionNeXtLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_InceptionNeXtLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_StarBlockLite(C3k2):
    """C3k2 variant using StarBlock-lite bottlenecks for bottom-up neck feature interaction."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_StarBlockLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_StarBlockLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_RepHELANLite(C3k2):
    """C3k2 variant using RepHELAN-lite bottlenecks for P4 neck enhancement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_RepHELANLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_RepHELANLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_RepHMSLite(C3k2):
    """C3k2 variant using RepHMS-lite bottlenecks for P4 directional neck enhancement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_RepHMSLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_RepHMSLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_MobileOneLite(C3k2):
    """C3k2 variant using MobileOne-lite bottlenecks for P4-only neck refinement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_MobileOneLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_MobileOneLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_RepMixerLite(C3k2):
    """C3k2 variant using RepMixer-lite bottlenecks for P4-only neck refinement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_RepMixerLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_RepMixerLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_DDFM(C3k2):
    """C3k2 variant using DDFM-lite bottlenecks for weak-boundary and directional detail modeling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_DDFM with the same arguments as C3k2."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DDFM(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_DDFM(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_GhostV2DFC(C3k2):
    """C3k2 variant with GhostNetV2 DFC bottlenecks for lightweight spatial context."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize a C3k2-compatible GhostNetV2 DFC block."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_GhostV2DFC(self.c, self.c, 2, shortcut, g)
            if c3k
            else GhostBottleneckV2DFC(self.c, self.c, shortcut)
            for _ in range(n)
        )


class C3k2_MSBlockLite(C3k2):
    """C3k2 variant using hierarchical lightweight MSBlock feature refinement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize a C3k2-compatible hierarchical MSBlock-Lite block."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_MSBlockLite(self.c, self.c, 2, shortcut, g)
            if c3k
            else HierarchicalMSBlockLite(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )


class C3k2_MSBlock(C3k2):
    """C3k2 variant using MSBlock bottlenecks for multi-scale local feature modeling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_MSBlock with the same arguments as C3k2."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_MSBlock(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_MSBlock(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_DCBLite(C3k2):
    """C3k2 variant using DCBLite bottlenecks for lightweight local detail enhancement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_DCBLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_DCBLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_EAConvLite(C3k2):
    """C3k2 variant using efficient attention convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_EAConvLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_EAConvLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_FADC(C3k2):
    """C3k2 variant using frequency-adaptive dilated convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_FADC with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_FADC(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_FDConv(C3k2):
    """C3k2 variant using frequency dynamic convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_FDConv with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_FDConv(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_ODConv(C3k2):
    """C3k2 variant using omni-dimensional dynamic convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_ODConv with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_ODConv(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_SFSConv(C3k2):
    """C3k2 variant using spatial-frequency selection convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_SFSConv with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_SFSConv(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_CrossConvLite(C3k2):
    """C3k2 variant using cross-shaped directional convolution bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_CrossConvLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_CrossConvLite(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_GCResidual(C3k2):
    """C3k2 with learnable residual global-context fusion at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ratio: float = 0.25,
        alpha_init: float = 0.1,
    ):
        """Initialize C3k2_GCResidual with the same core arguments as C3k2."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.gc = GCBlock(c2, c2, ratio)
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then blend in global context with a learnable residual scale."""
        y = super().forward(x)
        alpha = self.alpha_logit.sigmoid()
        return y + alpha * (self.gc(y) - y)


class C3k2_MCA(C3k2):
    """C3k2 followed by moment channel attention recalibration."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        reduction: int = 16,
    ):
        """Initialize C3k2_MCA with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.mca = MCAContext(c2, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then directly recalibrate features with MCA."""
        y = super().forward(x)
        return self.mca(y)


class SEContextLite(nn.Module):
    """Lightweight squeeze-excitation context recalibration."""

    def __init__(self, c1: int, reduction: int = 16):
        """Initialize SEContextLite."""
        super().__init__()
        hidden = max(8, c1 // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recalibrate channels using global average context."""
        return x * self.excite(self.pool(x))


class CAAContextLite(nn.Module):
    """Context anchor attention with lightweight strip depthwise context."""

    def __init__(self, c1: int, ratio: float = 0.125, k: int = 11):
        """Initialize CAAContextLite."""
        super().__init__()
        hidden = max(8, int(c1 * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = Conv(c1, hidden, 1, 1)
        self.horizontal = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, k), 1, (0, k // 2), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(hidden, hidden, (k, 1), 1, (k // 2, 0), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.expand = nn.Conv2d(hidden, c1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anchor-style global context and strip attention."""
        context = self.reduce(self.pool(x)).expand(-1, -1, x.shape[-2], x.shape[-1])
        attn = torch.sigmoid(self.expand(self.vertical(self.horizontal(context))))
        return x * attn


class C3k2_SEResidual(C3k2):
    """C3k2 followed by SE context recalibration at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        reduction: int = 16,
        alpha_init: float = 1.0,
    ):
        """Initialize C3k2_SEResidual with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.se = SEContextLite(c2, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then directly recalibrate features with SE context."""
        y = super().forward(x)
        return self.se(y)


class C3k2_CAAResidual(C3k2):
    """C3k2 with learnable residual context-anchor attention fusion at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ratio: float = 0.125,
        alpha_init: float = 0.8,
    ):
        """Initialize C3k2_CAAResidual with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.caa = CAAContextLite(c2, ratio)
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then blend in CAA context with a bounded residual scale."""
        y = super().forward(x)
        alpha = self.alpha_logit.sigmoid()
        return y + alpha * (self.caa(y) - y)


class FocalModulationLite(nn.Module):
    """Lightweight focal modulation with multi-level local and global context."""

    def __init__(self, c1: int, ratio: float = 0.25, kernels: tuple[int, ...] = (3, 5)):
        """Initialize FocalModulationLite."""
        super().__init__()
        hidden = max(8, int(c1 * ratio))
        self.reduce = Conv(c1, hidden, 1, 1)
        self.context_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden, hidden, k, 1, k // 2, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            for k in kernels
        )
        self.gates = nn.Conv2d(c1, len(kernels) + 1, 1, bias=True)
        self.modulator = nn.Sequential(
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c1, c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate features with gated local and global context."""
        ctx = self.reduce(x)
        gates = torch.softmax(self.gates(x), dim=1)
        ctx_all = torch.zeros_like(ctx)
        for i, branch in enumerate(self.context_branches):
            ctx_all = ctx_all + branch(ctx) * gates[:, i : i + 1]
        global_ctx = ctx.mean((2, 3), keepdim=True).expand_as(ctx)
        ctx_all = ctx_all + global_ctx * gates[:, len(self.context_branches) : len(self.context_branches) + 1]
        return self.out(x * self.modulator(ctx_all))


class C3k2_FocalModulationLite(C3k2):
    """C3k2 followed by lightweight focal modulation at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ratio: float = 0.25,
    ):
        """Initialize C3k2_FocalModulationLite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.focal = FocalModulationLite(c2, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then directly modulate features with focal context."""
        y = super().forward(x)
        return self.focal(y)


class SCSALite(nn.Module):
    """Lightweight spatial-channel synergistic attention for detection features."""

    def __init__(self, c1: int, ratio: float = 0.25, kernels: tuple[int, ...] = (3, 7)):
        """Initialize SCSALite.

        This keeps SCSA's spatial-prior to channel-recalibration flow while
        replacing heavier channel self-attention with lightweight channel gates.
        """
        super().__init__()
        hidden = max(8, int(c1 * ratio))
        self.h_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c1, c1, (k, 1), 1, (k // 2, 0), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )
            for k in kernels
        )
        self.w_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c1, c1, (1, k), 1, (0, k // 2), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )
            for k in kernels
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial priors first, then channel recalibration."""
        h_ctx = x.mean(dim=3, keepdim=True)
        w_ctx = x.mean(dim=2, keepdim=True)
        h_attn = sum(branch(h_ctx) for branch in self.h_convs)
        w_attn = sum(branch(w_ctx) for branch in self.w_convs)
        spatial = torch.sigmoid(h_attn.expand_as(x) + w_attn.expand_as(x))
        x_spatial = x * spatial
        return x_spatial * self.channel_gate(x_spatial)


class C3k2_SCSALite(C3k2):
    """C3k2 followed by lightweight spatial-channel synergistic attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ratio: float = 0.25,
    ):
        """Initialize C3k2_SCSALite with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.scsa = SCSALite(c2, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then directly refine features with SCSALite."""
        y = super().forward(x)
        return self.scsa(y)


class SHSABlock(nn.Module):
    """Single-head self-attention block with a lightweight convolutional bypass."""

    def __init__(self, c1: int, ratio: float = 0.25):
        """Initialize SHSABlock."""
        super().__init__()
        attn_c = max(16, int(c1 * ratio))
        attn_c = min(attn_c, c1)
        self.attn_c = attn_c
        self.conv_c = c1 - attn_c
        self.scale = attn_c**-0.5
        self.norm = nn.BatchNorm2d(attn_c)
        self.qkv = nn.Conv2d(attn_c, attn_c * 3, 1, bias=False)
        self.local = nn.Sequential(
            nn.Conv2d(attn_c, attn_c, 3, 1, 1, groups=attn_c, bias=False),
            nn.BatchNorm2d(attn_c),
            nn.SiLU(inplace=True),
        )
        self.proj = Conv(c1, c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply single-head attention to a channel subset and keep a local bypass."""
        xa, xb = torch.split(x, [self.attn_c, self.conv_c], dim=1) if self.conv_c else (x, None)
        b, c, h, w = xa.shape
        q, k, v = self.qkv(self.norm(xa)).flatten(2).chunk(3, dim=1)
        attn = (q.transpose(1, 2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        ya = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        ya = ya + self.local(xa)
        y = torch.cat((ya, xb), dim=1) if xb is not None else ya
        return self.proj(y)


class C3k2_SHSA(C3k2):
    """C3k2 followed by single-head self-attention refinement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ratio: float = 0.25,
    ):
        """Initialize C3k2_SHSA with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.shsa = SHSABlock(c2, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then directly refine features with SHSA."""
        y = super().forward(x)
        return self.shsa(y)


class MogaBlockLite(nn.Module):
    """Lightweight multi-order gated aggregation block inspired by MogaNet."""

    def __init__(self, c1: int, c2: int | None = None):
        """Initialize MogaBlockLite.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.norm = nn.BatchNorm2d(c2)
        splits = [c2 // 3, c2 // 3, c2 - 2 * (c2 // 3)]
        self.splits = splits
        self.dw3 = nn.Conv2d(splits[0], splits[0], 3, 1, 1, groups=splits[0], bias=False)
        self.dw5 = nn.Conv2d(splits[1], splits[1], 5, 1, 2, groups=splits[1], bias=False)
        self.dw7 = nn.Conv2d(splits[2], splits[2], 7, 1, 3, groups=splits[2], bias=False)
        self.gate = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())
        self.value = nn.Conv2d(c2, c2, 1, bias=False)
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-order depthwise aggregation with lightweight gating."""
        x = self.proj(x)
        x_norm = self.norm(x)
        x1, x2, x3 = torch.split(x_norm, self.splits, dim=1)
        agg = torch.cat((self.dw3(x1), self.dw5(x2), self.dw7(x3)), dim=1)
        return x + self.fuse(self.value(x_norm) * self.gate(agg))


class GnConvLite(nn.Module):
    """Lightweight recursive gated convolution inspired by HorNet gnConv."""

    def __init__(self, c1: int, c2: int | None = None):
        """Initialize GnConvLite.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.pw1 = nn.Conv2d(c2, c2 * 2, 1, bias=False)
        self.dw = nn.Conv2d(c2, c2, 7, 1, 3, groups=c2, bias=False)
        self.pw2 = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated high-order spatial interaction in a lightweight form."""
        x = self.proj(x)
        gate, value = self.pw1(x).chunk(2, dim=1)
        return x + self.pw2(value * torch.sigmoid(self.dw(gate)))


class C3k2_MogaResidual(C3k2):
    """C3k2 with learnable residual MogaBlock-lite fusion at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        alpha_init: float = 0.1,
    ):
        """Initialize C3k2_MogaResidual with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.moga = MogaBlockLite(c2, c2)
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then blend in Moga context with a learnable residual scale."""
        y = super().forward(x)
        alpha = self.alpha_logit.sigmoid()
        return y + alpha * (self.moga(y) - y)


class C3k2_HorNetResidual(C3k2):
    """C3k2 with learnable residual HorNet gnConv-lite fusion at the output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        alpha_init: float = 0.1,
    ):
        """Initialize C3k2_HorNetResidual with C3k2-compatible arguments."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.gnconv = GnConvLite(c2, c2)
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3k2, then blend in gnConv context with a learnable residual scale."""
        y = super().forward(x)
        alpha = self.alpha_logit.sigmoid()
        return y + alpha * (self.gnconv(y) - y)


class GRNLayer(nn.Module):
    """Global Response Normalization from ConvNeXt V2 for lightweight channel competition."""

    def __init__(self, c1, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c1, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class C3k2_GRN(C3k2):
    """C3k2 neck block with lightweight global response normalization."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.grn = GRNLayer(c2)

    def forward(self, x):
        return self.grn(super().forward(x))


class C3k2_DSConv(C3k2):
    """C3k2 variant using DSConv bottlenecks for elongated-structure modeling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_DSConv with the same arguments as C3k2."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DSConv(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_DSConv(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class C3k2_RFAConv(C3k2):
    """C3k2 variant using RFAConv bottlenecks for stronger local receptive-field modeling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2_RFAConv with the same arguments as C3k2."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(
            C3k_RFAConv(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_RFAConv(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class SEAM(nn.Module):
    """Separated and Enhancement Attention Module for feature recalibration."""

    def __init__(self, c1: int, c2: int | None = None, depth: int = 1, reduction: int = 16):
        """Initialize SEAM.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1`` for attention-only use.
            depth (int): Number of depthwise enhancement blocks.
            reduction (int): Channel reduction ratio in the attention MLP.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.enhance = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.GELU(),
                )
                for _ in range(max(depth, 1))
            )
        )
        hidden = max(c2 // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise feature enhancement followed by channel attention."""
        x = self.proj(x)
        y = self.enhance(x)
        b, c, _, _ = y.shape
        y = self.fc(self.pool(y).view(b, c)).view(b, c, 1, 1)
        return x * y


class ASFF2(nn.Module):
    """Two-input adaptive spatial feature fusion with channel-preserving output."""

    def __init__(self, channels: list[int] | tuple[int, int], compress_channels: int = 8):
        """Initialize ASFF2.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of the two input feature maps.
            compress_channels (int): Hidden channels used to generate spatial fusion weights.
        """
        super().__init__()
        assert len(channels) == 2, "ASFF2 expects exactly two input feature maps"
        c1, c2 = channels
        compress_channels = max(4, min(compress_channels, c1, c2))
        self.weight_level_0 = Conv(c1, compress_channels, 1, 1)
        self.weight_level_1 = Conv(c2, compress_channels, 1, 1)
        self.weight_levels = nn.Conv2d(compress_channels * 2, 2, 1, 1, 0)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two same-resolution feature maps with learned spatial weights."""
        x0, x1 = x
        weight_0 = self.weight_level_0(x0)
        weight_1 = self.weight_level_1(x1)
        weights = torch.softmax(self.weight_levels(torch.cat((weight_0, weight_1), 1)), dim=1)
        return torch.cat((x0 * weights[:, 0:1], x1 * weights[:, 1:2]), 1)


class AFPNFuse2(nn.Module):
    """Two-input adaptive fusion block for an AFPN-lite neck.

    The block aligns two same-resolution feature maps to the same channel count,
    predicts spatially adaptive branch weights, and outputs a weighted sum. It is
    intentionally lightweight so the backbone and detection head can stay
    unchanged during neck-only ablation experiments.
    """

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, compress_channels: int = 8):
        """Initialize AFPNFuse2.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of the two input feature maps.
            c2 (int): Output channel count after alignment and fusion.
            compress_channels (int): Hidden channels used to predict spatial fusion weights.
        """
        super().__init__()
        assert len(channels) == 2, "AFPNFuse2 expects exactly two input feature maps"
        c1, c1_b = channels
        compress_channels = max(4, min(compress_channels, c2))
        self.align0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align1 = Conv(c1_b, c2, 1, 1) if c1_b != c2 else nn.Identity()
        self.weight0 = Conv(c2, compress_channels, 1, 1)
        self.weight1 = Conv(c2, compress_channels, 1, 1)
        self.weight_levels = nn.Conv2d(compress_channels * 2, 2, 1, 1, 0)
        self.refine = Conv(c2, c2, 3, 1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two same-resolution feature maps with learned spatial weights."""
        x0, x1 = self.align0(x[0]), self.align1(x[1])
        weight_0 = self.weight0(x0)
        weight_1 = self.weight1(x1)
        weights = torch.softmax(self.weight_levels(torch.cat((weight_0, weight_1), 1)), dim=1)
        return self.refine(x0 * weights[:, 0:1] + x1 * weights[:, 1:2])


def _resize_neck_feature(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a pyramid feature while avoiding interpolation aliases during downsampling."""
    if x.shape[2:] == size:
        return x
    if x.shape[-2] > size[0] or x.shape[-1] > size[1]:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class NeckFeatureSelect(nn.Module):
    """Select one tensor from a multi-output neck for use in a YAML model graph."""

    def __init__(self, index: int):
        """Initialize a zero-parameter output selector."""
        super().__init__()
        self.index = int(index)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return the requested P3, P4, or P5 feature."""
        return x[self.index]


class _SPAFPNPyramidFusion(nn.Module):
    """Lightweight P3/P4/P5 intersection used by SPAFPN Pyramid Fusion."""

    def __init__(self, channels: list[int] | tuple[int, int, int], c2: int):
        super().__init__()
        assert len(channels) == 3, "SPAFPN Pyramid Fusion expects P3, P4, and P5"
        self.align = nn.ModuleList(Conv(c, c2, 1, 1) for c in channels)
        self.fuse = Conv(c2 * 3, c2, 1, 1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        target_size = x[1].shape[2:]
        features = [_resize_neck_feature(conv(feature), target_size) for conv, feature in zip(self.align, x)]
        return self.fuse(torch.cat(features, dim=1))


class _SPAFPNMultiConcat(nn.Module):
    """Decouple a SPAFPN global feature back to one pyramid scale."""

    def __init__(self, local_c: int, global_c: int, c2: int, adjacent_c: int | None = None):
        super().__init__()
        local_in = local_c + (adjacent_c or 0)
        self.local_conv = Conv(local_in, c2, 1, 1)
        self.global_conv = Conv(global_c, c2, 1, 1)
        self.has_adjacent = adjacent_c is not None

    def forward(
        self,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        adjacent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_size = local.shape[2:]
        if self.has_adjacent:
            if adjacent is None:
                raise ValueError("SPAFPN Multi-Concat requires an adjacent-scale feature")
            adjacent = _resize_neck_feature(adjacent, target_size)
            local = torch.cat((local, adjacent), dim=1)
        local = self.local_conv(local)
        global_feature = self.global_conv(_resize_neck_feature(global_feature, target_size))
        return local + local * global_feature


class SPAFPNC3k2Neck(nn.Module):
    """SPAFPN neck adapted to YOLO11 with C3k2 decoupling blocks.

    The topology follows SPAFPN's two-stage ``Pyramid Fusion -> Multi-Concat``
    design. Standard pooling/interpolation replaces the paper repository's
    optional DCNv3 and DySample operators so this ablation remains portable to
    the project's CPU-only validation environment.
    """

    def __init__(
        self,
        in_channels: list[int] | tuple[int, int, int],
        out_channels: list[int] | tuple[int, int, int],
        n: int = 1,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == 3, "SPAFPN expects three pyramid levels"
        c3, c4, c5 = (int(c) for c in out_channels)
        n = max(int(n), 1)
        self.input_proj = nn.ModuleList(Conv(c1, c2, 1, 1) for c1, c2 in zip(in_channels, out_channels))

        self.fusion_td = _SPAFPNPyramidFusion((c3, c4, c5), c4)
        self.td_p5 = _SPAFPNMultiConcat(c5, c4, c5)
        self.td_p4 = _SPAFPNMultiConcat(c4, c4, c4, adjacent_c=c5)
        self.td_p3 = _SPAFPNMultiConcat(c3, c4, c3, adjacent_c=c4)
        self.td_refine = nn.ModuleList(
            (C3k2(c5, c5, n, True), C3k2(c4, c4, n, False), C3k2(c3, c3, n, False))
        )

        self.fusion_bu = _SPAFPNPyramidFusion((c3, c4, c5), c4)
        self.bu_p3 = _SPAFPNMultiConcat(c3, c4, c3)
        self.bu_p4 = _SPAFPNMultiConcat(c4, c4, c4, adjacent_c=c3)
        self.bu_p5 = _SPAFPNMultiConcat(c5, c4, c5, adjacent_c=c4)
        self.bu_refine = nn.ModuleList(
            (C3k2(c3, c3, n, False), C3k2(c4, c4, n, False), C3k2(c5, c5, n, True))
        )

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return SPAFPN-enhanced P3, P4, and P5 features."""
        p3, p4, p5 = [proj(feature) for proj, feature in zip(self.input_proj, x)]

        global_td = self.fusion_td((p3, p4, p5))
        td5 = self.td_refine[0](self.td_p5(p5, global_td))
        td4 = self.td_refine[1](self.td_p4(p4, global_td, td5))
        td3 = self.td_refine[2](self.td_p3(p3, global_td, td4))

        global_bu = self.fusion_bu((td3, td4, td5))
        out3 = self.bu_refine[0](self.bu_p3(td3, global_bu))
        out4 = self.bu_refine[1](self.bu_p4(td4, global_bu, out3))
        out5 = self.bu_refine[2](self.bu_p5(td5, global_bu, out4))
        return [out3, out4, out5]


class _AdaHyperedgeGenerator(nn.Module):
    """Generate YOLOv13-style adaptive hyperedge participation weights."""

    def __init__(
        self,
        node_dim: int,
        num_hyperedges: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        context: str = "both",
    ):
        super().__init__()
        if node_dim % num_heads:
            raise ValueError("HyperACE node_dim must be divisible by num_heads")
        if context not in {"mean", "max", "both"}:
            raise ValueError("HyperACE context must be 'mean', 'max', or 'both'")
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context
        self.prototype_base = nn.Parameter(torch.empty(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        context_dim = node_dim * 2 if context == "both" else node_dim
        self.context_net = nn.Linear(context_dim, num_hyperedges * node_dim)
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, nodes, dim = x.shape
        if self.context == "mean":
            context = x.mean(dim=1)
        elif self.context == "max":
            context = x.max(dim=1).values
        else:
            context = torch.cat((x.mean(dim=1), x.max(dim=1).values), dim=-1)
        offsets = self.context_net(context).view(b, self.num_hyperedges, dim)
        prototypes = self.prototype_base.unsqueeze(0) + offsets
        node_heads = self.pre_head_proj(x).view(b, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        edge_heads = prototypes.view(b, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(node_heads, edge_heads.transpose(-1, -2)) / self.scaling
        logits = self.dropout(logits.mean(dim=1))
        return F.softmax(logits, dim=1)


class _AdaptiveHypergraphConv(nn.Module):
    """Aggregate vertices to hyperedges and distribute them back to vertices."""

    def __init__(self, channels: int, num_hyperedges: int, dropout: float = 0.1):
        super().__init__()
        heads = max(channels // 16, 1)
        while channels % heads:
            heads -= 1
        self.edge_generator = _AdaHyperedgeGenerator(channels, num_hyperedges, heads, dropout, "both")
        self.edge_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        nodes = x.flatten(2).transpose(1, 2)
        assignment = self.edge_generator(nodes)
        edges = self.edge_proj(torch.bmm(assignment.transpose(1, 2), nodes))
        nodes = self.node_proj(torch.bmm(assignment, edges)) + nodes
        return nodes.transpose(1, 2).reshape(b, c, h, w)


class _HyperACEC3AH(nn.Module):
    """CSP-style adaptive hypergraph branch from HyperACE."""

    def __init__(self, channels: int, num_hyperedges: int):
        super().__init__()
        self.cv1 = Conv(channels, channels, 1, 1)
        self.cv2 = Conv(channels, channels, 1, 1)
        self.hypergraph = _AdaptiveHypergraphConv(channels, num_hyperedges)
        self.cv3 = Conv(channels * 2, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.hypergraph(self.cv1(x)), self.cv2(x)), dim=1))


class _HyperACEDepthwiseBottleneck(nn.Module):
    """CPU-portable depthwise-separable low-order branch for HyperACE."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DWConv(channels, channels, 3, 1),
            Conv(channels, channels, 1, 1),
            DWConv(channels, channels, 7, 1),
            Conv(channels, channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _HyperACECore(nn.Module):
    """Fuse three scales and model high- and low-order correlations."""

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int],
        c2: int,
        num_hyperedges: int = 4,
        n: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        hidden = max(round(hidden / 16) * 16, 16)
        self.fuse = Conv(sum(channels), c2, 1, 1)
        self.cv1 = Conv(c2, hidden * 3, 1, 1)
        self.branch1 = _HyperACEC3AH(hidden, num_hyperedges)
        self.branch2 = _HyperACEC3AH(hidden, num_hyperedges)
        self.low_order = nn.ModuleList(_HyperACEDepthwiseBottleneck(hidden) for _ in range(max(int(n), 1)))
        self.cv2 = Conv(hidden * (4 + len(self.low_order)), c2, 1, 1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        target_size = x[1].shape[2:]
        fused = self.fuse(torch.cat([_resize_neck_feature(feature, target_size) for feature in x], dim=1))
        features = list(self.cv1(fused).chunk(3, dim=1))
        high_order_1 = self.branch1(features[1])
        high_order_2 = self.branch2(features[1])
        features.extend(block(features[-1]) for block in self.low_order)
        features[1] = high_order_1
        features.append(high_order_2)
        return self.cv2(torch.cat(features, dim=1))


class _FullPADTunnel(nn.Module):
    """Zero-initialized gated residual distribution from YOLOv13 FullPAD."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        return original + self.gate * enhanced


class HyperACEFullPADNeck(nn.Module):
    """Neck-only HyperACE and FullPAD adaptation for a GRN4 YOLO11 backbone."""

    def __init__(
        self,
        in_channels: list[int] | tuple[int, int, int],
        out_channels: list[int] | tuple[int, int, int],
        num_hyperedges: int = 4,
        n: int = 1,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == 3, "HyperACE neck expects three pyramid levels"
        c3, c4, c5 = (int(c) for c in out_channels)
        n = max(int(n), 1)
        self.input_proj = nn.ModuleList(Conv(c1, c2, 1, 1) for c1, c2 in zip(in_channels, out_channels))
        self.hyperace = _HyperACECore((c3, c4, c5), c4, int(num_hyperedges), n)
        self.ace_to_p3 = Conv(c4, c3, 1, 1)
        self.ace_to_p5 = Conv(c4, c5, 1, 1)
        self.input_tunnels = nn.ModuleList(_FullPADTunnel() for _ in range(3))
        self.neck_tunnels = nn.ModuleList(_FullPADTunnel() for _ in range(4))

        self.top_p4 = C3k2(c5 + c4, c4, n, False)
        self.top_p3 = C3k2(c4 + c3, c3, n, False)
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.bottom_p4 = C3k2(c3 + c4, c4, n, False)
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.bottom_p5 = C3k2(c4 + c5, c5, n, True)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return FullPAD-distributed P3, P4, and P5 neck outputs."""
        p3, p4, p5 = [proj(feature) for proj, feature in zip(self.input_proj, x)]
        ace = self.hyperace((p3, p4, p5))
        ace3 = self.ace_to_p3(_resize_neck_feature(ace, p3.shape[2:]))
        ace5 = self.ace_to_p5(_resize_neck_feature(ace, p5.shape[2:]))
        p3 = self.input_tunnels[0](p3, ace3)
        p4 = self.input_tunnels[1](p4, ace)
        p5 = self.input_tunnels[2](p5, ace5)

        top4 = self.top_p4(torch.cat((_resize_neck_feature(p5, p4.shape[2:]), p4), dim=1))
        top4 = self.neck_tunnels[0](top4, ace)
        top3 = self.top_p3(torch.cat((_resize_neck_feature(top4, p3.shape[2:]), p3), dim=1))
        top3 = self.neck_tunnels[1](top3, ace3)
        out4 = self.bottom_p4(torch.cat((self.down_p3(top3), top4), dim=1))
        out4 = self.neck_tunnels[2](out4, ace)
        out5 = self.bottom_p5(torch.cat((self.down_p4(out4), p5), dim=1))
        out5 = self.neck_tunnels[3](out5, ace5)
        return [top3, out4, out5]


class _SelectiveFusionElimination(nn.Module):
    """Select complementary cross-level information and suppress redundant responses."""

    def __init__(self, local_c: int, source_c: int, c2: int):
        super().__init__()
        self.local_proj = Conv(local_c, c2, 1, 1)
        self.source_proj = Conv(source_c, c2, 1, 1)
        self.context = DWConv(c2 * 2, c2 * 2, 3, 1)
        self.gates = nn.Conv2d(c2 * 2, c2 * 2, 1, 1, 0)
        self.refine = Conv(c2, c2, 3, 1)

    def forward(self, local: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        target_size = local.shape[2:]
        local = self.local_proj(local)
        source = self.source_proj(_resize_neck_feature(source, target_size))
        retain_gate, complement_gate = self.gates(self.context(torch.cat((local, source), dim=1))).sigmoid().chunk(2, 1)
        return self.refine(local * retain_gate + source * complement_gate)


class _WeightedParallelAggregation(nn.Module):
    """Learn a normalized balance between top-down and bottom-up SFE paths."""

    def __init__(self, channels: int, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2))
        self.refine = C3k2(channels, channels, n, False)

    def forward(self, top_down: torch.Tensor, bottom_up: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0)
        return self.refine(top_down * weights[0] + bottom_up * weights[1])


class SFEWPFPNNeck(nn.Module):
    """YOLO neck-only adaptation of WPDFPN's SFE and weighted parallel FPN.

    Classification/localization decoupling and the original two-stage detector
    head are deliberately excluded so the experiment changes only the neck.
    """

    def __init__(
        self,
        in_channels: list[int] | tuple[int, int, int],
        out_channels: list[int] | tuple[int, int, int],
        n: int = 1,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == 3, "SFE-WPFPN expects three pyramid levels"
        c3, c4, c5 = (int(c) for c in out_channels)
        n = max(int(n), 1)
        self.input_proj = nn.ModuleList(Conv(c1, c2, 1, 1) for c1, c2 in zip(in_channels, out_channels))

        self.td_p4 = _SelectiveFusionElimination(c4, c5, c4)
        self.td_p3 = _SelectiveFusionElimination(c3, c4, c3)
        self.bu_p4 = _SelectiveFusionElimination(c4, c3, c4)
        self.bu_p5 = _SelectiveFusionElimination(c5, c4, c5)
        self.aggregate = nn.ModuleList(
            (
                _WeightedParallelAggregation(c3, n),
                _WeightedParallelAggregation(c4, n),
                _WeightedParallelAggregation(c5, n),
            )
        )

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Run top-down and bottom-up SFE paths in parallel and aggregate them."""
        p3, p4, p5 = [proj(feature) for proj, feature in zip(self.input_proj, x)]
        td5 = p5
        td4 = self.td_p4(p4, td5)
        td3 = self.td_p3(p3, td4)
        bu3 = p3
        bu4 = self.bu_p4(p4, bu3)
        bu5 = self.bu_p5(p5, bu4)
        return [
            self.aggregate[0](td3, bu3),
            self.aggregate[1](td4, bu4),
            self.aggregate[2](td5, bu5),
        ]


class ZPool(nn.Module):
    """Channel max-average pooling used by TripletAttention."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool channel statistics into a two-channel descriptor."""
        return torch.cat((torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1)


class TripletAttentionGate(nn.Module):
    """One branch of convolutional triplet attention."""

    def __init__(self, kernel_size: int = 7):
        """Initialize a spatial attention gate over pooled descriptors."""
        super().__init__()
        assert kernel_size % 2 == 1, "TripletAttention kernel size must be odd"
        self.compress = ZPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate."""
        scale = torch.sigmoid(self.spatial(self.compress(x)))
        return x * scale


class TripletAttention(nn.Module):
    """Convolutional triplet attention for cross-dimension neck feature calibration."""

    def __init__(self, c1: int, kernel_size: int = 7, no_spatial: bool = False):
        """Initialize TripletAttention.

        Args:
            c1 (int): Input channels, kept for YAML parser compatibility.
            kernel_size (int): Spatial kernel size in each attention gate.
            no_spatial (bool): Whether to omit the H-W spatial branch.
        """
        super().__init__()
        self.cw = TripletAttentionGate(kernel_size)
        self.hc = TripletAttentionGate(kernel_size)
        self.hw = None if no_spatial else TripletAttentionGate(kernel_size)
        self.no_spatial = no_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three-branch cross-dimension attention."""
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
        if self.no_spatial:
            return 0.5 * (x_out1 + x_out2)
        return (x_out1 + x_out2 + self.hw(x)) / 3.0


class SimAM(nn.Module):
    """Parameter-free simple attention module for feature energy calibration."""

    def __init__(self, c1: int, e_lambda: float = 1e-4):
        """Initialize SimAM.

        Args:
            c1 (int): Input channels, kept for YAML parser compatibility.
            e_lambda (float): Stability term in the energy function.
        """
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free attention from feature variance energy."""
        n = x.shape[2] * x.shape[3] - 1
        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        denom = 4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / max(n, 1) + self.e_lambda)
        return x * self.activation(x_minus_mu_square / denom + 0.5)


class ECALayer(nn.Module):
    """Efficient Channel Attention for lightweight neck feature recalibration."""

    def __init__(self, c1: int, k_size: int = 3):
        """Initialize ECA with local cross-channel interaction."""
        super().__init__()
        assert k_size % 2 == 1, "ECALayer kernel size must be odd"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention."""
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class AFFFuse2(nn.Module):
    """Two-input attentional feature fusion for YOLO necks.

    AFFFuse2 aligns two adjacent-scale features to a shared channel width, then
    predicts local and global fusion weights from their sum. Unlike static BiFPN
    weights, the branch weight is input-adaptive for each image and spatial cell.
    """

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, reduction: int = 4):
        """Initialize AFFFuse2.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of two input feature maps.
            c2 (int): Output channel count after alignment and fusion.
            reduction (int): Channel reduction ratio for attention bottlenecks.
        """
        super().__init__()
        assert len(channels) == 2, "AFFFuse2 expects exactly two input feature maps"
        c1, c1_b = channels
        inter = max(c2 // reduction, 8)
        self.align0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align1 = Conv(c1_b, c2, 1, 1) if c1_b != c2 else nn.Identity()
        self.local_att = nn.Sequential(
            nn.Conv2d(c2, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, inter, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter, c2, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize feature map to target size when adjacent scales differ."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two features with attentional branch weights."""
        x0, x1 = x
        x0 = self._resize(x0, x1.shape[-2:])
        x0, x1 = self.align0(x0), self.align1(x1)
        xa = x0 + x1
        weight = self.sigmoid(self.local_att(xa) + self.global_att(xa))
        return 2.0 * x0 * weight + 2.0 * x1 * (1.0 - weight)


class BiFPNAdd2(nn.Module):
    """Two-input weighted BiFPN-style feature fusion.

    This block replaces channel concatenation with normalized learnable weights.
    It keeps the output channel count fixed, which reduces the following C3k2
    input width while preserving bidirectional multi-scale fusion behavior.
    """

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, eps: float = 1e-4, refine: bool = True):
        """Initialize BiFPNAdd2.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of the two input feature maps.
            c2 (int): Output channel count after channel alignment and fusion.
            eps (float): Numerical stability term for weight normalization.
            refine (bool): Whether to refine the fused feature with a lightweight depthwise convolution.
        """
        super().__init__()
        assert len(channels) == 2, "BiFPNAdd2 expects exactly two input feature maps"
        c1, c1_b = channels
        self.align0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align1 = Conv(c1_b, c2, 1, 1) if c1_b != c2 else nn.Identity()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.refine = DWConv(c2, c2, 3, 1) if refine else nn.Identity()

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize feature map to target size when adjacent scales differ."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two adjacent-scale feature maps with normalized positive weights."""
        x0, x1 = x
        size = x1.shape[-2:]
        x0 = self._resize(x0, size)
        x0, x1 = self.align0(x0), self.align1(x1)
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return self.refine(w[0] * x0 + w[1] * x1)


class SNIFuse2(nn.Module):
    """Soft nearest-neighbor interpolation fusion for a lightweight SNI neck.

    The first input is resized to the second input's spatial size using a
    learnable blend of nearest-neighbor and smoothed interpolation, then both
    branches are adaptively fused. This keeps the module lightweight while
    reducing hard upsampling artifacts at feature pyramid fusion points.
    """

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, compress_channels: int = 8):
        """Initialize SNIFuse2.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of the two input feature maps.
            c2 (int): Output channel count after alignment and fusion.
            compress_channels (int): Hidden channels used to predict spatial fusion weights.
        """
        super().__init__()
        assert len(channels) == 2, "SNIFuse2 expects exactly two input feature maps"
        c1, c1_b = channels
        compress_channels = max(4, min(compress_channels, c2))
        self.align0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align1 = Conv(c1_b, c2, 1, 1) if c1_b != c2 else nn.Identity()
        self.sni_alpha = nn.Parameter(torch.tensor(0.5))
        self.weight0 = Conv(c2, compress_channels, 1, 1)
        self.weight1 = Conv(c2, compress_channels, 1, 1)
        self.weight_levels = nn.Conv2d(compress_channels * 2, 2, 1, 1, 0)
        self.refine = Conv(c2, c2, 3, 1)

    def _sni_resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize with a learnable blend of nearest and smoothed interpolation."""
        if x.shape[-2:] == size:
            return x
        nearest = F.interpolate(x, size=size, mode="nearest")
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        smooth = F.interpolate(smooth, size=size, mode="bilinear", align_corners=False)
        alpha = self.sni_alpha.sigmoid()
        return alpha * nearest + (1.0 - alpha) * smooth

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two adjacent-scale feature maps with soft interpolation alignment."""
        x0, x1 = x
        x0 = self._sni_resize(x0, x1.shape[-2:])
        x0, x1 = self.align0(x0), self.align1(x1)
        weight_0 = self.weight0(x0)
        weight_1 = self.weight1(x1)
        weights = torch.softmax(self.weight_levels(torch.cat((weight_0, weight_1), 1)), dim=1)
        return self.refine(x0 * weights[:, 0:1] + x1 * weights[:, 1:2])


class FreqFusionLite(nn.Module):
    """Lightweight frequency-aware two-input fusion for YOLO necks.

    The block aligns two adjacent-scale features, separates each into low- and
    high-frequency components via average pooling, then fuses semantic low
    frequency and boundary high frequency components with learned weights.
    """

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, compress_channels: int = 8):
        """Initialize FreqFusionLite.

        Args:
            channels (list[int] | tuple[int, int]): Channel counts of the two input feature maps.
            c2 (int): Output channel count after alignment and fusion.
            compress_channels (int): Hidden channels used to predict frequency fusion weights.
        """
        super().__init__()
        assert len(channels) == 2, "FreqFusionLite expects exactly two input feature maps"
        c1, c1_b = channels
        compress_channels = max(4, min(compress_channels, c2))
        self.align0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align1 = Conv(c1_b, c2, 1, 1) if c1_b != c2 else nn.Identity()
        self.low0 = Conv(c2, compress_channels, 1, 1)
        self.low1 = Conv(c2, compress_channels, 1, 1)
        self.high0 = Conv(c2, compress_channels, 1, 1)
        self.high1 = Conv(c2, compress_channels, 1, 1)
        self.low_weights = nn.Conv2d(compress_channels * 2, 2, 1, 1, 0)
        self.high_weights = nn.Conv2d(compress_channels * 2, 2, 1, 1, 0)
        self.high_scale = nn.Parameter(torch.tensor(0.5))
        self.refine = Conv(c2, c2, 3, 1)

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize feature map to target size when adjacent scales differ."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    @staticmethod
    def _split_frequency(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split feature into low-frequency context and high-frequency detail."""
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        return low, high

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse two adjacent-scale features with low/high-frequency weighting."""
        x0, x1 = x
        x0 = self._resize(x0, x1.shape[-2:])
        x0, x1 = self.align0(x0), self.align1(x1)
        low0, high0 = self._split_frequency(x0)
        low1, high1 = self._split_frequency(x1)

        low_weights = torch.softmax(self.low_weights(torch.cat((self.low0(low0), self.low1(low1)), 1)), dim=1)
        high_weights = torch.softmax(self.high_weights(torch.cat((self.high0(high0), self.high1(high1)), 1)), dim=1)
        low = low0 * low_weights[:, 0:1] + low1 * low_weights[:, 1:2]
        high = high0 * high_weights[:, 0:1] + high1 * high_weights[:, 1:2]
        return self.refine(low + self.high_scale.sigmoid() * high)


class RFPNLiteFuse(nn.Module):
    """Weak residual top-down fusion inspired by rethinking feature-pyramid fusion."""

    def __init__(self, channels: list[int] | tuple[int, int], c2: int, init_weight: float = 0.25):
        """Initialize RFPNLiteFuse."""
        super().__init__()
        assert len(channels) == 2, "RFPNLiteFuse expects exactly two input feature maps"
        c_high, c_low = channels
        init_weight = min(max(float(init_weight), 1e-3), 1.0 - 1e-3)
        self.high_align = Conv(c_high, c2, 1, 1) if c_high != c2 else nn.Identity()
        self.low_align = Conv(c_low, c2, 1, 1) if c_low != c2 else nn.Identity()
        self.high_gate = nn.Parameter(torch.logit(torch.tensor(init_weight)))
        self.refine = nn.Sequential(DWConv(c2, c2, 3, 1), Conv(c2, c2, 1, 1))

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize feature map to target size when adjacent scales differ."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse high-level context as a weak residual into the low-level feature."""
        high, low = x
        high = self._resize(high, low.shape[-2:])
        return self.refine(self.low_align(low) + torch.sigmoid(self.high_gate) * self.high_align(high))


class CoordAtt(nn.Module):
    """Coordinate Attention for direction-aware feature recalibration.

    CoordAtt encodes long-range dependencies along height and width separately,
    which is useful for elongated targets such as reinforcement bars and timber.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        """Initialize CoordAtt.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1`` for attention-only use.
            reduction (int): Channel reduction ratio used by the shared transform.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        mip = max(8, c2 // reduction)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.conv1 = nn.Conv2d(c2, mip, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, c2, 1, 1, 0, bias=False)
        self.conv_w = nn.Conv2d(mip, c2, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention."""
        x = self.proj(x)
        identity = x
        _, _, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class GAM(nn.Module):
    """Global Attention Mechanism for channel-spatial feature recalibration.

    This module follows the common GAM design used by GSO-YOLO's GOM:
    channel attention is computed without global pooling, then spatial
    attention is applied with convolutional feature fusion.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 4):
        """Initialize GAM.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
            reduction (int): Channel reduction ratio used in both attention branches.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(c2 // reduction, 1)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.channel_attention = nn.Sequential(
            nn.Linear(c2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c2, bias=False),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c2, hidden, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention followed by spatial attention."""
        x = self.proj(x)
        b, c, h, w = x.shape
        channel_att = self.channel_attention(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        channel_att = channel_att.reshape(b, h, w, c).permute(0, 3, 1, 2).sigmoid()
        x = x * channel_att
        spatial_att = self.spatial_attention(x).sigmoid()
        return x * spatial_att


class ResCBAM(nn.Module):
    """Residual CBAM block for conservative feature recalibration.

    The residual path preserves weak target features while CBAM refines channel
    and spatial responses, making it safer than plain attention-only gating.
    """

    def __init__(self, c1: int, c2: int | None = None, kernel_size: int = 7, shortcut: bool = True):
        """Initialize ResCBAM.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1``.
            kernel_size (int): Spatial attention kernel size, either 3 or 7.
            shortcut (bool): Whether to add the residual connection.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.cbam = CBAM(c2, kernel_size)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual CBAM attention."""
        x = self.proj(x)
        y = self.cbam(x)
        return x + y if self.add else y


class DLU(nn.Module):
    """Dynamic Lightweight Upsampling inspired by Lighten CARAFE."""

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        compressed_channels: int = 64,
    ):
        """Initialize a pure PyTorch DLU upsampling module.

        Args:
            channels (int): Input and output feature channels.
            scale_factor (int): Upsampling ratio.
            up_kernel (int): Reassembly kernel size.
            encoder_kernel (int): Kernel size for kernel-space and offset generation.
            encoder_dilation (int): Dilation used by the generators.
            compressed_channels (int): Hidden channels after channel compression.
        """
        super().__init__()
        assert scale_factor >= 1, "scale_factor must be >= 1"
        assert up_kernel % 2 == 1 and up_kernel >= 1, "up_kernel must be a positive odd integer"
        compressed_channels = min(compressed_channels, channels)
        padding = (encoder_kernel - 1) * encoder_dilation // 2
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.kernel_space_generator = nn.Conv2d(
            compressed_channels,
            up_kernel * up_kernel,
            encoder_kernel,
            padding=padding,
            dilation=encoder_dilation,
        )
        self.conv_offset = nn.Conv2d(
            compressed_channels,
            2 * scale_factor * scale_factor,
            encoder_kernel,
            padding=padding,
            dilation=encoder_dilation,
        )
        self.init_weights()

    def init_weights(self):
        """Initialize DLU generators."""
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        nn.init.zeros_(self.channel_compressor.bias)
        nn.init.normal_(self.kernel_space_generator.weight, std=0.001)
        nn.init.zeros_(self.kernel_space_generator.bias)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def kernel_space_normalizer(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the source kernel space over the reassembly-kernel dimension."""
        b, c, h, w = mask.shape
        mask = mask.view(b, 1, c, h, w)
        mask = F.softmax(mask, dim=2)
        return mask.view(b, c, h, w).contiguous()

    def kernel_space_expander(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Expand low-resolution kernels to high-resolution kernels with learned guidance offsets."""
        b, _, h, w = offset.shape
        s = self.scale_factor
        offset = F.pixel_shuffle(offset, s).permute(0, 2, 3, 1)
        h_up, w_up = h * s, w * s
        dtype, device = offset.dtype, offset.device
        y = torch.repeat_interleave(torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype), s)
        x = torch.repeat_interleave(torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype), s)
        grid_y = y.view(h_up, 1).expand(h_up, w_up)
        grid_x = x.view(1, w_up).expand(h_up, w_up)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        norm_x = 2.0 / max(w - 1, 1)
        norm_y = 2.0 / max(h - 1, 1)
        offset = torch.stack((offset[..., 0] * norm_x, offset[..., 1] * norm_y), dim=-1)
        return F.grid_sample(mask, grid + offset, mode="bilinear", padding_mode="border", align_corners=True)

    def feature_reassemble(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reassemble features using high-resolution dynamic kernels."""
        b, c, h, w = x.shape
        k, s = self.up_kernel, self.scale_factor
        patches = F.unfold(x, kernel_size=k, padding=k // 2).view(b, c, k * k, h, w)
        patches = patches.repeat_interleave(s, dim=3).repeat_interleave(s, dim=4)
        mask = mask.view(b, 1, k * k, h * s, w * s)
        return (patches * mask).sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample features with dynamic lightweight content-aware reassembly."""
        compressed_x = self.channel_compressor(x)
        offset = self.conv_offset(compressed_x)
        mask = self.kernel_space_generator(compressed_x)
        mask = self.kernel_space_normalizer(mask)
        mask = self.kernel_space_expander(offset, mask)
        return self.feature_reassemble(x, mask)


class EMA(nn.Module):
    """Efficient Multi-scale Attention for lightweight spatial-channel feature calibration."""

    def __init__(self, channels: int, factor: int = 8):
        """Initialize EMA attention.

        Args:
            channels (int): Number of input and output channels.
            factor (int): Maximum channel grouping factor.
        """
        super().__init__()
        self.groups = max(1, min(factor, channels))
        while channels % self.groups != 0:
            self.groups -= 1
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped multi-scale attention to the input feature map."""
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class C3k2_EMA(C3k2):
    """C3k2 block enhanced with Efficient Multi-scale Attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        factor: int = 8,
    ):
        """Initialize C3k2_EMA module with the same arguments as C3k2 plus EMA grouping factor."""
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.ema = EMA(c2, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3k2 followed by EMA feature recalibration."""
        return self.ema(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split instead of chunk."""
        return self.ema(super().forward_split(x))


class MFAMBottleneck(nn.Module):
    """Multi-scale feature aggregation bottleneck for small-object feature enhancement."""

    def __init__(self, c: int, shortcut: bool = True):
        """Initialize a lightweight multi-scale aggregation block.

        Args:
            c (int): Input and output channels.
            shortcut (bool): Whether to add a residual connection.
        """
        super().__init__()
        self.cv1 = Conv(c, c, 1, 1)
        self.branch3 = DWConv(c, c, 3, 1)
        self.branch5 = DWConv(c, c, 5, 1)
        self.branch7 = DWConv(c, c, 7, 1)
        self.cv2 = Conv(3 * c, c, 1, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate 3x3, 5x5 and 7x7 receptive-field features."""
        y = self.cv1(x)
        y = self.cv2(torch.cat((self.branch3(y), self.branch5(y), self.branch7(y)), 1))
        return x + y if self.add else y


class MFAM(nn.Module):
    """C2f-style Multi-scale Feature Aggregation Module for YOLO neck feature fusion."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        """Initialize MFAM.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of MFAM bottlenecks.
            shortcut (bool): Whether MFAM bottlenecks use residual connections.
            e (float): Hidden channel expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MFAMBottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MFAM."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LSKBlock(nn.Module):
    """Large selective-kernel spatial attention block for degraded-scene detection features.

    The design follows the large selective kernel idea from LSKNet: combine a local
    depthwise branch and a larger dilated depthwise branch, then learn spatially
    varying selection weights. It is inserted as a lightweight residual feature
    recalibration block before detection heads.
    """

    def __init__(self, c1: int, c2: int | None = None, k1: int = 5, k2: int = 7, dilation: int = 3):
        """Initialize LSKBlock.

        Args:
            c1 (int): Input channels.
            c2 (int | None): Output channels. Defaults to ``c1`` for attention-only use.
            k1 (int): Local depthwise kernel size.
            k2 (int): Dilated depthwise kernel size.
            dilation (int): Dilation for the large-kernel branch.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.local = nn.Conv2d(c2, c2, k1, 1, autopad(k1), groups=c2, bias=False)
        self.context = nn.Conv2d(c2, c2, k2, 1, autopad(k2, d=dilation), dilation=dilation, groups=c2, bias=False)
        self.reduce_local = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.reduce_context = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.spatial_select = nn.Conv2d(2, 2, 7, 1, 3, bias=True)
        self.expand = nn.Conv2d(c2 // 2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply large selective-kernel feature recalibration."""
        x = self.proj(x)
        local = self.reduce_local(self.local(x))
        context = self.reduce_context(self.context(self.local(x)))
        pooled = torch.cat(
            (torch.mean(torch.cat((local, context), 1), dim=1, keepdim=True),
             torch.max(torch.cat((local, context), 1), dim=1, keepdim=True)[0]),
            dim=1,
        )
        weights = torch.sigmoid(self.spatial_select(pooled))
        y = local * weights[:, 0:1] + context * weights[:, 1:2]
        return x + self.act(self.bn(self.expand(y)))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k_DSConv(C3):
    """C3k block using DSConv bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3k_DSConv."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DSConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3k_DDFM(C3):
    """C3k block using DDFM-lite bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3k_DDFM."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DDFM(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3k_GhostV2DFC(C3):
    """C3k block using GhostNetV2 DFC bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize GhostNetV2 DFC bottlenecks inside a C3 container."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneckV2DFC(c_, c_, shortcut) for _ in range(n)))


class C3k_MSBlockLite(C3):
    """C3k block using hierarchical lightweight MSBlocks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize hierarchical MSBlock-Lite layers inside a C3 container."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(HierarchicalMSBlockLite(c_, c_, shortcut=shortcut) for _ in range(n)))


class C3k_MSBlock(C3):
    """C3k block using MSBlock bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3k_MSBlock."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_MSBlock(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3k_RFAConv(C3):
    """C3k block using RFAConv bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3k_RFAConv."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class RepDown(nn.Module):
    """YOLO-ULM reparameterizable parallel downsampling block."""

    def __init__(self, c1: int, c2: int, k: int = 7, s: int = 2):
        """Initialize pointwise projection and parallel 7x7/3x3 depthwise branches."""
        super().__init__()
        if k < 3 or k % 2 == 0:
            raise ValueError(f"RepDown kernel size must be odd and >= 3, got {k}")
        self.k = k
        self.s = s
        self.c2 = c2
        self.pw = Conv(c1, c2, 1, 1)
        self.dw_large = Conv(c2, c2, k, s, p=k // 2, g=c2, act=False)
        self.dw_small = Conv(c2, c2, 3, s, p=1, g=c2, act=False)
        self.act = Conv.default_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel depthwise downsampling during training."""
        x = self.pw(x)
        return self.act(self.dw_large(x) + self.dw_small(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the equivalent single-branch downsampling convolution."""
        return self.act(self.dw_reparam(self.pw(x)))

    @staticmethod
    def _fuse_branch(branch: Conv) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse one Conv-BN depthwise branch into a kernel and bias."""
        conv, bn = branch.conv, branch.bn
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        kernel = conv.weight * scale
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused large kernel and bias for deployment."""
        kernel_large, bias_large = self._fuse_branch(self.dw_large)
        kernel_small, bias_small = self._fuse_branch(self.dw_small)
        pad = (self.k - 3) // 2
        kernel_small = F.pad(kernel_small, [pad, pad, pad, pad])
        return kernel_large + kernel_small, bias_large + bias_small

    def fuse_convs(self) -> None:
        """Merge the parallel depthwise branches into one large-kernel convolution."""
        if hasattr(self, "dw_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dw_reparam = nn.Conv2d(
            self.c2,
            self.c2,
            self.k,
            self.s,
            self.k // 2,
            groups=self.c2,
            bias=True,
        ).to(kernel.device, kernel.dtype)
        self.dw_reparam.weight.data.copy_(kernel)
        self.dw_reparam.bias.data.copy_(bias)
        del self.dw_large
        del self.dw_small


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class LDownLite(nn.Module):
    """Lightweight two-branch downsampling for neck bottom-up paths."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        """Initialize LDownLite with local-detail and low-frequency context branches."""
        super().__init__()
        c_local = max(c2 // 2, 1)
        c_context = c2 - c_local
        self.local = nn.Sequential(DWConv(c1, c_local, k, s), Conv(c_local, c_local, 1, 1))
        self.context = nn.Sequential(nn.AvgPool2d(kernel_size=s, stride=s), Conv(c1, c_context, 1, 1))
        self.fuse = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample while preserving local detail and low-frequency context."""
        return self.fuse(torch.cat((self.local(x), self.context(x)), 1))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            semantic = self.semseg(feat)
            return (p, semantic)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
