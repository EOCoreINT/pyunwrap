"""
pyunwrap.models.ambiguity_net
================================

`AmbiguityNet`: a physics-informed U-Net that predicts the discrete integer
phase-ambiguity map ``k`` (rather than the continuous unwrapped phase
directly), plus an auxiliary uncertainty ("residue probability") map.

Architecture
------------
- **Encoder**: ResNet-34 (optionally ImageNet-pretrained), with its first
  conv/bn/relu/maxpool stage and four residual stages used as a 5-level
  feature pyramid for U-Net-style skip connections.
- **Decoder**: four upsampling blocks that each upsample the previous
  decoder output, concatenate the matching encoder skip connection, and
  apply a double conv block -- the standard U-Net decoder pattern.
- **Head 1 (primary)**: a 1-channel continuous ambiguity prediction, passed
  through `tanh` and scaled to `[-k_max, k_max]` (default +-10 integer
  wraps), matching the fact that `k` is a physically bounded, roughly
  symmetric quantity for typical InSAR scenes.
- **Head 2 (auxiliary)**: a 1-channel sigmoid-activated "residue
  probability" map used for uncertainty quantification (per-pixel confidence
  that the model's ambiguity resolution is trustworthy near that location).

Rounding & differentiability
-----------------------------
The hard requirement that the *reconstructed* phase obey
``phi_hat = wrapped_phase + 2*pi*round(k)`` is non-differentiable at the
rounding step. This is handled with a **straight-through estimator (STE)**:
the forward value is the rounded integer, but gradients flow through as if
rounding were the identity function. This lets the re-wrapping consistency
loss (Prompt 3, Component 2) backpropagate through the hard integer
constraint during training, exactly as it would be applied at inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchvision.models import ResNet34_Weights, resnet34

    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover
    _HAS_TORCHVISION = False


# --------------------------------------------------------------------------- #
# Straight-through rounding
# --------------------------------------------------------------------------- #


class _RoundSTE(torch.autograd.Function):
    """Straight-through estimator for `torch.round`.

    Forward: returns `round(x)`.
    Backward: passes the incoming gradient through unchanged (as if the
    forward op were the identity), which is the standard trick used to train
    through quantization/rounding bottlenecks (e.g. VQ-VAE, binarized nets).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with a straight-through gradient estimator."""
    return _RoundSTE.apply(x)


# --------------------------------------------------------------------------- #
# Decoder building blocks
# --------------------------------------------------------------------------- #


class ConvBnRelu(nn.Module):
    """3x3 Conv -> BatchNorm -> ReLU, the atomic block used throughout the decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """One U-Net decoder stage: upsample, concat skip connection, double conv.

    Args:
        in_channels: Channels coming from the previous (coarser) decoder stage.
        skip_channels: Channels of the corresponding encoder skip connection
            (0 if there is no skip at this stage).
        out_channels: Output channel count of this decoder stage.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Guard against off-by-one size mismatches from odd input
            # resolutions propagating through strided pooling layers.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# --------------------------------------------------------------------------- #
# ResNet-34 encoder wrapper
# --------------------------------------------------------------------------- #


class ResNet34Encoder(nn.Module):
    """Wraps `torchvision.models.resnet34` to expose a 5-level feature pyramid.

    Feature strides relative to the input (for a 256x256 input):
        stage0 (stem):  stride 2  -> 128x128, 64 channels
        stage1 (layer1): stride 4  -> 64x64,   64 channels
        stage2 (layer2): stride 8  -> 32x32,  128 channels
        stage3 (layer3): stride 16 -> 16x16,  256 channels
        stage4 (layer4): stride 32 -> 8x8,    512 channels
    """

    #: Output channel count of each feature stage, in encoder order.
    out_channels = (64, 64, 128, 256, 512)

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        """
        Args:
            in_channels: Number of input channels. `pyunwrap` uses 3
                (wrapped phase, coherence, amplitude), matching ImageNet's
                3-channel input so pretrained weights transfer directly to
                the first conv layer without modification.
            pretrained: Whether to attempt loading ImageNet-pretrained
                weights. Falls back to random (Kaiming) initialization with a
                warning if the weights cannot be downloaded (e.g. no network
                access), so the model always remains constructible offline.
        """
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for the ResNet-34 encoder.")

        weights = None
        if pretrained:
            try:
                weights = ResNet34_Weights.IMAGENET1K_V1
            except Exception as exc:  # pragma: no cover - depends on network access
                warnings.warn(
                    f"Could not resolve ImageNet weights spec ({exc}); "
                    "falling back to random initialization."
                )
                weights = None

        try:
            backbone = resnet34(weights=weights)
        except Exception as exc:  # pragma: no cover - network-dependent
            warnings.warn(
                f"Failed to download pretrained ResNet-34 weights ({exc}); "
                "falling back to random (Kaiming) initialization. Fine-tuning "
                "from scratch will require proportionally more training epochs."
            )
            backbone = resnet34(weights=None)

        if in_channels != 3:
            # Replace the stem conv to accept a different channel count,
            # averaging the pretrained RGB filters across channels as a
            # reasonable initialization for the new input depth.
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if weights is not None:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
            backbone.conv1 = new_conv

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns the 5-level feature pyramid, coarsest last.

        Args:
            x: Input tensor, shape [B, in_channels, H, W].

        Returns:
            List of 5 feature tensors: [stem, layer1, layer2, layer3, layer4].
        """
        stem = self.stem(x)  # stride 2
        pooled = self.maxpool(stem)  # stride 4
        l1 = self.layer1(pooled)  # stride 4
        l2 = self.layer2(l1)  # stride 8
        l3 = self.layer3(l2)  # stride 16
        l4 = self.layer4(l3)  # stride 32
        return [stem, l1, l2, l3, l4]


# --------------------------------------------------------------------------- #
# AmbiguityNet
# --------------------------------------------------------------------------- #


@dataclass
class AmbiguityNetOutput:
    """Structured output of `AmbiguityNet.forward`.

    Attributes:
        k_hat: Rounded integer ambiguity map (straight-through), [B, 1, H, W].
        k_continuous: Pre-rounding continuous ambiguity prediction, [B, 1, H, W].
        residue_prob: Sigmoid-activated residue/uncertainty probability map,
            [B, 1, H, W], values in [0, 1].
        phi_hat: Reconstructed unwrapped phase, radians, [B, 1, H, W]:
            `phi_hat = wrapped_phase_rad + 2*pi*k_hat`.
    """

    k_hat: torch.Tensor
    k_continuous: torch.Tensor
    residue_prob: torch.Tensor
    phi_hat: torch.Tensor


class AmbiguityNet(nn.Module):
    """Physics-informed U-Net that predicts the integer phase ambiguity map.

    Input: a 3-channel stack of (normalized wrapped phase in [-1, 1],
    coherence in [0, 1], normalized log-amplitude), matching
    `pyunwrap.data.dataloader.InSARTileDataset` output.

    Because the wrapped phase channel is stored normalized to [-1, 1] (see
    `pyunwrap.data.preprocessing.normalize_phase`), the model internally
    de-normalizes it (multiplies by pi) before using it to reconstruct
    `phi_hat`, so callers never need to pass the raw radian phase separately.

    Example:
        >>> model = AmbiguityNet(pretrained=False)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = model(x)
        >>> out.k_hat.shape
        torch.Size([2, 1, 256, 256])
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        k_max: float = 10.0,
        decoder_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
    ) -> None:
        """
        Args:
            in_channels: Number of stacked input channels (default 3: wrapped
                phase, coherence, amplitude).
            pretrained: Whether to initialize the ResNet-34 encoder from
                ImageNet-pretrained weights (network access required; falls
                back gracefully to random init otherwise).
            k_max: Maximum absolute ambiguity magnitude the primary head can
                express. The continuous k prediction is `tanh(logits) * k_max`,
                bounding the network's output to the physically expected range
                for typical scene sizes/deformation magnitudes; increase for
                scenes with very large expected phase gradients.
            decoder_channels: Output channel count of each of the 5 decoder
                stages (4 upsampling stages + the final full-resolution head
                stage), coarsest-to-finest.
        """
        super().__init__()
        self.k_max = k_max
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained)
        enc_ch = self.encoder.out_channels  # (64, 64, 128, 256, 512), stem..layer4

        c0, c1, c2, c3 = (
            decoder_channels[0],
            decoder_channels[1],
            decoder_channels[2],
            decoder_channels[3],
        )
        c4 = decoder_channels[4]

        # Decoder stages consume encoder features in reverse (coarsest first).
        # layer4 (512, stride32) -> up -> concat layer3(256) -> c0 (stride16)
        self.dec4 = DecoderBlock(in_channels=enc_ch[4], skip_channels=enc_ch[3], out_channels=c0)
        # c0 (stride16) -> up -> concat layer2(128) -> c1 (stride8)
        self.dec3 = DecoderBlock(in_channels=c0, skip_channels=enc_ch[2], out_channels=c1)
        # c1 (stride8) -> up -> concat layer1(64) -> c2 (stride4)
        self.dec2 = DecoderBlock(in_channels=c1, skip_channels=enc_ch[1], out_channels=c2)
        # c2 (stride4) -> up -> concat stem(64) -> c3 (stride2)
        self.dec1 = DecoderBlock(in_channels=c2, skip_channels=enc_ch[0], out_channels=c3)
        # c3 (stride2) -> up -> no skip available -> c4 (stride1, full resolution)
        self.dec0 = DecoderBlock(in_channels=c3, skip_channels=0, out_channels=c4)

        # Output heads, applied at full input resolution.
        self.ambiguity_head = nn.Conv2d(c4, 1, kernel_size=1)
        self.residue_head = nn.Conv2d(c4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> AmbiguityNetOutput:
        """Run the full forward pass.

        Args:
            x: Input tensor, shape [B, in_channels, H, W]. Channel 0 must be
                the wrapped phase normalized to [-1, 1] (see
                `pyunwrap.data.preprocessing.normalize_phase`).

        Returns:
            An `AmbiguityNetOutput` with the rounded/continuous ambiguity
            maps, the residue probability map, and the reconstructed
            unwrapped phase.
        """
        stem, l1, l2, l3, l4 = self.encoder(x)

        d = self.dec4(l4, l3)
        d = self.dec3(d, l2)
        d = self.dec2(d, l1)
        d = self.dec1(d, stem)
        d = self.dec0(d, skip=None)  # final upsample back to full input resolution

        if d.shape[-2:] != x.shape[-2:]:
            # Final safety net in case of odd input sizes not perfectly
            # divisible by the encoder's total stride (32).
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)

        k_logits = self.ambiguity_head(d)
        k_continuous = torch.tanh(k_logits) * self.k_max
        k_hat = round_ste(k_continuous)

        residue_prob = torch.sigmoid(self.residue_head(d))

        wrapped_phase_norm = x[:, 0:1, :, :]
        wrapped_phase_rad = wrapped_phase_norm * torch.pi
        phi_hat = wrapped_phase_rad + 2.0 * torch.pi * k_hat

        return AmbiguityNetOutput(
            k_hat=k_hat,
            k_continuous=k_continuous,
            residue_prob=residue_prob,
            phi_hat=phi_hat,
        )
