from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from acsconv.models.acsunet import _DecoderBlock
from acsconv.operators import ACSConv

from src.model.smp import ClassificationHead, SegmentationHead, SegmentationModel, Activation


class AcsConvnextWrapper(nn.Module):
    """"Wrapper to extract features from convnext model"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_stride = 32

    def forward(self, x):
        B, C, H, W, D = x.shape
        xs = []
        for i in range(4):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            xs.append(x)
        return xs


class DecoderBlock(_DecoderBlock):
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = super().forward(x)
        return x


class UNet3dAcsDecoder(nn.Module):
    def __init__(
        self, 
        encoder_channels, 
        decoder_mid_channels, 
        decoder_out_channels, 
    ):
        super().__init__()

        assert len(decoder_out_channels) == len(decoder_mid_channels)

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_out_channels
        mid_channels = decoder_mid_channels
        blocks = [
            DecoderBlock(in_ch + skip_ch, mid_ch, out_ch)
            for in_ch, skip_ch, mid_ch, out_ch in zip(in_channels, skip_channels, mid_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class GlobalAveragePooling(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class SegmentationHeadAcs(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv3d = ACSConv(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        pooling = GlobalAveragePooling(2)  # (B, C, D, H, W) -> (B, C, H, W)
        upsampling = nn.UpsamplingNearest2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv3d, pooling, upsampling, activation)


class UNet3dAcs(SegmentationModel):
    def __init__(
        self, 
        encoder,
        encoder_channels, 
        decoder_mid_channels, 
        decoder_out_channels, 
        upsampling,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = encoder
        
        self.decoder = UNet3dAcsDecoder(
            encoder_channels=encoder_channels,
            decoder_mid_channels=decoder_mid_channels,
            decoder_out_channels=decoder_out_channels,
        )

        self.segmentation_head = SegmentationHeadAcs(
            in_channels=decoder_out_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.initialize()

    def forward(self, x):
        # (B, C, H, W, D) -> (B, C, D, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return super().forward(x)
