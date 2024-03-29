from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, to_ntuple
from timm.layers.norm import fast_layer_norm, is_fast_norm, LayerNorm2d
from timm.models.convnext import ConvNeXtBlock, Downsample
from acsconv.utils import _triple_same
from acsconv.operators import ACSConv
from acsconv.converters import ACSConverter
from functools import partial

from src.model.smp import ClassificationHead, SegmentationModel, Activation



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


class LayerNorm3dAcs(nn.LayerNorm):
    """ LayerNorm for channels of '3D' spatial NCHWD tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x
    

class ConvNeXtBlockAcs(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ACSConverterTimm(ACSConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_stride = 32
    
    def convert_module(self, module):
        """
        A recursive function. 
        Treat the entire model as a tree and convert each leaf module to
            target_conv if it's Conv2d,
            3d counterparts if it's a pooling or normalization module,
            trilinear mode if it's a Upsample module.
        """
        for child_name, child in module.named_children(): 
            if isinstance(child, nn.Conv2d):
                arguments = nn.Conv2d.__init__.__code__.co_varnames[1:]

                # cleaning the tuple due to new PyTorch namming schema (or added new variables)
                arguments = [a for a in arguments if a not in ['device', 'dtype', 'factory_kwargs','kernel_size_', 'stride_', 'padding_', 'dilation_']]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs = self.convert_conv_kwargs(kwargs)
                setattr(module, child_name, self.__class__.target_conv(**kwargs))
            elif isinstance(child, LayerNorm2d):
                arguments = LayerNorm2d.__init__.__code__.co_varnames[1:]
                arguments = [a for a in arguments if a not in ['device', 'dtype', 'factory_kwargs']]
                
                # Map the arguments to the corresponding name in nn.LayerNorm
                timm_arg_name_to_nn_ln_attr_name = {
                    'num_channels': 'normalized_shape',
                    'affine': 'elementwise_affine',
                }
                timm_arg_name_to_nn_ln_attr_name_rev = {
                    v: k 
                    for k, v in timm_arg_name_to_nn_ln_attr_name.items()
                }
                arguments = [
                    timm_arg_name_to_nn_ln_attr_name[a] if a in timm_arg_name_to_nn_ln_attr_name else a
                    for a in arguments
                ]
                
                kwargs = {
                    timm_arg_name_to_nn_ln_attr_name_rev[k] if k in timm_arg_name_to_nn_ln_attr_name_rev else k: 
                    getattr(child, k) 
                    for k in arguments
                }
                setattr(module, child_name, LayerNorm3dAcs(**kwargs))
            elif hasattr(nn, child.__class__.__name__) and \
                ('pool' in child.__class__.__name__.lower() or 
                'norm' in child.__class__.__name__.lower()):
                if hasattr(nn, child.__class__.__name__.replace('2d', '3d')):
                    TargetClass = getattr(nn, child.__class__.__name__.replace('2d', '3d'))
                    arguments = TargetClass.__init__.__code__.co_varnames[1:]
                    arguments = [a for a in arguments if a not in ['device', 'dtype', 'factory_kwargs']]
                    kwargs = {k: getattr(child, k) for k in arguments}
                    if 'adaptive' in child.__class__.__name__.lower():
                        for k in kwargs.keys():
                            kwargs[k] = _triple_same(kwargs[k])
                    setattr(module, child_name, TargetClass(**kwargs))
                else:
                    raise Exception('No corresponding module in 3D for 2d module {}'.format(child.__class__.__name__))
            elif isinstance(child, nn.Upsample):
                arguments = nn.Upsample.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['mode'] = 'trilinear' if kwargs['mode']=='bilinear' else kwargs['mode']
                setattr(module, child_name, nn.Upsample(**kwargs))
            else:
                self.convert_module(child)
        return module


class SCSEModuleAcs(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            ACSConv(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            ACSConv(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(ACSConv(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    

class AttentionAcs(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModuleAcs(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, attention_type=None):
        super().__init__()
        self.attention1 = AttentionAcs(attention_type, in_channels=in_channels)
        self.decode = nn.Sequential(
            ACSConv(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            ACSConv(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention2 = AttentionAcs(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.decode(x)
        x = self.attention2(x)
        return x


class UNet3dAcsDecoder(nn.Module):
    def __init__(
        self, 
        encoder_channels, 
        decoder_mid_channels, 
        decoder_out_channels, 
        decoder_attention_type=None,
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
            DecoderBlock(in_ch + skip_ch, mid_ch, out_ch, attention_type=decoder_attention_type)
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


class GlobalPooling(nn.Module):
    def __init__(self, dim, agg='max', keepdim=False):
        super().__init__()
        self.dim = dim
        self.agg = agg
        self.keepdim = keepdim

    def forward(self, x):
        if self.agg == 'max':
            return x.max(dim=self.dim, keepdim=self.keepdim)[0]
        elif self.agg == 'mean':
            return x.mean(dim=self.dim, keepdim=self.keepdim)
        else:
            raise ValueError('Unknown aggregation {}'.format(self.agg))


class SegmentationHeadAcs(nn.Module):
    def __init__(self, in_channels, n_classes, depth=None, activation=None, kernel_size=3, upsampling=1):
        super().__init__()

        self.conv3d = ACSConv(in_channels, n_classes, kernel_size=kernel_size, padding=kernel_size // 2)
        if depth is not None:
            self.pooling = nn.Linear(depth, 1)
        else:
            self.pooling = GlobalPooling(dim=-1, agg='max', keepdim=True)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self, x):
        x = self.conv3d(x)
        
        # (B, C, D, H, W) -> (B, C, H, W, D)
        x = x.permute(0, 1, 3, 4, 2)
        # (B, C, H, W, D) -> (B, C, H, W, 1)
        x = self.pooling(x)
        # (B, C, H, W, 1) -> (B, C, H, W)
        x = x.squeeze(-1)

        x = self.upsampling(x)
        x = self.activation(x)
        
        return x


class UNet3dAcs(SegmentationModel):
    def __init__(
        self, 
        encoder,
        encoder_channels, 
        decoder_mid_channels, 
        decoder_out_channels, 
        upsampling,
        depth=None,
        decoder_attention_type=None,
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
            decoder_attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHeadAcs(
            in_channels=decoder_out_channels[-1],
            n_classes=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
            depth=depth,
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
