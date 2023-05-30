""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from copy import deepcopy
from enum import Enum
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, to_3tuple, trunc_normal_, _assert, ClassifierHead
from timm.models._builder import build_model_with_cfg, _filter_kwargs
from timm.models._features_fx import register_notrace_function
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['SwinTransformerV2']  # model_registry will add each entrypoint fn to this

_int_or_tuple_3_t = Union[int, Tuple[int, int, int]]



def window_partition(x, window_size: Tuple[int, int, int]):
    """
    Args:
        x: (B, H, W, D, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], D // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int, int], img_size: Tuple[int, int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], window_size[2], C)
        window_size (Tuple[int, int, int]): Window size
        img_size (Tuple[int, int, int]): Image size

    Returns:
        x: (B, H, W, D, C)
    """
    H, W, D = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1,  # B
        H // window_size[0], 
        W // window_size[1], 
        D // window_size[2], 
        window_size[0], 
        window_size[1], 
        window_size[2], 
        C
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(-1, H, W, D, C)
    return x



class Format(str, Enum):
    NCDHW = 'NCDHW'
    NCHWD = 'NCHWD'
    NHWDC = 'NHWDC'
    NCL = 'NCL'
    NLC = 'NLC'


def nchwd_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWDC:
        x = x.permute(0, 2, 3, 4, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwdc_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NCHWD:
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    elif fmt == Format.NCDHW:
        x = x.permute(0, 4, 3, 2, 1).contiguous()
    elif fmt == Format.NLC:
        x = x.flatten(1, 3)
    elif fmt == Format.NCL:
        x = x.flatten(1, 3).transpose(1, 2)
    return x


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHWD

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
            _assert(D == self.img_size[2], f"Input image depth ({D}) doesn't match model ({self.img_size[2]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        elif self.output_fmt != Format.NCHWD:
            x = nchwd_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            pretrained_window_size=[0, 0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww, Wd
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w,
            relative_coords_d])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2*Wd-1, 3
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        # TODO: check if it is correct way to introduce depth dimension
        relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wd, Wh*Ww*Wd
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], 
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wd,Wh*Ww*Wd,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wd, Wh*Ww*Wd
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2Block(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            qkv_bias=True,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = to_3tuple(input_resolution)
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int, int] = ws
        self.shift_size: Tuple[int, int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=to_3tuple(pretrained_window_size),
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W, D = self.input_resolution
            img_mask = torch.zeros((1, H, W, D, 1))  # 1 H W D 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    for d in (
                            slice(0, -self.window_size[2]),
                            slice(-self.window_size[2], -self.shift_size[2]),
                            slice(-self.shift_size[2], None)):
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _calc_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        target_window_size = to_3tuple(target_window_size)
        target_shift_size = to_3tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        B, H, W, D, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)  # B H' W' D' C

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B, H, W, D, C = x.shape
        x = x + self.drop_path1(self.norm1(self._attn(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = x.reshape(B, H, W, D, C)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 3 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 3 * dim
        self.reduction = nn.Linear(8 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        B, H, W, D, C = x.shape
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        _assert(D % 2 == 0, f"x depth ({D}) is not even.")
        # (B, H, W, D, C) -> (B, H // 2, 2, W // 2, 2, D // 2, 2, C)
        x = x.reshape(B, H // 2, 2, W // 2, 2, D // 2, 2, C)
        # (B, H // 2, 2, W // 2, 2, D // 2, 2, C) -> (B, H // 2, W // 2, D // 2, Cd, Cw, Ch, C)
        x = x.permute(0, 1, 3, 5, 6, 4, 2, 7)
        # (B, H // 2, W // 2, D // 2, Cd, Cw, Ch, C) -> (B, H // 2, W // 2, D // 2, C')
        x = x.flatten(4)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinTransformerV2Stage(nn.Module):
    """ A Swin Transformer V2 Stage.
    """

    def __init__(
            self,
            dim,
            out_dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            downsample=False,
            mlp_ratio=4.,
            qkv_bias=True,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
            output_nchw=False,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_3_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_3_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer stage (layer).
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Head dropout rate.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
        """
        super().__init__()

        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.output_fmt = 'NHWDC'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWDC',
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerV2Stage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_sizes[i],
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]

        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    native_checkpoint = 'head.fc.weight' in state_dict
    out_dict = {}
    import re
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table', 'attn_mask')]):
            continue  # skip buffers that should not be persistent
        if not native_checkpoint:
            # skip layer remapping for updated checkpoints
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = k.replace('head.', 'head.fc.')
        out_dict[k] = v
      
    return out_dict


def _create_swin_transformer_v2(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 1, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformerV2, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)
    return model



def map_pretrained_2d_to_3d(model_2d, model_3d):
    model_2d_state_dict = model_2d.state_dict()
    model_3d_state_dict = deepcopy(model_3d.state_dict())
    for key, value in model_2d_state_dict.items():
        if key in model_3d_state_dict:
            if value.shape == model_3d_state_dict[key].shape:
                model_3d_state_dict[key] = value
            else:
                print(f'{key}: {value.shape} -> {model_3d_state_dict[key].shape}')
                if key == 'patch_embed.proj.weight':
                    # Repeat the 2D patch embedding weight along the depth dimension
                    model_3d_state_dict[key] = value.unsqueeze(-1).repeat(
                        1, 1, 1, 1, model_3d_state_dict[key].shape[-1]
                    ) / model_3d_state_dict[key].shape[-1]
                elif key.endswith('cpb_mlp.0.weight'):
                    model_3d_state_dict[key][:, :2] = value
                elif key.endswith('downsample.reduction.weight'):
                    model_3d_state_dict[key] = value.repeat(1, 2)
                else:
                    raise ValueError(
                        f'{key}\'s shape {value.shape} is not equal '
                        f'to {model_3d_state_dict[key].shape} and '
                        f'no special handling is defined'
                    )
        else:
            raise ValueError(f'{key} not found in model_3d_state_dict')
    print(model_3d.load_state_dict(model_3d_state_dict))
    return model_3d


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256, 32), 'pool_size': (8, 8, 8),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


def _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H, W, D) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 4
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 4
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


default_cfgs = generate_default_cfgs({
    'swinv2_3d_base_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'swinv2_3d_base_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384, 32), pool_size=(12, 12), crop_pct=1.0,
    ),
    'swinv2_3d_large_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'swinv2_3d_large_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384, 32), pool_size=(12, 12), crop_pct=1.0,
    ),

    'swinv2_3d_tiny_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth',
    ),
    'swinv2_3d_tiny_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth',
    ),
    'swinv2_3d_small_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth',
    ),
    'swinv2_3d_small_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth',
    ),
    'swinv2_3d_base_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth',
    ),
    'swinv2_3d_base_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth',
    ),

    'swinv2_3d_base_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192, 32), pool_size=(6, 6)
    ),
    'swinv2_3d_large_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192, 32), pool_size=(6, 6)
    ),
})


@register_model
def swinv2_3d_tiny_window16_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_3d_tiny_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_tiny_window8_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_3d_tiny_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_small_window16_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_3d_small_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_small_window8_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_3d_small_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_base_window16_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_3d_base_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_base_window8_256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_3d_base_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_base_window12_192(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_3d_base_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_base_window12to16_192to256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_3d_base_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_base_window12to24_192to384(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(
        window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_3d_base_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_large_window12_192(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer_v2(
        'swinv2_3d_large_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_large_window12to16_192to256(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(
        window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_3d_large_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_3d_large_window12to24_192to384(pretrained=False, **kwargs) -> SwinTransformerV2:
    """
    """
    model_args = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_3d_large_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'swinv2_3d_base_window12_192_22k': 'swinv2_3d_base_window12_192.ms_in22k',
    'swinv2_3d_base_window12to16_192to256_22kft1k': 'swinv2_3d_base_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_3d_base_window12to24_192to384_22kft1k': 'swinv2_3d_base_window12to24_192to384.ms_in22k_ft_in1k',
    'swinv2_3d_large_window12_192_22k': 'swinv2_3d_large_window12_192.ms_in22k',
    'swinv2_3d_large_window12to16_192to256_22kft1k': 'swinv2_3d_large_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_3d_large_window12to24_192to384_22kft1k': 'swinv2_3d_large_window12to24_192to384.ms_in22k_ft_in1k',
})
