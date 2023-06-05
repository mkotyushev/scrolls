import contextlib
import logging
import math
import os
from pathlib import Path
import string
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import default_collate
from weakref import proxy
from timm.layers.format import nhwc_to, Format
from timm.models import FeatureListNet
from patchify import unpatchify, NonUniformStepSizeError
from torchvision.utils import make_grid
from scipy import interpolate, optimize
from tqdm import tqdm

from src.data.constants import N_SLICES, MAX_PIXEL_VALUE
from src.model.swin_transformer_v2_3d import nhwdc_to, Format as Format3d


logger = logging.getLogger(__name__)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")



class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger):
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class TempSetContextManager:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.old_value)


def surface_volume_collate_fn(batch):
    """Collate function for surface volume dataset.
    batch: list of dicts of key:str, value: np.ndarray | list | None
    output: dict of torch.Tensor
    """
    output = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            if v is None:
                continue

            if k == 'image':
                output[k].append(v)
            elif k == 'masks':
                for i, mask in enumerate(v):
                    output[f'mask_{i}'].append(mask)
            else:
                output[k].append(v)
    
    for k, v in output.items():
        if isinstance(v[0], str) or v[0].dtype == object:
            output[k] = v
        else:
            output[k] = default_collate(v)
    
    return output


def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model, format: Format | str = 'NHWC'):
        super().__init__()
        self.model = model
        self.output_stride = 32
        self.format = format if isinstance(format, Format) else Format(format)

    def __iter__(self):
        return iter(self.model)
    
    def forward(self, x):
        if self.format == Format('NHWC'):
            features = [nhwc_to(y, Format('NCHW')) for y in self.model(x)]
        else:
            features = self.model(x)
        return features


class FeatureExtractorWrapper3d(nn.Module):
    def __init__(self, model, output_format: Format3d | str = 'NHWDC'):
        super().__init__()
        self.model = model
        self.output_stride = 32
        self.output_format = output_format if isinstance(output_format, Format3d) else Format3d(output_format)

    def __iter__(self):
        return iter(self.model)
    
    def forward(self, x):
        # x here is always (B, C, H, W, D)

        # (B, C, D, H, W) -> (B, C, H, W, D)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        if self.output_format == Format3d('NHWDC'):
            features = [nhwdc_to(y, Format3d('NCDHW')) for y in self.model(x)]
        else:
            features = self.model(x)
        return features


def get_num_layers(model: FeatureListNet):
    return len([key for key in model if 'layers' in key])


def get_feature_channels(model, input_shape, output_format='NHWC'):
    is_training = model.training
    model.eval()
    
    x = torch.randn(1, *input_shape).to(next(model.parameters()).device)
    with torch.no_grad():
        y = model(x)
    channel_index = output_format.find('C')
    assert channel_index != -1, \
        f'output_format {output_format} not supported, must contain C'
    assert all(len(output_format) == len(y_.shape) for y_ in y), \
        f'output_format {output_format} does not match output shape {y[0].shape}'
    result = tuple(y_.shape[channel_index] for y_ in y)
    logger.info(f'feature channels: {result}')
    
    model.train(is_training)
    
    return result


def _unpatchify2d_avg(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=np.float16)
    counts = np.zeros(imsize, dtype=np.int32)

    n_h, n_w, p_h, p_w = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)

    # For each patch, add it to the image at the right location
    for i in range(n_h):
        for j in range(n_w):
            image[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += patches[i, j]
            counts[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += 1

    # Average
    counts[counts == 0] = 1
    image /= counts

    image = image.astype(patches.dtype)

    return image, counts


class PredictionTargetPreviewAgg(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: Optional[int] = 4, metrics=None, input_std=1, input_mean=0):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.metrics = metrics
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}
        self.input_std = input_std
        self.input_mean = input_mean

    def reset(self):
        # Note: metrics are reset in compute()
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        mask: torch.LongTensor,
        indices: torch.LongTensor, 
        pathes: list[str], 
        shape_patches: torch.LongTensor,
        shape_original: torch.LongTensor,
        shape_before_padding: torch.LongTensor,
        target: Optional[torch.Tensor] = None, 
    ):
        # To CPU & types
        input, probas, mask, indices, shape_patches, shape_before_padding = \
            ((input.cpu().numpy() * self.input_std + self.input_mean) * 255).astype(np.uint8), \
            probas.cpu().numpy().astype(np.float32), \
            mask.cpu().numpy().astype(np.uint8), \
            indices.cpu().long().numpy(), \
            shape_patches.cpu().long().numpy(), \
            shape_before_padding.cpu().long().numpy()
    
        if target is not None:
            target = target.cpu().numpy().astype(np.uint8)

        patch_size = probas.shape[-2:]

        # Place patches on the preview images
        for i in range(probas.shape[0]):
            path = '/'.join(pathes[i].split('/')[-2:])
            if f'proba_{path}' not in self.previews:
                shape = [
                    *shape_patches[i].tolist(),
                    *patch_size,
                ]
                self.previews[f'input_{path}'] = np.zeros(shape, dtype=np.uint8)
                self.previews[f'proba_{path}'] = np.zeros(shape, dtype=np.float32)
                self.previews[f'target_{path}'] = np.zeros(shape, dtype=np.uint8)
                self.previews[f'mask_{path}'] = np.zeros(shape, dtype=np.uint8)
                # hack to not change dict size later, actually computed in compute()
                self.previews[f'counts_{path}'] = None
                self.shapes[path] = shape_original[i].tolist()[:2]
                self.shapes_before_padding[path] = shape_before_padding[i].tolist()[:2]

            patch_index_w, patch_index_h = indices[i].tolist()

            self.previews[f'input_{path}'][patch_index_h, patch_index_w] = \
                input[i]
            self.previews[f'proba_{path}'][patch_index_h, patch_index_w] = \
                probas[i]
            self.previews[f'mask_{path}'][patch_index_h, patch_index_w] = \
                mask[i]
        
            if target is not None:
                self.previews[f'target_{path}'][patch_index_h, patch_index_w] = \
                    target[i]
    
    def compute(self):
        # Unpatchify
        for name in self.previews:
            path = '_'.join(name.split('_')[1:])
            shape_original = self.shapes[path]
            if name.startswith('proba_'):
                # Average overlapping patches
                self.previews[name], counts = _unpatchify2d_avg(
                    self.previews[name], 
                    shape_original
                )
                self.previews[name.replace('proba', 'counts')] = counts.astype(np.uint8)
            elif name.startswith('counts_'):
                # Do nothing
                pass
            else:
                # Just unpatchify
                self.previews[name] = unpatchify(
                    self.previews[name], 
                    shape_original
                )

        # Zero probas out where mask is zero
        for name in self.previews:
            if name.startswith('proba_'):
                mask = self.previews[name.replace('proba', 'mask')] == 0
                self.previews[name][mask] = 0

        # Crop to shape before padding
        for name in self.previews:
            path = '_'.join(name.split('_')[1:])
            shape_before_padding = self.shapes_before_padding[path]
            self.previews[name] = self.previews[name][
                :shape_before_padding[0], 
                :shape_before_padding[1],
            ]

        # Compute metrics if available
        metric_values = None
        if self.metrics is not None:
            preds, targets = [], []
            for name in self.previews:
                if name.startswith('proba_'):
                    path = '_'.join(name.split('_')[1:])
                    mask = self.previews[f'mask_{path}'] > 0
                    pred = self.previews[name][mask].flatten()
                    target = self.previews[f'target_{path}'][mask].flatten()

                    preds.append(pred)
                    targets.append(target)
            preds = torch.from_numpy(np.concatenate(preds))
            targets = torch.from_numpy(np.concatenate(targets))

            metric_values = {}
            for metric_name, metric in self.metrics.items():
                metric.update(preds, targets)
                metric_values[metric_name] = metric.compute()
                metric.reset()
        
        # Downscale and get captions
        captions, previews = [], []
        for name, preview in self.previews.items():
            if self.preview_downscale is not None:
                preview = cv2.resize(
                    preview,
                    dsize=(0, 0),
                    fx=1 / self.preview_downscale, 
                    fy=1 / self.preview_downscale, 
                    interpolation=cv2.INTER_LINEAR, 
                )
            captions.append(name)
            previews.append(preview)

        return metric_values, captions, previews
    

class PredictionTargetPreviewGrid(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: int = 4, n_images: int = 4):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.n_images = n_images
        self.previews = defaultdict(list)

    def reset(self):
        self.previews = defaultdict(list)

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        target: torch.Tensor, 
        pathes: list[str],
    ):
        # Add images until grid is full
        for i in range(probas.shape[0]):
            path = '/'.join(pathes[i].split('/')[-2:])
            if len(self.previews[f'input_{path}']) < self.n_images:
                # Get preview images
                inp = F.interpolate(
                    input[i].float().unsqueeze(0),
                    scale_factor=1 / self.preview_downscale, 
                    mode='bilinear',
                    align_corners=False, 
                ).cpu()
                proba = F.interpolate(
                    probas[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                    scale_factor=1 / self.preview_downscale, 
                    mode='bilinear', 
                    align_corners=False, 
                ).cpu()
                targ = F.interpolate(
                    target[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                    scale_factor=1 / self.preview_downscale,
                    mode='bilinear',
                    align_corners=False, 
                ).cpu()

                self.previews[f'input_{path}'].append(inp)
                self.previews[f'proba_{path}'].append((proba * 255).byte())
                self.previews[f'target_{path}'].append((targ * 255).byte())
    
    def compute(self):
        captions = list(self.previews.keys())
        preview_grids = [
            make_grid(
                torch.cat(v, dim=0), 
                nrow=int(self.n_images ** 0.5)
            ).float()
            for v in self.previews.values()
        ]

        return captions, preview_grids
    

def convert_to_grayscale(model, backbone_name):
    if backbone_name == 'swinv2_tiny_window8_256.ms_in1k':
        model.patch_embed.proj.in_channels = 1
        model.patch_embed.proj.weight = nn.Parameter(model.patch_embed.proj.weight.mean(dim=1, keepdim=True))
    elif backbone_name == 'convnext_small.in12k_ft_in1k_384':
        model.stem_0.in_channels = 1
        model.stem_0.weight = nn.Parameter(model.stem_0.weight.mean(dim=1, keepdim=True))
    else:
        raise ValueError(f'backbone {backbone_name} not supported')

    return model


def calculate_minmax_mean_per_z(dataset):
    _, mean = get_z_dataset_mean_per_z(dataset, z_start=0)
    min_, max_ = mean.min(), mean.max()
    return min_, max_


def calculate_statistics(volume, scroll_mask, mode, normalize):
    assert mode in ['volume', 'volume_mean_per_z']
    assert normalize in ['minmax', 'meanstd', 'quantile']

    H, W, D = volume.shape
    
    # Get only scroll area
    scroll_mask_flattened_xy = (scroll_mask > 0).flatten()
    volume_flattened_xy = volume.reshape(H * W, D)
    volume_flattened_xy_scroll = volume_flattened_xy[
        scroll_mask_flattened_xy
    ]

    # On which array to calculate statistics
    volume_mean_per_z = volume_flattened_xy_scroll.mean(0)
    if mode == 'volume':
        arr = volume_flattened_xy_scroll
    elif mode == 'volume_mean_per_z':
        arr = volume_mean_per_z

    # Calculate statistics
    if normalize == 'minmax':
        subtract = arr.min()
        divide = arr.max() - subtract
    elif normalize == 'meanstd':
        subtract = arr.mean()
        divide = arr.std()
    elif normalize == 'quantile':
        subtract = np.quantile(arr, 0.05)
        divide = np.quantile(arr, 0.95) - subtract

    return subtract, divide


def normalize_volume(volume, scroll_mask, mode='volume_mean_per_z', normalize='quantile', precomputed=None):
    assert mode in ['volume', 'volume_mean_per_z']
    assert normalize in ['minmax', 'meanstd', 'quantile']

    # Calculate statistics
    if precomputed is not None:
        subtract, divide = precomputed
    else:
        subtract, divide = calculate_statistics(volume, scroll_mask, mode, normalize)

    # Normalize
    volume = (volume - subtract) / divide

    return volume


def get_z_dataset_mean_per_z(dataset, z_start):
    depth = dataset[0]['image'].shape[2]
    z = np.arange(z_start, z_start + depth)

    sum_, count = np.zeros(depth, dtype=np.uint64), 0
    for item in tqdm(dataset, desc='get_z_dataset_mean_per_z'):
        volume, scroll_mask = item['image'], item['masks'][0]
        scroll_mask = (scroll_mask > 0)[:, :, None]
        sum_ += np.sum(volume, where=scroll_mask, axis=(0, 1))
        count += np.sum(scroll_mask, axis=(0, 1))

    mean = sum_ / count
    logger.info(f'get_z_dataset_mean_per_z: sum: {sum_}, count: {count}, mean: {mean}')
    
    return z, mean


def get_z_volume_mean_per_z(volume, scroll_mask, z_start):
    z = np.arange(z_start, z_start + volume.shape[2])
    H, W, D = volume.shape
    scroll_mask_flattened_xy = (scroll_mask > 0).flatten()
    volume_flattened_xy = volume.reshape(H * W, D)
    volume_mean_per_z = volume_flattened_xy[
        scroll_mask_flattened_xy
    ].mean(0)

    return z, volume_mean_per_z


def fit_x_shift_scale(x, y, x_target, y_target, model='no_y_scale'):
    """Fit x_shift and scale so that f(x_target),
    where f(x) is the interpolation of y(x * scale + x_shift),
    is close to y_target in the least square sense.

    Use scipy.interpolate.interp1d to get y(x_target) from y, x * scale + x_shift 
    and scipy.optimize.least_squares to find x_shift and scale.
    
    Model 'no_y':
        - shift corresponds to different leveling of the 
        papirus w. r. t. the default, so it needs to be compensated
        by shifting the x
        - scale corresponds to different width of the papirus
        w. r. t. the default.
    Model 'independent_y_scale': 
        Same as model 'no_y' but additionally scale the y as well via
        independent scale parameter: optical density of the papira 
        could be different.
    Model 'independent_y_shift_scale': 
        Same as model 'independent_y_scale' but additionally scale and 
        shift the y as well via independent scale parameter: optical 
        density of the papira could be different.
    Model 'beer_lambert_law':
        Same as model 1 but additionally scale the y as well via
        exponential of minus independent total absorbance (absorbance * traveled length) 
        parameter multiplied by the change of x scale.
    """
    assert model in ['no_y', 'independent_y_scale', 'independent_y_shift_scale', 'beer_lambert_law']
    
    # Not sure if least_squares is scale invariant, so scale variables closer to 0
    X_SHIFT_MULTIPLIER = 30
    X_SCALE_MULTIPLIER = 1
    Y_SHIFT_MULTIPLIER = 2e4
    Y_SCALE_MULTIPLIER = 1
    ABSORBANCE_MULTIPLIER = 2

    if model == 'no_y':
        def fun(p):
            x_shift, x_scale = p
            x_shift *= X_SHIFT_MULTIPLIER
            x_scale *= X_SCALE_MULTIPLIER
            x_scaled_shifted = x * x_scale + x_shift
            f = interpolate.interp1d(x_scaled_shifted, y, bounds_error=False, fill_value='extrapolate')
            return f(x_target) - y_target
        x_shift, x_scale = optimize.least_squares(
            fun=fun, 
            x0=[0, 1],
            bounds=([-1, 0.1], [1, 10]),
        ).x
        y_scale = 1
        y_shift = 0
        
        x_shift *= X_SHIFT_MULTIPLIER
        x_scale *= X_SCALE_MULTIPLIER
    elif model == 'independent_y_scale':
        def fun(p):
            x_shift, x_scale, y_scale = p
            x_shift *= X_SHIFT_MULTIPLIER
            x_scale *= X_SCALE_MULTIPLIER
            y_scale *= Y_SCALE_MULTIPLIER
            x_scaled_shifted = x * x_scale + x_shift
            f = interpolate.interp1d(x_scaled_shifted, y, bounds_error=False, fill_value='extrapolate')
            mult = y_scale
            return f(x_target) * mult - y_target
        x_shift, x_scale, y_scale = optimize.least_squares(
            fun=fun, 
            x0=[0, 1, 1],
            bounds=([-1, 0.1, 0.1], [1, 10, 3]),
        ).x
        y_shift = 0
        x_shift *= X_SHIFT_MULTIPLIER
        x_scale *= X_SCALE_MULTIPLIER
        y_scale *= Y_SCALE_MULTIPLIER
    elif model == 'independent_y_shift_scale':
        def fun(p):
            x_shift, x_scale, y_shift, y_scale = p
            x_shift *= X_SHIFT_MULTIPLIER
            x_scale *= X_SCALE_MULTIPLIER
            y_shift *= Y_SHIFT_MULTIPLIER
            y_scale *= Y_SCALE_MULTIPLIER
            x_scaled_shifted = x * x_scale + x_shift
            f = interpolate.interp1d(x_scaled_shifted, y, bounds_error=False, fill_value='extrapolate')
            return f(x_target) * y_scale + y_shift - y_target
        x_shift, x_scale, y_scale, y_shift = optimize.least_squares(
            fun=fun, 
            x0=[0, 1, 1, 0],
            bounds=([-1, 0.1, 0.1, -1], [1, 10, 3, 1]),
        ).x
        x_shift *= X_SHIFT_MULTIPLIER
        x_scale *= X_SCALE_MULTIPLIER
        y_shift *= Y_SHIFT_MULTIPLIER
        y_scale *= Y_SCALE_MULTIPLIER
    elif model == 'beer_lambert_law':
        def fun(p):
            x_shift, x_scale, total_absorbance = p
            x_shift *= X_SHIFT_MULTIPLIER
            x_scale *= X_SCALE_MULTIPLIER
            total_absorbance *= ABSORBANCE_MULTIPLIER
            x_scaled_shifted = x * x_scale + x_shift
            f = interpolate.interp1d(x_scaled_shifted, y, bounds_error=False, fill_value='extrapolate')
            mult = np.exp(-total_absorbance * (x_scale - 1))
            return f(x_target) * mult - y_target
        x_shift, x_scale, total_absorbance = optimize.least_squares(
            fun=fun, 
            x0=[0, 1, 1e-3],
            bounds=([-1, 0.1, 0.0], [1, 10, 1]),
        ).x
        x_shift *= X_SHIFT_MULTIPLIER
        x_scale *= X_SCALE_MULTIPLIER
        total_absorbance *= ABSORBANCE_MULTIPLIER
        y_scale = np.exp(-total_absorbance * (x_scale - 1))
        y_shift = 0
    
    return x_shift, x_scale, y_shift, y_scale



# https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
def interpolate_masked_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def build_nan_or_outliers_mask(arr):
    is_nan = np.isnan(arr)
    is_outlier = (arr < np.quantile(arr[~is_nan], 0.05)) | (arr > np.quantile(arr[~is_nan], 0.95))

    return is_nan, is_outlier


def scale_shift_volume(volume, z_shift, z_scale, center_crop_z):
    H, W, D = volume.shape
    volume_flattened_xy = volume.reshape(H * W, D)

    # Crop
    center_z_scaled = (D // 2 - z_shift) / z_scale
    center_crop_z_scaled = center_crop_z / z_scale
    
    z_start = max(math.floor(center_z_scaled - center_crop_z_scaled / 2), 0)
    z_end = min(math.ceil(center_z_scaled + center_crop_z_scaled / 2), D)
    
    volume_flattened_xy = volume_flattened_xy[:, z_start:z_end]

    # Scale
    volume_flattened_xy = cv2.resize(
        volume_flattened_xy, 
        (center_crop_z, H*W),
        fx=1, 
        fy=1,
        interpolation=cv2.INTER_LINEAR,
    )

    volume = volume_flattened_xy.reshape(H, W, center_crop_z)
    return volume


# Fast run length encoding, from https://www.kaggle.com/code/hackerpoet/even-faster-run-length-encoder/script
def rle(img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths


# https://stackoverflow.com/questions/49555991/
# can-i-create-a-local-numpy-random-seed
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path, images_output_dir=None, image_postfix=None):
        super().__init__('batch_and_epoch')
        self.output_path = output_path
        self.images_output_dir = images_output_dir
        if image_postfix is None:
            # Generate random alphanumeric string, seed is calculated from
            # sys.argv to make it reproducible but different for different
            # runs. Use numpy random
            seed = hash(tuple(sys.argv)) % (2 ** 32)
            with temp_seed(seed):
                image_postfix = ''.join(
                    np.random.choice(list(string.ascii_letters + string.digits))
                    for _ in range(10)
                )

        self.image_postfix = image_postfix
        self.aggregator = PredictionTargetPreviewAgg(
            preview_downscale=None,
            metrics=None,
        )

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        _, y_pred = pl_module.extract_targets_and_probas_for_metric(prediction, batch)
        self.aggregator.update(
            batch['image'][..., batch['image'].shape[-1] // 2],
            y_pred, 
            target=None, 
            mask=batch['mask_0'],
            pathes=batch['path'],
            indices=batch['indices'], 
            shape_patches=batch['shape_patches'],
            shape_original=batch['shape_original'],
            shape_before_padding=batch['shape_before_padding'],
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Get predictions as images
        _, captions, previews = self.aggregator.compute()
        self.aggregator.reset()
        
        ids, probas = [], []
        for caption, preview in zip(captions, previews):
            if caption.startswith('proba_'):
                ids.append(caption.split('/')[-1])
                probas.append(preview)

        # Sort by id
        ids, probas = zip(*sorted(zip(ids, probas), key=lambda x: x[0]))
        
        # Save
        with open(self.output_path, 'w') as f:
            print("Id,Predicted", file=f)
            for i, (id_, proba) in tqdm(enumerate(zip(ids, probas))):
                starts_ix, lengths = rle(proba)
                inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
                print(f"{id_}," + inklabels_rle, file=f, end="\n" if i != len(ids) - 1 else "")
        
        # Save images
        if self.images_output_dir is not None:
            for i, (id_, proba) in tqdm(enumerate(zip(ids, probas))):
                out_path = os.path.join(self.images_output_dir, f'{id_}_{self.image_postfix}.png')
                cv2.imwrite(out_path, (proba * 255).astype(np.uint8))



def calculate_pad_width(image_shape, image_ndim, step):
    return [
        (0, math.ceil(image_shape[i] / step[i]) * step[i] - image_shape[i])
        if image_shape[i] % step[i] != 0 else 
        (0, 0)
        for i in range(image_ndim)
    ]


def pad_divisible_2d(image: np.ndarray, size: tuple, step: Optional[tuple] = None):
    """Pad 2D or 3D image to be divisible by size."""
    assert image.ndim in (2, 3)
    assert len(size) == image.ndim

    if step is None:
        step = size

    pad_width = calculate_pad_width(image.shape, image.ndim, step)
    return np.pad(
        image, 
        pad_width, 
        'constant', 
        constant_values=0
    )


def read_data(surface_volume_dirs, z_start=None, z_end=None):
    if z_start is None:
        z_start = 0
    if z_end is None:
        z_end = N_SLICES

    # Volumes
    volumes = []
    for root in surface_volume_dirs:
        root = Path(root)
        volume = None
        for i, layer_index in tqdm(enumerate(range(z_start, z_end))):
            v = cv2.imread(
                str(root / 'surface_volume' / f'{layer_index:02}.tif'),
                cv2.IMREAD_UNCHANGED
            )
            if volume is None:
                volume = np.zeros((*v.shape, z_end - z_start), dtype=np.uint16)
            volume[..., i] = v
        volumes.append(volume)

    # Masks: binary masks of scroll regions
    scroll_masks = []
    for root in surface_volume_dirs:
        root = Path(root)
        scroll_masks.append(
            (
                cv2.imread(
                    str(root / 'mask.png'),
                    cv2.IMREAD_GRAYSCALE
                ) > 0
            ).astype(np.uint8)
        )

    # (Optional) IR images: grayscale images
    ir_images = []
    for root in surface_volume_dirs:
        root = Path(root)
        path = root / 'ir.png'
        if path.exists():
            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )
            ir_images.append(image)
    if len(ir_images) == 0:
        ir_images = None
    
    # (Optional) labels: binary masks of ink
    ink_masks = []
    for root in surface_volume_dirs:
        root = Path(root)
        path = root / 'inklabels.png'
        if path.exists():
            image = (
                cv2.imread(
                    str(path),
                    cv2.IMREAD_GRAYSCALE
                ) > 0
            ).astype(np.uint8)
            ink_masks.append(image)
    if len(ink_masks) == 0:
        ink_masks = None

    # Calculate statistics
    subtracts, divides = [], []
    for volume, scroll_mask in zip(volumes, scroll_masks):
        subtract, divide = calculate_statistics(
            volume, 
            scroll_mask, 
            mode='volume_mean_per_z', 
            normalize='minmax'
        )
        subtracts.append(subtract)
        divides.append(divide)

    logger.info(f'Loaded {len(volumes)} volumes from {surface_volume_dirs} dirs')
    logger.info(f'Statistics: subtracts={subtracts}, divides={divides}')

    return \
        volumes, \
        scroll_masks, \
        ir_images, \
        ink_masks, \
        subtracts, \
        divides


def calc_mean_std(volumes, scroll_masks):
    # mean, std across all volumes
    sums, sums_sq, ns = [], [], []
    for volume, scroll_mask in zip(volumes, scroll_masks):
        scroll_mask = scroll_mask > 0
        sums.append((volume[scroll_mask] / MAX_PIXEL_VALUE).sum())
        sums_sq.append(((volume[scroll_mask] / MAX_PIXEL_VALUE) ** 2).sum())
        ns.append(scroll_mask.sum() * N_SLICES)
    mean = sum(sums) / sum(ns)

    sum_sq = 0
    for sum_, sum_sq_, n in zip(sums, sums_sq, ns):
        sum_sq += (sum_sq_ - 2 * sum_ * mean + mean ** 2 * n)
    std = np.sqrt(sum_sq / sum(ns))

    return mean, std


def get_num_samples_and_weights(scroll_masks, img_size):
    assert all(
        [
            scroll_mask.min() == 0 and scroll_mask.max() == 1
            for scroll_mask in scroll_masks
        ]
    )
    areas = [
        scroll_mask.sum()
        for scroll_mask in scroll_masks
    ]
    num_samples = sum(
        [
            math.ceil(area / (img_size ** 2))
            for area in areas
        ]
    )
    weights = np.array(areas) / sum(areas)
    return num_samples, weights


def rotate_limit_to_min_scale(rotate_limit_deg, proj=True):
    rotate_limit_rad = np.deg2rad(rotate_limit_deg)
    if proj:
        scale = np.sqrt(1 / (1 - np.sin(2 * rotate_limit_rad)))
    else:
        scale = np.cos(rotate_limit_rad) + np.sin(rotate_limit_rad)
    return scale


def copy_crop_pad_2d(img, size, bbox, fill_value=0):
    """Create new image by cropping bbox from img with right and bottom padding to size.
        bbox: (h_start, w_start, h_end, w_end)
    """
    assert len(size) == 2
    img_cropped = np.full((*size, *img.shape[2:]), fill_value=fill_value, dtype=img.dtype)
    img_cropped[:bbox[2]-bbox[0], :bbox[3]-bbox[1]] = img[
        bbox[0]:bbox[2],
        bbox[1]:bbox[3]
    ]
    return img_cropped
