import math
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

from src.model.unet_3d_acs import AcsConvnextWrapper


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
            return [nhwc_to(y, Format('NCHW')) for y in self.model(x)]
        else:
            return self.model(x)


def get_num_layers(model: FeatureListNet):
    return len([key for key in model if 'layers' in key])


def get_feature_channels(model: FeatureListNet | FeatureExtractorWrapper | AcsConvnextWrapper, input_shape):
    is_training = model.training
    model.eval()
    x = torch.randn(1, *input_shape)
    y = model(x)
    if isinstance(model, (FeatureExtractorWrapper, AcsConvnextWrapper)):
        channel_index = 1
    else:
        channel_index = 3
    result = tuple(y_.shape[channel_index] for y_ in y)
    model.train(is_training)
    return result


def _unpatchify2d_avg(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
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

    return image, counts


class PredictionTargetPreviewAgg(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: Optional[int] = 4, metrics=None):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.metrics = metrics
        self.previews = {}
        self.shapes = {}

    def reset(self):
        # Note: metrics are reset in compute()
        self.previews = {}
        self.shapes = {}

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.LongTensor,
        indices: torch.LongTensor, 
        pathes: list[str], 
        shape_patches: torch.LongTensor,
        shape_original: torch.LongTensor,
    ):
        # To CPU & types
        input, probas, target, mask, indices, shape_patches = \
            input.cpu().float().numpy(), \
            probas.cpu().float().numpy(), \
            target.cpu().float().numpy(), \
            mask.cpu().long().numpy(), \
            indices.cpu().long().numpy(), \
            shape_patches.cpu().long().numpy()

        patch_size = probas.shape[-2:]

        # Place patches on the preview images
        for i in range(probas.shape[0]):
            path = '/'.join(pathes[i].split('/')[-2:])
            if f'proba_{path}' not in self.previews:
                shape = [
                    *shape_patches[i].tolist(),
                    *patch_size,
                ]
                self.previews[f'input_{path}'] = np.zeros(shape, dtype=np.float32)
                self.previews[f'proba_{path}'] = np.zeros(shape, dtype=np.float32)
                self.previews[f'target_{path}'] = np.zeros(shape, dtype=np.float32)
                self.previews[f'mask_{path}'] = np.zeros(shape, dtype=np.float32)
                # hack to not change dict size later, actually computed in compute()
                self.previews[f'counts_{path}'] = None
                self.shapes[path] = shape_original[i].tolist()[:2]

            patch_index_w, patch_index_h = indices[i].tolist()

            self.previews[f'input_{path}'][patch_index_h, patch_index_w] = \
                input[i]
            self.previews[f'proba_{path}'][patch_index_h, patch_index_w] = \
                probas[i]
            self.previews[f'target_{path}'][patch_index_h, patch_index_w] = \
                target[i]
            self.previews[f'mask_{path}'][patch_index_h, patch_index_w] = \
                mask[i]
    
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
                self.previews[name.replace('proba', 'counts')] = counts.astype(np.float32)
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

        # Compute metrics if available
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


def get_z_volume_mean_per_z(volume, scroll_mask):
    z = np.arange(volume.shape[2])
    H, W, D = volume.shape
    scroll_mask_flattened_xy = (scroll_mask > 0).flatten()
    volume_flattened_xy = volume.reshape(H * W, D)
    volume_mean_per_z = volume_flattened_xy[
        scroll_mask_flattened_xy
    ].mean(0)

    return z, volume_mean_per_z


def fit_x_shift_scale(x, y, x_target, y_target):
    """Fit x_shift and x_scale so that y_target
    is close to y(x_target) in the least square sense.

    Use scipy.interpolate.interp1d to get y(x_target) from y, x * x_scale + x_shift 
    and scipy.optimize.least_squares to find x_shift and x_scale.
    """
    def fun(p):
        x_shift, x_scale = p
        x_scaled_shifted = x * x_scale + x_shift
        f = interpolate.interp1d(x_scaled_shifted, y, bounds_error=False, fill_value='extrapolate')
        return f(x_target) - y_target
    x_shift, x_scale = optimize.least_squares(
        fun=fun, 
        x0=[0, 1],
        bounds=([-30, 0.1], [30, 10]),
    ).x
    return x_shift, x_scale


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


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path, write_interval):
        super().__init__(write_interval)
        self.output_path = output_path
        self.aggregator = PredictionTargetPreviewAgg(
            preview_downscale=None,
            metrics=None,
        )

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        y, y_pred = pl_module.extract_targets_and_probas_for_metric(prediction, batch)
        self.aggregator.update(
            batch['image'][..., batch['image'].shape[-1] // 2],
            y_pred, 
            y, 
            mask=batch['mask_0'],
            pathes=batch['path'],
            indices=batch['indices'], 
            shape_patches=batch['shape_patches'],
            shape_original=batch['shape_original'],
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Get predictions as images
        captions, previews = self.aggregator.compute()
        self.aggregator.reset()
        
        ids, probas = [], []
        for caption, preview in zip(captions, previews):
            if caption.startswith('proba_'):
                ids.append(caption.split('/')[-1])
                probas.append(preview)

        # Sort by id
        ids, probas = zip(*sorted(zip(ids, probas), key=lambda x: int(x[0])))
        
        # Save
        with open(self.output_path, 'w') as f:
            print("Id,Predicted\n", file=f)
            for i, (id_, proba) in enumerate(zip(ids, probas)):
                starts_ix, lengths = rle(proba)
                inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
                print(f"{id_}," + inklabels_rle, file=f, end="\n" if i != len(ids) - 1 else "")
