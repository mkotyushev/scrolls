import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Optional, Union
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import default_collate
from weakref import proxy
from timm.layers.format import nhwc_to, Format
from timm.models import FeatureListNet
from patchify import unpatchify
from torchvision.utils import make_grid

from src.model.unet_3d_acs import AcsConvnextWrapper


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return



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


class PredictionTargetPreviewAgg(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: int = 4):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.previews = {}

    def reset(self):
        self.previews = {}

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        target: torch.Tensor, 
        indices: torch.LongTensor, 
        pathes: list[str], 
        shape_patches: torch.LongTensor,
    ):
        # Get preview images
        input = F.interpolate(
            input.float(),
            scale_factor=1 / self.preview_downscale, 
            mode='bilinear',
            align_corners=False, 
        )
        probas = F.interpolate(
            probas.float().unsqueeze(1),  # interpolate as (N, C, H, W)
            scale_factor=1 / self.preview_downscale, 
            mode='bilinear', 
            align_corners=False, 
        ).squeeze(1)
        target = F.interpolate(
            target.float().unsqueeze(1),  # interpolate as (N, C, H, W)
            scale_factor=1 / self.preview_downscale,
            mode='bilinear',
            align_corners=False, 
        ).squeeze(1)

        # To CPU * types
        input, probas, target, indices, shape_patches = \
            input.cpu(), \
            probas.cpu(), \
            target.cpu(), \
            indices.cpu().long(), \
            shape_patches.cpu().long()

        patch_size = probas.shape[-2:]

        # Place patches on the preview images
        for i in range(probas.shape[0]):
            path = '/'.join(pathes[i].split('/')[-2:])
            if f'proba_{path}' not in self.previews:
                shape = [
                    *shape_patches[i].tolist(),
                    *patch_size,
                ]
                self.previews[f'input_{path}'] = torch.zeros(shape, dtype=torch.float32)
                self.previews[f'proba_{path}'] = torch.zeros(shape, dtype=torch.uint8)
                self.previews[f'target_{path}'] = torch.zeros(shape, dtype=torch.uint8)

            patch_index_w, patch_index_h = indices[i].tolist()

            self.previews[f'input_{path}'][patch_index_h, patch_index_w] = \
                input[i]
            self.previews[f'proba_{path}'][patch_index_h, patch_index_w] = \
                (probas[i] * 255).byte()
            self.previews[f'target_{path}'][patch_index_h, patch_index_w] = \
                (target[i] * 255).byte()
    
    def compute(self):
        captions, previews_unpatchified = [], []
        for name, preview in self.previews.items():
            previews_unpatchified.append(
                unpatchify(
                    preview.numpy(), 
                    (
                        preview.shape[0] * preview.shape[2],  # H' * H
                        preview.shape[1] * preview.shape[3],  # W' * W
                    )
                )
            )
            captions.append(name)

        return captions, previews_unpatchified
    

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
