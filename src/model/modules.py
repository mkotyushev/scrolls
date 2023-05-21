import logging
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict
from lightning import LightningModule
from typing import Any, Dict, Literal, Optional, Union
from torch import Tensor
from lightning.pytorch.cli import instantiate_class
from torchmetrics import Metric
from torchmetrics.classification import BinaryFBetaScore
from lightning.pytorch.utilities import grad_norm
from unittest.mock import patch
from acsconv.operators import ACSConv

from src.model.smp import Unet, patch_first_conv
from src.model.swin_transformer_v2_pseudo_3d import (
    SwinTransformerV2Pseudo3d, 
    map_pretrained_2d_to_pseudo_3d, 
)
from src.model.unet_2d_agg import Unet2dAgg
from src.model.unet_2d import Unet2d
from src.model.unet_3d_acs import AcsConvnextWrapper, UNet3dAcs, ACSConverterTimm
from src.utils.utils import (
    FeatureExtractorWrapper, 
    PredictionTargetPreviewAgg, 
    PredictionTargetPreviewGrid, 
    get_feature_channels, 
    state_norm,
    convert_to_grayscale
)


logger = logging.getLogger(__name__)


class BaseModule(LightningModule):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = None
        self.cat_metrics = None

        self.configure_metrics()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def configure_metrics(self):
        """Configure task-specific metrics."""

    def bootstrap_metric(self, probas, targets, metric: Metric):
        """Calculate metric on bootstrap samples."""

    @staticmethod
    def check_batch_dims(batch):
        assert all(map(lambda x: len(x) == len(batch[0]), batch)), \
            f'All entities in batch must have the same length, got ' \
            f'{list(map(len, batch))}'

    def remove_nans(self, y, y_pred):
        nan_mask = torch.isnan(y_pred)
        
        if nan_mask.ndim > 1:
            nan_mask = nan_mask.any(dim=1)
        
        if nan_mask.any():
            if not self.hparams.skip_nan:
                raise ValueError(
                    f'Got {nan_mask.sum()} / {nan_mask.numel()} nan values in update_metrics. '
                    f'Use skip_nan=True to skip them.'
                )
            logger.warning(
                f'Got {nan_mask.sum()} / {nan_mask.numel()} nan values in update_metrics. '
                f'Dropping them & corresponding targets.'
            )
            y_pred = y_pred[~nan_mask]
            y = y[~nan_mask]
        return y, y_pred

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y, y_pred = batch[1].detach(), preds[:, 1].detach().float()
        y, y_pred = self.remove_nans(y, y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        return y, y_pred

    def update_metrics(self, span, preds, batch):
        """Update train metrics."""
        y, y_proba = self.extract_targets_and_probas_for_metric(preds, batch)
        self.cat_metrics[span]['probas'].update(y_proba)
        self.cat_metrics[span]['targets'].update(y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            # TODO change to >= somehow
            if self.current_epoch == self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()

    def unfreeze_only_selected(self):
        """
        Unfreeze only layers selected by 
        model.finetuning.unfreeze_layer_names_*.
        """
        if self.hparams.finetuning is not None:
            for name, param in self.named_parameters():
                selected = False

                if 'unfreeze_layer_names_startswith' in self.hparams.finetuning:
                    selected = selected or any(
                        name.startswith(pattern) 
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_startswith']
                    )

                if 'unfreeze_layer_names_contains' in self.hparams.finetuning:
                    selected = selected or any(
                        pattern in name
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_contains']
                    )
                logger.info(f'Param {name}\'s requires_grad == {selected}.')
                param.requires_grad = selected

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'tl_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch[0].shape[0],
            )
        self.update_metrics('train_metrics', preds, batch)

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'vl_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch[0].shape[0],
            )
        self.update_metrics('val_metrics', preds, batch)
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def log_metrics_and_reset(
        self, 
        prefix, 
        on_step=False, 
        on_epoch=True, 
        prog_bar_names=None,
        reset=True,
    ):
        # Get metric span: train or val
        span = None
        if prefix == 'train':
            span = 'train_metrics'
        elif prefix in ['val', 'val_ds']:
            span = 'val_metrics'
        
        # Get concatenated preds and targets
        # and reset them
        probas, targets = \
            self.cat_metrics[span]['probas'].compute().cpu(),  \
            self.cat_metrics[span]['targets'].compute().cpu()
        if reset:
            self.cat_metrics[span]['probas'].reset()
            self.cat_metrics[span]['targets'].reset()

        # Calculate and log metrics
        for name, metric in self.metrics.items():
            metric_value = None
            if prefix == 'val_ds':  # bootstrap
                if self.hparams.n_bootstrap > 0:
                    metric_value = self.bootstrap_metric(probas[:, 1], targets, metric)
                else:
                    logger.warning(
                        f'prefix == val_ds but n_bootstrap == 0. '
                        f'No bootstrap metrics will be calculated '
                        f'and logged.'
                    )
            else:
                metric.update(probas[:, 1], targets)
                metric_value = metric.compute()
                metric.reset()
            
            prog_bar = False
            if prog_bar_names is not None:
                prog_bar = (name in prog_bar_names)

            if metric_value is not None:
                self.log(
                    f'{prefix}_{name}',
                    metric_value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                )

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""

    def get_lr_decayed(self, lr, layer_index, layer_name):
        """
        Get lr decayed by 
            - layer index as (self.hparams.lr_layer_decay ** layer_index) if
              self.hparams.lr_layer_decay is float 
              (useful e. g. when new parameters are in classifer head)
            - layer name as self.hparams.lr_layer_decay[layer_name] if
              self.hparams.lr_layer_decay is dict
              (useful e. g. when pretrained parameters are at few start layers 
              and new parameters are the most part of the model)
        """
        if isinstance(self.hparams.lr_layer_decay, dict):
            for key in self.hparams.lr_layer_decay:
                if layer_name.startswith(key):
                    return lr * self.hparams.lr_layer_decay[key]
            return lr
        elif isinstance(self.hparams.lr_layer_decay, float):
            if self.hparams.lr_layer_decay == 1.0:
                return lr
            else:
                return lr * (self.hparams.lr_layer_decay ** layer_index)

    def build_parameter_groups(self):
        """Get parameter groups for optimizer."""
        names, params = list(zip(*self.named_parameters()))
        num_layers = len(params)
        grouped_parameters = [
            {
                'params': param, 
                'lr': self.get_lr_decayed(
                    self.hparams.optimizer_init['init_args']['lr'], 
                    num_layers - layer_index - 1,
                    name
                )
            } for layer_index, (name, param) in enumerate(self.named_parameters())
        ]
        logger.info(
            f'Number of layers: {num_layers}, '
            f'min lr: {names[0]}, {grouped_parameters[0]["lr"]}, '
            f'max lr: {names[-1]}, {grouped_parameters[-1]["lr"]}'
        )
        return grouped_parameters

    def configure_optimizer(self):
        optimizer = instantiate_class(args=self.build_parameter_groups(), init=self.hparams.optimizer_init)
        return optimizer

    def configure_lr_scheduler(self, optimizer):
        # Convert milestones from total persents to steps
        # for PiecewiceFactorsLRScheduler
        if (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'step'
        ):
            total_steps = len(self.trainer.fit_loop._data_source.dataloader()) * self.trainer.max_epochs
            grad_accum_steps = self.trainer.accumulate_grad_batches
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps / grad_accum_steps) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        elif (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'epoch'
        ):
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * self.trainer.max_epochs) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        
        scheduler = instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.pl_lrs_cfg,
        }

        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams.lr_scheduler_init is None:
            return optimizer

        scheduler = self.configure_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'grad_2.0_norm_total' in norms:
                self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'state_2.0_norm_total' in norms:
                self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])


backbone_name_to_params = {
    'swinv2': {
        'window_size': (8, 8),
        # TODO: SWIN v2 has patch size 4, upsampling at 
        # the last step degrades quality
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'format': 'NHWC',
    },
    'convnext': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'convnextv2': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
}


def build_segmentation(backbone_name, type_, in_channels=1, decoder_attention_type=None, img_size=256):
    """Build segmentation model."""
    backbone_param_key = backbone_name.split('_')[0]
    create_model_kwargs = {}
    if backbone_param_key == 'swinv2':
        create_model_kwargs['img_size'] = img_size
    encoder_2d = timm.create_model(
        backbone_name, 
        features_only=True,
        pretrained=True,
        **create_model_kwargs,
    )
    if type_ == 'pseudo_3d':
        with patch('timm.models.swin_transformer_v2.SwinTransformerV2', SwinTransformerV2Pseudo3d):
            encoder_pseudo_3d = timm.create_model(
                backbone_name, 
                features_only=True,
                pretrained=False,
                window_size=(*backbone_name_to_params[backbone_param_key]['window_size'], in_channels // 4),
                img_size=(img_size, img_size, in_channels),
            )
        encoder = map_pretrained_2d_to_pseudo_3d(encoder_2d, encoder_pseudo_3d)
        patch_first_conv(
            encoder, 
            new_in_channels=1,
            default_in_channels=3, 
            pretrained=True,
            conv_type=nn.Conv3d,
        )
        encoder = FeatureExtractorWrapper(encoder, format=backbone_name_to_params[backbone_param_key]['format'])
        
        model = Unet(
            encoder=encoder,
            encoder_channels=get_feature_channels(
                encoder, 
                input_shape=(1, img_size, img_size, in_channels)
            ),
            decoder_channels=backbone_name_to_params[backbone_param_key]['decoder_channels'],
            classes=1,
            upsampling=backbone_name_to_params[backbone_param_key]['upsampling'],
        )
    elif type_.startswith('2d'):
        encoder = encoder_2d

        if type_.startswith('2d_agg'):
            in_channels = 1
        patch_first_conv(
            encoder, 
            new_in_channels=in_channels,
            default_in_channels=3, 
            pretrained=True,
            conv_type=nn.Conv2d,
        )

        encoder = FeatureExtractorWrapper(
            encoder, 
            format=backbone_name_to_params[backbone_param_key]['format']
        )
        unet = Unet(
            encoder=encoder,
            encoder_channels=get_feature_channels(
                encoder, 
                input_shape=(in_channels, img_size, img_size)
            ),
            decoder_channels=backbone_name_to_params[backbone_param_key]['decoder_channels'],
            classes=1,
            upsampling=backbone_name_to_params[backbone_param_key]['upsampling'],
            decoder_attention_type=decoder_attention_type,
        )
        
        if type_.startswith('2d_agg'):
            agg = type_.split('2d_agg_')[-1]
            model = Unet2dAgg(unet, agg=agg)
        else:
            model = Unet2d(unet)
    elif type_.startswith('3d_acs'):
        encoder = encoder_2d
        patch_first_conv(
            encoder, 
            new_in_channels=1,
            default_in_channels=3, 
            pretrained=True,
            conv_type=nn.Conv2d,
        )

        # Only convert here, not wrap because 
        # ACSConverterTimm is not a nn.Module
        ACSConverterTimm(encoder)

        encoder = FeatureExtractorWrapper(
            encoder, 
            format=backbone_name_to_params[backbone_param_key]['format']
        )

        model = UNet3dAcs(
            encoder=encoder,
            encoder_channels=get_feature_channels(
                encoder,
                input_shape=(1, img_size, img_size, in_channels)
            ),
            decoder_mid_channels=backbone_name_to_params[backbone_param_key]['decoder_mid_channels'],
            decoder_out_channels=backbone_name_to_params[backbone_param_key]['decoder_out_channels'],
            classes=1,
            depth=in_channels // backbone_name_to_params[backbone_param_key]['upsampling'],
            decoder_attention_type=decoder_attention_type,
            upsampling=backbone_name_to_params[backbone_param_key]['upsampling'],
        )
    else:
        raise NotImplementedError(f'Unknown type {type_}.')

    # TODO: compile model, now blows up with
    # AssertionError: expected size 64==64, stride 4096==1 at dim=1
    # model = torch.compile(model)

    return model


class SegmentationModule(BaseModule):
    def __init__(
        self, 
        type_: str = 'pseudo_3d',
        decoder_attention_type: Literal[None, 'scse'] = None,
        backbone_name: str = 'swinv2_tiny_window8_256.ms_in1k',
        in_channels: int = 6,
        label_smoothing: float = 0.0,
        pos_weight: float = 1.0,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
        img_size=256,
    ):
        super().__init__(
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning,
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
            n_bootstrap=n_bootstrap,
            skip_nan=skip_nan,
            prog_bar_names=prog_bar_names,
        )
        self.save_hyperparameters()
        self.model = build_segmentation(
            backbone_name, 
            type_, 
            in_channels=in_channels,
            decoder_attention_type=decoder_attention_type,
            img_size=img_size,
        )

        if finetuning is not None and finetuning['unfreeze_before_epoch'] == 0:
            self.unfreeze()
        else:
            self.unfreeze_only_selected()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""
        preds = self.model(batch['image'])

        # 3d_acs_weights outputs probabilities, not logits
        weight = torch.where(
            batch['mask_2'] == 1,
            torch.tensor(self.hparams.pos_weight, dtype=torch.float32, device=batch['mask_2'].device),
            torch.tensor(1.0, dtype=torch.float32, device=batch['mask_2'].device),
        ).flatten()

        losses = {
            'bce': F.binary_cross_entropy_with_logits(
                preds.squeeze(1).float().flatten(),
                batch['mask_2'].float().flatten(),
                reduction='mean',
                weight=weight,
            ),
        }
        total_loss = sum(losses.values())
        return total_loss, losses, preds

    def configure_metrics(self):
        """Configure task-specific metrics."""
        self.metrics = ModuleDict(
            {
                'train_metrics': ModuleDict(
                    {
                        'f05': BinaryFBetaScore(beta=0.5),
                        'preview': PredictionTargetPreviewGrid(preview_downscale=16, n_images=9),
                    }
                ),
                'val_metrics': ModuleDict(
                    {
                        'f05': BinaryFBetaScore(beta=0.5),
                        'preview': PredictionTargetPreviewAgg(preview_downscale=16),
                    }
                ),
            }
        )
        self.cat_metrics = None

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'tl_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch['image'].shape[0],
            )
        
        for metric_name, metric in self.metrics['train_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewGrid):  # Epoch-level
                metric.update(
                    batch['image'][..., batch['image'].shape[-1] // 2],
                    y_pred, 
                    y, 
                    pathes=batch['path'],
                )
            else:
                y, y_pred = self.extract_targets_and_probas_for_metric(preds, batch)
                metric.update(y_pred.flatten(), y.flatten())
                self.log(
                    f't_{metric_name}',
                    metric.compute(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch['image'].shape[0],
                )
                metric.reset()

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'vl_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch['image'].shape[0],
            )
        
        y, y_pred = self.extract_targets_and_probas_for_metric(preds, batch)
        y_masked = y.flatten()[batch['mask_0'].flatten() == 1]
        y_pred_masked = y_pred.flatten()[batch['mask_0'].flatten() == 1]
        for metric in self.metrics['val_metrics'].values():
            if isinstance(metric, PredictionTargetPreviewAgg) and batch['indices'] is not None:
                metric.update(
                    batch['image'][..., batch['image'].shape[-1] // 2],
                    y_pred, 
                    y, 
                    pathes=batch['path'],
                    indices=batch['indices'], 
                    shape_patches=batch['shape_patches'],
                )
            else:
                metric.update(y_pred_masked.flatten(), y_masked.flatten())
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        if self.metrics is None:
            return

        for metric_name, metric in self.metrics['train_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewGrid):
                captions, previews = metric.compute()
                self.trainer.logger.log_image(
                    key=f't_{metric_name}',	
                    images=previews,
                    caption=captions,
                    step=self.current_epoch,
                )
                metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return

        for metric_name, metric in self.metrics['val_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewAgg):
                captions, previews = metric.compute()
                self.trainer.logger.log_image(
                    key=f'v_{metric_name}',	
                    images=previews,
                    caption=captions,
                    step=self.current_epoch,
                )
            else:
                self.log(
                    f'v_{metric_name}',
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            metric.reset()

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y, y_pred = batch['mask_2'].detach(), preds.detach().float()
        y, y_pred = self.remove_nans(y, y_pred)
        y_pred = torch.sigmoid(y_pred).squeeze(1)
        return y, y_pred
