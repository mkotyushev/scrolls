import logging
import math
import albumentations as A
from torch.utils.data.sampler import WeightedRandomSampler
from typing import List, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.constants import MAX_PIXEL_VALUE
from src.data.datasets import (
    InMemorySurfaceVolumeDataset,
    OnlineSurfaceVolumeDataset, 
)
from src.data.transforms import (
    CopyPastePositive,
    CutMix,
    MixUp,
    RandomCropVolumeInside2dMask, 
    CenterCropVolume, 
    ResizeVolume, 
    ToCHWD,
    ToFloatMasks,
)
from src.utils.utils import (
    surface_volume_collate_fn,
    read_data, 
    calc_mean_std,
    get_num_samples_and_weights,
)


logger = logging.getLogger(__name__)


class SurfaceVolumeDatamodule(LightningDataModule):
    """Base datamodule for surface volume data."""
    def __init__(
        self,
        surface_volume_dirs: List[str] | str = './data/train',	
        surface_volume_dirs_test: Optional[List[str] | str] = None,	
        z_shift_scale_pathes_test: Optional[List[str] | str] = None,
        val_dir_indices: Optional[List[int] | int] = None,
        z_start: int = 24,
        z_end: int = 48,
        img_size: int = 256,
        img_size_z: int = 64,
        z_scale_limit: float = 2.0,
        resize_xy: str = 'crop',
        use_imagenet_stats: bool = True,
        use_mix: bool = False,
        batch_size: int = 32,
        batch_size_full: int = 32,
        batch_size_full_apply_epoch: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()

        if isinstance(surface_volume_dirs, str):
            surface_volume_dirs = [surface_volume_dirs]
        if isinstance(surface_volume_dirs_test, str):
            surface_volume_dirs_test = [surface_volume_dirs_test]
        if isinstance(val_dir_indices, int):
            val_dir_indices = [val_dir_indices]
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.train_transform_mix = None
        self.val_transform = None
        self.test_transform = None

        # Train dataset scale is min volume scale for surface_volume_dirs
        volume_scales = []
        for root in self.hparams.surface_volume_dirs:
            scale = 1.0
            if 'fragments_downscaled_2' in root:
                scale = 2.0
            else:
                logger.warning(f'Unknown scale for {root}, assuming 1.0') 
            volume_scales.append(scale)
        self.train_dataset_scale = min(volume_scales)
        logger.info(f'train_dataset_scale: {self.train_dataset_scale}')

        # Imagenets mean and std
        self.train_volume_mean = sum(IMAGENET_DEFAULT_MEAN) / 3
        self.train_volume_std = sum(IMAGENET_DEFAULT_STD) / 3

        self.collate_fn = surface_volume_collate_fn

    def build_transforms(self) -> None:        
        train_pre_resize_transform = []

        # Rotation limits: so as reflection padding
        # is used, no need to limit rotation
        rotate_limit_degrees_xy = 45

        if self.hparams.resize_xy == 'crop':
            train_pre_resize_transform = [ 
                RandomCropVolumeInside2dMask(
                    base_size=self.hparams.img_size, 
                    base_depth=self.hparams.img_size_z,
                    scale=(0.5, 2.0),
                    ratio=(0.9, 1.1),
                    scale_z=(1 / self.hparams.z_scale_limit, self.hparams.z_scale_limit),
                    always_apply=True,
                    crop_mask_index=0,
                )
            ]
        elif self.hparams.resize_xy == 'resize':
            # Simply resize to img_size later
            train_pre_resize_transform = []
        elif self.hparams.resize_xy == 'none':  # memory hungry
            pass
        else:
            raise ValueError(f'Unknown resize_xy: {self.hparams.resize_xy}')

        # If mix is used, then train_transform_mix is used
        # (additional costly sampling & augmentation from dataset)
        # and post transform is done in train_transform_mix
        # otherwise post transform is done in train_transform 
        post_transform = []
        if not self.hparams.use_mix:
            post_transform = [
                ToTensorV2(),
                ToCHWD(always_apply=True),
            ]

        self.train_transform = A.Compose(
            [
                *train_pre_resize_transform,
                A.Rotate(
                    p=0.5, 
                    limit=rotate_limit_degrees_xy, 
                    crop_border=False,
                ),
                ResizeVolume(
                    height=self.hparams.img_size, 
                    width=self.hparams.img_size,
                    depth=self.hparams.img_size_z,
                    always_apply=True,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ], 
                    p=0.4
                ),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.CoarseDropout(
                    max_holes=1, 
                    max_width=int(self.hparams.img_size * 0.3), 
                    max_height=int(self.hparams.img_size * 0.3), 
                    mask_fill_value=0, p=0.5
                ),
                A.Normalize(
                    max_pixel_value=MAX_PIXEL_VALUE,
                    mean=self.train_volume_mean,
                    std=self.train_volume_std,
                    always_apply=True,
                ),
                *post_transform,
            ],
        )
        self.train_transform_mix = None
        if self.hparams.use_mix:
            self.train_transform_mix = A.Compose(
                [
                    ToFloatMasks(),
                    A.OneOf(
                        [
                            CutMix(
                                width=int(self.hparams.img_size * 0.3), 
                                height=int(self.hparams.img_size * 0.3), 
                                p=1.0,
                                always_apply=False,
                            ),
                            MixUp(alpha=3.0, beta=3.0, p=1.0, always_apply=False),
                            CopyPastePositive(mask_index=2, p=1.0, always_apply=False),
                        ],
                        p=0.5,
                    ),
                    ToTensorV2(),
                    ToCHWD(always_apply=True),
                ],
            )
        self.val_transform = self.test_transform = A.Compose(
            [
                CenterCropVolume(
                    height=None, 
                    width=None,
                    depth=math.ceil(self.hparams.img_size_z * self.hparams.z_scale_limit),
                    strict=True,
                    always_apply=True,
                    p=1.0,
                ),
                ResizeVolume(
                    height=self.hparams.img_size, 
                    width=self.hparams.img_size,
                    depth=self.hparams.img_size_z,
                    always_apply=True,
                ),
                A.Normalize(
                    max_pixel_value=MAX_PIXEL_VALUE,
                    mean=self.train_volume_mean,
                    std=self.train_volume_std,
                    always_apply=True,
                ),
                ToTensorV2(),
                ToCHWD(always_apply=True),
            ],
        )

    def setup(self, stage: str = None) -> None:
        self.build_transforms()

        # Controls whether val dataset will be cropped to patches (img_size)
        # with step (img_size // 2) or whole volume is provided (None)
        val_test_patch_size = \
            None \
            if self.hparams.resize_xy in ['resize', 'none'] else \
            self.hparams.img_size
        val_test_patch_step = \
            None \
            if self.hparams.resize_xy in ['resize', 'none'] else \
            self.hparams.img_size // 2

        # Train
        train_surface_volume_dirs, val_surface_volume_dirs = [], []
        if (
            self.hparams.surface_volume_dirs is not None and 
            self.hparams.val_dir_indices is not None
        ):
            train_surface_volume_dirs = [
                d for i, d in enumerate(self.hparams.surface_volume_dirs)
                if i not in self.hparams.val_dir_indices
            ]
            val_surface_volume_dirs = [
                d for i, d in enumerate(self.hparams.surface_volume_dirs)
                if i in self.hparams.val_dir_indices
            ]

        if self.train_dataset is None and train_surface_volume_dirs:
            volumes, \
            scroll_masks, \
            ir_images, \
            ink_masks, \
            subtracts, \
            divides = \
                read_data(
                    train_surface_volume_dirs, 
                    z_start=self.hparams.z_start,
                    z_end=self.hparams.z_end,
                )
            
            # Update mean and std
            if not self.hparams.use_imagenet_stats:
                self.train_volume_mean, self.train_volume_std = \
                    calc_mean_std(volumes, scroll_masks)
            
            self.train_dataset = InMemorySurfaceVolumeDataset(
                volumes=volumes, 
                scroll_masks=scroll_masks, 
                pathes=train_surface_volume_dirs,
                ir_images=ir_images, 
                ink_masks=ink_masks,
                transform=self.train_transform,
                transform_mix=self.train_transform_mix,
                # Patches are generated in dataloader randomly 
                # or whole volume is provided
                patch_size=None,
                subtracts=subtracts,
                divides=divides,
            )

        if self.val_dataset is None and val_surface_volume_dirs:
            volumes, \
            scroll_masks, \
            ir_images, \
            ink_masks, \
            subtracts, \
            divides = \
                read_data(
                    val_surface_volume_dirs, 
                    z_start=self.hparams.z_start,
                    z_end=self.hparams.z_end,
                )
            
            self.val_dataset = InMemorySurfaceVolumeDataset(
                volumes=volumes, 
                scroll_masks=scroll_masks, 
                pathes=val_surface_volume_dirs,
                ir_images=ir_images, 
                ink_masks=ink_masks,
                transform=self.val_transform,
                patch_size=val_test_patch_size,
                patch_step=val_test_patch_step,
                subtracts=subtracts,
                divides=divides,
            )

        if self.test_dataset is None and self.hparams.surface_volume_dirs_test is not None:
            self.test_dataset = OnlineSurfaceVolumeDataset(
                pathes=self.hparams.surface_volume_dirs_test,
                z_shift_scale_pathes=self.hparams.z_shift_scale_pathes_test,
                z_start=self.hparams.z_start,
                z_end=self.hparams.z_end,
                transform=self.test_transform,
                patch_size=val_test_patch_size,
                patch_step=val_test_patch_step,
            )
        
        # To rebuild normalization
        self.reset_transforms()

    def reset_transforms(self):
        self.build_transforms()

        if self.train_dataset is not None:
            self.train_dataset.transform = self.train_transform
            self.train_dataset.transform_mix = self.train_transform_mix
        if self.val_dataset is not None:
            self.val_dataset.transform = self.val_transform
        if self.test_dataset is not None:
            self.test_dataset.transform = self.test_transform

    def train_dataloader(self) -> DataLoader:
        batch_size = self.hparams.batch_size
        if self.hparams.batch_size_full_apply_epoch is not None and self.trainer is not None:
            if self.trainer.current_epoch >= self.hparams.batch_size_full_apply_epoch:
                batch_size = self.hparams.batch_size_full
        
        # Will sample patches num_samples times 
        # (where num_samples == max (mask == 1 area) / (crop area) per all masks)
        # each epoch for 'crop' or will provide whole volume once
        # for 'resize' or 'none'
        # 
        # For 'crop' such sampling provides uniform sampling of volumes, 
        # but oversampling volumes (in patches terms) with smaller area.
        sampler, shuffle = None, True
        if self.hparams.resize_xy == 'crop':
            num_samples, weights = get_num_samples_and_weights(
                scroll_masks=self.train_dataset.scroll_masks,
                img_size=self.hparams.img_size,
            )
            sampler = WeightedRandomSampler(
                weights=weights, 
                replacement=True, 
                num_samples=num_samples,
            )
            shuffle = None

            if self.trainer.current_epoch == 0:
                logger.info(f'num_samples: {num_samples}, weights: {weights}')
        
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        batch_size = self.hparams.batch_size
        if self.hparams.batch_size_full_apply_epoch is not None and self.trainer is not None:
            if self.trainer.current_epoch >= self.hparams.batch_size_full_apply_epoch:
                batch_size = self.hparams.batch_size_full
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "test dataset is not defined"
        batch_size = self.hparams.batch_size
        if self.hparams.batch_size_full_apply_epoch is not None and self.trainer is not None:
            if self.trainer.current_epoch >= self.hparams.batch_size_full_apply_epoch:
                batch_size = self.hparams.batch_size_full
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
