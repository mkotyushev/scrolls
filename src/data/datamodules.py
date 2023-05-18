import logging
import cv2
import albumentations as A
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from src.data.datasets import InMemorySurfaceVolumeDataset
from src.data.transforms import (
    RandomCropVolumeInside2dMask, 
    CenterCropVolume, 
    RandomScaleResize, 
    RotateX, 
    ToCHWD, 
    ToWritable
)
from src.utils.utils import surface_volume_collate_fn


logger = logging.getLogger(__name__)


N_SLICES = 65
MAX_PIXEL_VALUE = 65536
def read_data(surface_volume_dirs):
    # Volumes
    volumes = []
    for root in surface_volume_dirs:
        root = Path(root)
        volume = []
        for i in range(N_SLICES):
            volume.append(
                cv2.imread(
                    str(root / 'surface_volume' / f'{i:02}.tif'),
                    cv2.IMREAD_UNCHANGED
                )
            )
        volume = np.stack(volume).transpose(1, 2, 0)
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

    logger.info(f'Loaded {len(volumes)} volumes from {surface_volume_dirs} dirs')

    return volumes, scroll_masks, ir_images, ink_masks


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


class SurfaceVolumeDatamodule(LightningDataModule):
    """Base datamodule for surface volume data."""
    def __init__(
        self,
        surface_volume_dirs: List[str] | str = './data/train',	
        surface_volume_dirs_test: Optional[List[str] | str] = None,	
        val_dir_indices: Optional[List[int] | int] = None,
        crop_size: int = 256,
        crop_size_z: int = 48,
        img_size: int = 256,
        img_size_z: int = 64,
        resize_xy: str = 'crop',
        use_imagenet_stats: bool = True,
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
        self.val_transform = None
        self.test_transform = None

        # Imagenets mean and std
        self.train_volume_mean = sum((0.485, 0.456, 0.406)) / 3
        self.train_volume_std = sum((0.229, 0.224, 0.225)) / 3

        self.collate_fn = surface_volume_collate_fn

    def build_transforms(self) -> None:
        train_resize_transform, val_resize_transform = [], []
        if self.hparams.resize_xy == 'crop':
            # Crop to crop_size and resize to img_size
            train_resize_transform = [ 
                RandomCropVolumeInside2dMask(
                    height=self.hparams.crop_size, 
                    width=self.hparams.crop_size, 
                    depth=None,
                    always_apply=True,
                    crop_mask_index=0,
                ),
                A.Resize(
                    height=self.hparams.img_size, 
                    width=self.hparams.img_size, 
                    always_apply=True,
                )
            ]
            # Resize anyway: crop is done in dataset
            # for that case
            val_resize_transform = [
                A.Resize(
                    height=self.hparams.img_size, 
                    width=self.hparams.img_size, 
                    always_apply=True,
                )
            ]
        elif self.hparams.resize_xy == 'resize':
            # Simply resize to img_size
            train_resize_transform = val_resize_transform = [
                A.Resize(
                    height=self.hparams.img_size, 
                    width=self.hparams.img_size, 
                    always_apply=True,
                )
            ]
        elif self.hparams.resize_xy == 'none':  # memory hungry
            pass
        else:
            raise ValueError(f'Unknown resize_xy: {self.hparams.resize_xy}')

        self.train_transform = A.Compose(
            [
                CenterCropVolume(
                    height=None, 
                    width=None,
                    depth=self.hparams.crop_size_z,
                    always_apply=True,
                ),
                *train_resize_transform,
                ToWritable(),
                RotateX(p=0.5, limit=1, value=0, mask_value=0),
                A.Rotate(p=0.5, limit=30, value=0, mask_value=0),
                RandomScaleResize(p=0.5, scale_limit=0.1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
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
        self.val_transform = self.test_transform = A.Compose(
            [
                CenterCropVolume(
                    height=None, 
                    width=None,
                    depth=self.hparams.crop_size_z,
                    always_apply=True,
                ),
                *val_resize_transform,
                ToWritable(),
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

        if self.train_dataset is None:
            if (
                self.val_dataset is None and 
                self.hparams.val_dir_indices is not None and
                self.hparams.val_dir_indices
            ):
                # Train
                train_surface_volume_dirs = [
                    d for i, d in enumerate(self.hparams.surface_volume_dirs)
                    if i not in self.hparams.val_dir_indices
                ]
                volumes, scroll_masks, ir_images, ink_masks = \
                    read_data(train_surface_volume_dirs)
                
                # Update mean and std
                if not self.hparams.use_imagenet_stats:
                    self.train_volume_mean, self.train_volume_std = \
                        calc_mean_std(volumes, scroll_masks)

                # Will sample patches from each volume 200 times
                # each epoch for 'crop' or will provide whole volume once
                # for 'resize' or 'none'
                train_n_repeat = 200 if self.hparams.resize_xy == 'crop' else 1
                
                self.train_dataset = InMemorySurfaceVolumeDataset(
                    volumes=volumes, 
                    scroll_masks=scroll_masks, 
                    pathes=train_surface_volume_dirs,
                    ir_images=ir_images, 
                    ink_masks=ink_masks,
                    transform=self.train_transform,
                    # Patches are generated in dataloader randomly 
                    # or whole volume is provided
                    patch_size=None,
                    n_repeat=train_n_repeat,
                )

                val_surface_volume_dirs = [
                    d for i, d in enumerate(self.hparams.surface_volume_dirs)
                    if i in self.hparams.val_dir_indices
                ]
                volumes, scroll_masks, ir_images, ink_masks = \
                    read_data(val_surface_volume_dirs)
                
                # Controls whether val dataset will be cropped to patches (crop_size)
                # or whole volume is provided (None)
                val_patch_size = \
                    None \
                    if self.hparams.resize_xy in ['resize', 'none'] else \
                    self.hparams.crop_size
                
                self.val_dataset = InMemorySurfaceVolumeDataset(
                    volumes=volumes, 
                    scroll_masks=scroll_masks, 
                    pathes=val_surface_volume_dirs,
                    ir_images=ir_images, 
                    ink_masks=ink_masks,
                    transform=self.val_transform,
                    patch_size=val_patch_size,
                    n_repeat=1,  # no repeats for validation
                )
            else:
                volumes, scroll_masks, ir_images, ink_masks = \
                    read_data(self.hparams.surface_volume_dirs)
                
                # Update mean and std
                if not self.hparams.use_imagenet_stats:
                    self.train_volume_mean, self.train_volume_std = \
                        calc_mean_std(volumes, scroll_masks)

                self.train_dataset = InMemorySurfaceVolumeDataset(
                    volumes=volumes, 
                    scroll_masks=scroll_masks, 
                    pathes=self.hparams.surface_volume_dirs,
                    ir_images=ir_images, 
                    ink_masks=ink_masks,
                    transform=self.train_transform,
                    patch_size=None,  # patches are generated in dataloader randomly
                    n_repeat=200,  # sample patches from each volume 200 times
                )
                self.val_dataset = None
        if (
            self.test_dataset is None and 
            self.hparams.surface_volume_dirs_test is not None
        ):
            volumes, scroll_masks, ir_images, ink_masks = \
                read_data(self.hparams.surface_volume_dirs_test)
            self.test_dataset = InMemorySurfaceVolumeDataset(
                volumes=volumes, 
                scroll_masks=scroll_masks, 
                pathes=self.hparams.surface_volume_dirs_test,
                ir_images=ir_images, 
                ink_masks=ink_masks,
                transform=self.test_transform,
                patch_size=self.crop_size,  # patch without overlap
                n_repeat=1,  # no repeats for test
            )
        
        # To rebuild normalization
        self.reset_transforms()

    def reset_transforms(self):
        self.build_transforms()

        if self.train_dataset is not None:
            self.train_dataset.transform = self.train_transform
        if self.val_dataset is not None:
            self.val_dataset.transform = self.val_transform
        if self.test_dataset is not None:
            self.test_dataset.transform = self.test_transform

    def train_dataloader(self) -> DataLoader:
        batch_size = self.hparams.batch_size
        if self.hparams.batch_size_full_apply_epoch is not None and self.trainer is not None:
            if self.trainer.current_epoch >= self.hparams.batch_size_full_apply_epoch:
                batch_size = self.hparams.batch_size_full
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            sampler=None,
            shuffle=True
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
