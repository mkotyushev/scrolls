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
from src.data.transforms import RandomCropVolumeInside2dMask, CenterCropVolume, RotateX, ToCHWD, ToWritable
from src.utils.utils import surface_volume_collate_fn


N_SLICES = 65
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

    return volumes, scroll_masks, ir_images, ink_masks


class SurfaceVolumeDatamodule(LightningDataModule):
    """Base datamodule for surface volume data."""
    def __init__(
        self,
        surface_volume_dirs: List[str] | str = './data/train',	
        surface_volume_dirs_test: Optional[List[str] | str] = None,	
        val_dir_indices: Optional[List[int] | int] = None,
        crop_size: int = 256,
        crop_size_z: int = 48,
        batch_size: int = 32,
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

        self.collate_fn = surface_volume_collate_fn

    def build_transforms(self) -> None:
        self.train_transform = A.Compose(
            [
                RandomCropVolumeInside2dMask(
                    height=2 * self.hparams.crop_size, 
                    width=2 * self.hparams.crop_size, 
                    depth=2 * self.hparams.crop_size_z,
                    always_apply=True,
                    crop_mask_index=0,
                ),
                ToWritable(),
                RotateX(p=0.5, limit=10),
                A.Rotate(p=0.5, limit=30),
                A.RandomScale(p=0.5, scale_limit=0.2),
                CenterCropVolume(
                    height=self.hparams.crop_size, 
                    width=self.hparams.crop_size,
                    depth=self.hparams.crop_size_z,
                    always_apply=True,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    max_pixel_value=65536,
                    mean=sum((0.485, 0.456, 0.406)) / 3,
                    std=sum((0.229, 0.224, 0.225)) / 3,
                    always_apply=True,
                ),
                ToTensorV2(),
                ToCHWD(always_apply=True),
            ],
        )
        self.val_transform = self.test_transform = A.Compose(
            [
                ToWritable(),  # TODO: remove this
                CenterCropVolume(
                    height=self.hparams.crop_size, 
                    width=self.hparams.crop_size,
                    depth=self.hparams.crop_size_z,
                    always_apply=True,
                ),
                A.Normalize(
                    max_pixel_value=65536,
                    mean=sum((0.485, 0.456, 0.406)) / 3,
                    std=sum((0.229, 0.224, 0.225)) / 3,
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
                self.train_dataset = InMemorySurfaceVolumeDataset(
                    volumes=volumes, 
                    scroll_masks=scroll_masks, 
                    pathes=train_surface_volume_dirs,
                    ir_images=ir_images, 
                    ink_masks=ink_masks,
                    transform=self.train_transform,
                    patch_size=None,  # patches are generated in dataloader randomly
                    n_repeat=200,  # sample patches from each volume 200 times
                )

                val_surface_volume_dirs = [
                    d for i, d in enumerate(self.hparams.surface_volume_dirs)
                    if i in self.hparams.val_dir_indices
                ]
                volumes, scroll_masks, ir_images, ink_masks = \
                    read_data(val_surface_volume_dirs)
                self.val_dataset = InMemorySurfaceVolumeDataset(
                    volumes=volumes, 
                    scroll_masks=scroll_masks, 
                    pathes=train_surface_volume_dirs,
                    ir_images=ir_images, 
                    ink_masks=ink_masks,
                    transform=self.val_transform,
                    patch_size=self.hparams.crop_size,  # patch without overlap
                    n_repeat=1,  # no repeats for validation
                )
            else:
                volumes, scroll_masks, ir_images, ink_masks = \
                    read_data(self.hparams.surface_volume_dirs)
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            sampler=None,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
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
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
