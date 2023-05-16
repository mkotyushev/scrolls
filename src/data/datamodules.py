import cv2
import albumentations as A
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from src.data.datasets import InMemorySurfaceVolumeDataset, SubsetWithTransformAndRepeats
from src.data.transforms import RandomCropVolume, CenterCropVolume, RotateX, ToCHWD, ToWritable
from src.utils.utils import surface_volume_collate_fn


N_SLICES = 65
def read_volumes(shared_storage, surface_volume_dirs):
    # Volumes
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
        shared_storage.append(volume)


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
                RandomCropVolume(
                    height=2 * self.hparams.crop_size, 
                    width=2 * self.hparams.crop_size, 
                    depth=2 * self.hparams.crop_size_z,
                    always_apply=True,
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
            shared_storage = []
            read_volumes(shared_storage, self.hparams.surface_volume_dirs)

            if (
                self.val_dataset is None and 
                self.hparams.val_dir_indices is not None and
                self.hparams.val_dir_indices
            ):
                dataset = InMemorySurfaceVolumeDataset(
                    volumes_shared_storage=shared_storage,
                    surface_volume_dirs=self.hparams.surface_volume_dirs,
                    transform=None,
                )
                train_indices = sorted(list(
                    set(range(len(dataset))) - 
                    set(self.hparams.val_dir_indices)
                ))
                self.train_dataset = SubsetWithTransformAndRepeats(
                    dataset,
                    train_indices,
                    transform=self.train_transform,
                    n_repeat=200,
                )
                self.val_dataset = SubsetWithTransformAndRepeats(
                    dataset,
                    self.hparams.val_dir_indices,
                    transform=self.val_transform,
                    n_repeat=1,
                )
            else:
                self.train_dataset = InMemorySurfaceVolumeDataset(
                    volumes_shared_storage=shared_storage,
                    surface_volume_dirs=self.hparams.surface_volume_dirs,
                    transform=self.train_transform,
                )
                self.val_dataset = None
        if (
            self.test_dataset is None and 
            self.hparams.surface_volume_dirs_test is not None
        ):
            shared_storage = []
            read_volumes(shared_storage, self.hparams.surface_volume_dirs_test)

            self.test_dataset = InMemorySurfaceVolumeDataset(
                volumes_shared_storage=shared_storage,
                surface_volume_dirs=self.hparams.surface_volume_dirs_test,
                transform=self.test_transform,
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