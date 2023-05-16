import numpy as np
from typing import Callable, Optional, Tuple
from patchify import patchify
from timm.layers import to_2tuple


def pad_divisible_2d(image: np.ndarray, size: tuple):
    """Pad 2D or 3D image to be divisible by size."""
    assert image.ndim in (2, 3)
    assert len(size) == image.ndim

    pad_width = [
        (0, size[i] - image.shape[i] % size[i])
        if image.shape[i] % size[i] != 0 else 
        (0, 0)
        for i in range(image.ndim)
    ]
    return np.pad(
        image, 
        pad_width, 
        'constant', 
        constant_values=0
    )


class InMemorySurfaceVolumeDataset:
    """Dataset for surface volumes."""
    def __init__(
        self, 
        volumes: np.ndarray,
        scroll_masks: np.ndarray,
        pathes: np.ndarray,
        ir_images: Optional[np.ndarray] = None,
        ink_masks: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        n_repeat: int = 1,
    ):
        self.volumes = volumes
        self.scroll_masks = scroll_masks
        self.ir_images = ir_images
        self.ink_masks = ink_masks
        self.pathes = pathes

        self.shape_patches = None
        self.indices = None

        self.transform = transform
        self.patch_size = to_2tuple(patch_size)
        self.n_repeat = n_repeat

        # Patchify
        if patch_size is not None:
            self.volumes, \
            self.scroll_masks, \
            self.ir_images, \
            self.ink_masks, \
            self.pathes, \
            self.shape_patches, \
            self.indices = \
                self.patchify_data()

    def __len__(self) -> int:
        return len(self.volumes) * self.n_repeat
    
    def patchify_data(self):
        """Split data into patches."""
        step = self.patch_size

        (
            volumes, 
            scroll_masks, 
            ir_images, 
            ink_masks, 
            pathes, 
            shape_patches, 
            indices,
        ) = [], [], [], [], [], [], []
        for i in range(len(self.volumes)):
            # Patchify

            # Volume
            volume_patch_size = (*self.patch_size, self.volumes[i].shape[2])
            volume_step = (*step, self.volumes[i].shape[2])
            volume = pad_divisible_2d(self.volumes[i], volume_patch_size)
            volume_patches = patchify(
                volume, 
                volume_patch_size, 
                step=volume_step
            )

            # Scroll mask
            scroll_mask = pad_divisible_2d(self.scroll_masks[i], self.patch_size)
            scroll_mask_patches = patchify(scroll_mask, self.patch_size, step=step)
            
            # IR image
            ir_image_patches = None
            if self.ir_images is not None:
                ir_image = pad_divisible_2d(self.ir_images[i], self.patch_size)
                ir_image_patches = patchify(ir_image, self.patch_size, step=step)
            
            # Ink mask
            ink_mask_patches = None
            if self.ink_masks is not None:
                ink_mask = pad_divisible_2d(self.ink_masks[i], self.patch_size)
                ink_mask_patches = patchify(ink_mask, self.patch_size, step=step)
            
            # Pathes
            pathes_patches = np.full_like(
                volume_patches[..., 0, 0, 0],
                self.pathes[i],
                dtype=object
            )

            # Shape patches
            shape_patches_patches = np.tile(
                np.array(volume_patches.shape[:2])[None, None, :],
                [*volume_patches.shape[:2], 1],
            )

            # Indices
            indices_patches = np.meshgrid(
                np.arange(volume_patches.shape[0]),
                np.arange(volume_patches.shape[1]),
            )
            indices_patches = np.stack(indices_patches, axis=-1)
            
            # Flatten patches
            volume_patches = volume_patches.reshape(-1, *volume_patches.shape[-3:])
            scroll_mask_patches = scroll_mask_patches.reshape(-1, *scroll_mask_patches.shape[-2:])
            if ir_image_patches is not None:
                ir_image_patches = ir_image_patches.reshape(-1, *ir_image_patches.shape[-2:])
            if ink_mask_patches is not None:
                ink_mask_patches = ink_mask_patches.reshape(-1, *ink_mask_patches.shape[-2:])
            pathes_patches = pathes_patches.flatten()
            shape_patches_patches = shape_patches_patches.reshape(-1, *shape_patches_patches.shape[-1:])
            indices_patches = indices_patches.reshape(-1, *indices_patches.shape[-1:])

            # Drop empty patches (no 1s in scroll mask)
            mask = (scroll_mask_patches > 0).any(axis=(-1, -2))
            volume_patches = volume_patches[mask]
            scroll_mask_patches = scroll_mask_patches[mask]
            if ir_image_patches is not None:
                ir_image_patches = ir_image_patches[mask]
            if ink_mask_patches is not None:
                ink_mask_patches = ink_mask_patches[mask]
            pathes_patches = pathes_patches[mask]
            shape_patches_patches = shape_patches_patches[mask]
            indices_patches = indices_patches[mask]
            
            # Append
            volumes.append(volume_patches)
            scroll_masks.append(scroll_mask_patches)
            if ir_image_patches is not None:
                ir_images.append(ir_image_patches)
            if ink_mask_patches is not None:
                ink_masks.append(ink_mask_patches)
            pathes.append(pathes_patches)
            shape_patches.append(shape_patches_patches)
            indices.append(indices_patches)

        # Concatenate
        volumes = np.concatenate(volumes, axis=0)
        scroll_masks = np.concatenate(scroll_masks, axis=0)
        if len(ir_images) > 0:
            ir_images = np.concatenate(ir_images, axis=0)
        else:
            ir_images = None
        if len(ink_masks) > 0:
            ink_masks = np.concatenate(ink_masks, axis=0)
        else:
            ink_masks = None
        pathes = np.concatenate(pathes, axis=0)
        shape_patches = np.concatenate(shape_patches, axis=0)
        indices = np.concatenate(indices, axis=1)

        return volumes, scroll_masks, ir_images, ink_masks, pathes, shape_patches, indices
    
    def __getitem__(
        self, 
        idx
    ) -> Tuple[
        np.ndarray,  # volume, (H, W, D)
        np.ndarray,  # mask, (H, W)
        Optional[np.ndarray],  # optional IR image, (H, W)
        Optional[np.ndarray],  # optional labels, (H, W)
    ]:
        idx = idx % len(self.volumes)  # for repeated case

        # Always here
        image = self.volumes[idx]
        masks = [self.scroll_masks[idx]]
        path = self.pathes[idx]

        # Only in train
        if self.ir_images is not None:
            masks.append(self.ir_images[idx])
        if self.ink_masks is not None:
            masks.append(self.ink_masks[idx])

        # Only in val / test
        indices = None
        if self.indices is not None:
            indices = self.indices[idx]
        shape_patches = None
        if self.shape_patches is not None:
            shape_patches = self.shape_patches[idx]

        output = {
            'image': image,
            'masks': masks,
            'path': path,
            'indices': indices,
            'shape_patches': shape_patches,
        }

        if self.transform is not None:
            output = self.transform(**output)

        return output
