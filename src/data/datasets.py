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
            self.indices = \
                self.patchify_data()

    def __len__(self) -> int:
        return len(self.volumes) * self.n_repeat
    
    def patchify_data(self):
        """Split data into patches."""
        step = self.patch_size

        volumes, scroll_masks, ir_images, ink_masks, pathes, indices = [], [], [], [], [], []
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
                volume_patches[..., 0, 0],
                self.pathes[i],
                dtype=object
            )

            # Indices
            indices_patches = np.meshgrid(
                np.arange(volume_patches.shape[0]),
                np.arange(volume_patches.shape[1]),
            )
            indices_patches = np.stack(indices_patches).transpose(1, 2, 0)
            
            # Flatten patches
            volume_patches = volume_patches.reshape(-1, *volume_patches.shape[-3:])
            scroll_mask_patches = scroll_mask_patches.reshape(-1, *scroll_mask_patches.shape[-2:])
            if ir_image_patches is not None:
                ir_image_patches = ir_image_patches.reshape(-1, *ir_image_patches.shape[-2:])
            if ink_mask_patches is not None:
                ink_mask_patches = ink_mask_patches.reshape(-1, *ink_mask_patches.shape[-2:])
            pathes_patches = pathes_patches.flatten()
            indices_patches = indices_patches.reshape(-1, 2)
            
            # Append
            volumes.append(volume_patches)
            scroll_masks.append(scroll_mask_patches)
            if ir_image_patches is not None:
                ir_images.append(ir_image_patches)
            if ink_mask_patches is not None:
                ink_masks.append(ink_mask_patches)
            pathes.append(pathes_patches)
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
        indices = np.concatenate(indices, axis=1)

        print(volumes.shape, scroll_masks.shape, ir_images.shape, ink_masks.shape, pathes.shape, indices.shape)

        return volumes, scroll_masks, ir_images, ink_masks, pathes, indices
    
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

        image = self.volumes[idx]
        masks = [self.scroll_masks[idx]]
        if self.ir_images is not None:
            masks.append(self.ir_images[idx])
        if self.ink_masks is not None:
            masks.append(self.ink_masks[idx])

        if self.transform is not None:
            return self.transform(image=image, masks=masks)
        else:
            return {
                'image': image,
                'masks': masks,
            }
