import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from typing import Any, Callable, Dict, List, Optional, Tuple
from patchify import patchify
from timm.layers import to_2tuple

from src.utils.utils import (
    normalize_volume, 
    get_z_volume_mean_per_z, 
    fit_x_shift_scale, 
    build_nan_or_outliers_mask, 
    interpolate_masked_pixels
)


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
        subtracts: Optional[List[int]] = None,
        divides: Optional[List[int]] = None,
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
        self.subtracts = subtracts
        self.divides = divides

        # Patchify
        if patch_size is not None:
            self.volumes, \
            self.scroll_masks, \
            self.ir_images, \
            self.ink_masks, \
            self.pathes, \
            self.shape_patches, \
            self.indices, \
            self.subtracts, \
            self.divides = \
                self.patchify_data()

    def __len__(self) -> int:
        return len(self.volumes)
    
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
            subtracts,
            divides,
        ) = [], [], [], [], [], [], [], [], []
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
            ).squeeze(2)  # bug in patchify

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
            pathes_patches = np.full(
                volume_patches.shape[:2],
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
                np.arange(volume_patches.shape[1]),
                np.arange(volume_patches.shape[0]),
            )
            indices_patches = np.stack(indices_patches, axis=-1)

            # Subtracts
            subtracts_patches = None
            if self.subtracts is not None:
                subtracts_patches = np.full(
                    volume_patches.shape[:2],
                    self.subtracts[i],
                    dtype=np.float32
                )
            
            # Divides
            divides_patches = None
            if self.divides is not None:
                divides_patches = np.full(
                    volume_patches.shape[:2],
                    self.divides[i],
                    dtype=np.float32
                )

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
            if subtracts_patches is not None:
                subtracts_patches = subtracts_patches[mask]
            if divides_patches is not None:
                divides_patches = divides_patches[mask]
            
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
            if subtracts_patches is not None:
                subtracts.append(subtracts_patches)
            if divides_patches is not None:
                divides.append(divides_patches)

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
        if len(subtracts) > 0:
            subtracts = np.concatenate(subtracts, axis=0)
        else:
            subtracts = None
        if len(divides) > 0:
            divides = np.concatenate(divides, axis=0)
        else:
            divides = None

        return \
            volumes, \
            scroll_masks, \
            ir_images, \
            ink_masks, \
            pathes, \
            shape_patches, \
            indices, \
            subtracts, \
            divides
    
    def __getitem__(
        self, 
        idx
    ) -> Dict[str, Any]:
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

        # Optional
        subtract = None
        if self.subtracts is not None:
            subtract = self.subtracts[idx]
        divide = None
        if self.divides is not None:
            divide = self.divides[idx]

        output = {
            'image': image,  # volume, (H, W, D)
            'masks': masks,  # masks, (H, W) each
            'path': path,  # path, 1
            'indices': indices,  # indices, 2
            'shape_patches': shape_patches,  # shape_patches, 2
            'subtract': subtract,  # subtract, 1
            'divide': divide,  # divide, 1
        }

        if self.transform is not None:
            output = self.transform(**output)

        return output


# Constants of fragment 2
CENTER_Z = 32
Z_TARGET = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64])
VOLUME_MEAN_PER_Z_TARGET = np.array([ 5.13208528e-01,  5.16408885e-01,  5.19434831e-01,  5.21984080e-01,
        5.23897714e-01,  5.25018397e-01,  5.25379059e-01,  5.25026003e-01,
        5.23951138e-01,  5.22097850e-01,  5.19750996e-01,  5.17430883e-01,
        5.15448979e-01,  5.14234214e-01,  5.14447837e-01,  5.16901802e-01,
        5.22654541e-01,  5.32695096e-01,  5.47928805e-01,  5.69591645e-01,
        5.99385291e-01,  6.39242319e-01,  6.90646470e-01,  7.53476899e-01,
        8.24894873e-01,  8.98176349e-01,  9.61862669e-01,  1.00000000e+00,
        9.94738465e-01,  9.31891064e-01,  8.08179900e-01,  6.36992343e-01,
        4.45879897e-01,  2.67190902e-01,  1.26850405e-01,  3.78383233e-02,
       -3.99358636e-16,  3.82105731e-03,  3.56873080e-02,  8.23619561e-02,
        1.33404854e-01,  1.82043457e-01,  2.24758615e-01,  2.60212419e-01,
        2.88509731e-01,  3.10495925e-01,  3.27333343e-01,  3.40104985e-01,
        3.49733725e-01,  3.56971222e-01,  3.62402574e-01,  3.66477129e-01,
        3.69507893e-01,  3.71695538e-01,  3.73246001e-01,  3.74380452e-01,
        3.75340047e-01,  3.76180927e-01,  3.76906288e-01,  3.77474479e-01,
        3.77896668e-01,  3.78197379e-01,  3.78455692e-01,  3.78703337e-01,
        3.78895042e-01])
MINMAX_SUBTRACT, MINMAX_DIVIDE = (22060.014794403214, 9586.303729423242)


def build_z_shift_scale_maps(
    pathes,
    volumes, 
    scroll_masks, 
    subtracts,
    divides,
    z_start=0,
    crop_z_span=8,
    mode='volume_mean_per_z', 
    normalize='minmax', 
    patch_size=(128, 128),
    sigma=0.5,
):
    # Center crop
    z_target = Z_TARGET[CENTER_Z - crop_z_span:CENTER_Z + crop_z_span + 1]
    volume_mean_per_z_target = VOLUME_MEAN_PER_Z_TARGET[CENTER_Z - crop_z_span:CENTER_Z + crop_z_span + 1]

    z_shifts_all, z_scales_all = [], []
    for i, path in enumerate(pathes):
        # Build dataset with patchification
        dataset = InMemorySurfaceVolumeDataset(
            volumes=[volumes[i]],
            scroll_masks=[scroll_masks[i]],
            pathes=[path],
            ir_images=None,
            ink_masks=None,
            transform=None,
            patch_size=patch_size,
            subtracts=[subtracts[i]],
            divides=[divides[i]],
        )

        z_shifts, z_scales = None, None
        for item in tqdm(dataset):
            # For each patch, calculate z_shift and z_scale
            # and store them in z_shifts and z_scales maps
            if z_shifts is None:
                shape = item['shape_patches'].tolist()
                z_shifts = np.full(shape, fill_value=np.nan, dtype=np.float32)
                z_scales = np.full(shape, fill_value=np.nan, dtype=np.float32)

            if not item['masks'][0].all():
                continue

            indices = item['indices']
            volume = normalize_volume(
                item['image'], 
                item['masks'][0], 
                mode=mode, 
                normalize=normalize,
                precomputed=(item['subtract'], item['divide']),
            )
            z, volume_mean_per_z = get_z_volume_mean_per_z(
                volume, item['masks'][0]
            )
            z = z + z_start
        
            z_shift, z_scale = fit_x_shift_scale(z, volume_mean_per_z, z_target, volume_mean_per_z_target)

            z_shifts[indices[1], indices[0]] = z_shift
            z_scales[indices[1], indices[0]] = z_scale

        # Clear outliers & nans
        z_shifts_nan, z_shifts_outliers = build_nan_or_outliers_mask(z_shifts)
        z_scales_nan, z_scales_outliers = build_nan_or_outliers_mask(z_scales)
        
        z_shifts[z_shifts_nan] = z_shifts[~z_shifts_nan].mean()
        z_scales[z_scales_nan] = z_scales[~z_scales_nan].mean()

        mask = z_shifts_outliers | z_scales_outliers
        z_shifts = interpolate_masked_pixels(z_shifts, mask, method='linear')
        z_scales = interpolate_masked_pixels(z_scales, mask, method='linear')

        # Apply filtering
        z_shifts = gaussian_filter(z_shifts, sigma=sigma)
        z_scales = gaussian_filter(z_scales, sigma=sigma)

        # Upscale z_shifts and z_scales maps to the 
        # original (padded) volume size

        # Note: shape could be not equal to the original volume size
        # because of the padding used in patchification
        shape = \
            z_shifts.shape[1] * patch_size[1], \
            z_shifts.shape[0] * patch_size[0]
        z_shifts = cv2.resize(
            z_shifts,
            shape,
            interpolation=cv2.INTER_LINEAR,
        )
        z_scales = cv2.resize(
            z_scales,
            shape,
            interpolation=cv2.INTER_LINEAR,
        )

        # Crop z_shifts and z_scales maps to the
        # original volume size (padding is always 'after')
        z_shifts = z_shifts[
            :volumes[i].shape[0],
            :volumes[i].shape[1],
        ]
        z_scales = z_scales[
            :volumes[i].shape[0],
            :volumes[i].shape[1],
        ]

        z_shifts_all.append(z_shifts)
        z_scales_all.append(z_scales)
    
    return z_shifts_all, z_scales_all
