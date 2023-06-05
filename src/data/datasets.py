import logging
import cv2
import pyvips
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from typing import Any, Callable, Dict, List, Optional, Tuple
from patchify import patchify
from timm.layers import to_2tuple

from src.data.constants import N_SLICES, Z_TARGET, VOLUME_MEAN_PER_Z_TARGET, CENTER_Z
from src.scripts.z_shift_scale.scale import (
    apply_z_shift_scale, 
    build_z_shift_scale_transform, 
    calculate_input_z_range
)
from src.utils.utils import (
    calculate_pad_width,
    copy_crop_pad_2d,
    normalize_volume, 
    get_z_volume_mean_per_z, 
    fit_x_shift_scale, 
    build_nan_or_outliers_mask, 
    interpolate_masked_pixels,
    pad_divisible_2d
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
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
        transform_mix: Optional[Callable] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        patch_step: Optional[Tuple[int, int]] = None,
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
        self.transform_mix = transform_mix
        if patch_step is None:
            patch_step = patch_size
        self.patch_size = to_2tuple(patch_size) if patch_size is not None else None
        self.patch_step = to_2tuple(patch_step) if patch_step is not None else None
        self.patch_indices = None
        self.subtracts = subtracts
        self.divides = divides
        self.shape_original = None
        self.shape_before_padding = None

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
            self.divides, \
            self.shape_original, \
            self.patch_indices, \
            self.shape_before_padding = \
                self.patchify_data()

    def __len__(self) -> int:
        if self.patch_indices is None:
            return len(self.volumes)
        else:
            return sum(map(len, self.patch_indices))
    
    def patchify_data(self):
        """Split data into patches."""
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
            shape_original,
            patch_indices,
            shape_before_padding,
        ) = [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(len(self.volumes)):
            # Patchify

            # Shape before padding
            shape_before_padding_patches = self.volumes[i].shape

            # Volume
            volume_patch_size = (*self.patch_size, self.volumes[i].shape[2])
            volume_patch_step = (*self.patch_step, self.volumes[i].shape[2])
            volume = pad_divisible_2d(self.volumes[i], volume_patch_size, step=volume_patch_step)
            volume_patches = patchify(
                volume, 
                volume_patch_size, 
                step=volume_patch_step
            ).squeeze(2)  # bug in patchify

            shape_before_padding_patches = np.tile(
                np.array(shape_before_padding_patches)[None, None, :],
                [*volume_patches.shape[:2], 1],
            )

            # Scroll mask
            scroll_mask = pad_divisible_2d(self.scroll_masks[i], self.patch_size, step=self.patch_step)
            scroll_mask_patches = patchify(scroll_mask, self.patch_size, step=self.patch_step)
            
            # IR image
            ir_image_patches = None
            if self.ir_images is not None:
                ir_image = pad_divisible_2d(self.ir_images[i], self.patch_size, step=self.patch_step)
                ir_image_patches = patchify(ir_image, self.patch_size, step=self.patch_step)
            
            # Ink mask
            ink_mask_patches = None
            if self.ink_masks is not None:
                ink_mask = pad_divisible_2d(self.ink_masks[i], self.patch_size, step=self.patch_step)
                ink_mask_patches = patchify(ink_mask, self.patch_size, step=self.patch_step)
            
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

            # Original shape
            shape_original_patches = np.tile(
                np.array(volume.shape)[None, None, :],
                [*volume_patches.shape[:2], 1],
            )

            # Generate indices, omit empty patches
            mask = (scroll_mask_patches > 0).any(axis=(-1, -2))
            patch_indices.append(np.argwhere(mask))
            
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
            shape_original.append(shape_original_patches)
            shape_before_padding.append(shape_before_padding_patches)

        if len(ir_images) == 0:
            ir_images = None
        if len(ink_masks) == 0:
            ink_masks = None
        if len(subtracts) == 0:
            subtracts = None
        if len(divides) == 0:
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
            divides, \
            shape_original, \
            patch_indices, \
            shape_before_padding

    def get_item_single(self, idx) -> Dict[str, Any]:
        if self.patch_indices is not None:
            idx_outer, idx_inner = 0, 0
            for i in range(len(self.patch_indices)):
                if idx < len(self.patch_indices[i]):
                    idx_outer = i
                    idx_inner = tuple(self.patch_indices[i][idx])
                    break
                else:
                    idx -= len(self.patch_indices[i])
            
            # Always here
            image = self.volumes[idx_outer][idx_inner]
            masks = [self.scroll_masks[idx_outer][idx_inner]]
            path = self.pathes[idx_outer][idx_inner]

            # Only in train
            if self.ir_images is not None:
                masks.append(self.ir_images[idx_outer][idx_inner])
            if self.ink_masks is not None:
                masks.append(self.ink_masks[idx_outer][idx_inner])

            # Only in val / test
            indices = None
            if self.indices is not None:
                indices = self.indices[idx_outer][idx_inner]
            shape_patches = None
            if self.shape_patches is not None:
                shape_patches = self.shape_patches[idx_outer][idx_inner]

            # Optional
            subtract = None
            if self.subtracts is not None:
                subtract = self.subtracts[idx_outer][idx_inner]
            divide = None
            if self.divides is not None:
                divide = self.divides[idx_outer][idx_inner]
            shape_original = None
            if self.shape_original is not None:
                shape_original = self.shape_original[idx_outer][idx_inner]
            shape_before_padding = None
            if self.shape_before_padding is not None:
                shape_before_padding = self.shape_before_padding[idx_outer][idx_inner]
        else:
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
            shape_original = None
            if self.shape_original is not None:
                shape_original = self.shape_original[idx]
            shape_before_padding = None
            if self.shape_before_padding is not None:
                shape_before_padding = self.shape_before_padding[idx]

        output = {
            'image': image,  # volume, (H, W, D)
            'masks': masks,  # masks, (H, W) each
            'path': path,  # path, 1
            'indices': indices,  # indices, 2
            'shape_patches': shape_patches,  # shape_patches, 2
            'subtract': subtract,  # subtract, 1
            'divide': divide,  # divide, 1
            'shape_original': shape_original,  # shape_original, 3
            'shape_before_padding': shape_before_padding,  # shape_before_padding, 3
        }

        if self.transform is not None:
            output = self.transform(**output)

        return output

    def __getitem__(
        self, 
        idx
    ) -> Dict[str, Any]:
        output = self.get_item_single(idx)
        if self.transform_mix is None:
            return output
        
        output_keys = list(output.keys())
        output1 = self.get_item_single(idx)  # same volume, different random crop is assumed
        output.update({f'{k}1': v for k, v in output1.items()})
        
        output = self.transform_mix(**output)
        return {k: v for k, v in output.items() if k in output_keys}


class SurfaceVolumeDatasetTest:
    def __init__(
        self, 
        pathes, 
        z_start,
        z_end,
        transform=None, 
        patch_size: int | Tuple[int, int] = 256, 
        patch_step: None | int | Tuple[int, int] = 128,
        do_z_shift_scale: bool = True,
    ):
        self.pathes = pathes
        self.z_start = z_start
        self.z_end = z_end
        self.transform = transform

        if patch_step is None:
            patch_step = patch_size
        self.patch_size = to_2tuple(patch_size)
        self.patch_step = to_2tuple(patch_step)
        self.do_z_shift_scale = do_z_shift_scale

        self.build_data()

    def build_data(self):
        # Load data
        self.volumes = []
        self.scroll_masks = []
        self.ir_images = []
        self.ink_masks = []
        self.z_shifts = []
        self.z_scales = []
        for root in self.pathes:
            root = Path(root)
            
            # Volume
            volume = []
            for layer_index in range(0, N_SLICES):
                volume.append(
                    pyvips.Image.new_from_file(str(root / 'surface_volume' / f'{layer_index:02}.tif'))
                )
            self.volumes.append(volume)

            # Scroll mask
            self.scroll_masks.append(
                (
                    cv2.imread(
                        str(root / 'mask.png'),
                        cv2.IMREAD_GRAYSCALE
                    ) > 0
                ).astype(np.uint8)
            )

            # IR image
            if (root / 'ir.png').exists():
                self.ir_images.append(
                    cv2.imread(
                        str(root / 'ir.png'),
                        cv2.IMREAD_GRAYSCALE
                    )
                )
            
            # Ink mask
            if (root / 'inklabels.png').exists():
                self.ink_masks.append(
                    (
                        cv2.imread(
                            str(root / 'inklabels.png'),
                            cv2.IMREAD_GRAYSCALE
                        ) > 0
                    ).astype(np.uint8)
                )
        
            # Z shift and scale maps
            self.z_shifts.append(np.load(str(root / 'z_shift.npy')))
            self.z_scales.append(np.load(str(root / 'z_scale.npy')))

        # Build index
        self.shape_patches = []
        self.index_to_patch_info: List[Dict[int, Any]] = []
        for i, scroll_mask in enumerate(self.scroll_masks):
            scroll_mask = self.scroll_masks[i]
            
            n_h_starts = 0
            for j, h_start in enumerate(range(0, scroll_mask.shape[0], self.patch_step[0])):
                n_h_starts += 1
                n_w_starts = 0
                # Note: although openslide handles out of bounds by zero padding, 
                # for convenience we clamp h_end and w_end and pad images manually later
                h_end = min(h_start + self.patch_size[0], scroll_mask.shape[0])
                for k, w_start in enumerate(range(0, scroll_mask.shape[1], self.patch_step[1])):
                    n_w_starts += 1
                    w_end = min(w_start + self.patch_size[1], scroll_mask.shape[1])
                    if scroll_mask[h_start:h_end, w_start:w_end].sum() > 0:
                        patch_info = {
                            'outer_index': i,
                            'indices': (k, j),  # Note: (w, h)
                            'bbox': (h_start, w_start, h_end, w_end),
                        }
                        self.index_to_patch_info.append(patch_info)
                    # Only single patch out of bounds is allowed
                    # as in patchify (if step < size, multiple such patches are possible)
                    if w_start + self.patch_size[1] >= scroll_mask.shape[1]:
                        break
                # Only single patch out of bounds is allowed
                # as in patchify (if step < size, multiple such patches are possible)
                if h_start + self.patch_size[0] >= scroll_mask.shape[0]:
                    break

            self.shape_patches.append((n_h_starts, n_w_starts))
        
    def __len__(self):
        return len(self.index_to_patch_info)

    def __getitem__(self, idx):
        patch_info = self.index_to_patch_info[idx]
        outer_index = patch_info['outer_index']

        # Read part of volume
        # Note: zero padding is applied to z_scale which could lead to zero division
        # but it is handled later in transform
        z_shift = copy_crop_pad_2d(self.z_shifts[outer_index], self.patch_size, patch_info['bbox'])
        z_scale = copy_crop_pad_2d(self.z_scales[outer_index], self.patch_size, patch_info['bbox'])
        
        z_start_input, z_end_input = 0, N_SLICES
        if self.do_z_shift_scale:
            z_start_input, z_end_input = calculate_input_z_range(
                self.z_start, 
                self.z_end, 
                z_shift,
                z_scale,
            )

        image = np.full(
            (*self.patch_size, z_end_input - z_start_input), 
            fill_value=0, 
            dtype=np.uint16
        )
        volume = self.volumes[outer_index]
        for i, layer_index in enumerate(range(z_start_input, z_end_input)):
            slice_ = volume[layer_index].crop(
                patch_info['bbox'][1],
                patch_info['bbox'][0],
                patch_info['bbox'][3] - patch_info['bbox'][1],
                patch_info['bbox'][2] - patch_info['bbox'][0],
            ).numpy()
            image[..., i][:slice_.shape[0], :slice_.shape[1]] = slice_

        # Apply z shift and scale
        if self.do_z_shift_scale:
            z_shift_scale_transform = build_z_shift_scale_transform(
                image.shape, 
                z_start_input, 
                self.z_start, 
                z_shift, 
                z_scale, 
            )
            image = apply_z_shift_scale(
                image,
                z_shift_scale_transform,
                self.z_start,
                self.z_end,
            )

        # Get masks
        scroll_mask = copy_crop_pad_2d(self.scroll_masks[outer_index], self.patch_size, patch_info['bbox'])
        masks = [scroll_mask]

        if self.ir_images is not None:
            ir_image = copy_crop_pad_2d(self.ir_images[outer_index], self.patch_size, patch_info['bbox'])
            masks.append(ir_image)
        if self.ink_masks is not None:
            ink_mask = copy_crop_pad_2d(self.ink_masks[outer_index], self.patch_size, patch_info['bbox'])
            masks.append(ink_mask)

        # Get shapes of full volume before and after padding
        path = self.pathes[outer_index]
        indices = np.array(patch_info['indices'])
        shape_patches = np.array(self.shape_patches[outer_index])
        shape_before_padding = np.array(self.scroll_masks[outer_index].shape)
        pad_width = calculate_pad_width(shape_before_padding, 2, self.patch_step)
        # TODO: fix naming
        shape_original = np.array([s + p[1] for s, p in zip(shape_before_padding, pad_width)])
        
        output = {
            'image': image,  # volume, (H, W, D)
            'masks': masks,  # masks, (H, W) each
            'path': path,  # path, 1
            'indices': indices,  # indices, 2
            'shape_patches': shape_patches,  # shape_patches, 2
            'subtract': None,  # subtract, 1
            'divide': None,  # divide, 1
            'shape_original': shape_original,  # shape_original, 3
            'shape_before_padding': shape_before_padding,  # shape_before_padding, 3
        }

        if self.transform is not None:
            output = self.transform(**output)

        return output


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
    sigma=None,
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
            patch_step=tuple(s // 2 for s in patch_size),
            subtracts=[subtracts[i]],
            divides=[divides[i]],
        )

        z_shifts, z_scales, shape_patches, shape_original, shape_before_padding = \
            None, None, None, None, None
        for j in tqdm(range(len(dataset))):
            item = dataset[j]

            # For each patch, calculate z_shift and z_scale
            # and store them in z_shifts and z_scales maps
            if z_shifts is None:
                shape_patches = item['shape_patches'].tolist()
                shape_original = item['shape_original'].tolist()
                shape_before_padding = item['shape_before_padding'].tolist()
                z_shifts = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)
                z_scales = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)

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
        if sigma is not None:
            z_shifts = gaussian_filter(z_shifts, sigma=sigma)
            z_scales = gaussian_filter(z_scales, sigma=sigma)

        # Upscale z_shifts and z_scales maps to the 
        # original (padded) volume size

        # Note: shape could be not equal to the original volume size
        # because of the padding used in patchification
        z_shifts = cv2.resize(
            z_shifts,
            shape_original[:2][::-1],
            interpolation=cv2.INTER_LINEAR,
        )
        z_scales = cv2.resize(
            z_scales,
            shape_original[:2][::-1],
            interpolation=cv2.INTER_LINEAR,
        )

        # Crop z_shifts and z_scales maps to the
        # original volume size (padding is always 'after')
        z_shifts = z_shifts[
            :shape_before_padding[0],
            :shape_before_padding[1],
        ]
        z_scales = z_scales[
            :shape_before_padding[0],
            :shape_before_padding[1],
        ]

        z_shifts_all.append(z_shifts)
        z_scales_all.append(z_scales)
    
    return z_shifts_all, z_scales_all
