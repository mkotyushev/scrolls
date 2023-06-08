import argparse
import git
import logging
import os
import sys
import cv2
import imagesize
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.data.constants import (
    N_SLICES,
    VOLUME_MEAN_PER_Z_TARGET_NORMALIZED,
    Z_TARGET,
    Z_TARGET_FIT_START_INDEX, 
    Z_TARGET_FIT_END_INDEX,
)
from src.data.datasets import OnlineSurfaceVolumeDataset
from src.utils.utils import (
    get_z_dataset_mean_per_z,
    get_z_volume_mean_per_z, 
    fit_x_shift_scale, 
    build_nan_or_outliers_mask, 
    interpolate_masked_pixels,
)


# Usage: python src/scripts/z_shift_scale/build_maps.py --input_dir /workspace/data/fragments/train/2 --downscaled_input_dir /workspace/data/fragments_downscaled_2/train/2 --output_dir . --patch_size 256 --downscale_factor 2

logger = logging.getLogger(__name__)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def build_maps_wrt_self(
    path,
    z_start=0,
    z_end=N_SLICES,
    patch_size=256,
    overlap_divider=2,
    sigma=None,
):
    # Dataset with all slices, patchification and no overlap
    dataset = OnlineSurfaceVolumeDataset(
        pathes=[path],
        map_pathes=None,
        do_scale=False,
        z_start=0,
        z_end=N_SLICES,
        transform=None,
        patch_size=patch_size,
        patch_step=patch_size,
    )
    z_target, volume_mean_per_z_target = get_z_dataset_mean_per_z(dataset, z_start=0)

    z_target = z_target[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]
    volume_mean_per_z_target = volume_mean_per_z_target[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]

    min_, max_ = volume_mean_per_z_target.min(), volume_mean_per_z_target.max()
    subtract = min_
    divide = max_ - min_
    logger.info(f'subtract: {subtract}, divide: {divide}')
    volume_mean_per_z_target = (volume_mean_per_z_target - subtract) / divide
    
    # Dataset with partial slices, patchification and overlap
    dataset = OnlineSurfaceVolumeDataset(
        pathes=[path],
        map_pathes=None,
        do_scale=False,
        z_start=z_start,
        z_end=z_end,
        transform=None,
        patch_size=patch_size,
        patch_step=patch_size // overlap_divider,
    )

    z_shifts, z_scales, y_shifts, y_scales, shape_patches, shape_original, shape_before_padding = \
        None, None, None, None, None, None, None
    for j in tqdm(range(len(dataset))):
        item = dataset[j]

        # For each patch, calculate shifts and scales
        # and store them in maps
        if z_shifts is None:
            shape_patches = item['shape_patches'].tolist()
            shape_original = item['shape_original'].tolist()
            shape_before_padding = item['shape_before_padding'].tolist()
            z_shifts = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)
            z_scales = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)
            y_shifts = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)
            y_scales = np.full(shape_patches, fill_value=np.nan, dtype=np.float32)

        z, volume_mean_per_z = get_z_volume_mean_per_z(
            item['image'], item['masks'][0], z_start
        )
        volume_mean_per_z = (volume_mean_per_z - subtract) / divide
        z_shift, z_scale, y_shift, y_scale = fit_x_shift_scale(
            z, 
            volume_mean_per_z, 
            z_target, 
            volume_mean_per_z_target,
            model='x_shift',
        )

        indices = item['indices']
        z_shifts[indices[1], indices[0]] = z_shift
        z_scales[indices[1], indices[0]] = z_scale
        y_shifts[indices[1], indices[0]] = y_shift
        y_scales[indices[1], indices[0]] = y_scale

    # Clear outliers & nans
    z_shifts_nan, z_shifts_outliers, z_shifts_q05, z_shifts_q95 = build_nan_or_outliers_mask(z_shifts)
    z_scales_nan, z_scales_outliers, z_scales_q05, z_scales_q95 = build_nan_or_outliers_mask(z_scales)
    y_shifts_nan, y_shifts_outliers, y_shifts_q05, y_shifts_q95 = build_nan_or_outliers_mask(y_shifts)
    y_scales_nan, y_scales_outliers, y_scales_q05, y_scales_q95 = build_nan_or_outliers_mask(y_scales)
    
    z_shifts[z_shifts_nan] = z_shifts[~z_shifts_nan].mean()
    z_scales[z_scales_nan] = z_scales[~z_scales_nan].mean()
    y_shifts[y_shifts_nan] = y_shifts[~y_shifts_nan].mean()
    y_scales[y_scales_nan] = y_scales[~y_scales_nan].mean()

    mask = z_shifts_outliers | z_scales_outliers | y_scales_outliers | y_shifts_outliers
    z_shifts = interpolate_masked_pixels(z_shifts, mask, method='linear')
    z_scales = interpolate_masked_pixels(z_scales, mask, method='linear')
    y_shifts = interpolate_masked_pixels(y_shifts, mask, method='linear')
    y_scales = interpolate_masked_pixels(y_scales, mask, method='linear')

    # interpolate_masked_pixels does not take into account edge & large gaps
    # so clean again
    mask = \
        (z_shifts < z_shifts_q05) | (z_shifts > z_shifts_q95) | \
        (z_scales < z_scales_q05) | (z_scales > z_scales_q95) | \
        (y_shifts < y_shifts_q05) | (y_shifts > y_shifts_q95) | \
        (y_scales < y_scales_q05) | (y_scales > y_scales_q95)
    z_shifts = np.where(mask, z_shifts[~mask].mean(), z_shifts)
    z_scales = np.where(mask, z_scales[~mask].mean(), z_scales)
    y_shifts = np.where(mask, y_shifts[~mask].mean(), y_shifts)
    y_scales = np.where(mask, y_scales[~mask].mean(), y_scales)

    # Apply filtering
    if sigma is not None:
        z_shifts = gaussian_filter(z_shifts, sigma=sigma)
        z_scales = gaussian_filter(z_scales, sigma=sigma)
        y_shifts = gaussian_filter(y_shifts, sigma=sigma)
        y_scales = gaussian_filter(y_scales, sigma=sigma)

    # Upscale maps to the original (padded) volume size

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
    y_shifts = cv2.resize(
        y_shifts,
        shape_original[:2][::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    y_scales = cv2.resize(
        y_scales,
        shape_original[:2][::-1],
        interpolation=cv2.INTER_LINEAR,
    )

    # Crop maps to the original volume size 
    # (padding is always 'after')
    z_shifts = z_shifts[
        :shape_before_padding[0],
        :shape_before_padding[1],
    ]
    z_scales = z_scales[
        :shape_before_padding[0],
        :shape_before_padding[1],
    ]
    y_shifts = y_shifts[
        :shape_before_padding[0],
        :shape_before_padding[1],
    ]
    y_scales = y_scales[
        :shape_before_padding[0],
        :shape_before_padding[1],
    ]
    
    return z_shifts, z_scales, y_shifts, y_scales


def build_maps_wrt_ideal(
    path,
    map_pathes,
    z_start=0,
    z_end=N_SLICES,
    patch_size=256,
):
    # Get ideal curve
    z_target = Z_TARGET[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]
    volume_mean_per_z_target = VOLUME_MEAN_PER_Z_TARGET_NORMALIZED[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]

    min_, max_ = volume_mean_per_z_target.min(), volume_mean_per_z_target.max()
    subtract = min_
    divide = max_ - min_
    logger.info(f'subtract: {subtract}, divide: {divide}')
    volume_mean_per_z_target = (volume_mean_per_z_target - subtract) / divide

    # Get curve for the fragment
    dataset = OnlineSurfaceVolumeDataset(
        pathes=[path],
        do_scale=True,
        map_pathes=map_pathes,
        z_start=z_start,
        z_end=z_end,
        transform=None,
        patch_size=patch_size,
        patch_step=patch_size,
    )
    shape_before_padding = dataset[0]['shape_before_padding'].tolist()

    # Fit z_shifts and z_scales
    z, volume_mean_per_z = get_z_dataset_mean_per_z(dataset, z_start=0)
    volume_mean_per_z = (volume_mean_per_z - subtract) / divide
    z_shift, z_scale, _, _ = fit_x_shift_scale(
        z, 
        volume_mean_per_z, 
        z_target, 
        volume_mean_per_z_target,
        model='no_y',
    )

    # Set y_shifts and y_scales according to the stats
    min_, max_ = volume_mean_per_z_target.min(), volume_mean_per_z_target.max()
    y_shift = -min_
    y_scale = max_ - min_

    z_shifts, z_scales, y_shifts, y_scales = \
        np.full(shape_before_padding[:2], fill_value=z_shift, dtype=np.float32), \
        np.full(shape_before_padding[:2], fill_value=z_scale, dtype=np.float32), \
        np.full(shape_before_padding[:2], fill_value=y_shift, dtype=np.float32), \
        np.full(shape_before_padding[:2], fill_value=y_scale, dtype=np.float32)
    
    return z_shifts, z_scales, y_shifts, y_scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--downscaled_input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--downscale_factor', type=int, default=2)
    parser.add_argument('--overlap_divider', type=int, default=2)
    parser.add_argument(
        '--model', 
        type=str, 
        choices=[
            'no_y', 
            'independent_y_scale', 
            'independent_y_shift_scale', 
            'beer_lambert_law',
            'beer_lambert_law_independent_y_shift',
            'x_shift',
        ], 
        default='no_y'
    )

    args = parser.parse_args()

    # Read original image size
    original_width, original_height = imagesize.get(args.input_dir / 'mask.png')

    # Build z shift maps wrt self
    z_shift_wrt_self, z_scale_wrt_self, y_shift_wrt_self, y_scale_wrt_self = build_maps_wrt_self(
        path=args.downscaled_input_dir,
        z_start=17,
        z_end=50,
        patch_size=args.patch_size,
        overlap_divider=args.overlap_divider,
        sigma=None,
    )
    assert np.allclose(z_scale_wrt_self, 1.0), 'z_scale_wrt_self should be 1.0'
    assert np.allclose(y_shift_wrt_self, 0.0), 'y_shift_wrt_self should be 0.0'
    assert np.allclose(y_scale_wrt_self, 1.0), 'y_scale_wrt_self should be 1.0'

    # Save to temp dir
    temp_dir = Path('/tmp/build_maps')
    temp_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / 'z_shift.npy',
        z_shift_wrt_self,
    )
    np.save(
        args.output_dir / 'z_scale.npy',
        z_scale_wrt_self,
    )
    np.save(
        args.output_dir / 'y_shift.npy',
        y_shift_wrt_self,
    )
    np.save(
        args.output_dir / 'y_scale.npy',
        y_scale_wrt_self,
    )

    # Build z shift & scale maps and scalar y shift & scale maps wrt ideal
    z_shift_wrt_ideal, z_scale_wrt_ideal, y_shift_wrt_ideal, y_scale_wrt_ideal = build_maps_wrt_ideal(
        path=args.input_dir,
        map_pathes=temp_dir,
        z_start=17,
        z_end=50,
        patch_size=args.patch_size,
    )
    assert np.allclose(y_shift_wrt_ideal, y_shift_wrt_ideal[0, 0]), \
        'y_shift_wrt_ideal should be all the same'
    assert np.allclose(y_scale_wrt_ideal, y_scale_wrt_ideal[0, 0]), \
        'y_scale_wrt_ideal should be all the same'

    # Remove temp dir
    for path in temp_dir.iterdir():
        path.unlink()
    temp_dir.rmdir()

    # Build superposition z shift and scale maps:
    # first wrt self is applied, then wrt ideal
    # z_input = 
    #   = (z - shift) / scale =
    #   = ((z - shift_wrt_self) / scale_wrt_self) - shift_wrt_ideal) / scale_wrt_ideal;
    # thus:
    #   scale = scale_wrt_self * scale_wrt_ideal
    #   shift = shift_wrt_self + shift_wrt_ideal * scale_wrt_self
    z_shift = z_shift_wrt_self + z_shift_wrt_ideal * z_scale_wrt_self
    z_scale = z_scale_wrt_self * z_scale_wrt_ideal

    # Use only y shift & scale wrt ideal
    y_shift = y_shift_wrt_ideal
    y_scale = y_scale_wrt_ideal

    # Upscale to original size
    z_shift = cv2.resize(z_shift, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    z_scale = cv2.resize(z_scale, (original_width, original_height), interpolation=cv2.INTER_LINEAR) 
    y_shift = cv2.resize(y_shift, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    y_scale = cv2.resize(y_scale, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(
        args.output_dir / 'z_shift.npy',
        z_shift,
    )
    np.save(
        args.output_dir / 'z_scale.npy',
        z_scale,
    )
    np.save(
        args.output_dir / 'y_shift.npy',
        y_shift,
    )
    np.save(
        args.output_dir / 'y_scale.npy',
        y_scale,
    )

    # Save run command & commit id
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(args.output_dir / 'build_maps_args.txt', 'w') as f:
        f.write(f'git commit id: {sha}\n')
        f.write(' '.join(['python'] + sys.argv))


if __name__ == '__main__':
    main()
