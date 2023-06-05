import argparse
import logging
import os
import sys
import cv2
import imagesize
import numpy as np
from pathlib import Path

from src.data.constants import (
    Z_TARGET, 
    Z_TARGET_FIT_START_INDEX, 
    Z_TARGET_FIT_END_INDEX,
    VOLUME_MEAN_PER_Z_TARGET,
    VOLUME_MEAN_PER_Z_TARGET_NORMALIZED,
)
from src.data.datasets import build_maps

# Usage: python src/scripts/z_shift_scale/build_maps.py --input_dir /workspace/data/fragments/train/2 --downscaled_input_dir /workspace/data/fragments_downscaled_2/train/2 --output_dir . --patch_size 256 --downscale_factor 2

logger = logging.getLogger(__name__)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S',
)

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
    choices=['no_y', 'independent_y_scale', 'independent_y_shift_scale', 'beer_lambert_law'], 
    default='no_y'
)
parser.add_argument('--normalize', action='store_true')

args = parser.parse_args()

# Read original image size
original_width, original_height = imagesize.get(args.input_dir / 'mask.png')

# Build maps
z_target = Z_TARGET[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]
volume_mean_per_z_target = VOLUME_MEAN_PER_Z_TARGET_NORMALIZED if args.normalize else VOLUME_MEAN_PER_Z_TARGET
volume_mean_per_z_target = volume_mean_per_z_target[Z_TARGET_FIT_START_INDEX:Z_TARGET_FIT_END_INDEX]

z_shift, z_scale, y_shift, y_scale = build_maps(
    path=args.downscaled_input_dir,
    z_target=z_target,
    volume_mean_per_z_target=volume_mean_per_z_target,
    z_start=17,
    z_end=50,
    patch_size=(args.patch_size, args.patch_size),
    overlap_divider=args.overlap_divider,
    model=args.model,
    sigma=None,
    normalize=args.normalize,
)

# Upscale to original size
z_shift = cv2.resize(z_shift, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
z_scale = cv2.resize(z_scale, (original_width, original_height), interpolation=cv2.INTER_LINEAR) 
y_shift = cv2.resize(y_shift, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
y_scale = cv2.resize(y_scale, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

# Save
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

# Save run command
with open(args.output_dir / 'build_maps_args.txt', 'w') as f:
    f.write(' '.join(['python'] + sys.argv))
