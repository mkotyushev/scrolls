import argparse
import logging
import cv2
import imagesize
import numpy as np
from pathlib import Path

from src.data.datamodules import read_data
from src.data.datasets import build_z_shift_scale_maps

# Usage: python src/scripts/build_z_shift_scale.py --input_dir /workspace/data/fragments/train/2 --downscaled_input_dir /workspace/data/fragments_downscaled_2/train/2 --output_dir . --patch_size 128 --downscale_factor 2

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=Path, required=True)
parser.add_argument('--downscaled_input_dir', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--downscale_factor', type=int, default=2)
args = parser.parse_args()

# Read original image size
original_width, original_height = imagesize.get(args.input_dir / 'mask.png')

# Read downscaled data
volumes, scroll_masks, ir_images, ink_masks, subtracts, divides = \
    read_data([args.downscaled_input_dir])

# Build z shift and scale maps
z_shifts, z_scales = build_z_shift_scale_maps(
    pathes=[args.downscaled_input_dir],
    volumes=[volumes[0]],
    scroll_masks=[scroll_masks[0]],
    subtracts=[subtracts[0]],
    divides=[divides[0]],
    z_start=0,
    crop_z_span=8,
    mode='volume_mean_per_z', 
    normalize='minmax', 
    patch_size=(args.patch_size, args.patch_size),
    sigma=0.5,
)

# Upscale to original size
z_shifts = [
    cv2.resize(z_shift, (original_width, original_height), interpolation=cv2.INTER_LINEAR) 
    for z_shift in z_shifts
]
z_scales = [
    cv2.resize(z_scale, (original_width, original_height), interpolation=cv2.INTER_LINEAR) 
    for z_scale in z_scales
]

# Save
np.save(
    args.output_dir / 'z_shift.npy',
    z_shifts[0],
)
np.save(
    args.output_dir / 'z_scale.npy',
    z_scales[0],
)
