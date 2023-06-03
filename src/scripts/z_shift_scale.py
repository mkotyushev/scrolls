
import argparse
import gc
import logging
import cv2
from pathlib import Path
import numpy as np
from scipy.ndimage import geometric_transform

from src.data.datamodules import read_data
from src.data.datasets import build_z_shift_scale_maps

# Usage: python src/scripts/z_shift_scale.py --input_dir /workspace/data/fragments/train/2 --output_dir /workspace/data/fragments_z_shift_scale/train/2 --patch_size 256 --z_start 20 --z_end 44

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--z_start', type=int, default=None)
parser.add_argument('--z_end', type=int, default=None)
args = parser.parse_args()

# Copy all files (except .tif) to output directory keeping the same structure
for path in args.input_dir.glob('**/*'):
    if path.is_dir():
        continue
    if path.suffix != '.tif':
        path_out = args.output_dir / path.relative_to(args.input_dir)
        logger.info(f'Copying {path} to {path_out}')

        path_out.parent.mkdir(parents=True, exist_ok=True)
        path_out.write_bytes(path.read_bytes())

# Read data
volumes, scroll_masks, ir_images, ink_masks, subtracts, divides = \
    read_data([args.input_dir])

# Build z shift and scale maps
z_shifts, z_scales = build_z_shift_scale_maps(
    pathes=[args.input_dir],
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

def z_shift_scale_map(x):
    shift, scale = z_shifts[0][x[0], x[1]], z_scales[0][x[0], x[1]]       
    z = (x[2] - shift) / scale  # assuming non-zero scale
    return (
        x[0], 
        x[1], 
        z
    )

# Apply z shift and scale maps
volume_transformed = geometric_transform(
    volumes[0],
    z_shift_scale_map,
    order=1,
)

# Convert to uint16
volume_transformed = volume_transformed.astype(np.uint16)

# Get z range
z_start, z_end = 0, volume_transformed.shape[2]
if args.z_start is not None:
    z_start = z_start
if args.z_end is not None:
    z_end = z_end

# Save transformed volume
for z in range(z_start, z_end):
    path_out = \
        args.output_dir / \
        'surface_volume' / f'{z:02}.tif'
    logger.info(f'Saving layer {z} to {path_out}')
    
    path_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path_out),
        volume_transformed[:, :, z],
    )
