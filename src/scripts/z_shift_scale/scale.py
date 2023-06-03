
import argparse
import ctypes
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import geometric_transform
from scipy import LowLevelCallable

from src.data.datamodules import N_SLICES, read_data

# Usage: python src/scripts/z_shift_scale/scale.py --input_dir /workspace/data/fragments/train/2 --output_dir /workspace/data/fragments_z_shift_scale/train/2 --z_shift_path z_shift.npy --z_scale_path z_scale.npy --z_start 20 --z_end 44

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--z_shift_path', type=Path, required=True)
parser.add_argument('--z_scale_path', type=Path, default=True)
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

# Read z shift and scale maps
z_shifts = [np.load(args.z_shift_path)]
z_scales = [np.load(args.z_scale_path)]

logger.info(f'z_shifts: ({z_shifts[0].min()}, {z_shifts[0].max()})')
logger.info(f'z_scales: ({z_scales[0].min()}, {z_scales[0].max()})')

# Get z range
z_start, z_end = 0, N_SLICES
if args.z_start is not None:
    z_start = z_start
if args.z_end is not None:
    z_end = z_end

# Build transform
mapping_lib_path = os.path.abspath(__file__).replace('scale.py', 'mapping.so')
lib = ctypes.CDLL(mapping_lib_path)
lib.mapping.restype = ctypes.c_int
lib.mapping.argtypes = (
    ctypes.POINTER(ctypes.c_longlong), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_void_p
)

# Read data (partially):
# input_coordinates[2] = (z_start + output_coordinates[2] - shift) / scale;
z_start_input = np.floor(((z_start - z_shifts[0]) / z_scales[0]).min()).astype(np.int32)
z_end_input = (np.ceil(((z_end - z_shifts[0]) / z_scales[0]).max()) + 1).astype(np.int32)
z_start_input = max(0, z_start_input)
z_end_input = min(N_SLICES, z_end_input)

logger.info(f'Reading slices {z_start_input} to {z_end_input}')
volumes, scroll_masks, ir_images, ink_masks, subtracts, divides = \
    read_data([args.input_dir], z_start=z_start_input, z_end=z_end_input)

# z_start, n_rows, n_cols, flattened shift array, flattened scale array
# as single array double
user_data = np.concatenate(
    [
        np.array([z_start, z_start_input, volumes[0].shape[0], volumes[0].shape[1]], dtype=np.double),
        z_shifts[0].flatten().astype(np.double),
        z_scales[0].flatten().astype(np.double),
    ],
    dtype=np.double,
)
user_data = np.ascontiguousarray(user_data)
user_data = ctypes.c_void_p(user_data.__array_interface__['data'][0])
func = LowLevelCallable(lib.mapping, user_data)

# Apply z shift and scale maps
volume_transformed = geometric_transform(
    volumes[0],
    func,
    output_shape=(volumes[0].shape[0], volumes[0].shape[1], z_end - z_start),
    order=1,
)

# Convert to uint16
volume_transformed = volume_transformed.astype(np.uint16)

# Save transformed volume
for z in range(volume_transformed.shape[2]):
    path_out = \
        args.output_dir / \
        'surface_volume' / f'{z:02}.tif'
    logger.info(f'Saving layer {z} to {path_out}')
    
    path_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path_out),
        volume_transformed[:, :, z],
    )
