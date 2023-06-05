
import argparse
import ctypes
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import geometric_transform
from scipy import LowLevelCallable

from src.data.constants import N_SLICES
from src.utils.utils import read_data

# Usage: python src/scripts/z_shift_scale/scale.py --input_dir /workspace/data/fragments/train/2 --output_dir /workspace/data/fragments_z_shift_scale/train/2 --z_shift_path z_shift.npy --z_scale_path z_scale.npy --z_start 20 --z_end 44

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def build_z_shift_scale_transform(volume_shape, z_start_input, z_start, z_shift, z_scale):
    # Build transform
    mapping_lib_path = os.path.abspath(__file__).replace('scale.py', 'mapping.so')
    lib = ctypes.CDLL(mapping_lib_path)
    lib.mapping.restype = ctypes.c_int
    lib.mapping.argtypes = (
        ctypes.POINTER(ctypes.c_long), 
        ctypes.POINTER(ctypes.c_double), 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_void_p
    )

    # z_start, n_rows, n_cols, flattened shift array, flattened scale array
    # as single array double
    user_data = np.concatenate(
        [
            np.array([z_start, z_start_input, volume_shape[0], volume_shape[1]], dtype=np.double),
            z_shift.flatten().astype(np.double),
            z_scale.flatten().astype(np.double),
        ],
        dtype=np.double,
    )
    user_data = (ctypes.c_double*user_data.shape[0]).from_buffer(user_data)
    user_data = ctypes.cast(user_data, ctypes.c_void_p)
    func = LowLevelCallable(lib.mapping, user_data)

    return func


def apply_z_shift_scale(volume, func, z_start, z_end):
    # Apply z shift and scale maps
    volume_transformed = geometric_transform(
        volume,
        func,
        output_shape=(volume.shape[0], volume.shape[1], z_end - z_start),
        order=1,
    )
    
    # Convert to uint16
    volume_transformed = volume_transformed.astype(np.uint16)

    return volume_transformed


def calculate_input_z_range(z_start, z_end, z_shift, z_scale):
    # input_coordinates[2] = (z_start + output_coordinates[2] - shift) / scale - z_start_input;
    z_start_input, z_end_input = 0, N_SLICES

    # Remove outliers
    mask = np.isclose(z_scale, 0.0)
    z_scale = np.where(mask, 1.0, z_scale)
    z_shift = np.where(mask, 0.0, z_shift)

    # Get input z range required to calculate output z
    z_start_input = np.floor(((z_start - z_shift) / z_scale).min() - 1).astype(np.int32)
    z_end_input = np.ceil(((z_end - z_shift) / z_scale).max() + 1).astype(np.int32)

    # Clip input z range
    z_start_input = max(0, z_start_input)
    z_end_input = min(N_SLICES, z_end_input)

    return z_start_input, z_end_input


def main():
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
    logger.info(
        f'Input shape: {volumes[0].shape}, '
        f'output shape: {(volumes[0].shape[0], volumes[0].shape[1], z_end - z_start)}, '
        f'z_shifts shape: {z_shifts[0].shape}, '
        f'z_scales shape: {z_scales[0].shape}'
    )

    # Get z range
    z_start, z_end = 0, N_SLICES
    if args.z_start is not None:
        z_start = args.z_start
    if args.z_end is not None:
        z_end = args.z_end

    # Read data (partially):
    z_start_input, z_end_input = calculate_input_z_range(z_start, z_end, z_shifts[0], z_scales[0])
    logger.info(f'Reading slices {z_start_input} to {z_end_input}')
    volumes, *_ = \
        read_data([args.input_dir], z_start=z_start_input, z_end=z_end_input)

    # Build transform
    func = build_z_shift_scale_transform(
        volumes[0].shape, 
        z_start_input, 
        z_start, 
        z_shifts[0], 
        z_scales[0], 
    )

    # Apply z shift and scale maps
    logger.info(f'Transforming slices {z_start_input} - {z_end_input} to {z_start} - {z_end}')
    volume_transformed = apply_z_shift_scale(
        volumes[0],
        func,
        z_start_input, 
        z_end_input,
        z_start,
        z_end,
    )

    # Save transformed volume
    for z in range(volume_transformed.shape[2]):
        path_out = \
            args.output_dir / \
            'surface_volume' / f'{z_start + z:02}.tif'
        logger.info(f'Saving layer {z_start + z} to {path_out}')
        
        path_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(path_out),
            volume_transformed[:, :, z],
        )