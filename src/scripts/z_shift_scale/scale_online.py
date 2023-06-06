
import argparse
import logging
import os
import numpy as np
from pathlib import Path

from src.data.constants import N_SLICES
from src.data.datasets import OnlineSurfaceVolumeDataset

# Usage: python src/scripts/z_shift_scale/scale_online.py --input_dir /workspace/data/fragments/train/2 --map_path /workspace/data/fragments/train/2 --output_dir /workspace/data/fragments_z_shift_scale/train/2 --patch_size 384 --z_start 20 --z_end 44

logger = logging.getLogger(__name__)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--map_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--patch_size', type=int, default=384)
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

    # Read maps
    z_shifts = [np.load(args.map_path / 'z_shift.npy')]
    z_scales = [np.load(args.map_path / 'z_scale.npy')]
    y_shifts = [np.load(args.map_path / 'y_shift.npy')]
    y_scales = [np.load(args.map_path / 'y_scale.npy')]

    logger.info(f'z_shifts: ({z_shifts[0].min()}, {z_shifts[0].max()})')
    logger.info(f'z_scales: ({z_scales[0].min()}, {z_scales[0].max()})')
    logger.info(f'y_shifts: ({y_shifts[0].min()}, {y_shifts[0].max()})')
    logger.info(f'y_scales: ({y_scales[0].min()}, {y_scales[0].max()})')

    # Get z range
    z_start, z_end = 0, N_SLICES
    if args.z_start is not None:
        z_start = args.z_start
    if args.z_end is not None:
        z_end = args.z_end

    # Dataset with all slices, patchification and no overlap
    dataset = OnlineSurfaceVolumeDataset(
        pathes=[args.input_dir],
        map_pathes=[args.map_path],
        do_scale=True,
        z_start=z_start,
        z_end=z_end,
        transform=None,
        patch_size=args.patch_size,
        patch_step=args.patch_size,
        skip_empty_scroll_mask=False,
    )

    # Dump
    dataset.dump(args.output_dir, only_volume=True)


if __name__ == '__main__':
    main()
