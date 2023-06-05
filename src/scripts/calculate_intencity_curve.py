
# Build dataset with patchification and no overlap
import argparse
import logging
import os
from pathlib import Path

from src.data.constants import N_SLICES, Z_TARGET_FIT_END_INDEX, Z_TARGET_FIT_START_INDEX
from src.data.datasets import SurfaceVolumeDatasetTest
from src.utils.utils import get_z_dataset_mean_per_z


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
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', action='store_true')
args = parser.parse_args()


dataset = SurfaceVolumeDatasetTest(
    pathes=[args.input_dir],
    z_shift_scale_pathes=None,
    do_z_shift_scale=False,
    z_start=0,
    z_end=N_SLICES,
    transform=None,
    patch_size=args.patch_size,
    patch_step=args.patch_size,
)

# Calculate intensity per z
z, volume_mean_per_z = get_z_dataset_mean_per_z(dataset, z_start=0)
if args.normalize:
    volume_mean_per_z = (volume_mean_per_z - volume_mean_per_z.min()) / (volume_mean_per_z.max() - volume_mean_per_z.min())

print(f'z: {z}, volume_mean_per_z: {volume_mean_per_z}')