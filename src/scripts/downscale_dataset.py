# Create in args.out_dir the same dir structure as in args.in_dir, 
# copy all the files from args.in_dir to args.out_dir if it is 
# not .tif or .png file, downscale all the .tif or .png files with 
# the same dir structure by scale factor (> 1) args.downscale_factor.
#
# Use pathlib.Path to work with paths not os.path.
# Use argparse to work with command line arguments.
#
# Usage: python src/scripts/downscale_dataset.py /workspace/data/fragments/ /workspace/data/fragments_downscaled_2/ 2

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def downscale_image(in_path, out_path, downscale_factor):
    """Downsample image from in_path by downscale_factor and save it to out_path"""
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(
        img,
        (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(out_path, img)


parser = argparse.ArgumentParser(description='Downscale dataset')
parser.add_argument('in_dir', type=Path, help='input directory')
parser.add_argument('out_dir', type=Path, help='output directory')
parser.add_argument('downscale_factor', type=int, help='downscale factor')
args = parser.parse_args()

for path in tqdm(args.in_dir.glob('**/*')):
    if path.is_dir():
        path_out = args.out_dir / path.relative_to(args.in_dir)
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        if path.suffix in ['.png', '.tif']:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            downscale_image(str(path), str(path_out), args.downscale_factor)
        else:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(path.read_bytes())
