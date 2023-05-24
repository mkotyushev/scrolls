# Create in args.out_dir 4 dirs with the same dir structure as in args.in_dir, 
# copy all the files from args.in_dir to args.out_dir / {i} if it is 
# not .tif or .png file, crop {i}th quater of each .tif or .png file with 
# the same dir structure and filename.
#
# Use pathlib.Path to work with paths not os.path.
# Use argparse to work with command line arguments.
#
# Usage: python src/scripts/crop_fragment.py /workspace/data/fragments/train/2 /workspace/data/fragments_cropped/train/

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def crop_image(in_path, in_fragment_root, out_fragments_root):
    """Crop image from in_path and save it to out_path / {i}"""
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    h_center, w_center = img.shape[0] // 2, img.shape[1] // 2

    img_crop = img[:h_center, :w_center]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}a' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_crop)

    img_crop = img[:h_center, w_center:]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}b' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_crop)

    img_crop = img[h_center:, :w_center]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}c' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_crop)

    img_crop = img[h_center:, w_center:]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}d' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_crop)


parser = argparse.ArgumentParser(description='Crop fragment')
parser.add_argument('in_dir', type=Path, help='input directory (root of single fragment)')
parser.add_argument('out_dir', type=Path, help='output directory (root of fragments)')
args = parser.parse_args()

for path in tqdm(args.in_dir.glob('**/*')):
    if path.is_dir():
        path_out = args.out_dir / path.relative_to(args.in_dir)
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        if path.suffix in ['.png', '.tif']:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            crop_image(path, args.in_dir, args.out_dir)
        else:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(path.read_bytes())
