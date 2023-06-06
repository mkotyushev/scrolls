# Create in args.out_dir 4 dirs with the same dir structure as in args.in_dir, 
# copy all the files from args.in_dir to args.out_dir / {i} if it is 
# not .tif or .png file, crop {i}th quater of each .tif or .png file with 
# the same dir structure and filename.
#
# Use pathlib.Path to work with paths not os.path.
# Use argparse to work with command line arguments.
#
# Usage: python src/scripts/crop_fragment.py /workspace/data/fragments/train/2 /workspace/data/fragments_cropped/train/ mass

import argparse
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm


def crop_image(in_path, in_fragment_root, out_fragments_root, center=None):
    """Crop image from in_path and save it to out_path / {i}"""
    if in_path.suffix in ['.png', '.tif']:
        img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    elif in_path.suffix == '.npy':
        img = np.load(in_path)
    else:
        raise ValueError(f'Unknown file type: {in_path.suffix}')
    
    if center is None:
        h_center, w_center = img.shape[0] // 2, img.shape[1] // 2
    else:
        h_center, w_center = center

    img_crop = img[:h_center, :w_center]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}a' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if in_path.suffix in ['.png', '.tif']:
        cv2.imwrite(str(out_path), img_crop)
    elif in_path.suffix == '.npy':
        np.save(out_path, img_crop)

    img_crop = img[:h_center, w_center:]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}b' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix in ['.png', '.tif']:
        cv2.imwrite(str(out_path), img_crop)
    elif in_path.suffix == '.npy':
        np.save(out_path, img_crop)

    img_crop = img[h_center:, :w_center]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}c' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix in ['.png', '.tif']:
        cv2.imwrite(str(out_path), img_crop)
    elif in_path.suffix == '.npy':
        np.save(out_path, img_crop)

    img_crop = img[h_center:, w_center:]
    out_path = (
        out_fragments_root / 
        f'{in_fragment_root.stem}d' / 
        Path(in_path).relative_to(in_fragment_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix in ['.png', '.tif']:
        cv2.imwrite(str(out_path), img_crop)
    elif in_path.suffix == '.npy':
        np.save(out_path, img_crop)


def get_scroll_mask_path(img_path):
    if img_path.suffix in ['.png', '.npy']:
        return img_path.parent / 'mask.png'
    elif img_path.suffix == '.tif':
        return img_path.parent.parent / 'mask.png'


parser = argparse.ArgumentParser(description='Crop fragment')
parser.add_argument('in_dir', type=Path, help='input directory (root of single fragment)')
parser.add_argument('out_dir', type=Path, help='output directory (root of fragments)')
parser.add_argument('mode', type=str, choices=['geom', 'mass'], help='crop center is geom or mass center')
args = parser.parse_args()

for path in tqdm(args.in_dir.glob('**/*')):
    if path.is_dir():
        path_out = args.out_dir / path.relative_to(args.in_dir)
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        if path.suffix in ['.png', '.tif', '.npy']:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            center = None
            if args.mode == 'mass':
                scroll_mask = cv2.imread(str(get_scroll_mask_path(path)), cv2.IMREAD_GRAYSCALE) > 0
                mass_h, mass_w = np.where(scroll_mask)
                center = np.floor(mass_h.mean()).astype(np.int32), np.floor(mass_w.mean()).astype(np.int32)
            crop_image(path, args.in_dir, args.out_dir, center=center)
        else:
            path_out = args.out_dir / path.relative_to(args.in_dir)
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(path.read_bytes())
