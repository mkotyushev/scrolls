import argparse
import logging
import cv2
from pathlib import Path
from tqdm import tqdm

# Usage: python src/scripts/merge_cropped_subfragments.py /workspace/data/fragments_z_shift_scale_cropped/train/2a /workspace/data/fragments_z_shift_scale_cropped/train/2b /workspace/data/fragments_z_shift_scale_cropped/train/2c /workspace/data/fragments_z_shift_scale_cropped/train/2d /workspace/data/fragments_z_shift_scale_cropped_merged/train/2

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

def merge_images(in_path, in_fragment_root_a, in_fragment_root_b, in_fragment_root_c, in_fragment_root_d, out_fragments_root):
    """Merge images / {i} from in_path and save it to out_path"""
    rel_path = in_path.relative_to(in_fragment_root_a)
    img_a = cv2.imread(str(in_fragment_root_a / rel_path), cv2.IMREAD_UNCHANGED)
    img_b = cv2.imread(str(in_fragment_root_b / rel_path), cv2.IMREAD_UNCHANGED)
    img_c = cv2.imread(str(in_fragment_root_c / rel_path), cv2.IMREAD_UNCHANGED)
    img_d = cv2.imread(str(in_fragment_root_d / rel_path), cv2.IMREAD_UNCHANGED)
    img = cv2.vconcat([cv2.hconcat([img_a, img_b]), cv2.hconcat([img_c, img_d])])
    cv2.imwrite(str(out_fragments_root / rel_path), img)
    logger.info(
        f'Wrote {in_fragment_root_a / rel_path}, '
        f'{in_fragment_root_b / rel_path}, '
        f'{in_fragment_root_c / rel_path}, '
        f'{in_fragment_root_d / rel_path} '
        f'to {out_fragments_root / rel_path}'
    )


parser = argparse.ArgumentParser(description='Merge cropped fragment')
parser.add_argument('in_dir_a', type=Path, help='input directory a')
parser.add_argument('in_dir_b', type=Path, help='input directory b')
parser.add_argument('in_dir_c', type=Path, help='input directory c')
parser.add_argument('in_dir_d', type=Path, help='input directory d')
parser.add_argument('out_dir', type=Path, help='output directory (root of merged fragment)')
args = parser.parse_args()

for path in tqdm(args.in_dir_a.glob('**/*')):
    if path.is_dir():
        path_out = args.out_dir / path.relative_to(args.in_dir_a)
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        if path.suffix in ['.png', '.tif']:
            merge_images(path, args.in_dir_a, args.in_dir_b, args.in_dir_c, args.in_dir_d, args.out_dir)
        else:
            path_out = args.out_dir / path.relative_to(args.in_dir_a)
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(path.read_bytes())
