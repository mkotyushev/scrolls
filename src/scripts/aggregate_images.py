import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.utils import rle


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser(description='Downscale dataset')
parser.add_argument('proba_dir', type=Path, help='proba directory')
parser.add_argument('output_path', type=Path, help='output filepath')
args = parser.parse_args()

ids = sorted(list(set([path.stem.split('_')[0] for path in args.proba_dir.glob('*.png')])))

probas = []
for id_ in ids:
    proba_pathes = list(args.proba_dir.glob(f'{id_}_*.png'))

    logger.info(f'Processing {id_}, proba_pathes: {proba_pathes}...')

    proba = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) / 255
        for path in proba_pathes
    ]

    proba_agg = np.stack(proba, axis=0).mean(axis=0)
    probas.append(proba_agg)

# Save
with open(args.output_path, 'w') as f:
    print("Id,Predicted", file=f)
    for i, (id_, proba) in tqdm(enumerate(zip(ids, probas))):
        starts_ix, lengths = rle(proba)
        inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
        print(f"{id_}," + inklabels_rle, file=f, end="\n" if i != len(ids) - 1 else "")
