import argparse
import gc
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.utils import rle


logger = logging.getLogger(__name__)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser(description='Downscale dataset')
parser.add_argument('proba_dir', type=Path, help='proba directory')
parser.add_argument('output_dir', type=Path, default='.', help='output directory')
args = parser.parse_args()

ids = sorted(list(set([path.stem.split('_')[0] for path in args.proba_dir.glob('*.png')])))

probas = []
for id_ in ids:
    proba_pathes = list(args.proba_dir.glob(f'{id_}_*.png'))

    logger.info(f'Processing {id_}, proba_pathes: {proba_pathes}...')

    proba = None
    for path in proba_pathes:
        gc.collect()
        if proba is None:
            proba = np.array(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)).astype(np.uint32)
        else:
            proba += np.array(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)).astype(np.uint32)
    proba = proba.astype(np.float64) / (255 * len(proba_pathes))
    probas.append(proba)

# Save images
for i, (id_, proba) in enumerate(zip(ids, probas)):
    cv2.imwrite(str(args.output_dir / f'{id_}_mean.png'), (proba * 255).astype(np.uint8))

# Save
with open(args.output_dir / 'submission.csv', 'w') as f:
    print("Id,Predicted", file=f)
    for i, (id_, proba) in tqdm(enumerate(zip(ids, probas))):
        starts_ix, lengths = rle(proba)
        inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
        print(f"{id_}," + inklabels_rle, file=f, end="\n" if i != len(ids) - 1 else "")
