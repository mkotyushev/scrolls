import cv2
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
from torch.utils.data import Subset, Dataset


class InMemorySurfaceVolumeDataset:
    """Dataset for surface volumes."""
    N_SLICES = 65
    def __init__(
        self, 
        volumes_shared_storage,
        surface_volume_dirs: List[str], 
        transform: Optional[Callable] = None,
    ):
        self.surface_volume_dirs = surface_volume_dirs
        self.transform = transform
        self.volumes = volumes_shared_storage

        # Masks: binary masks of scroll regions
        self.scroll_masks = []
        for root in self.surface_volume_dirs:
            root = Path(root)
            self.scroll_masks.append(
                cv2.imread(
                    str(root / 'mask.png'),
                    cv2.IMREAD_GRAYSCALE
                )
            )

        # (Optional) IR images: grayscale images
        self.ir_images = []
        for root in self.surface_volume_dirs:
            root = Path(root)
            path = root / 'ir.png'
            image = None
            if path.exists():
                image = cv2.imread(
                    str(path),
                    cv2.IMREAD_GRAYSCALE
                )
            self.ir_images.append(image)
        
        # (Optional) labels: binary masks of ink
        self.ink_masks = []
        for root in self.surface_volume_dirs:
            root = Path(root)
            path = root / 'inklabels.png'
            image = None
            if path.exists():
                image = cv2.imread(
                    str(path),
                    cv2.IMREAD_GRAYSCALE
                )
            self.ink_masks.append(image)

    def __len__(self) -> int:
        return len(self.volumes)
    
    def __getitem__(
        self, 
        idx
    ) -> Tuple[
        np.ndarray,  # volume
        np.ndarray,  # mask
        Optional[np.ndarray],  # optional IR image 
        Optional[np.ndarray],  # optional labels
    ]:
        image = self.volumes[idx]
        masks = [(self.scroll_masks[idx] > 0).astype(np.uint8)]
        if self.ir_images[idx] is not None:
            masks.append(self.ir_images[idx])
        if self.ink_masks[idx] is not None:
            masks.append((self.ink_masks[idx] > 0).astype(np.uint8))

        if self.transform is not None:
            return self.transform(image=image, masks=masks)
        else:
            return {
                'image': image,
                'masks': masks,
            }


class SubsetWithTransformAndRepeats(Subset):
    def __init__(
        self, 
        dataset: Dataset, 
        indices: Sequence[int],
        transform=None,
        n_repeat: int = 1,
    ):
        super().__init__(dataset, indices)
        self.transform = transform
        self.n_repeat = n_repeat
        
    def __len__(self) -> int:
        return super().__len__() * self.n_repeat

    def __getitem__(self, index):
        item = super().__getitem__(index % super().__len__())
        if self.transform is not None:
            item = self.transform(**item)
        return item
