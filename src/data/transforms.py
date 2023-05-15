import cv2
import numpy as np
from albumentations import Rotate, ImageOnlyTransform, DualTransform


class RotateX(Rotate):
    """Rotate along X axis on randomly selected angle."""
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # (H, W, D) -> (D, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Rotate as usual image
        img = super().apply(img, **params)

        # (D, H, W) -> (H, W, D)
        img = np.transpose(img, (1, 2, 0))

        return img
    
    def apply_to_mask(self, img, angle=0, **params):
        # Resize by H axis on cos angle
        img = cv2.resize(
            img, 
            (int(img.shape[1] * np.cos(np.deg2rad(angle))), img.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        return img
    

class RandomCropZ(ImageOnlyTransform):
    """Crop randomly selected slice along Z axis."""
    def __init__(self, crop_size: int):
        super().__init__(always_apply=True, p=1)
        self.crop_size = crop_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        z = np.random.randint(0, img.shape[2] - self.crop_size)
        return img[:, :, z:z + self.crop_size]


class Copy(DualTransform):
    """Copy image."""
    def __init__(self):
        super().__init__(always_apply=True, p=1)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.copy()

    def apply_to_mask(self, img, **params):
        return img.copy()
