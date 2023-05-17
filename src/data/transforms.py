import random
import cv2
import numpy as np
import torch
from typing import Dict
from albumentations import Rotate, DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops import functional as F


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
        img_new = np.full_like(img, self.mask_value)
        h_offset = int(img.shape[0] * (1 - np.cos(np.deg2rad(angle))) / 2)
        img_shrinked = cv2.resize(
            img, 
            (0, 0),
            fx=1, 
            fy=np.cos(np.deg2rad(angle)),
            interpolation=cv2.INTER_LINEAR,
        )
        img_new[h_offset:h_offset + img_shrinked.shape[0], :] = img_shrinked
        return img_new


class RandomCropVolumeInside2dMask:
    def __init__(
        self,
        height, 
        width, 
        depth,
        always_apply=True,
        p=1.0,
        crop_mask_index=0,
    ):
        self.height = height
        self.width = width
        self.depth = depth
        self.always_apply = always_apply
        self.p = p
        self.crop_mask_index = crop_mask_index
    
    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        # Get crop mask
        crop_mask = kwargs['masks'][self.crop_mask_index]  # (H, W)

        # Crop mask so that it is not on the border with
        # height // 2, width // 2 offsets
        crop_mask = crop_mask[
            self.height // 2:-self.height // 2,
            self.width // 2:-self.width // 2,
        ]

        # Get indices of non-zero elements
        # TODO: consider caching / precomputing
        nonzero_indices = np.nonzero(crop_mask)

        # Get random index
        random_index = random.randint(0, nonzero_indices[0].shape[0] - 1)

        # Add height // 2, width // 2 offsets to center index
        center_index = (
            nonzero_indices[0][random_index] + self.height // 2,
            nonzero_indices[1][random_index] + self.width // 2,
        )

        # Get crop indices
        w_start = center_index[1] - self.width // 2
        w_end = center_index[1] + self.width // 2
        h_start = center_index[0] - self.height // 2
        h_end = center_index[0] + self.height // 2

        z_start_max = max(kwargs['image'].shape[2] - self.depth, 0)
        z_start = np.random.randint(0, z_start_max) if z_start_max > 0 else 0
        z_end = min(z_start + self.depth, kwargs['image'].shape[2])

        # Crop data
        kwargs['image'] = kwargs['image'][h_start:h_end, w_start:w_end, z_start:z_end]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = kwargs['masks'][i][h_start:h_end, w_start:w_end]

        return kwargs


class RandomCropVolume(DualTransform):
    """Crop a random part of the input.
    If image, crop is 3D, if mask, crop is 2D.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        depth (int): depth of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, depth, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth

    def apply(self, img, h_start=0, w_start=0, d_start=0, is_mask=False, **params):
        img = F.random_crop(img, self.height, self.width, h_start, w_start)
        if not is_mask:
            z = np.random.randint(d_start, max(img.shape[2] - self.depth, d_start + 1))
            img = img[:, :, z:min(z + self.depth, img.shape[2])]
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, is_mask=True, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, self.height, self.width, **params)

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random(), "d_start": random.random()}

    def get_transform_init_args_names(self):
        return ("height", "width", "depth")


class CenterCropVolume(DualTransform):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        depth (int): depth of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, height, width, depth, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth

    def apply(self, img, is_mask=False, **params):
        img = F.center_crop(img, self.height, self.width)
        if not is_mask:
            z = max((img.shape[2] - self.depth) // 2, 0)
            img = img[:, :, z:min(z + self.depth, img.shape[2])]
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, is_mask=True, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_center_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_center_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ("height", "width", "depth")


class ToCHWD(ImageOnlyTransform):
    def apply(self, img: torch.Tensor, **params) -> torch.Tensor:
        # (D, H, W) (grayscale) ->
        # (1, H, W, D)
        return img.unsqueeze(0).permute(0, 2, 3, 1)


class ToWritable(DualTransform):
    def apply(self, img, **params):
        img = img.copy()
        img.setflags(write=True)
        return img
