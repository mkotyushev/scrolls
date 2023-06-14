import itertools
import logging
import math
import random
import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from albumentations import Rotate, DualTransform, ImageOnlyTransform, Resize, RandomScale
from albumentations.augmentations.crops import functional as F_crops
from albumentations.augmentations.geometric import functional as F_geometric
from albumentations.augmentations import functional as F
from torchvision.transforms import functional as F_torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


logger = logging.getLogger(__name__)


class RotateZ(Rotate):
    """Rotate around X axis on randomly selected angle."""
    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        # (H, W, D) -> (D, H, W)
        kwargs['image'] = np.transpose(kwargs['image'], (2, 0, 1))

        # Rotate as usual image
        kwargs = super().__call__(*args, force_apply=force_apply, **kwargs)

        # (D, H, W) -> (H, W, D)
        kwargs['image'] = np.transpose(kwargs['image'], (1, 2, 0))

        return kwargs

    def apply_to_mask(self, img, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, **params):
        # Project mask to XZ plane.

        # cv2's y corresponds to H dimention which is shrinked
        # by cos(angle): mask is projected to XZ plane.
        H, W = img.shape[:2]
        img_out = cv2.resize(
            img, 
            (0, 0),
            fx=1, 
            fy=np.cos(np.deg2rad(angle)),
            interpolation=cv2.INTER_NEAREST,
        )

        # Pad mask to the same size as image by H dimention.
        top_pad = (H - img_out.shape[0]) // 2
        bottom_pad = H - img_out.shape[0] - top_pad
        img_out = cv2.copyMakeBorder(
            img_out,
            top=top_pad,
            bottom=bottom_pad,
            left=0,
            right=0,
            borderType=self.border_mode,
        )

        # Crop mask: if border is cropped, then
        # mask should not only be projected to XZ plane,
        # but also cropped to the same size as image by H dimention.

        # Here only H changes and H corresponds
        # to y here and x in image case because of transpose
        # (see __call__ method).
        if self.crop_border:
            img_out = F_crops.crop(img_out, x_min=0, y_min=x_min, x_max=W, y_max=x_max)

        return img_out

class RandomCropVolumeInside2dMask:
    """Crop a random part of the input.
    """
    def __init__(
        self,
        base_size: Optional[int] = None,
        base_depth: Optional[int] = None,
        scale: Tuple[float, float] = (1.0, 1.0),
        scale_z: Tuple[float, float] = (1.0, 1.0),
        ratio: Tuple[float, float] = (1.0, 1.0),
        value: int = 0,
        mask_value: int = 0,
        crop_mask_index=0,
        always_apply=True,
        p=1.0,
    ):
        if not (base_size is not None or base_depth is not None):
            logger.warning(
                f"Eigher base_size or base_depth should be not None. "
                f"Got base_size={base_size}, base_depth={base_depth}. "
                f"No transform will be performed."
            )
        
        assert scale[0] > 0 and scale[1] > 0, f"scale should be positive. Got {scale}"
        assert scale_z[0] > 0 and scale_z[1] > 0, f"scale_z should be positive. Got {scale_z}"
        assert ratio[0] > 0 and ratio[1] > 0, f"ratio should be positive. Got {ratio}"
        assert scale[0] <= scale[1], f"scale[0] should be less or equal than scale[1]. Got {scale}"
        assert scale_z[0] <= scale_z[1], f"scale_z[0] should be less or equal than scale_z[1]. Got {scale_z}"
        assert ratio[0] <= ratio[1], f"ratio[0] should be less or equal than ratio[1]. Got {ratio}"

        self.base_size = base_size
        self.base_depth = base_depth
        self.scale = scale
        self.scale_z = scale_z
        self.ratio = ratio
        self.value = value
        self.mask_value = mask_value
        self.crop_mask_index = crop_mask_index

        self.always_apply = always_apply
        self.p = p
    
    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs
        
        # Get random scale and ratio
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        h_start, h_end = 0, kwargs['image'].shape[0]
        w_start, w_end = 0, kwargs['image'].shape[1]
        if self.base_size is not None:
            # Get height and width
            height = int(self.base_size * scale)
            width = int(self.base_size * scale * ratio)       

            # Get crop mask
            crop_mask = kwargs['masks'][self.crop_mask_index]  # (H, W)
            
            # Crop the mask to ensure that the crop is inside the mask
            h_shift, w_shift = height // 2 + 1, width // 2 + 1
            crop_mask = crop_mask[
                h_shift:-h_shift,
                w_shift:-w_shift,
            ]

            # Get indices of non-zero elements
            nonzero_indices = np.nonzero(crop_mask)

            # Get random index
            random_index = random.randint(0, nonzero_indices[0].shape[0] - 1)
            h_center, w_center = nonzero_indices[0][random_index], nonzero_indices[1][random_index]

            # Shift indices back to compensate crop above
            h_center += h_shift
            w_center += w_shift

            # Get crop indices
            h_start = h_center - height // 2
            h_end = h_start + height
            w_start = w_center - width // 2
            w_end = w_start + width

            # Ensure that crop is inside the image
            assert h_start >= 0 and h_end <= kwargs['image'].shape[0], \
                f"h_start={h_start} and h_end={h_end} should be in [0, {kwargs['image'].shape[0]}]"
            assert w_start >= 0 and w_end <= kwargs['image'].shape[1], \
                f"w_start={w_start} and w_end={w_end} should be in [0, {kwargs['image'].shape[1]}]"

        scale_z = random.uniform(self.scale_z[0], self.scale_z[1])
        z_start, z_end = 0, kwargs['image'].shape[2]
        if self.base_depth is not None:
            # Get depth
            depth = int(self.base_depth * scale_z)

            # Get random z_start
            z_start_max = max(kwargs['image'].shape[2] - depth, 1)
            z_start = np.random.randint(0, z_start_max)

            # Get z_end
            z_end = min(z_start + depth, kwargs['image'].shape[2])

        # Crop data
        kwargs['image'] = kwargs['image'][h_start:h_end, w_start:w_end, z_start:z_end].copy()
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = kwargs['masks'][i][h_start:h_end, w_start:w_end].copy()

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
        img = F_crops.random_crop(img, self.height, self.width, h_start, w_start)
        if not is_mask:
            z = np.random.randint(d_start, max(img.shape[2] - self.depth, d_start + 1))
            img = img[:, :, z:min(z + self.depth, img.shape[2])]
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, is_mask=True, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_bbox(self, bbox, **params):
        return F_crops.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F_crops.keypoint_random_crop(keypoint, self.height, self.width, **params)

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
        strict (bool): if True, will fail if image size is less than crop size by any dimention, 
            otherwise will crop as much as possible. Default: True.
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

    def __init__(self, height, width, depth, strict=True, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        if not ((height is not None and width is not None) or depth is not None):
            logger.warning(
                f"Eigher height and width or depth should be not None. "
                f"Got height={height}, width={width}, depth={depth}. "
                f"No transform will be performed."
            )
        self.height = height
        self.width = width
        self.depth = depth
        self.strict = strict

    def apply(self, img, is_mask=False, **params):
        if self.height is not None and self.width is not None:
            height, width = img.shape[0], img.shape[1]
            if not self.strict:
                if height < self.height:
                    logger.warning(
                        f"Height of the image is {height} but height of the crop is {self.height}. "
                        f"All the height of the image will be used."
                    )
                if width < self.width:
                    logger.warning(
                        f"Width of the image is {width} but width of the crop is {self.width}. "
                        f"All the width of the image will be used."
                    )
                height, width = \
                    min(img.shape[0], self.height), \
                    min(img.shape[1], self.width)
            img = F_crops.center_crop(img, height, width)
        if not is_mask and self.depth is not None:
            if img.shape[2] < self.depth:
                logger.warning(
                    f"Depth of the image is {img.shape[2]} but depth of the crop is {self.depth}. "
                    f"All the depth of the image will be used."
                )

            if self.strict:
                assert img.shape[2] >= self.depth, \
                    f"Depth of the image is {img.shape[2]} but depth of the crop is {self.depth}. " \

            z_start = max((img.shape[2] - self.depth) // 2, 0)
            z_end = min(z_start + self.depth, img.shape[2])
            img = img[:, :, z_start:z_end]
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, is_mask=True, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_bbox(self, bbox, **params):
        return F_crops.bbox_center_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F_crops.keypoint_center_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ("height", "width", "depth")
    

class RandomZSlices(ImageOnlyTransform):
    """Randomly select N (permuted) slices from the input.

    Args:
        n_slices (int): number of slices to select.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, n_slices, permuted=True, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.n_slices = n_slices
        self.permuted = permuted

    def apply(self, img, **params):
        indices = np.random.permutation(img.shape[2])[:self.n_slices]
        if not self.permuted:
            indices = np.sort(indices)
        return img[:, :, indices]

    def get_transform_init_args_names(self):
        return ("n_slices", )


class UniformZSlices(ImageOnlyTransform):
    """Select N (uniformly) slices from the input.

    Args:
        n_slices (int): number of slices to select.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, n_slices, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.n_slices = n_slices

    def apply(self, img, **params):
        indices = np.linspace(0, img.shape[2] - 1, self.n_slices).astype(int)
        return img[:, :, indices]

    def get_transform_init_args_names(self):
        return ("n_slices", )


class ResizeVolume(Resize):
    def __init__(self, height, width, depth=None, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(height, width, interpolation, always_apply, p)
        self.depth = depth

    def apply(self, img, interpolation=cv2.INTER_LINEAR, is_mask=False, **params):
        # Interpolate 2D
        img = super().apply(img, interpolation=interpolation, **params)
        if is_mask:
            return img
    
        if self.depth is not None:
            # Interpolate depth
            # (H, W, D) -> (D, H, W)
            img = np.transpose(img, (2, 0, 1))
            
            # Resize as usual image
            img = F_geometric.resize(img, height=self.depth, width=img.shape[1], interpolation=interpolation)

            # (D, H, W) -> (H, W, D)
            img = np.transpose(img, (1, 2, 0))

        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, is_mask=True, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})



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

class RandomScaleResize(RandomScale):
    """Same as RandomScale but resize to original size after scaling."""
    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        original_h, original_w = img.shape[:2]
        img = super().apply(img, scale, interpolation, **params)
        return F_geometric.resize(img, height=original_h, width=original_w, interpolation=interpolation)


class SubtractDivide:
    """Subtracttion and division is applied by the formula: 
    `img = (img - subtract) / divide`

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, 
        subtract: float = 0.0, 
        divide: float = 1.0,
        always_apply=True,
        p=1.0, 
    ):
        self.subtract = subtract
        self.divide = divide
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs
        
        subtract, divide = self.subtract, self.divide
        
        if (
            "subtract" in kwargs and kwargs["subtract"] is not None and
            "divide" in kwargs and kwargs["divide"] is not None
        ):
            subtract = kwargs["subtract"]
            divide = kwargs["divide"]
        else:
            logger.warning(
                f"Either subtract or divide values are not provided. "
                f"Subtract is set to {subtract} and divide is set to {divide}."
            )
        
        kwargs['image'] = F.normalize(kwargs['image'], subtract, divide, 1)

        return kwargs


class CopyPastePositive:
    """Copy masked area from one image to another.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self, 
        mask_index: int = 2,
        always_apply=True,
        p=1.0, 
    ):
        self.mask_index = mask_index
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        mask = (kwargs['masks'][self.mask_index] > 0) & (kwargs['masks1'][self.mask_index] <= 0)

        kwargs['image'][mask] = kwargs['image1'][mask]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][mask] = kwargs['masks1'][i][mask]

        return kwargs


# https://github.com/albumentations-team/albumentations/pull/1409/files
class MixUp:
    def __init__(
        self,
        alpha = 32.,
        beta = 32.,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h1, w1, _ = kwargs['image'].shape
        h2, w2, _ = kwargs['image1'].shape
        if h1 != h2 or w1 != w2:
            raise ValueError("MixUp transformation expects both images to have identical shape.")
        
        r = np.random.beta(self.alpha, self.beta)
        
        kwargs['image'] = (kwargs['image'] * r + kwargs['image1'] * (1 - r))
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = (kwargs['masks'][i] * r + kwargs['masks1'][i] * (1 - r))
        
        return kwargs


class CutMix:
    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h, w, _ = kwargs['image'].shape
        h1, w1, _ = kwargs['image1'].shape
        if (
            h < self.height or 
            w < self.width or 
            h1 < self.height or 
            w1 < self.width
        ):
            raise ValueError("CutMix transformation expects both images to be at least {}x{} pixels.".format(self.max_height, self.max_width))

        # Get random bbox
        h_start = random.randint(0, h - self.height)
        w_start = random.randint(0, w - self.width)
        h_end = h_start + self.height
        w_end = w_start + self.width

        # Copy image and masks region
        kwargs['image'][h_start:h_end, w_start:w_end] = kwargs['image1'][h_start:h_end, w_start:w_end]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][h_start:h_end, w_start:w_end] = kwargs['masks1'][i][h_start:h_end, w_start:w_end]
        
        return kwargs


class ToFloatMasks:
    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = kwargs['masks'][i].astype(np.float32)
        return kwargs


class EnlargePositive:
    """Scale up masked area."""
    def __init__(
        self, 
        mask_index: int = 2,
        scale_limit: float = 0.1,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0, 
    ):
        self.mask_index = mask_index
        self.scale_limit = scale_limit
        self.interpolation = interpolation
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs
            
        # Get random scale
        scale = 1 + random.uniform(0, self.scale_limit)

        # Get crop bbox around center of mass
        mask = kwargs['masks'][self.mask_index]
        center_h = math.floor(mask.mean(axis=1) * scale)
        center_w = math.floor(mask.mean(axis=0) * scale)
        h_start = max(0, center_h - mask.shape[0] // 2)
        w_start = max(0, center_w - mask.shape[1] // 2)
        h_end = min(kwargs['image'].shape[0], h_start + mask.shape[0])
        w_end = min(kwargs['image'].shape[1], w_start + mask.shape[1])

        # Copy image and masks
        image_larger = kwargs['image'].copy()
        masks_larger = [mask.copy() for mask in kwargs['masks']]

        # Scale up and crop around center of mass
        image_larger = cv2.resize(
            image_larger, 
            (0, 0),
            fx=scale, 
            fy=scale,
            interpolation=self.interpolation,
        )
        image_larger = F_crops.crop(
            image_larger, 
            xmin=w_start,
            ymin=h_start,
            xmax=w_end,
            ymax=h_end,
        )

        for i in range(len(masks_larger)):
            masks_larger[i] = cv2.resize(
                masks_larger[i], 
                (0, 0),
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            masks_larger[i] = F_crops.crop(
                masks_larger[i], 
                xmin=w_start,
                ymin=h_start,
                xmax=w_end,
                ymax=h_end,
            )

        # Get enlarged mask
        mask = masks_larger[self.mask_index] > 0

        # Copy masked area
        kwargs['image'][h_start:h_end, w_start:w_end][mask] = image_larger[mask]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][h_start:h_end, w_start:w_end][mask] = masks_larger[i][mask]

        return kwargs


class TtaHorizontalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
        return batch.flip(3)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(3)


class TtaVerticalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
        return batch.flip(2)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(2)


class TtaRotate90:
    def __init__(self, n_rot) -> None:
        assert n_rot % 4 != 0, f"n_rot should not be divisible by 4. Got {n_rot}"
        self.n_rot = n_rot % 4
    
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
        return batch.rot90(self.n_rot, (2, 3))

    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.rot90(-self.n_rot, (2, 3))


class TtaRotate:
    def __init__(self, limit_degrees: int = 90, fill_value=0.0) -> None:
        self.limit_degrees = limit_degrees
        self.fill_value = fill_value
        self.angle = None

    @staticmethod
    def _rotate(batch: torch.Tensor, angle: int, fill: float=0) -> torch.Tensor:
        N, C, H, W, D = batch.shape
        # (N, C, H, W, D) -> (N, C, D, H, W)
        batch = batch.permute(0, 1, 4, 2, 3)
        # (N, C, D, H, W) -> (N, C * D, H, W)
        batch = batch.reshape(N, C * D, H, W)
        batch = F_torchvision.rotate(
            batch,
            angle,
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            expand=False,
            fill=fill,
        )
        # (N, C * D, H, W) -> (N, C, D, H, W)
        batch = batch.reshape(N, C, D, H, W)
        # (N, C, D, H, W) -> (N, C, H, W, D)
        batch = batch.permute(0, 1, 3, 4, 2)
        return batch

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        assert self.angle is None, "TtaRotate should be applied only once."
        # (N, C, H, W, D)
        self.angle = random.randint(-self.limit_degrees, self.limit_degrees) 
        return TtaRotate._rotate(batch, self.angle, fill=self.fill_value)

    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        assert self.angle is not None, "TtaRotate should be applied before TtaRotate.apply_inverse_to_pred."
        # (N, C = 1, H, W) -> (N, C = 1, H, W, D = 1)
        batch_pred = batch_pred.unsqueeze(-1)
        # Fill with NaNs to ignore them in averaging
        batch_pred = TtaRotate._rotate(batch_pred, -self.angle, fill=torch.nan)
        # (N, C = 1, H, W, D = 1) -> (N, C = 1, H, W)
        batch_pred = batch_pred.squeeze(-1)
        self.angle = None
        return batch_pred



class Tta:
    def __init__(self, model, n_random_replays=1, use_flip=True, use_rotate=True):
        assert n_random_replays > 0 or use_flip or use_rotate, \
            "At least one of n_random_replays > 0, "\
            "use_flip or use_rotate should be True."
        self.model = model
        
        # Imagenet normalization during training is assumed
        fill_value = (0.0 - sum(IMAGENET_DEFAULT_MEAN) / 3) / (sum(IMAGENET_DEFAULT_STD) / 3)
        
        # All possible combinations of
        # - flips
        # - rotations on 90 degrees
        # - n_replays rotations on random angle
        rotates90 = [None]
        if use_rotate:
            rotates90 = [
                None,
                TtaRotate90(1),
                TtaRotate90(2),
                TtaRotate90(3),
            ]
        flips = [None]
        if use_flip:
            flips = [
                None,
                TtaHorizontalFlip(),
                TtaVerticalFlip(),
            ]
        rotates = [None]
        if n_random_replays > 0:
            rotates = [None] + [
                TtaRotate(limit_degrees=45, fill_value=fill_value) 
                for _ in range(n_random_replays)
            ]
        self.transforms = [
            rotates90,
            flips,
            rotates,
        ]

    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        preds = []

        # Apply TTA
        for transform_chain in itertools.product(*self.transforms):
            # Direct transform
            batch_aug = batch.clone()
            for transform in transform_chain:
                if transform is not None:
                    batch_aug = transform.apply(batch_aug)
            
            # Predict
            pred_aug = self.model(batch_aug)
            if pred_aug.ndim == 3:
                pred_aug = pred_aug.unsqueeze(1)

            # Inverse transform
            # Note: order of transforms is reversed
            for transform in reversed(transform_chain):
                if transform is not None:
                    pred_aug = transform.apply_inverse_to_pred(pred_aug)
            
            preds.append(pred_aug)

        return preds

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        preds = self.predict(batch)
        
        # Average predictions, ignoring NaNs
        preds = torch.stack(preds, dim=0)
        preds = torch.nanmean(preds, dim=0)
        
        return preds
