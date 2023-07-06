import torchvision.transforms as transforms
import numpy as np
from typing import Tuple


class SageDINOTransform(object):
    def __init__(self):
        self.rgb_transform = lambda rh, rw: transforms.Compose(
            [
                transforms.ToTensor(),
                # these parameters are from augmentation in vicreg
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                # crop the image to match the size of thermal image\
                # keep the same aspect ratio 4/3
                # TODO try to train the NN with different ratio
                transforms.CenterCrop((rh, rw)),
#                 transforms.CenterCrop((1800, 2400)),
                transforms.Resize((600, 800))
            ]
        )
        self.thermal_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image_pair: Tuple[np.ndarray, np.ndarray, str]):
        ratio = 4 / 3
        rh = int(image_pair[0].shape[0] / 1.1)
        rw = int(rh * ratio)
        rgb_out = self.rgb_transform(rh, rw)(image_pair[0])
        thermal_out = self.thermal_transform(image_pair[1])
        return rgb_out, thermal_out, image_pair[2]


    
    
    def expand_dim(image, bounds=None):
        """Expand a single channel image into three channels

        image : np.ndarray
            the image data to transform
        bounds : Tuple or None
            If not None, Separate the image into 3 channels given
            the two bounds. The bound is a tuple (lower, upper).
                channel 1: (-Inf, lower]
                channel 2: (lower, upper]
                channel 3: (upper, Inf)
            The temperate higher or lower than the bounds are repalced
            by the bound values.
        """
        if len(image.shape) > 2:
            raise ValueError(f"Image should be 1 channel, but instead has {image.shape[-1]} channels")
        newim = np.stack([image for i in range(3)], axis=2)
        if bounds is not None:
            assert len(bounds) == 2, f"Bounds should have two elements, but instead has {len(bounds)} elements"
            all_bounds = [-np.inf, *bounds, np.inf]
            for i in range(1, 4):
                pick_lo = newim[:, :, i-1] > all_bounds[i-1]
                pick_hi = newim[:, :, i-1] <= all_bounds[i]
                newim[~pick_lo, i-1] = all_bounds[i-1]
                newim[~pick_hi, i-1] = all_bounds[i]
        return newim
        