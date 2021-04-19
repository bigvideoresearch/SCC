__all__ = ['ResizeBeforeCrop']


import torchvision.transforms.functional as TF
from .interpolate_mapping import interpolate_any2int


STD_RATIO = 256 / 224
STD_PADDING = 32


class ResizeBeforeCrop:

    def __init__(self, center_crop_size, pre_size=None, mode='ratio',
                 crop_ratio=STD_RATIO, crop_padding=STD_PADDING,
                 interpolation='bilinear'):
        common_setting = {224: 256, 299: 320, 331: 352}  # commonly used setting for ImageNet in research

        if pre_size is not None:
            self.size = pre_size
        elif crop_ratio is not None or crop_padding is not None:
            if mode == 'ratio':
                self.size = round(center_crop_size * crop_ratio)
            elif mode == 'padding':
                self.size = center_crop_size + STD_PADDING
        elif center_crop_size in common_setting:
            self.size = common_setting[center_crop_size]
        else:
            msg = '\n'.join([
                'invalid arguments:'
                '- pre_size is None',
                '- crop_ratio is None',
                '- center_crop_size ({}) not in common_setting: {}'.format(
                    center_crop_size,
                    common_setting,
                ),
            ])
            raise ValueError(msg)

        self.interpolation = interpolate_any2int[interpolation]

    def __call__(self, img):
        return TF.resize(img, self.size, self.interpolation)
