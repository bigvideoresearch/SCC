__all__ = ['ImageNetPolicy', 'CIFAR10Policy', 'EfficientNetPolicy', 'RandAugment']


import random
from . import autoaugment_operators as op
from .interpolate_mapping import interpolate_any2int
from torchvision.transforms import Compose, RandomChoice


class RandomApply:

    def __init__(self, operator, prob):
        self.operator = operator
        self.prob = prob

    def __call__(self, img):
        if random.random() <= self.prob:
            img = self.operator(img)
        return img


class AutoAugmentPolicy:

    def __init__(self, sub_policies, interpolation=None):
        if interpolation is not None:
            resample = interpolate_any2int[interpolation]
            for sub_policy in sub_policies:
                for operator, _ in sub_policy:
                    if isinstance(operator, op.ResampleOp):
                        operator.set_resample_mode(resample)
        self.transform = RandomChoice([
            Compose([
                RandomApply(operator, prob)
                for operator, prob in sub_policy
            ])
            for sub_policy in sub_policies
        ])

    def __call__(self, img):
        return self.transform(img)


class CIFAR10Policy(AutoAugmentPolicy):

    def __init__(self, interpolation=None):
        super(CIFAR10Policy, self).__init__([
            [[op.Invert(7), 0.1], [op.Contrast(6), 0.2]],
            [[op.Rotate(2), 0.7], [op.TranslateX(9), 0.3]],
            [[op.Sharpness(1), 0.8], [op.Sharpness(3), 0.9]],
            [[op.ShearY(8), 0.5], [op.TranslateY(9), 0.7]],
            [[op.AutoContrast(8), 0.5], [op.Equalize(2), 0.9]],
            [[op.ShearY(7), 0.2], [op.Posterize(7), 0.3]],
            [[op.Color(3), 0.4], [op.Brightness(7), 0.6]],
            [[op.Sharpness(9), 0.3], [op.Brightness(9), 0.7]],
            [[op.Equalize(5), 0.6], [op.Equalize(1), 0.5]],
            [[op.Contrast(7), 0.6], [op.Sharpness(5), 0.6]],
            [[op.Color(7), 0.7], [op.TranslateX(8), 0.5]],
            [[op.Equalize(7), 0.3], [op.AutoContrast(8), 0.4]],
            [[op.TranslateY(3), 0.4], [op.Sharpness(6), 0.2]],
            [[op.Brightness(6), 0.9], [op.Color(8), 0.2]],
            [[op.Solarize(2), 0.5], [op.Invert(3), 0.0]],
            [[op.Equalize(0), 0.2], [op.AutoContrast(0), 0.6]],
            [[op.Equalize(8), 0.2], [op.Equalize(4), 0.6]],
            [[op.Color(9), 0.9], [op.Equalize(6), 0.6]],
            [[op.AutoContrast(4), 0.8], [op.Solarize(8), 0.2]],
            [[op.Brightness(3), 0.1], [op.Color(0), 0.7]],
            [[op.Solarize(5), 0.4], [op.AutoContrast(3), 0.9]],
            [[op.TranslateY(9), 0.9], [op.TranslateY(9), 0.7]],
            [[op.AutoContrast(2), 0.9], [op.Solarize(3), 0.8]],
            [[op.Equalize(8), 0.8], [op.Invert(3), 0.1]],
            [[op.TranslateY(9), 0.7], [op.AutoContrast(1), 0.9]],
        ], interpolation)


class SVHNPolicy(AutoAugmentPolicy):

    def __init__(self, interpolation=None):
        super(SVHNPolicy, self).__init__([
            [[op.ShearX(4), 0.9], [op.Invert(3), 0.2]],
            [[op.ShearY(8), 0.9], [op.Invert(5), 0.7]],
            [[op.Equalize(5), 0.6], [op.Solarize(6), 0.6]],
            [[op.Invert(3), 0.9], [op.Equalize(3), 0.6]],
            [[op.Equalize(1), 0.6], [op.Rotate(3), 0.9]],
            [[op.ShearX(4), 0.9], [op.AutoContrast(3), 0.8]],
            [[op.ShearY(8), 0.9], [op.Invert(5), 0.4]],
            [[op.ShearY(5), 0.9], [op.Solarize(6), 0.2]],
            [[op.Invert(6), 0.9], [op.AutoContrast(1), 0.8]],
            [[op.Equalize(3), 0.6], [op.Rotate(3), 0.9]],
            [[op.ShearX(4), 0.9], [op.Solarize(3), 0.3]],
            [[op.ShearY(8), 0.8], [op.Invert(4), 0.7]],
            [[op.Equalize(5), 0.9], [op.TranslateY(6), 0.6]],
            [[op.Invert(4), 0.9], [op.Equalize(7), 0.6]],
            [[op.Contrast(3), 0.3], [op.Rotate(4), 0.8]],
            [[op.Invert(5), 0.8], [op.TranslateY(2), 0.0]],
            [[op.ShearY(6), 0.7], [op.Solarize(8), 0.4]],
            [[op.Invert(4), 0.6], [op.Rotate(4), 0.8]],
            [[op.ShearY(7), 0.3], [op.TranslateX(3), 0.9]],
            [[op.ShearX(6), 0.1], [op.Invert(5), 0.6]],
            [[op.Solarize(2), 0.7], [op.TranslateY(7), 0.6]],
            [[op.ShearY(4), 0.8], [op.Invert(8), 0.8]],
            [[op.ShearX(9), 0.7], [op.TranslateY(3), 0.8]],
            [[op.ShearY(5), 0.8], [op.AutoContrast(3), 0.7]],
            [[op.ShearX(2), 0.7], [op.Invert(5), 0.1]],
        ], interpolation)


class ImageNetPolicy(AutoAugmentPolicy):

    def __init__(self, interpolation=None):
        super(ImageNetPolicy, self).__init__([
            [[op.Posterize(8), 0.4], [op.Rotate(9), 0.6]],
            [[op.Solarize(5), 0.6], [op.AutoContrast(5), 0.6]],
            [[op.Equalize(8), 0.8], [op.Equalize(3), 0.6]],
            [[op.Posterize(7), 0.6], [op.Posterize(6), 0.6]],
            [[op.Equalize(7), 0.4], [op.Solarize(4), 0.2]],
            [[op.Equalize(4), 0.4], [op.Rotate(8), 0.8]],
            [[op.Solarize(3), 0.6], [op.Equalize(7), 0.6]],
            [[op.Posterize(5), 0.8], [op.Equalize(2), 1.0]],
            [[op.Rotate(3), 0.2], [op.Solarize(8), 0.6]],
            [[op.Equalize(8), 0.6], [op.Posterize(6), 0.4]],
            [[op.Rotate(8), 0.8], [op.Color(0), 0.4]],
            [[op.Rotate(9), 0.4], [op.Equalize(2), 0.6]],
            [[op.Equalize(7), 0.0], [op.Equalize(8), 0.8]],
            [[op.Invert(4), 0.6], [op.Equalize(8), 1.0]],
            [[op.Color(4), 0.6], [op.Contrast(8), 1.0]],
            [[op.Rotate(8), 0.8], [op.Color(2), 1.0]],
            [[op.Color(8), 0.8], [op.Solarize(7), 0.8]],
            [[op.Sharpness(7), 0.4], [op.Invert(8), 0.6]],
            [[op.ShearX(5), 0.6], [op.Equalize(9), 1.0]],
            [[op.Color(0), 0.4], [op.Equalize(3), 0.6]],
            [[op.Equalize(7), 0.4], [op.Solarize(4), 0.2]],
            [[op.Solarize(5), 0.6], [op.AutoContrast(5), 0.6]],
            [[op.Invert(4), 0.6], [op.Equalize(8), 1.0]],
            [[op.Color(4), 0.6], [op.Contrast(8), 1.0]],
            [[op.Equalize(8), 0.8], [op.Equalize(3), 0.6]],
        ], interpolation)


class EfficientNetPolicy(AutoAugmentPolicy):
    '''
    This is the so-called "AutoAugment Policy V0" from EfficientNet 's repo:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    '''

    def __init__(self, interpolation=None):
        super(EfficientNetPolicy, self).__init__([
            [[op.Equalize(1), 0.8], [op.ShearY(4), 0.8]],
            [[op.Color(9), 0.4], [op.Equalize(3), 0.6]],
            [[op.Color(1), 0.4], [op.Rotate(8), 0.6]],
            [[op.Solarize(3), 0.8], [op.Equalize(7), 0.4]],
            [[op.Solarize(2), 0.4], [op.Solarize(2), 0.6]],
            [[op.Color(0), 0.2], [op.Equalize(8), 0.8]],
            [[op.Equalize(8), 0.4], [op.SolarizeAdd(3), 0.8]],
            [[op.ShearX(9), 0.2], [op.Rotate(8), 0.6]],
            [[op.Color(1), 0.6], [op.Equalize(2), 1.0]],
            [[op.Invert(9), 0.4], [op.Rotate(0), 0.6]],
            [[op.Equalize(9), 1.0], [op.ShearY(3), 0.6]],
            [[op.Color(7), 0.4], [op.Equalize(0), 0.6]],
            [[op.Posterize(6), 0.4], [op.AutoContrast(7), 0.4]],
            [[op.Solarize(8), 0.6], [op.Color(9), 0.6]],
            [[op.Solarize(4), 0.2], [op.Rotate(9), 0.8]],
            [[op.Rotate(7), 1.0], [op.TranslateY(9), 0.8]],
            [[op.ShearX(0), 0.0], [op.Solarize(4), 0.8]],
            [[op.ShearY(0), 0.8], [op.Color(4), 0.6]],
            [[op.Color(0), 1.0], [op.Rotate(2), 0.6]],
            [[op.Equalize(4), 0.8], [op.Equalize(8), 0.0]],
            [[op.Equalize(4), 1.0], [op.AutoContrast(2), 0.6]],
            [[op.ShearY(7), 0.4], [op.SolarizeAdd(7), 0.6]],
            [[op.Posterize(2), 0.8], [op.Solarize(10), 0.6]],
            [[op.Solarize(8), 0.6], [op.Equalize(1), 0.6]],
            [[op.Color(6), 0.8], [op.Rotate(5), 0.4]],
        ], interpolation)


class RandAugment:
    '''
    Reference:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    Not exactly the same as the RandAugment Paper.
    '''

    def __init__(self, num_ops, magnitude_level, interpolation=None):
        self.ops = [
            op.AutoContrast(magnitude_level),
            op.Equalize(magnitude_level),
            op.Invert(magnitude_level),
            op.Rotate(magnitude_level),
            op.Posterize(magnitude_level),
            op.Solarize(magnitude_level),
            op.Color(magnitude_level),
            op.Contrast(magnitude_level),
            op.Brightness(magnitude_level),
            op.Sharpness(magnitude_level),
            op.ShearX(magnitude_level),
            op.ShearY(magnitude_level),
            op.TranslateX(magnitude_level),
            op.TranslateY(magnitude_level),
            op.Cutout(magnitude_level),
            op.SolarizeAdd(magnitude_level),
        ]
        if interpolation is not None:
            resample = interpolate_any2int[interpolation]
            for operator in self.ops:
                if isinstance(operator, op.ResampleOp):
                    operator.set_resample_mode(resample)
        self.num_ops = num_ops

    def __call__(self, img):
        for _ in range(self.num_ops):
            if random.random() <= 0.5:
                op = random.choice(self.ops)
                img = op(img)
        return img
