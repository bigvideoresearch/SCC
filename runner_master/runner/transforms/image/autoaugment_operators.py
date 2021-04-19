'''
References:
    https://arxiv.org/abs/1805.09501
    https://github.com/DeepVoltaire/Autoaugment/blob/master/autoaugment.py
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
'''


import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms.functional as F
from .interpolate_mapping import interpolate_int2str, interpolate_any2int


class ResampleOp:

    def set_resample_mode(self, resample):
        self.resample = resample


class ShearX(ResampleOp):

    def __init__(self, level, upper=0.3, resample=0):
        self.magnitude = level * upper / 10
        self.resample = resample

    def __call__(self, img):
        m = self.magnitude * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, m, 0, 0, 1, 0), self.resample, fillcolor=(128, 128, 128))


class ShearY(ResampleOp):

    def __init__(self, level, upper=0.3, resample=0):
        self.magnitude = level * upper / 10
        self.resample = resample

    def __call__(self, img):
        m = self.magnitude * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, m, 1, 0), self.resample, fillcolor=(128, 128, 128))


class TranslateX(ResampleOp):

    def __init__(self, level, upper=150 / 331, resample=0):
        self.magnitude = level * upper / 10
        self.resample = resample

    def __call__(self, img):
        m = self.magnitude * img.size[0] * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, m, 0, 1, 0), self.resample, fillcolor=(128, 128, 128))


class TranslateY(ResampleOp):

    def __init__(self, level, upper=150 / 331, resample=0):
        self.magnitude = level * upper / 10
        self.resample = resample

    def __call__(self, img):
        m = self.magnitude * img.size[1] * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, m), self.resample, fillcolor=(128, 128, 128))


class Rotate(ResampleOp):

    def __init__(self, level, upper=30, resample=0):
        self.magnitude = level * upper / 10
        self.resample = resample

    def __call__(self, img):
        m = self.magnitude * random.choice([-1, 1])
        rot = img.convert("RGBA").rotate(m, self.resample)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


class AutoContrast:

    def __init__(self, level):
        pass

    def __call__(self, img):
        return ImageOps.autocontrast(img)


class Invert:

    def __init__(self, level):
        pass

    def __call__(self, img):
        return ImageOps.invert(img)


class Equalize:

    def __init__(self, level):
        pass

    def __call__(self, img):
        return ImageOps.equalize(img)


class Solarize:

    def __init__(self, level, upper=256):
        '''From https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py:
        self.magnitude = level * upper / 10
        However under this implementation, higher magnitude level actually leads to weaker regularization.
        So I invert it by "256 -".
        '''
        self.magnitude = 256 - level * upper / 10

    def __call__(self, img):
        return ImageOps.solarize(img, self.magnitude)


class SolarizeAdd:
    '''
    The original Autoaugment paper does not mention this operator.
    It comes from the so-called Policy V0 in the efficientnet repo:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L181
    '''
    def __init__(self, level, upper=110):
        self.magnitude = level * upper / 10

    def __call__(self, img):
        img = np.array(img)
        added_img = (img.astype('float64') + self.magnitude).clip(0, 255).astype('uint8')
        img = np.where(img < 128, added_img, img)
        return Image.fromarray(img)


class Posterize:

    def __init__(self, level, upper=4):
        self.magnitude = max(1, 8 - int(level * upper / 10))

    def __call__(self, img):
        return ImageOps.posterize(img, self.magnitude)


class Enhance:

    def __init__(self, level, upper=0.9):
        self.magnitude = level * upper / 10

    def __call__(self, img):
        magnitude = max(0, 1 + self.magnitude * random.choice([-1, 1]))
        return self.enhance(img, magnitude)

    def enhance(self, img, magnitude):
        raise NotImplementedError


class Contrast(Enhance):

    def enhance(self, img, magnitude):
        return ImageEnhance.Contrast(img).enhance(magnitude)


class Color(Enhance):

    def enhance(self, img, magnitude):
        return ImageEnhance.Color(img).enhance(magnitude)


class Brightness(Enhance):

    def enhance(self, img, magnitude):
        return ImageEnhance.Brightness(img).enhance(magnitude)


class Sharpness(Enhance):

    def enhance(self, img, magnitude):
        return ImageEnhance.Brightness(img).enhance(self.magnitude)


class Cutout:

    def __init__(self, level, upper=60 / 331, fillcolor=0):
        self.magnitude = level * upper / 10
        self.fillcolor = fillcolor

    def __call__(self, img):
        w, h = img.size
        x = np.random.randint(w)
        y = np.random.randint(h)
        offset = int(self.magnitude * min(img.size) / 2)
        x1 = np.clip(x - offset, 0, w)
        x2 = np.clip(x + offset, 0, w)
        y1 = np.clip(y - offset, 0, h)
        y2 = np.clip(y + offset, 0, h)

        paste = Image.new(img.mode, (x2 - x1, y2 - y1), self.fillcolor)
        img.paste(paste, (x1, y1, x2, y2))
        return img


class RandAugmentResizedCrop:

    def __init__(self, size, level,
                 scale_anchors=(0.08, 0.6457, 1.0),
                 ratio_anchors=(3 / 4, 1.0, 4 / 3),
                 priority='ratio', interpolation='bilinear'):
        if scale_anchors[2] > 1:
            raise RuntimeError('max scale should not > 1, but got {}'.format(scale_anchors[2]))
        if ratio_anchors[0] > ratio_anchors[2]:
            raise RuntimeError('ratio[0] > ratio[2]: {} '.format(ratio_anchors))
        if priority not in ('scale', 'ratio'):
            raise RuntimeError('priority should be "scale" or "ratio" , but got {}'.format(repr(priority)))

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.level = level

        min_scale, std_scale, max_scale = scale_anchors
        min_ratio, std_ratio, max_ratio = ratio_anchors
        if level > 10:
            self.scale = (min_scale, max_scale)
            self.ratio = (min_ratio, max_ratio)
        else:
            power = level / 10
            lower_scale = (min_scale / std_scale) ** power * std_scale
            upper_scale = (max_scale / std_scale) ** power * std_scale
            self.scale = (lower_scale, upper_scale)
            lower_ratio = (min_ratio / std_ratio) ** power * std_ratio
            upper_ratio = (max_ratio / std_ratio) ** power * std_ratio
            self.ratio = (lower_ratio, upper_ratio)

        self.priority = priority
        self.interpolation = interpolate_any2int[interpolation]

    @staticmethod
    def get_params(img, scale, ratio, priority):
        full_area = img.size[0] * img.size[1]
        part_area = random.uniform(*scale) * full_area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        w = math.sqrt(part_area * aspect_ratio)
        h = math.sqrt(part_area / aspect_ratio)

        rel_w = w / img.size[0]
        rel_h = h / img.size[1]
        assert rel_w <= 1 or rel_h <= 1
        if rel_w > 1 or rel_h > 1:
            if priority == 'scale':
                if rel_w > 1:
                    w /= rel_w
                    h *= rel_w
                elif rel_h > 1:
                    w *= rel_h
                    h /= rel_h
            elif priority == 'ratio':
                rel = max(rel_w, rel_h)
                w /= rel
                h /= rel
            else:
                raise RuntimeError('invalid argument priority: {}'.format(priority))

        w = round(w)
        h = round(h)
        if w > img.size[0] or h > img.size[1]:
            raise RuntimeError('w = {}, h = {}, img size {}'.format(w, h, img.size))

        i = random.randint(0, img.size[1] - h)
        j = random.randint(0, img.size[0] - w)
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio, self.priority)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = interpolate_int2str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', level={0}'.format(self.level)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
