from .. import image as run_trans
import torchvision.transforms as tvs_trans
from .interpolate_mapping import interpolate_any2int


STD_RATIO = 256 / 224


class PartialRemainTransform:

    def __init__(self, trans1, trans2, partial=False):
        if partial:
            self.partial = tvs_trans.Compose(trans1)
            self.remain = tvs_trans.Compose(trans2)
        else:
            self.partial = tvs_trans.Compose(trans1 + trans2)
            self.remain = tvs_trans.Compose([])

    def __call__(self, image):
        return self.partial(image)


class ValidImagenet(PartialRemainTransform):

    def __init__(self, image_size, ratio=STD_RATIO, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        if ratio is None:
            pre_size = {224: 256, 299: 320, 331: 352}[image_size]
        else:
            pre_size = round(image_size * ratio)
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.Resize(pre_size, interpolation=interpolation),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(ValidImagenet, self).__init__(trans1, trans2, partial)


class TrainImagenet(PartialRemainTransform):

    def __init__(self, image_size, ratio=STD_RATIO, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.RandomResizedCrop(size=image_size, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.FancyPCA(),
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainImagenet, self).__init__(trans1, trans2, partial)


class TrainColorless(PartialRemainTransform):

    def __init__(self, image_size, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.RandomResizedCrop(size=image_size, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainColorless, self).__init__(trans1, trans2, partial)


class TrainSinglescale(PartialRemainTransform):

    def __init__(self, image_size, ratio=STD_RATIO, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        pre_size = round(image_size * ratio)
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.Resize(pre_size, interpolation=interpolation),
            tvs_trans.RandomCrop(image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.FancyPCA(),
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainSinglescale, self).__init__(trans1, trans2, partial)

class TrainAutoAugmentImageNetPolicy(PartialRemainTransform):

    def __init__(self, image_size, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.RandomResizedCrop(size=image_size, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            run_trans.ImageNetPolicy(),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainAutoAugmentImageNetPolicy, self).__init__(trans1, trans2, partial)


class TrainAutoAugmentEfficientNetPolicy(PartialRemainTransform):

    def __init__(self, image_size, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.RandomResizedCrop(size=image_size, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            run_trans.EfficientNetPolicy(),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainAutoAugmentEfficientNetPolicy, self).__init__(trans1, trans2, partial)


class TrainRandAugment(PartialRemainTransform):

    def __init__(self, image_size, num_ops, magnitude_level, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            tvs_trans.RandomResizedCrop(size=image_size, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            run_trans.RandAugment(num_ops, magnitude_level),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainRandAugment, self).__init__(trans1, trans2, partial)


class TrainRandAugmentResizedCrop(PartialRemainTransform):

    def __init__(self, image_size, num_ops, magnitude_level, interpolation='bilinear', partial=False):
        interpolation = interpolate_any2int[interpolation]
        trans1 = [
            run_trans.Convert('RGB'),
            run_trans.RandAugmentResizedCrop(image_size, magnitude_level, interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),
            run_trans.RandAugment(num_ops, magnitude_level),
            tvs_trans.ToTensor(),
        ]
        trans2 = [
            run_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        super(TrainRandAugmentResizedCrop, self).__init__(trans1, trans2, partial)
