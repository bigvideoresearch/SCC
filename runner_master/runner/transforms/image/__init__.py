from torchvision.transforms import *

from .fancy_pca import *
from .convert import *

from .autoaugment_operators import *
from .autoaugment_policies import *

from .normalize import *
from .common import *
from .resize_before_crop import *


def get(name, *args, **kwargs):
    if name not in locals():
        raise RuntimeError('transform [{}] not found in runner.transforms.image')
    TransformClass = locals()[name]
    transform = TransformClass(*args, **kwargs)
    return transform
