from .single_label import *


def get(name, *args, **kwargs):
    if name not in locals():
        raise RuntimeError('transform [{}] not found in runner.transforms.extra')
    TransformClass = locals()[name]
    transform = TransformClass(*args, **kwargs)
    return transform
