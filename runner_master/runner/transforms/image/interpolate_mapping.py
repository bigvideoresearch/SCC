__all__ = [
    'interpolate_int2str',
    'interpolate_str2int',
    'interpolate_any2str',
    'interpolate_any2int',
]


from PIL import Image


pairs = [
    (Image.NEAREST, 'nearest'),
    (Image.LANCZOS, 'lanczos'),
    (Image.BILINEAR, 'bilinear'),
    (Image.BICUBIC, 'bicubic'),
    (Image.BOX, 'box'),
    (Image.HAMMING, 'hamming'),
]
interpolate_int2str = dict()
interpolate_str2int = dict()
interpolate_any2str = dict()
interpolate_any2int = dict()
for i, s in pairs:
    interpolate_int2str[i] = s
    interpolate_str2int[s] = i
    interpolate_any2str[i] = s
    interpolate_any2str[s] = s
    interpolate_any2int[i] = i
    interpolate_any2int[s] = i
