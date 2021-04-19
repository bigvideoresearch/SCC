__all__ = ['ImglistDatasetV2']


import os
import io
import torch
import logging
from PIL import Image, ImageFile
from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImglistDatasetV2(BaseDataset):

    def __init__(self, imglist, root, reader,
                 transform_image, transform_extra, maxlen=None,
                 dummy_read=False, dummy_size=None, **kwargs):
        super(ImglistDatasetV2, self).__init__(**kwargs)
        self.imglist = imglist
        self.root = root
        self.reader = reader
        self.transform_image = transform_image
        self.transform_extra = transform_extra
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError('if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', maxsplit=1)
        if len(tokens) == 1:
            image_name = tokens[0]
            extra_str = ''
        else:
            image_name, extra_str = tokens
        if self.root != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)
        sample = dict()
        try:
            if not self.dummy_read:
                filebytes = self.reader(path)
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                self.get_dummy_image(sample)
            else:
                self.get_real_image(sample, buff)
            for key, value in self.transform_extra(extra_str).items():
                sample[key] = value
        except Exception as e:
            if self.skip_broken:
                logging.warn('[{}] broken'.format(path))
            else:
                print('[{}] broken'.format(path))
            raise e

        return sample

    def get_dummy_image(self, sample):
        sample['data'] = torch.rand(self.dummy_size)

    def get_real_image(self, sample, buff):
        image = Image.open(buff)
        sample['data'] = self.transform_image(image)
