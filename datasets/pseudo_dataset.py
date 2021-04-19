import torch
import numpy as np
from runner_master import runner
from runner_master.runner.data import datasets
import os, io, logging
from PIL import Image
from runner_master.runner.data.datasets.base_dataset import BaseDataset

class PseudoDataset(BaseDataset):
    # This patch loads pseudo labels from directory as sample['pseudo_label'].
    # Pseudo label file should have an itentical filename and related path to image inputs.
    def __init__(self, pseudo_root, imglist, root, reader,
                 transform_image, transform_extra, maxlen=None,
                 dummy_read=False, dummy_size=None, **kwargs):
        super(PseudoDataset, self).__init__(**kwargs)
        self.pseudo_root = pseudo_root
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

    def get_dummy_image(self, sample):
        sample['data'] = torch.rand(self.dummy_size)

    def get_real_image(self, sample, buff):
        image = Image.open(buff)
        sample['data'] = self.transform_image(image)


    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', maxsplit=1)
        if len(tokens) != 2:
            raise RuntimeError('split tokens < 2')
        image_name, extra_str = tokens
        if self.root != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)
        sample = dict()
        sample['image_name'] = image_name
        pseudo_name = image_name.split('.jpg')[0] + '.npy'
        pseudo_path = os.path.join(self.pseudo_root, pseudo_name)
        try:
            if not self.dummy_read:
                filebytes = self.reader(path)
                buff = io.BytesIO(filebytes)
                filebytes_pseudo = self.reader(pseudo_path)
                buff_pseudo = io.BytesIO(filebytes_pseudo)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff)
                sample['data'] = self.transform_image(image)
                sample['pseudo_label'] = torch.FloatTensor(np.load(buff_pseudo))
            for key, value in self.transform_extra(extra_str).items():
                sample[key] = value
        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample

runner.patch_dataset('PseudoDataset', PseudoDataset)
