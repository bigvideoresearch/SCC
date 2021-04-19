from runner_master import runner
import os
import io
import torch
import logging
from PIL import Image, ImageFile
from runner_master.runner.data import datasets
# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagenameDataset(datasets.ImglistDatasetV2):

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', maxsplit=1)
        #if len(tokens) != 2:
        #    raise RuntimeError('split tokens < 2')

        image_name, extra_str = tokens[0], tokens[1]
        if self.root != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)
        sample = dict()
        sample['image_name'] = image_name
        try:
            if not self.dummy_read:
                filebytes = self.reader(path)
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff)
                sample['data'] = self.transform_image(image)
            for key, value in self.transform_extra(extra_str).items():
                sample[key] = value
        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample

runner.patch_dataset('ImagenameDataset', ImagenameDataset)
