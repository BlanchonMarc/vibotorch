
'''
Creating a Dataloader for segmentation taks
'''

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class DatasetLoaderSegmentation(Dataset):
    ''' Derived class to manage Segmentation Dataset Image Loading
    '''

    def __init__(self, root,
                 images='images', imagext='png', imageconv='RGB',
                 targets='targets', targetext='png', targetconv='P'):
        # parameters copies
        self.images_root = os.path.join(root, images)
        self.labels_root = os.path.join(root, targets)
        self.ext = [imagext, targetext]

        # Extract the filenames without extansions
        self.filenames = [os.path.basename(os.path.splitext(fname)[0])
                          for fname in os.listdir(self.labels_root)
                          if any(filename.endswith(ext) for ext in self.ext)]

        # Alphabetical Order
        self.filenames.sort()

    def __getitem__(self, index):
        '''Convert the images and return to be passed in dataloader

        Dataloader will recieve the images and transform
        It will also allow the paralellization / optimization
        '''

        # Images
        with open(os.path.join(self.images_root, self.filenames[index] \
                               + self.ext[0]), 'rb') as f:
            image = Image.open(f).convert(imageconv)

        # Targets
        with open(os.path.join(self.labels_root, self.filenames[index] \
                               + self.ext[1]), 'rb') as f:
            target = Image.open(f).convert(targetconv)

        return image, target

    def __len__(self):
        return len(self.filenames)
