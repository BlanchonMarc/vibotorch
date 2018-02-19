'''
Creating a Dataloader for segmentation taks
'''

import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderSegmentation(Dataset):
    """
        A generic data loader for image segmentation where the data
        are organised as:

        ....

        Parameters
        ----------
        root : str
            Root folder containing the segmentation data.
        images : str, default='images'
            Name of the folder containing input images_root
        imagext : str, default='png'
            Extensions of the images contained in the images folder
        imageconv : str, default='RGB'
            Conversion color space for inputs
        labels : str, default='images'
            Name of the folder containing input labels_root
        labelsext : str, default='png'
            Extensions of the images contained in the labels folder
        labelsconv : str, default='P'
            Conversion color space for labels

        Attributes
        ----------
        images_root : List[str]
            Images names with full path
        labels_root : List[str]
            Labels names with full path
        ext : List[str]
            Extansion List
        conv: List[str]
            Conversion List
    """

    def __init__(self, root,
                 images='images', imagext='png', imageconv='RGB',
                 labels='labels', labelsext='png', labelsconv='P'):

        images_root = os.path.join(root, images + '/*.' + imagext)
        labels_root = os.path.join(root, labels + '/*.' + labelsext)

        self.image_filenames = sorted(glob.glob(images_root))
        self.label_filenames = sorted(glob.glob(labels_root))

        self.ext = [imagext, labelsext]
        self.conv = [imageconv, labelsconv]

        if not all([self._get_filename(imf) == self._get_filename(lf)
                    for imf, lf in zip(self.image_filenames,
                                       self.label_filenames)]):
                raise ValueError(
                    'Image names in Images and labels have to be identical')

    def _get_filename(self, path):
        return os.path.basename(os.path.splitext(path)[0])

    def __getitem__(self, index):
        '''Get an image and a label'''
        with open(self.image_filenames[index], 'rb') as f:
            image = Image.open(f).convert(self.conv[0])

        with open(self.label_filenames[index], 'rb') as f:
            labels = Image.open(f).convert(self.conv[1])

        return image, labels

    def __len__(self):
        return len(self.image_filenames)
