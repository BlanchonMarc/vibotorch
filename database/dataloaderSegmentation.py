'''
Creating a Dataloader for segmentation taks
'''

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable


class ImageFolderSegmentation(Dataset):
    """
        A generic data loader for image segmentation where the data
        are organised as:

        ....

        Parameters
        ----------
        root : str
            Root folder containing the segmentation data.
        images_path : str
            path of the images with selector
            image_path = '/image/*.png'
        label_path : str
            path of the labals with selector
        conversion : str
            conversion for input images
        transform : Composed Transformation
            transformation applied on input images
        label_transform : Composed Transformation
            transformation applied on label images

        Attributes
        ----------
        image_filenames : list of str
            images names with full path
        label_filenames : list of str
            label names with full path
        conv: list of str
            conversion List

        Examples
        --------

        >>> from dataloaderSegmentation import ImageFolderSegmentation
        >>> image_path = '/image/*.png'
        >>> label_path = '/label/*.png'
        >>> data = ImageFolderSegmentation(image_path=image_path,
        ...                                label_path=label_path)


    """

    def __init__(self, images_path, label_path, conversion='RGB',
                 transform=None,
                 label_transform=None):

        self.image_filenames = sorted(glob.glob(images_path))
        self.label_filenames = sorted(glob.glob(label_path))

        self.conversion = conversion

        if not all([self._get_filename(imf) == self._get_filename(lf)
                    for imf, lf in zip(self.image_filenames,
                                       self.label_filenames)]):
                raise ValueError(
                    'Image names in Images and label have to be identical')
        self.transform = transform
        self.label_transform = label_transform

    def _get_filename(self, path):
        return os.path.basename(os.path.splitext(path)[0])

    def _pil_loader(self, path, conversion=None):
        with open(path, 'rb') as f:
            if conversion is not None:
                return Image.open(f).convert(conversion)
            else:
                return Image.open(f).convert('P')

    def __getitem__(self, index):
        '''Get an image and a label'''

        image = self._pil_loader(path=self.image_filenames[index],
                                 conversion='RGB')
        label = self._pil_loader(path=self.label_filenames[index])

        if self.transform is not None:
            image = self.transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace(
                                         '\n', '\n' + ' ' * len(tmp)))
        tmp = '    Label Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.label_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderSegmentationX(Dataset):
    """
        A generic data loader for image segmentation where the data
        are organised as:

        ....

        Parameters
        ----------
        root : str
            Root folder containing the segmentation data.
        images_path : str
            path of the images with selector
            image_path = '/image/*.png'
        label_path : str
            path of the labals with selector
        conversion : str
            conversion for input images
        transform : Composed Transformation
            transformation applied on input images
        label_transform : Composed Transformation
            transformation applied on label images

        Attributes
        ----------
        image_filenames : list of str
            images names with full path
        label_filenames : list of str
            label names with full path
        conv: list of str
            conversion List

        Examples
        --------

        >>> from dataloaderSegmentation import ImageFolderSegmentation
        >>> image_path = '/image/*.png'
        >>> label_path = '/label/*.png'
        >>> data = ImageFolderSegmentation(image_path=image_path,
        ...                                label_path=label_path)


    """

    def __init__(self, images_path, label_path, conversion='RGB',
                 transform=None,
                 label_transform=None,
                 use_cuda=False):

        self.image_filenames = sorted(glob.glob(images_path))
        self.label_filenames = sorted(glob.glob(label_path))

        self.conversion = conversion

        if not all([self._get_filename(imf) == self._get_filename(lf)
                    for imf, lf in zip(self.image_filenames,
                                       self.label_filenames)]):
                raise ValueError(
                    'Image names in Images and label have to be identical')
        self.transform = transform
        self.label_transform = label_transform
        self.cuda = use_cuda

    def _get_filename(self, path):
        return os.path.basename(os.path.splitext(path)[0])

    def _pil_loader(self, path, conversion=None):
        with open(path, 'rb') as f:
            if conversion is not None:
                return Image.open(f).convert(conversion)
            else:
                return Image.open(f).convert('P')

    def __getitem__(self, index):
        '''Get an image and a label'''

        image = self._pil_loader(path=self.image_filenames[index],
                                 conversion='RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.cuda:
            return Variable(image.cuda())
        else:
            return image

    def __len__(self):
        return len(self.image_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace(
                                         '\n', '\n' + ' ' * len(tmp)))
        tmp = '    Label Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.label_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderSegmentationY(Dataset):
    """
        A generic data loader for image segmentation where the data
        are organised as:

        ....

        Parameters
        ----------
        root : str
            Root folder containing the segmentation data.
        images_path : str
            path of the images with selector
            image_path = '/image/*.png'
        label_path : str
            path of the labals with selector
        conversion : str
            conversion for input images
        transform : Composed Transformation
            transformation applied on input images
        label_transform : Composed Transformation
            transformation applied on label images

        Attributes
        ----------
        image_filenames : list of str
            images names with full path
        label_filenames : list of str
            label names with full path
        conv: list of str
            conversion List

        Examples
        --------

        >>> from dataloaderSegmentation import ImageFolderSegmentation
        >>> image_path = '/image/*.png'
        >>> label_path = '/label/*.png'
        >>> data = ImageFolderSegmentation(image_path=image_path,
        ...                                label_path=label_path)


    """

    def __init__(self, images_path, label_path, conversion='RGB',
                 transform=None,
                 label_transform=None,
                 use_cuda=False):

        self.image_filenames = sorted(glob.glob(images_path))
        self.label_filenames = sorted(glob.glob(label_path))

        self.conversion = conversion

        if not all([self._get_filename(imf) == self._get_filename(lf)
                    for imf, lf in zip(self.image_filenames,
                                       self.label_filenames)]):
                raise ValueError(
                    'Image names in Images and label have to be identical')
        self.transform = transform
        self.label_transform = label_transform
        self.cuda = use_cuda

    def _get_filename(self, path):
        return os.path.basename(os.path.splitext(path)[0])

    def _pil_loader(self, path, conversion=None):
        with open(path, 'rb') as f:
            if conversion is not None:
                return Image.open(f).convert(conversion)
            else:
                return Image.open(f).convert('P')

    def __getitem__(self, index):
        '''Get an image and a label'''

        label = self._pil_loader(path=self.label_filenames[index])

        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.cuda:
            return Variable(label.cuda())
        else:
            return label

    def __len__(self):
        return len(self.image_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace(
                                         '\n', '\n' + ' ' * len(tmp)))
        tmp = '    Label Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.label_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str
