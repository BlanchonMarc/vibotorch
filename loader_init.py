from database.dataloaderSegmentation import ImageFolderSegmentation
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NormalizeInput:
    def __call__(self, _input):
        img = np.array(_input, dtype=np.uint8)
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        mean = np.array([122.67892, 104.00699, 116.66877])
        img -= mean
        img = img.astype(float) / 255.0
        # # NHWC -> NCHW
        # img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        return img


class load_label:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input):
        # return torch.from_numpy(np.array(_input)).long().unsqueeze(0)
        return torch.from_numpy(np.array(_input, dtype=np.uint8)).long()


def loader_init(image_path, label_path, image_path2, label_path2,
                batch_size, num_workers):

    transform = Compose([
        ToTensor(),
        # NormalizeInput(),
        Normalize([0.3120, 0.3018, 0.2944], [0.2169, 0.2097, 0.2087]),
    ])
    label_transform = Compose([
        load_label(),
    ])

    var = ImageFolderSegmentation(images_path=image_path,
                                  label_path=label_path,
                                  transform=transform,
                                  label_transform=label_transform)

    trainloader = torch.utils.data.DataLoader(var, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    var2 = ImageFolderSegmentation(images_path=image_path2,
                                   label_path=label_path2,
                                   transform=transform,
                                   label_transform=label_transform)

    valloader = torch.utils.data.DataLoader(var2, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)

    return trainloader, valloader
