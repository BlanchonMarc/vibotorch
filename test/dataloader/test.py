import sys
sys.path.append('../../database/')
sys.path.append('../../nn/')
import torch
import torchvision
import numpy as np
from dataloaderSegmentation import ImageFolderSegmentation
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
from nn import SegNet
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm


class load_label:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input):
        # return torch.from_numpy(np.array(_input)).long().unsqueeze(0)
        return torch.from_numpy(np.array(_input, dtype=np.uint8)).long()


transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.5, .5, .5], [.5, .5, .5]),
])
label_transform = Compose([
    CenterCrop(256),
    load_label(),
    # ToTensor(),
    # to_long(),
    # to_label(),
    # relabel(255, 31),
])

image_path = '/data/scene-segmentation/CamVid/test/*.png'

label_path = '/data/scene-segmentation/CamVid/testannot/*.png'

var = ImageFolderSegmentation(images_path=image_path,
                              label_path=label_path,
                              transform=transform,
                              label_transform=label_transform)

valloader = torch.utils.data.DataLoader(var, batch_size=1,
                                        shuffle=False, num_workers=10)

n_classes = 12
running_metrics = runningScore(n_classes=n_classes)
model = SegNet(n_classes=n_classes)
state = convert_state_dict(torch.load('segnet_Camvid_best_model.pkl')
                           ['model_state'])
model.load_state_dict(state)
model.eval()

for i, (images, labels) in tqdm(enumerate(valloader)):
        model.cuda()
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        np.savetxt('/pred/pred' + i + '.txt', pred)
        np.savetxt('/gt/gt' + i + '.txt', gt)
