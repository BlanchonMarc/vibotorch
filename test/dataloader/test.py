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


class to_label:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input):
        return torch.from_numpy(np.array(_input)).long().unsqueeze(0)


class relabel:
    """Class to relabel along each channels a torch.Tensor"""
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, _input):
        assert isinstance(_input,
                          torch.LongTensor), 'tensor needs to be LongTensor'

        _input[_input == self.olabel] = self.nlabel
        return _input

transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.5, .5, .5], [.5, .5, .5]),
])
label_transform = Compose([
    CenterCrop(256),
    to_label(),
    relabel(255, 21),
])


image_path = '/data/scene_segmentation/CamVid/train/*.png'

label_path = '/data/scene_segmentation/CamVid/trainannot/*.png'

var = ImageFolderSegmentation(images_path=image_path,
                              label_path=label_path,
                              transform=transform,
                              label_transform=label_transform)

trainloader = torch.utils.data.DataLoader(var, batch_size=3,
                                          shuffle=False, num_workers=10)

model = SegNet(in_channels=3, n_classes=22)
model = torch.nn.DataParallel(model,
                              device_ids=range(torch.cuda.device_count()))
model.cuda()

weight = torch.ones(22)
weight[0] = 0

criterion = nn.NLLLoss2d(weight)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

cuda_activated = True

for epoch in range(500):
    running_loss = 0.0
    for step, (images, labels) in enumerate(trainloader):
        if cuda_activated:
            images = images.cuda()
            labels = labels.cuda()

        images = autograd.Variable(images)
        labels = autograd.Variable(labels)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(F.log_softmax(outputs, dim=1), labels[:, 0])
        loss.backward()
        optimizer.step()
        print('step: ' + step)

        # print statistics
        running_loss = loss.data[0]
        print(running_loss)
        if step % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
