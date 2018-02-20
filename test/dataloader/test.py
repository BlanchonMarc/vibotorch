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

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                     lp.flatten(),
                                                     self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist)
        iu = iu / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,
                }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


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


image_path2 = '../../../../data/scene_segmentation/CamVid/val/*.png'

label_path2 = '../../../../data/scene_segmentation/CamVid/valannot/*.png'

var2 = ImageFolderSegmentation(images_path=image_path2,
                               label_path=label_path2,
                               transform=transform,
                               label_transform=label_transform)

valloader = torch.utils.data.DataLoader(var2, batch_size=3,
                                        shuffle=False, num_workers=10)


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# fig = plt.figure()
# # show images
# imshow2(torchvision.utils.make_grid(images),
#         torchvision.utils.make_grid(labels[:,0].type(torch.DoubleTensor)))
#
# plt.show()
# plt.close(fig)

running_metrics = runningScore(n_classes=21)
model = SegNet(in_channels=3, n_classes=22)
model = torch.nn.DataParallel(model,
                              device_ids=range(torch.cuda.device_count()))
model.cuda()

weight = torch.ones(n_classes)
weight[0] = 0

criterion = nn.NLLLoss2d(weight)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

cuda_activated = True

for epoch in range(500):
    running_loss = 0.0
    for step, (images, targets) in enumerate(trainloader):
        if cuda_activated:
            images = images.cuda()
            labels = labels.cuda()

        images = autograd.Variable(images)
        targets = autograd.Variable(targets)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(F.log_softmax(outputs, dim=1), targets[:, 0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.data[0]
        print(running_loss)
        if step % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
