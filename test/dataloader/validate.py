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


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """


    dict_ = state_dict
    temp = ''
    for k, v in state_dict.iteritems():
        name = k[7:]  # remove `module.`
        dict_[name] = v
        temp = k

    _ = dict_.pop(temp, None)
    return dict_


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


image_path = '/data/scene-segmentation/CamVid/test/*.png'

label_path = '/data/scene-segmentation/CamVid/testannot/*.png'

var = ImageFolderSegmentation(images_path=image_path,
                              label_path=label_path,
                              transform=transform,
                              label_transform=label_transform)

valloader = torch.utils.data.DataLoader(var, batch_size=16,
                                        shuffle=False, num_workers=10)


running_metrics = runningScore(n_classes=22)
model = SegNet()
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

        running_metrics.update(gt, pred)

score, class_iou = running_metrics.get_scores()

for k, v in score.items():
    print(k, v)

for i in range(n_classes):
    print(i, class_iou[i])
