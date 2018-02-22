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


class load_label:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input):
        # return torch.from_numpy(np.array(_input)).long().unsqueeze(0)
        return torch.from_numpy(np.array(_input, dtype=np.int8)).long()


class to_long:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input):
        # return torch.from_numpy(np.array(_input)).long().unsqueeze(0)
        return _input.type(torch.LongTensor)


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
    load_label(),
    # ToTensor(),
    # to_long(),
    # to_label(),
    # relabel(255, 31),
])


image_path = '/data/scene-segmentation/CamVid/train/*.png'

label_path = '/data/scene-segmentation/CamVid/trainannot/*.png'

var = ImageFolderSegmentation(images_path=image_path,
                              label_path=label_path,
                              transform=transform,
                              label_transform=label_transform)

trainloader = torch.utils.data.DataLoader(var, batch_size=16,
                                          shuffle=False, num_workers=10)


image_path2 = '/data/scene-segmentation/CamVid/test/*.png'

label_path2 = '/data/scene-segmentation/CamVid/testannot/*.png'

var2 = ImageFolderSegmentation(images_path=image_path2,
                               label_path=label_path2,
                               transform=transform,
                               label_transform=label_transform)

valloader = torch.utils.data.DataLoader(var2, batch_size=16,
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
n_classes = 12

running_metrics = runningScore(n_classes=n_classes)
model = SegNet(in_channels=3, n_classes=n_classes)
model = torch.nn.DataParallel(model,
                              device_ids=range(torch.cuda.device_count()))
model.cuda()
epochs = [10]
lrs = [0.001]
best_iou = -100.0
criterion = nn.CrossEntropyLoss()
for ep in epochs:

    for lr in lrs:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        for epoch in range(ep):  # loop over the dataset multiple times
            model.train()
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                # output = output.view(output.size(0), output.size(1), -1)
                # output = torch.transpose(output, 1, 2).contiguous()
                # output = output.view(-1, output.size(2))
                # labels = labels.view(-1)
                loss = criterion(output, labels[:, 0])
                loss.backward()
                optimizer.step()

                # print statistics
                if i % 1 == 0:    # print every 20 mini-batches
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1,
                                                        ep,
                                                        loss.data[0]))
            print('Finished Training')
            model.eval()

            for i_val, (images_val,
                        labels_val) in tqdm(enumerate(valloader)):
                images_val = Variable(images_val.cuda(), volatile=True)
                labels_val = Variable(labels_val.cuda(), volatile=True)

                outputs = model(images_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                groundtruth = labels_val.data.cpu().numpy()
                running_metrics.update(groundtruth, pred)
            score, class_iou = running_metrics.get_scores()
            for k, v in score.items():
                print(k, v)

            running_metrics.reset()

            if score['Mean IoU : \t'] >= best_iou:
                best_iou = score['Mean IoU : \t']
                state = {'epoch': epoch + 1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), }
                torch.save(state,
                           "{}_{}_best_model.pkl".format('segnet', 'Camvid'))
