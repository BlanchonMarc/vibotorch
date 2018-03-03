import sys
sys.path.append('../../database/')
sys.path.append('../../nn/')
sys.path.append('../../utils/')
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
from metrics import evaluation
from compute_weight import WeightComputationMedian
from compute_weight import NormalizedWeightComputationMedian


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


transform = Compose([
    CenterCrop(256),
    ToTensor(),
    # NormalizeInput(),
    Normalize([0.4119, 0.4251, 0.4327], [0.2741, 0.2851, 0.2828]),
])
label_transform = Compose([
    CenterCrop(256),
    load_label(),
])


image_path = '/data/scene-segmentation/CamVid/train/*.png'

label_path = '/data/scene-segmentation/CamVid/trainannot/*.png'

var = ImageFolderSegmentation(images_path=image_path,
                              label_path=label_path,
                              transform=transform,
                              label_transform=label_transform)

trainloader = torch.utils.data.DataLoader(var, batch_size=10,
                                          shuffle=True, num_workers=10,
                                          pin_memory=True)


image_path2 = '/data/scene-segmentation/CamVid/test/*.png'

label_path2 = '/data/scene-segmentation/CamVid/testannot/*.png'

var2 = ImageFolderSegmentation(images_path=image_path2,
                               label_path=label_path2,
                               transform=transform,
                               label_transform=label_transform)

valloader = torch.utils.data.DataLoader(var2, batch_size=10,
                                        shuffle=False, num_workers=10,
                                        pin_memory=True)


n_classes = 12

model = SegNet(in_channels=3, n_classes=n_classes)
model.init_encoder()
# model = torch.nn.DataParallel(model,
#                              device_ids=range(torch.cuda.device_count()))
model.cuda()
epochs = [300]
lrs = [0.001]

metrics = evaluation(n_classes=n_classes, lr=lrs[0], modelstr="SegNet",
                     textfile="newlog.txt")


weights = NormalizedWeightComputationMedian(labels_path=label_path,
                                            n_classes=n_classes)

weights = torch.from_numpy(weights).float().cuda()

criterion = nn.CrossEntropyLoss(weight=weights, reduce=True,
                                size_average=True).cuda()

# criterion = nn.CrossEntropyLoss().cuda()

for ep in epochs:

    for lr in lrs:
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(ep):
            model.train()
            aver_Loss = 0
            n_it = 0
            save_epoch = 0
            for i, data in tqdm(enumerate(trainloader, 0)):
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # mask = labels in (1,2,3)
                optimizer.zero_grad()

                output = model(inputs)

                # loss = criterion(torch.masked_select(output, mask), labels)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                print("[%d/%d] Loss: %.4f" % (epoch + 1,
                                              ep, loss.data))
                save_epoch = epoch + 1
                aver_Loss += loss.data
                n_it = i
            aver_Loss = aver_Loss / n_it
            print("Averaged Loss Ep[[%d/%d]] : %f" % (save_epoch,
                                                      ep,
                                                      aver_Loss))

            model.eval()

            for i_val, (images_val,
                        labels_val) in tqdm(enumerate(valloader)):
                images_val = Variable(images_val.cuda(), volatile=True)
                labels_val = Variable(labels_val.cuda(), volatile=True)

                outputs = model(images_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                groundtruth = labels_val.data.cpu().numpy()
                metrics(groundtruth.ravel(), pred.ravel())
            metrics.estimate(epoch, ep, model, optimizer)
            metrics.print_major_metric()
            metrics.reset()
metrics.close()
