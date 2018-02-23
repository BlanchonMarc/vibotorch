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
    Normalize([.485, .456, .406], [.229, .224, .225]),
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


n_classes = 12

model = SegNet(in_channels=3, n_classes=n_classes)
model = torch.nn.DataParallel(model,
                              device_ids=range(torch.cuda.device_count()))
model.cuda()
epochs = [2]
lrs = [0.001]

metrics = evaluation(n_classes=n_classes, lr=lrs[0], modelstr="SegNet",
                     textfile="newlog.txt")


weights = WeightComputationMedian(labels_path=label_path, n_classes=n_classes)

weights = torch.from_numpy(weights).float()

criterion = nn.CrossEntropyLoss(weight=weights, reduce=True,
                                size_average=True).cuda()
for ep in epochs:

    for lr in lrs:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        for epoch in range(ep):
            model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                if i % 1 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1,
                                                        ep,
                                                        loss.data))
            print('Epoch ' + str(epoch + 1) + " Done")
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

            metrics.reset()
    metrics.close()
