from structures import routine as Struct
from segmentation.models import nn as NeuralNet
from loader_init import loader_init as Loader
import torch
from torch import nn
from utils import compute_weight as cw


if __name__ == '__main__':

        '''Server'''
        trainimage = '~/OutdoorPola/train/*.png'

        trainlabel = '~/OutdoorPola/trainannot/*.png'

        valimage = '~/OutdoorPola/test/*.png'

        vallabel = '~/OutdoorPola/testannot/*.png'

        '''Local'''

        # trainimage = '../NeuralNetwork/Datasets/CamVid/train/*.png'
        #
        # trainlabel = '../NeuralNetwork/Datasets/CamVid/trainannot/*.png'
        #
        # valimage = '../NeuralNetwork/Datasets/CamVid/test/*.png'
        #
        # vallabel = '../NeuralNetwork/Datasets/CamVid/testannot/*.png'

        trainloader, valloader = Loader(trainimage,
                                        trainlabel,
                                        valimage,
                                        vallabel,
                                        batch_size=10,
                                        num_workers=10)

        n_classes = 8
        weights = cw.NormalizedWeightComputationMedian(labels_path=trainlabel,
                                                       n_classes=n_classes)
        weights = torch.from_numpy(weights).float().cuda()
        criterion = nn.CrossEntropyLoss(weight=weights, reduce=True,
                                        size_average=True)
        model = NeuralNet.SegNet(in_channels=3,
                                 n_classes=n_classes)

        dic = {
            'model': model,
            'trainloader': trainloader,
            'valloader': valloader,
            'n_classes': n_classes,
            'max_epochs': 500,
            'lr': 0.0001,
            'loss': nn.CrossEntropyLoss(reduce=True, size_average=True),
            'cuda': True,
            'logfile': 'log.txt',
            # 'stop_criterion': True,
            # 'brute_force': 3.00,
            # 'percent_loss': 0.99,
            # 'till_convergence': True,
        }

        trainer = Struct.Routine(dic)

        trainer.fit()
