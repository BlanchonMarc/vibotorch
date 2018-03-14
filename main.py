from structures import routine as Struct
from segmentation.models import nn as NeuralNet
from loader_init import loader_init as Loader
from torch import nn


if __name__ == '__main__':

        '''Server'''
        # trainimage = '/data/scene-segmentation/CamVid/train/*.png'

        # trainlabel = '/data/scene-segmentation/CamVid/trainannot/*.png'

        # valimage = '/data/scene-segmentation/CamVid/test/*.png'

        # vallabel = '/data/scene-segmentation/CamVid/testannot/*.png'

        '''Local'''

        trainimage = '../NeuralNetwork/Datasets/CamVid/train/*.png'

        trainlabel = '../NeuralNetwork/Datasets/CamVid/trainannot/*.png'

        valimage = '../NeuralNetwork/Datasets/CamVid/test/*.png'

        vallabel = '../NeuralNetwork/Datasets/CamVid/testannot/*.png'

        trainloader, valloader = Loader(trainimage,
                                        trainlabel,
                                        valimage,
                                        vallabel,
                                        batch_size=10,
                                        num_workers=10)

        n_classes = 12
        model = NeuralNet.SegNet(in_channels=3, n_classes=n_classes)

        dic = {
            'model': model,
            'trainloader': trainloader,
            'valloader': valloader,
            'n_classes': n_classes,
            'max_epochs': 400,
            'lr': 0.0001,
            'loss': nn.CrossEntropyLoss(reduce=True, size_average=True),
            'cuda': True,
            'logfile': 'log.txt',
            'brute_force': 3.00,
        }

        trainer = Struct.Routine(dic)

        trainer.fit()
