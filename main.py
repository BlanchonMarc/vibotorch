from structures import routine as Struct
from segmentation.models import nn as NeuralNet
from loader_init import loader_init as Loader
import torch
from torch import nn
from utils import compute_weight as cw


if __name__ == '__main__':
        #torch.backends.cudnn.enabled=False
        '''Server'''
        trainimage = '../../rgbData/train/*.png'

        trainlabel = '../../rgbData/trainannot/*.png'

        valimage = '../../rgbData/test/*.png'

        vallabel = '../../rgbData/testannot/*.png'

        '''Local'''

        # trainimage = '/Users/marc/Documents/OutdoorPola/train/*.png'
        #
        # trainlabel = '/Users/marc/Documents/OutdoorPola/trainannot/*.png'
        #
        # valimage = '/Users/marc/Documents/OutdoorPola/test/*.png'
        #
        # vallabel = '/Users/marc/Documents/OutdoorPola/testannot/*.png'

        trainloader, valloader = Loader(trainimage,
                                        trainlabel,
                                        valimage,
                                        vallabel,
                                        batch_size=25,
                                        num_workers=8
)

        n_classes = 11
        #weights = cw.NormalizedWeightComputationMedian(labels_path=trainlabel,
        #                                               n_classes=n_classes)
        #weights = torch.from_numpy(weights).float()
        criterion = nn.CrossEntropyLoss(reduce=True,
                                        size_average=True).cuda()
        model = NeuralNet.SegNet(in_channels=3,
                                 n_classes=n_classes)
        model.init_encoder()
        model.cuda()

        dic = {
            'model': model,
            'trainloader': trainloader,
            'valloader': valloader,
            'n_classes': n_classes,
            'max_epochs': 500,
            'lr': 0.0001,
            'loss': criterion,
            'cuda': True,
            'logfile': 'log.txt',
            # 'stop_criterion': True,
            # 'brute_force': 3.00,
            # 'percent_loss': 0.99,
            # 'till_convergence': True,
        }

        trainer = Struct.Routine(dic)

        trainer.fit()
