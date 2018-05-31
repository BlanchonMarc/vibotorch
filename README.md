# ViboTorch
This library is a wrapper of PyTorch Library mainly developed for segmentation task.

## Structure

The wrapper is structured as follow:

```
vibotorch
    |__ database
    |__ procedure/experiments
    |__ segmentation/models
        |__ layer
        |__ nn
    |__ structures
    |__ test
    |__ trainer
    |__ utils
    |
    |
    |loader_init
    |main
```

## Methods

Here will be described the different parts of the library.

##### database

This folder contains the loader giving ability to assign image(input) to image(gt) [segmentation] contrary to native PyTorch image(input) to label(gt) [classification].

##### procedure/experiments

Experiments are the different codes that were tester before the development of the wrapper and contains train test procedures.

##### segmentation/models

This is the container for the models, layer contains different layers and nn the implemented networks (SegNet, UNet, UpNet, Multi-Model Segnet).

##### structures

In structures, the routine is the core of the library, this is the pipeline between parameters passed as a dict and PyTorch Library.


##### trainer

trainer contains the previous version of the wrapper.

##### utils

The utils are all the functionalities implemented that can allow processing or facility. Utils also embed the metrics.

##### loader_init

loader_init allow a usage of the database structure in less parameters.

##### main

The main is the launcher for the procedure.

## How to use

### Install

Multiple packages are necessary to use this wrapper (here all the commands are for conda installation and if not, the commands should be executed in the desired environment):

PyTorch:
```
conda install pytorch torchvision -c pytorch
```

Numpy:
```
conda install -c anaconda numpy
```

tqdm:
```
conda install -c conda-forge tqdm
```

[Augmentor](https://github.com/mdbloice/Augmentor)(utils):
```
pip install Augmentor
```

[scikit-learn](https://github.com/scikit-learn/scikit-learn)(metrics):
```
conda install scikit-learn
```

Pillow:
```
conda install -c anaconda pillow
```

Glob:
```
 conda install -c anaconda glob2
 ```


### Running the Code (main.py)

```
trainimage = '/Users/marc/Documents/OutdoorPola/train/*.png'

trainlabel = '/Users/marc/Documents/OutdoorPola/trainannot/*.png'

valimage = '/Users/marc/Documents/OutdoorPola/test/*.png'

vallabel = '/Users/marc/Documents/OutdoorPola/testannot/*.png'

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
```
