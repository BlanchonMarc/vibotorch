import torch
from torch import nn
import sys
import torch.optim as optim

sys.path.append('../')
from database import dataloaderSegmentation
from utils import metrics
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import glob


class Routine(object):
    '''Object allowing training of neural networks models'''
    def __init__(self, d=None):
        self.dict = d

        if d is None:
            raise AttributeError('''Input dictionnary have to be instanciated
                                 with at least a model and loaders''')
        else:
            self._lr = 0.001

            self._n_ep = 10

            self._cuda = False

            self._loss = nn.CrossEntropyLoss(reduce=True, size_average=True)

            self._logname = None

            self._n_classes = 1

            self._stop_crit = False

            self._dict_estimation()

            self._opt = optim.Adam(self._model.parameters(), lr=self._lr)

            if self._cuda:
                self._set_Cuda()

            if self._logname is not None:
                self.metrics = metrics.evaluation(n_classes=self._n_classes,
                                                  lr=self._lr,
                                                  modelstr="Model",
                                                  textfile=self._logname)
            else:
                warnings.warn("Without log there will be no metrics estimation",
                              RuntimeWarning,
                              stacklevel=2)

    def fit(self):
        '''Train the model'''

        breaker = False
        Loss_store = []
        metricArray = []
        counterPercent = 0
        firstLoss = 0.0
        firstPass = True
        check_metrics = False
        terminate_metrics = True

        for epoch in range(self._n_ep):
            self._model.train()
            aver_Loss = 0
            n_it = 0
            save_epoch = 0
            for i, data in tqdm(enumerate(self._trainloader, 0)):
                inputs, labels = data
                if self._cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                self._opt.zero_grad()

                output = self._model(inputs)

                loss = self._loss(output, labels)
                loss.backward()
                self._opt.step()

                save_epoch = epoch + 1
                aver_Loss += loss.data
                n_it = i
            aver_Loss = aver_Loss / n_it
            print("Averaged Loss Ep[[%d/%d]] : %f" % (save_epoch,
                                                      self._n_ep,
                                                      aver_Loss))

            if firstPass:
                firstLoss = aver_Loss
                firstPass = False

            Loss_store.append(aver_Loss)
            if self._stop_crit:
                if self._brute_crit is not None:
                    if (aver_Loss <= self._brute_crit).cpu().numpy():
                        breaker = True
                elif self._perc_crit is not None:
                    criterion_break = (self._perc_crit * firstLoss)
                    if (criterion_break < aver_Loss).cpu().numpy():
                        if not counterPercent == 0:
                            diff = some_list[-1] - some_list[-2]
                            if (diff <= (criterion_break * self._perc_crit)).cpu().numpy():
                                counterPercent += 1
                            else:
                                counterPercent = 0
                        counterPercent += 1
                    else:
                        counterPercent = 0

                    if counterPercent == 2:
                        breaker = True

                elif self._conv_crit:
                    check_metrics = True

            self._model.eval()

            for i_val, (images_val,
                        labels_val) in tqdm(enumerate(self._valloader)):
                if self._cuda:
                    images_val = Variable(images_val.cuda(), volatile=True)
                    labels_val = Variable(labels_val.cuda(), volatile=True)
                    outputs = self._model(images_val)
                    pred = outputs.data.max(1)[1].cpu().numpy()
                    groundtruth = labels_val.data.cpu().numpy()
                else:
                    images_val = Variable(images_val, volatile=True)
                    labels_val = Variable(labels_val, volatile=True)
                    outputs = self._model(images_val)
                    pred = outputs.data.max(1)[1].numpy()
                    groundtruth = labels_val.data.numpy()

                if self._logname is not None:
                    self.metrics(groundtruth.ravel(), pred.ravel())
                    metricArray.append(self.metrics.IoU)
                    if check_metrics:
                        if not check_metrics[-1] >= check_metrics[-2]:
                            breaker = True
                            terminate_metrics = False
                            break

            if self._logname is not None and terminate_metrics:
                self.metrics.estimate(epoch, self._n_ep, self._model, self._opt)
                self.metrics.print_major_metric()
                self.metrics.reset()
            if breaker:
                print('Stopping Criterion Reached')
                if self._logname is not None:
                    self.metrics.close()
                break
            else:
                if self._logname is not None:
                    self.metrics.close()

        if not breaker:
            print('Stopping Criterion have not been Reached')

    def test(self, loadertest):
        '''Test the model with one or multiple inputs'''
        self._model.eval()

        out = np.ndarray(shape=(len(loadertest), 2))

        for i, (images, labels) in tqdm(enumerate(loadertest)):
                if self._cuda:
                    self._model.cuda()
                    images = Variable(images.cuda(), volatile=True)
                else:
                    images = Variable(images, volatile=True)

                outputs = self._model(images)
                pred = outputs.data.max(1)[1].cpu().numpy()
                if i == 0:
                    out = np.ndarray(shape=(len(loadertest),
                                            pred.shape[0],
                                            pred.shape[1]))
                out[i] = pred

        return out

    def predict(self, input_, modelIn=None):
        '''Predict the model with one or multiple inputs'''
        if model is not None:
            self._model.load_state_dict(modelIn)

        self._model.eval()
        output = self._model(input_)
        return output

    def _dict_estimation(self):
        '''Initialization according to input dictionaty'''

        '''Mandatory Arguments'''
        if 'model' in self.dict:
            self._model = self.dict['model']
        else:
            sys.stderr.write('Input dictionnary have to specify a model \n')
            raise SystemExit(1)

        if 'inputTransform' in self.dict and 'targetTransform' in self.dict:
            transforms = True
        else:
            transforms = False

        if 'traininput' in self.dict and 'traintarget' in self.dict:
            trainpaths = True
        else:
            trainpaths = False

        if 'valinput' in self.dict and 'valtarget' in self.dict:
            valpaths = True
        else:
            valpaths = False

        if 'trainloader' in self.dict:
            self._trainloader = self.dict['trainloader']
            print('maPREMIEREbite')
        else:
            if trainpaths and transforms:
                self._trainloader = self._load_(self.dict['traininput'],
                                                self.dict['traintarget'],
                                                self.dict['inputTransform'],
                                                self.dict['targetTransform']
                                                )
            else:
                sys.stderr.write('''Input dictionnary have to specify
                                a trainloader or two path to train dataset''')
                raise SystemExit(1)

        if 'valloader' in self.dict:
            self._valloader = self.dict['valloader']
            print('mabite')
        else:
            if valpaths and transforms:
                self._valloader = self._load_(self.dict['valinput'],
                                              self.dict['valtarget'],
                                              self.dict['inputTransform'],
                                              self.dict['targetTransform']
                                              )
            else:
                sys.stderr.write('''Input dictionnary have to specify
                                a trainloader or two path to train dataset''')
                raise SystemExit(1)

        '''Optional Arguments'''

        if 'lr' in self.dict:
            self._lr = self.dict['lr']

        if 'max_epochs' in self.dict:
            self._n_ep = self.dict['max_epochs']

        if 'n_classes' in self.dict:
            self._n_classes = self.dict['n_classes']

        if 'loss' in self.dict:
            self._loss = self.dict['loss']

        if 'cuda' in self.dict:
            self._cuda = self.dict['cuda']

        if 'logfile' in self.dict:
            self._logname = self.dict['logfile']
            if 'n_classes' in self.dict:
                self._n_classes = self.dict['n_classes']

        if 'stop_criterion' in self.dict:
            self._stop_crit = self.dict['stop_criterion']

            if 'brute_force' in self.dict:
                self._brute_crit = self.dict['brute_force']
                self._perc_crit = None
                self._conv_crit = False
            elif 'percent_loss' in self.dict:
                self._brute_crit = None
                self._perc_crit = self.dict['percent_loss']
                self._conv_crit = False
            elif 'till_convergence' in self.dict:
                self._brute_crit = None
                self._perc_crit = None
                self._conv_crit = self.dict['till_convergence']
            else:
                self._brute_crit = None
                self._perc_crit = None
                self._conv_crit = False

    def _set_Cuda(self):
        '''If Cuda True Set every prior object as cuda objects'''
        self._loss = self._loss.cuda()
        self._model = self._model.cuda()

    def _load_(self, inputpath, targetpath, transformin, transformtar):
        '''Load the data from two folder path of the dataset'''
        var = ImageFolderSegmentation(images_path=inputpath,
                                      label_path=targetpath,
                                      transform=transformin,
                                      label_transform=transformtar)

        if 'shuffle' in self.dict:
            shuffle = self.dict['shuffle']
        else:
            shuffle = False

        if 'batch_size' in self.dict:
            self.batch_size = self.dict['batch_size']
        else:
            self.batch_size = 10

        if 'workers' in self.dict:
            self.workers = self.dict['workers']
        else:
            self.workers = 10

        if inputpath == 'valinput':
            shuffle = False

        loader = torch.utils.data.DataLoader(var, batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             num_workers=self.workers,
                                             pin_memory=True)
