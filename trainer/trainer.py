import torch
import sys
sys.path.append('../utils/')
sys.path.append('../segmentation/models/')
from tracker import Tracker
from nn import *


class Trainer:
    def __init__(self,
                 in_ = None,
                 lab_ = None,
                 n_classes : int = 1,
                 model : None = None,
                 model_name : str = 'SegNet',
                 optimizer : None = None,
                 optimizer_name : str = 'SGD',
                 learning_rate : float = .001,
                 momentum : float = .9,
                 weight_decay : float = 0.0,
                 n_epoch : int = 100,
                 weight = None,
                 loss = None,
                 loss_name : str = 'NLLLoss2d',
                 track = None,
                 cuda = False) -> None:
        self._data = in_
        self._label = lab_
        self._n_cl = n_classes
        self._mod = model
        self._mod_nm = model_name
        self._opt = optimizer
        self._opt_name = optimizer_name
        self._lr = learning_rate
        self._mom = momentum
        self._w_dec = weight_decay
        self._n_ep = n_epoch
        self._wght = weight
        self._loss = loss
        self._loss_nm = loss_name
        self._log = track
        self._cuda = cuda

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, in_ : torch.Tensor) -> None:
        self._data = in_

    @property
    def label(self) -> torch.Tensor:
        return self._label

    @label.setter
    def label(self, label_ : torch.Tensor) -> None:
        self._label = label_

    @property
    def nclasses(self) -> int:
        return self._n_cl

    @nclasses.setter
    def nclasses(self, n_classes : int) -> None:
        self._n_cl = n_classes

    @property
    def model(self):
        return self._mod

    @model.setter
    def model(self, model ) -> None:
        self._mod = model

    @property
    def model_name(self) -> str:
        return self._mod_nm

    @model_name.setter
    def model_name(self, model_name : str ) -> None:
        self._mod_nm = model_name

    @property
    def optimizer(self):
        return self._opt

    @optimizer.setter
    def optimizer(self, optimizer) -> None:
        self._opt = optimizer

    @property
    def optimizer_name(self) -> str:
        return self._opt

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name : str) -> None:
        self._opt_name = optimizer_name

    @property
    def learning_rate(self) -> float:
        return self._lr

    @learning_rate.setter
    def learning_rate(self, lr : float) -> None:
        self._lr = lr

    @property
    def momentum(self) -> float:
        return self._mom

    @momentum.setter
    def momentum(self, momentum : float) -> None:
        self._mom = momentum

    @property
    def weight_decay(self) -> float:
        return self._w_dec

    @momentum.setter
    def weight_decay(self, weight_decay : float) -> None:
        self._w_dec = momentum

    @property
    def n_epoch(self) -> int:
        return self._n_ep

    @n_epoch.setter
    def n_epoch(self, n_epoch : int) -> None:
        self._n_ep = n_epoch

    @property
    def weight(self) -> int:
        return self._wght

    @weight.setter
    def weight(self, weight) -> None:
        self._wght = weight

    @property
    def loss(self) -> int:
        return self._loss

    @loss.setter
    def loss(self, loss) -> None:
        self._loss = loss

    @property
    def loss_name(self) -> int:
        return self._loss_nm

    @loss_name.setter
    def loss_name(self, name : str) -> None:
        self._loss_nm = name

    @property
    def log(self) -> int:
        return self._log

    @log.setter
    def log(self, fname : str = 'track.log') -> None:
        self._log = Tracker(fname)

    @property
    def cuda(self) -> int:
        return self._cuda

    @cuda.setter
    def cuda(self, act : bool ) -> None:
        self._cuda = act

    def __call__(self) -> None:
        if not self._log == None:
            is_tracked = True

        if self._mod == None:
            if self._opt_name == "SegNet":
                self._mod = SegNet(self._data, self._n_cl)

        if self._opt == None:
            if self._opt_name == "SGD":
                self._opt = torch.optim.SGD(self._mod.parameters(),
                                            lr = self._lr,
                                            momentum = self._mom,
                                            weight_decay = self._w_dec)
            elif self._opt_name == "Adam":
                self._opt = torch.optim.Adam(self._mod.parameters(),
                                            lr = self._lr,
                                            weight_decay = self._w_dec)

        if self._wght == None:
            self._wght = torch.ones(self._n_cl)
            self._wght[0] = 0

        if self._loss == None:
            if self._loss_nm == "CrossEntropyLoss":
                self._loss = torch.nn.CrossEntropyLoss(self._wght)
            elif self._loss_nm == "NLLLoss":
                self._loss = torch.nn.NLLLoss(self._wght)
            elif self._loss_nm == "NLLLoss2d":
                self._loss = torch.nn.NLLLoss2d(self._wght)

        if self._cuda:
            self._mod.cuda()

        for epoch in range(self._n_ep):
            running_loss = 0.0
            for indx in range(len(self._data)):
                if self._cuda:
                    images = self._data[indx].cuda()
                    labels = self._label[indx].cuda()

                images = autograd.Variable(images)
                labels = autograd.Variable(labels)

                self._opt.zero_grad()

                outputs = self._mod(images)

                loss = self._loss(F.log_softmax(outputs, dim=1), labels[:, 0])
                loss.backward()
                self._opt.step()

                # print statistics
                running_loss = loss.data[0]
                if step % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, indx + 1, running_loss / 20))
                    running_loss = 0.0

        print('Finished Training')
