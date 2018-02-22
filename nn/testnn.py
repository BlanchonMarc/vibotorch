from nn import U_Net
import torch
import torch.nn as nn
from torch import autograd, optim

# Execution
batch_size = 1
input_size = 3
num_classes = 8
learning_rate = 0.0001
nb = 500

input_ = autograd.Variable(torch.rand(1, 3, 256, 256))
target = autograd.Variable(torch.rand(1, 3, 256, 256)).long()


model = U_Net(in_channels=input_size, n_classes=num_classes)

opt = optim.Adam(params=model.parameters(), lr=learning_rate)


for epoch in range(1):
    print('epoch: ' + str(epoch))
    out = model(input_)

    print(out)

    # Loss definition - cross entropy
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, target[:, 0])

    # nll loss
    # loss = F.nll_loss(out, target[:, 0])

    print ('Loss : ' + str(loss.data))

    model.zero_grad()
    loss.backward()

    opt.step()
