# Testing the DataLoader

from database.database import DatabaseLoader,DatabaseTorch

root_dataset = '/Users/marc/Github/NeuralNetwork/Datasets/CamVid/'
inputs = ['train/', 'val/', 'test/']
checkings = ['trainannot/', 'valannot/', 'testannot/']
Db = DatabaseTorch(root=root_dataset, train_folders=inputs, test_folders=checkings)

data_dict = Db(batch_size = 3, shuffle = False, num_workers = 1)

# print(data_dict['train'])


# Testing the Network class

# from nn.nn import SegNet
# import torch
#
# # Execution
# batch_size = 1
# input_size = 8
# num_classes = 8
# learning_rate = 0.0001
# nb = 64
#
# input = torch.autograd.Variable(torch.rand(batch_size, input_size, nb, nb))
# target = torch.autograd.Variable(torch.rand(batch_size, num_classes, nb, nb)).long()
#
#
# model = SegNet(in_channels=input_size, n_classes=num_classes)
#
# opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#
#
# for epoch in range(1000):
#     out = model(input)
#
#     # Loss definition - cross entropy
#     criterion = torch.nn.CrossEntropyLoss()
#     loss = criterion(out, target[:, 0])
#
#     # nll loss
#     # loss = F.nll_loss(out, target[:, 0])
#
#     print ('Loss : ' + str(loss.data))
#
#     model.zero_grad()
#     loss.backward()
#
#     opt.step()
