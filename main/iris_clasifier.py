import torch
import pandas as pd
from torch import nn, optim
import numpy as np
import torch.utils.data.dataloader as DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from main import IrisDataSet
from main.Net import ClasifyNet

trainPath = "./datasets/iris.csv"
testPath = "./datasets/iris_test.csv"

epoches = 200


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=1)


def compute_accuracy(x, y):
    z = x == y
    length = z.shape[0]
    length = torch.tensor(length).float()
    print("length = {}".format(length))
    summ = z.sum(dim=0).float()
    print("summ = {}".format(summ))
    return summ / length


if __name__ == '__main__':
    # net = ClasifyNet()
    # net.apply(weights_init)
    net = torch.load("./test.pth")
    criterion = nn.CrossEntropyLoss(size_average = True)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # 数据准备
    train_dataset = IrisDataSet(trainPath)
    train_data_loader = DataLoader.DataLoader(train_dataset, batch_size=150, shuffle=True, num_workers=4)

    index = 1
    num_x = []
    num_y = []
    for epoch in range(epoches):
        for i, item in enumerate(train_data_loader):
            optimizer.zero_grad()
            data, label = item
            data = Variable(data.float())
            label = Variable(label.long())
            y = net(data)
            y_temp = y.argmax(dim=1)
            accuracy = compute_accuracy(label, y_temp)

            print("y = {}".format(accuracy))
            loss = criterion(y, label)
            loss.backward()
            optimizer.step()
            if i == 0:
                print("losss = {}".format(loss.item()))
                num_x.append(index)
                index += 1
                num_y.append(loss)
    plt.plot(num_x, num_y)
    plt.show()
    torch.save(net, f='./test.pth')
