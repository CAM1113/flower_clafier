import torch
import torch.utils.data.dataloader as DataLoader
from torch.autograd import Variable
from main import IrisDataSet
from main.iris_clasifier import ClasifyNet

testPath = "./datasets/iris_test.csv"
test_dataset = IrisDataSet(testPath)
test_data_loader = DataLoader.DataLoader(test_dataset, batch_size=100, shuffle=False)

net = torch.load(f='./test.pth')


def compute_accuracy(x, y):
    z = x == y
    length = z.shape[0]
    length = torch.tensor(length).float()
    print("length = {}".format(length))
    summ = z.sum(dim=0).float()
    print("summ = {}".format(summ))
    return summ / length


for index, items in enumerate(test_data_loader):
    data, label = items
    data = Variable(data.view(-1, 4).float())
    label = Variable(label.long())
    y = net(data)
    y = torch.argmax(y, dim=1)
    print("y = {}".format(y))
    print('label = {}'.format(label))
    accuracy = compute_accuracy(label, y)
    print("accuracy = {}".format(accuracy))

net.eval()
