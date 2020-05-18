import torch
import pandas as pd
import torch.utils.data.dataset as Dataset


def init_data(path):
    data = pd.read_csv(path)
    sLength = data["Sepal.Length"].values
    sWidth = data["Sepal.Width"].values
    pLength = data["Petal.Length"].values
    pWidth = data["Petal.Width"].values
    label = data["Species"].values
    sLength = torch.from_numpy(sLength).view(-1, 1)
    sWidth = torch.from_numpy(sWidth).view(-1, 1)
    pLength = torch.from_numpy(pLength).view(-1, 1)
    pWidth = torch.from_numpy(pWidth).view(-1, 1)
    data = torch.cat((sLength, sWidth, pLength, pWidth), 1)
    nuLabel = torch.zeros(label.shape).long()
    for index, item in enumerate(label):
        if item == 'setosa':
            nuLabel[index] = 0
        if item == 'versicolor':
            nuLabel[index] = 1
        if item == 'virginica':
            nuLabel[index] = 2
    return data.t(), nuLabel


class IrisDataSet(Dataset.Dataset):
    def __init__(self, path):
        data, label = init_data(path)
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = self.data[:, item]
        label = self.label[item]
        return data, label

    def __len__(self):
        return self.data.shape[1]


if __name__ == '__main__':
    dataset = IrisDataSet(path="./datasets/iris.csv")
    temp = dataset[0]
    data,label = temp
    print(data,label)
