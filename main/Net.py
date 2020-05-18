from torch import nn, optim


class ClasifyNet(nn.Module):
    def __init__(self):
        super(ClasifyNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=100, bias=True),
            nn.BatchNorm1d(num_features=100),
            nn.LeakyReLU(),

            nn.Linear(in_features=100, out_features=50, bias=True),
            nn.BatchNorm1d(num_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=3, bias=False),

        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.l1(x)
        return self.soft(y)
