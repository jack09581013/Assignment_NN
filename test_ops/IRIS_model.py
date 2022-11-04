import torch.nn as nn
import torch

class IRIS(nn.Module):
    def __init__(self):
        super(IRIS, self).__init__()
        self.linear_1 = nn.Sequential(nn.Linear(4, 2),
                                    # nn.BatchNorm1d(2),
                                    nn.LeakyReLU(inplace=True))
        self.linear_2 = nn.Sequential(nn.Linear(2, 2),
                                    nn.LeakyReLU(inplace=True))
        self.linear_3 = nn.Sequential(nn.Linear(2, 4),
                                    nn.LeakyReLU(inplace=True))
        self.linear_4 = nn.Sequential(nn.Linear(4, 3),
                                    nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)

        return x

    def predict_class(self, x):
        x = self.forward(x)
        return torch.argmax(x, dim=1)