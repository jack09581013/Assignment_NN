import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self):
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Conv2d(6, 6, 3, padding=1, groups=3)
        self.conv.weight = nn.Parameter(torch.randn(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.randn(self.conv.bias.shape))
        # self.conv.weight = nn.Parameter(torch.full(self.conv.weight.shape, 1.0))
        # self.conv.bias = nn.Parameter(torch.full(self.conv.bias.shape, 1.0))
        # print(self.conv.weight.shape)  # torch.Size([6, 2, 3, 3])
        # print(self.conv.bias.shape)  # torch.Size([6])

    def forward(self, x):
        x = self.conv(x)
        return x

def depth_wise_conv(x, weight, bias, group):
    batch, Cin, height, width = x.shape
    assert Cin % group == 0
    Cout_w, Cin_w, kernel_size_1, kernel_size_2 = weight.shape

    x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
    y = torch.zeros((batch, Cout_w, height, width))

    for b in range(batch):
        for i in range(height):
            for j in range(width):
                for Cout_idx in range(Cout_w):
                    sum_v = 0
                    g = Cout_idx // Cin_w
                    for Cin_idx in range(Cin_w):
                        for k1 in range(kernel_size_1):
                            for k2 in range(kernel_size_2):
                                sum_v += weight[Cout_idx, Cin_idx, k1, k2] * \
                                         x[b, g * Cin_w + Cin_idx, i + k1, j + k2]
                    y[b, Cout_idx, i, j] = sum_v + bias[Cout_idx]
    return y


if __name__ == '__main__':
    torch.manual_seed(0)
    model = DepthWiseConv()
    x = torch.randn(1, 6, 5, 5)
    # x = torch.full((1, 6, 5, 5), 1.0)
    x[0, 0, :, :] = 2
    y1 = model(x)
    y2 = depth_wise_conv(x, model.conv.weight, model.conv.bias, 3)
    print((y1 - y2).mean())
    print(y1[0, :2, 1, :])
    print(y1[0, :2, 1, :])