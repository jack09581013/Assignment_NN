import torch
import torch.nn as nn
import torch.nn.functional as F
import answer.optimizer as opt


class BatchNorm1d:
    def __init__(self, num_batches, gamma=1, beta=0, eps=1e-5, momentum=0.1, optimizer=opt.Adam):
        self.num_batches = num_batches
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_batches)
        self.running_var = torch.zeros(num_batches)
        self.mode = 'train'
        self.gamma_opt = optimizer()
        self.beta_opt = optimizer()

    def forward(self, x):
        if self.mode == 'train':
            sample_mean = x.mean(dim=0)
            sample_var = x.var(dim=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

            self.std = torch.sqrt(sample_var + self.eps)
            self.x_centered = x - sample_mean
            self.x_norm = self.x_centered / self.std
            out = self.gamma * self.x_norm + self.beta

        elif self.mode == 'test':
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        return out

    def backward_1(self, g):
        dgamma = (g * self.x_norm).sum(dim=0)
        dbeta = g.sum(dim=0)

        dx_norm = g * self.gamma
        dx_centered = dx_norm / self.std
        dmean = -(dx_centered.sum(dim=0) + 2 / self.num_batches * self.x_centered.sum(dim=0))
        dstd = (dx_norm * self.x_centered * -self.std ** (-2)).sum(dim=0)
        dvar = dstd / 2 / self.std
        dx = dx_centered + (dmean + dvar * 2 * self.x_centered) / self.num_batches

        self.gamma = self.gamma_opt.get(self.gamma, dgamma)
        self.beta = self.beta_opt.get(self.beta, dbeta)

        # self.gamma -= 0.1 * dgamma
        # self.beta -= 0.1 * dbeta

        return dx, dgamma, dbeta

    def backward_2(self, g):
        dgamma = (g * self.x_norm).sum(dim=0)
        dbeta = g.sum(dim=0)

        dx_norm = g * self.gamma
        dx = 1 / self.num_batches / self.std * (self.num_batches * dx_norm -
                                                dx_norm.sum(dim=0) - self.x_norm *
                                                (dx_norm * self.x_norm).sum(dim=0))

        self.gamma = self.gamma_opt.get(self.gamma, dgamma)
        self.beta = self.beta_opt.get(self.beta, dbeta)

        # self.gamma -= 0.1 * dgamma
        # self.beta -= 0.1 * dbeta

        return dx, dgamma, dbeta


class BatchNorm1d_Simple:
    """Tutorial: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html"""

    def __init__(self, gamma=1, beta=0, eps=1e-5, momentum=0.1, optimizer=opt.Adam):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum
        self.gamma_opt = optimizer()
        self.beta_opt = optimizer()

    def forward(self, x):
        N, D = x.shape

        # step1: calculate mean
        mu = 1. / N * torch.sum(x, dim=0)

        # step2: subtract mean vector of every trainings example
        xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = 1. / N * torch.sum(sq, dim=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = torch.sqrt(var + self.eps)

        # step6: invert sqrtwar
        ivar = 1. / sqrtvar

        # step7: execute normalization
        xhat = xmu * ivar

        # step8: Nor the two transformation steps
        gammax = self.gamma * xhat

        # step9
        out = gammax + self.beta

        # store intermediate
        self.cache = (xhat, self.gamma, xmu, ivar, sqrtvar, var)

        return out

    def backward(self, g):
        # unfold the variables stored in cache
        xhat, gamma, xmu, ivar, sqrtvar, var = self.cache

        # get the dimensions of the input/output
        N, D = g.shape

        # step9
        dbeta = torch.sum(g, dim=0)
        dgammax = g  # not necessary, but more understandable

        # step8
        dgamma = torch.sum(dgammax * xhat, dim=0)
        dxhat = dgammax * gamma

        # step7
        divar = torch.sum(dxhat * xmu, dim=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / torch.sqrt(var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * torch.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1 + dxmu2, dim=0)

        # step1
        dx2 = 1. / N * torch.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        self.gamma = self.gamma_opt.get(self.gamma, dgamma)
        self.beta = self.beta_opt.get(self.beta, dbeta)

        # self.gamma -= 0.1 * dgamma
        # self.beta -= 0.1 * dbeta

        return dx, dgamma, dbeta

if __name__ == '__main__':

    model = BatchNorm1d(2)
    # model = BatchNorm1d_Simple()

    x = torch.FloatTensor([[1, 5], [3, 2], [5, 7]])
    y = torch.FloatTensor([[3, 7], [18, 8], [5, 2]])
    predict = None

    for i in range(6000):
        predict = model.forward(x)
        loss = ((predict - y) ** 2).mean()
        g = 2 / y.size(0) * (predict - y)

        if isinstance(model, BatchNorm1d):
            model.backward_2(g)
        elif isinstance(model, BatchNorm1d_Simple):
            model.backward(g)
        else:
            raise NotImplementedError()

    print(predict)
    # tensor([[7.6667, 5.2895],
    #         [8.6667, 8.6842],
    #         [9.6667, 3.0263]])
