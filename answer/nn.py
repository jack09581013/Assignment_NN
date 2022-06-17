import numpy as np
import optimizer as opt


class Layer:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, g):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, input, output, optimizer=opt.Adam):
        self.w = np.random.randn(input, output)
        self.b = np.random.randn(output)
        self.w_opt = optimizer()
        self.b_opt = optimizer()

    def forward(self, x):
        self.x = x
        return x.dot(self.w) + self.b

    def backward(self, g):
        xg = g.dot(self.w.T)
        wg = self.x.T.dot(g)
        bg = g.sum(axis=0)

        self.w = self.w_opt.get(self.w, wg)
        self.b = self.b_opt.get(self.b, bg)
        return xg


class BatchNorm1d:
    def __init__(self, num_batches, gamma=1, beta=0, eps=1e-5, momentum=0.1, optimizer=opt.Adam):
        self.num_batches = num_batches
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_batches)
        self.running_var = np.zeros(num_batches)
        self.mode = 'train'
        self.gamma_opt = optimizer()
        self.beta_opt = optimizer()

    def forward(self, x):
        if self.mode == 'train':
            sample_mean = x.mean(axis=0)
            sample_var = x.var(axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

            self.std = np.sqrt(sample_var + self.eps)
            self.x_centered = x - sample_mean
            self.x_norm = self.x_centered / self.std
            out = self.gamma * self.x_norm + self.beta

        elif self.mode == 'test':
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        return out

    def backward(self, g):
        dgamma = (g * self.x_norm).sum(axis=0)
        dbeta = g.sum(axis=0)

        dx_norm = g * self.gamma
        dx = 1 / self.num_batches / self.std * (self.num_batches * dx_norm -
                                                dx_norm.sum(axis=0) - self.x_norm *
                                                (dx_norm * self.x_norm).sum(axis=0))

        self.gamma = self.gamma_opt.get(self.gamma, dgamma)
        self.beta = self.beta_opt.get(self.beta, dbeta)

        # self.gamma -= 0.1 * dgamma
        # self.beta -= 0.1 * dbeta

        return dx


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
        mu = 1. / N * np.sum(x, axis=0)

        # step2: subtract mean vector of every trainings example
        xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = 1. / N * np.sum(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + self.eps)

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
        dbeta = np.sum(g, axis=0)
        dgammax = g  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        self.gamma = self.gamma_opt.get(self.gamma, dgamma)
        self.beta = self.beta_opt.get(self.beta, dbeta)

        # self.gamma -= 0.1 * dgamma
        # self.beta -= 0.1 * dbeta

        return dx


class ReLU(Layer):
    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, g):
        g[self.mask] = 0
        return g


class LeakyReLU(Layer):
    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0.1 * x[self.mask]
        return x

    def backward(self, g):
        g[self.mask] = 0.1 * g[self.mask]
        return g


class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, g):
        return g * (self.out * (1 - self.out))


def softmax(x):
    max = x.max(axis=1).reshape(-1, 1)
    e = np.exp(x - max)
    return e / np.sum(e, axis=1).reshape(-1, 1)


def cross_entropy_error(y, t):
    epsilon = 1e-6
    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + epsilon)) / batch_size


class SoftmaxWithLoss:
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)

    def backward(self):
        batch_size = self.t.shape[0]
        xg = (self.y - self.t) / batch_size
        return xg

    def predict_class(self, x):
        self.y = softmax(x)
        return np.argmax(self.y, axis=1)


class MSE_Loss:
    def forward(self, x, t):
        self.x = x
        self.t = t
        return ((x - t) ** 2).mean()

    def backward(self):
        return 2 / self.t.size * (self.x - self.t)

    def predict_class(self, x):
        return np.argmax(x, axis=1)


class Model:
    def __init__(self, loss):
        self.layers = []

        if loss == 'mse':
            self.loss = MSE_Loss()
        elif loss == 'cross_entropy':
            self.loss = SoftmaxWithLoss()
        else:
            raise Exception('Cannot find loss: ' + loss)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y, t):
        loss = self.loss.forward(y, t)
        g = self.loss.backward()
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        return loss

    def predict_class(self, x):
        x = self.forward(x)
        return self.loss.predict_class(x)
