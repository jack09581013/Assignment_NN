import numpy as np

class LearningRate:
    def __init__(self, lr=0.005):
        # initialize the values of the parameters
        self.lr = lr

    def get(self, x, gradient):
        return x - gradient * self.lr

class Adam:
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    def __init__(self):
        # initialize the values of the parameters

        self.m_t = 0
        self.v_t = 0
        self.t = 0

    def get(self, x, gradient):
        self.t += 1
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * gradient  # updates the moving averages of the gradient
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (gradient ** 2)  # updates the moving averages of the squared gradient
        self.m_cap = self.m_t / (1 - (self.beta_1 ** self.t))  # calculates the bias-corrected estimates
        self.v_cap = self.v_t / (1 - (self.beta_2 ** self.t))  # calculates the bias-corrected estimates
        return x - (self.alpha * self.m_cap) / (np.sqrt(self.v_cap) + self.epsilon)  # updates the parameters