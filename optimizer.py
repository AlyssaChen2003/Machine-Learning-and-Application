import numpy as np


# class SGD:
#     def __init__(self, params, learning_rate=1e-3):
#         self.learning_rate = learning_rate
#         self.params = params

#     def step(self):
#         for param in self.params:
#             param['value'] -= self.learning_rate * param['grad']


class Adam:
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.params_grad = params

    def step(self):
        if self.m is None:
            self.m, self.v = [], []
            for param in self.params_grad:
                self.m.append(np.zeros_like(param['value']))
                self.v.append(np.zeros_like(param['grad']))

        self.iter += 1
        learning_rate_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(self.params_grad)):
            self.m[i] += (1 - self.beta1) * (self.params_grad[i]['grad'] - self.m[i])
            self.v[i] += (1 - self.beta2) * (self.params_grad[i]['grad'] ** 2 - self.v[i])
            self.params_grad[i]['value'] -= learning_rate_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

