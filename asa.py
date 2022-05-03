# P(x,y)->A(y)

import torch
import numpy as np

v = torch.as_tensor(np.asarray([[0., 1., 2., 3.]]))

u = torch.as_tensor(np.asarray([[3., 2., 1., 0.]]))


class g_P(torch.nn.Module):
    def __init__(self):
        super(g_P, self).__init__()
        self.l1 = torch.nn.Linear(4, 1)
        self.l2 = torch.nn.Linear(4, 1)

    def forward(self, u, v):
        a = torch.tanh(self.l1(u))
        b = torch.tanh(self.l2(v))
        return torch.sigmoid(a + b)


class g_A(torch.nn.Module):
    def __init__(self):
        super(g_A, self).__init__()
        self.l1 = torch.nn.Linear(4, 1)
        self.l2 = torch.nn.Linear(4, 1)

    def forward(self, v):
        b = self.l1(v)
        return torch.sigmoid(b)


def Neg(x):
    return 1 - x


def Disj(a, b):
    return torch.maximum(a, b)


class formula(torch.nn.Module):
    def __init__(self):
        super(formula, self).__init__()
        self.P = g_P()
        self.A = g_A()

    def forward(self, v,u):
        return Disj(Neg(self.P(v,u)),self.A(u))


print()
