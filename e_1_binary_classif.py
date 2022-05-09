import torch

from ltn_pytorch import Theory

K = Theory()


class A_gr(torch.nn.Module):
    def __init__(self):
        super(A_gr, self).__init__()
        self.l1 = torch.nn.Linear(2, 8, True)
        self.a1 = torch.relu
        self.l2 = torch.nn.Linear(8, 8, True)
        self.a2 = torch.relu
        self.l3 = torch.nn.Linear(8, 1, False)
        self.a3 = torch.sigmoid

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x


# domains
K.domain('point', 'fp32', (2,))
# vars
# K.variable('x', 'point', )

K.variable('x_pos', 'point')
K.variable('x_neg', 'point')
# constants
# K.constant(name, domain, grounding)
# functions
# K.function(name, domain_in, domain_out, grounding)
# predicates
K.predicate('A', ['point'], A_gr(), True)
# axioms
K.axiom('FORALL(x_pos):A(x_pos)')
K.axiom('FORALL(x_neg):!A(x_neg)')

# optimizer
opt = torch.optim.Adam(K.parameters())
# dataset
## generate data
N_pts_total = 1100
generated = []
import random

for _ in range(N_pts_total):
    xy = (random.uniform(0., 1.), random.uniform(0., 1.))
    l = ((xy[0] - 0.5) ** 2 + (xy[1] - 0.5) ** 2) ** 0.5 < 0.2
    generated.append((xy, l))
## split
test_data = generated[:50]
train_data = generated[50:100]
validation = generated[100:]

## dataset
from torch.utils.data import Dataset


class PointsDataset(Dataset):
    def __init__(self, lst):
        self.items = lst

    def __getitem__(self, item):
        it = self.items[item]
        return {'x': it[0], 'label': it[1]}

    def __len__(self):
        return len(self.items)


train_dataset = PointsDataset(train_data)
test_dataset = PointsDataset(test_data)
##dataloader

from torch.utils.data import DataLoader

batchsz = 32

train_loader = DataLoader(train_dataset, batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batchsz, shuffle=False)
## objective function and loss function
for ep in range(2000):
    for batch in train_loader:
        x_pos_i = [i for i, l in enumerate(batch['label']) if l]
        x_neg_i = [i for i, l in enumerate(batch['label']) if not l]

        if len(x_pos_i) == 0 or len(x_neg_i) == 0:
            print('empty var case')
            continue

        x_pos = torch.stack([batch['x'][0][x_pos_i], batch['x'][1][x_pos_i]], 1).to(torch.float32)
        x_neg = torch.stack([batch['x'][0][x_neg_i], batch['x'][1][x_neg_i]], 1).to(torch.float32)

        opt.zero_grad()

        sat, loss = K(x_pos=x_pos, x_neg=x_neg)
        print('ep', ep, 'sat:', sat.cpu().data.numpy(), '\tloss:', loss.cpu().data.numpy(), )
        loss.backward()
        opt.step()
# quering
# 1.
# K.evaluate_formula('A(x)', x=[[]])
# 2.
# y = K.A(x=[])
# 3.
y = K.predicates['A'](x=torch.as_tensor([[0.5, 0.5]]))
# 4.
# y = K('A(x)', x=[[]])
# y = K('A', x=[[]])
print(y)
