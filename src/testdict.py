# %%

import torch
from networks.MLP import MLP
import numpy as np
model = MLP(200)
m1 = model.state_dict()
m1['fc1.weight'] = torch.tensor(np.ones_like(m1['fc1.weight']),dtype=torch.float32)
X = torch.randn((1,784))
pred = model(X)
y=torch.randn((10,1))
l = (y-pred).sum()
l
l.backward()
for k,v in model.named_parameters():
    print(k,v.grad)
model.load_state_dict(m1)
for k,v in model.named_parameters():
    print(k,v)

# %%
