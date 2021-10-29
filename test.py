# %%
from typing import OrderedDict
import torch
outputs = torch.tensor([[1,2],[3,4]])
ret,pred = torch.max(outputs,1)
targets =torch.tensor([0,1]) 
(pred==targets).sum().item()
# %%
import numpy as np
m2 = np.random.randn(4,6)
m3 = np.random.randn(4,6)
m4 = np.concatenate((m2,m3),axis=1)
U,S,Vh = np.linalg.svd(m4,full_matrices=False)

S = np.diag(S)
# print(U.shape,S.shape,Vh.shape)
temp = np.matmul(U,S)
m4_hat = np.matmul(temp,Vh[:,:6])
a2 = np.linalg.norm(m2)
a3 = np.linalg.norm(m3)
a2,a3,np.linalg.norm(m4_hat)
# %%
from src.networks.MLP import MLP
import torch.nn as nn
S =  nn.Sequential()
S.add_module('linear',nn.Linear(100,784))
n1 = MLP(200)
S.add_module('MLp',n1)
n2 = MLP(200)
n3 = MLP(200)
n1_dict = n1.state_dict()
n2_dict = n2.state_dict()
n3_dict ={}
for k in n1_dict:
    n3_dict[k] =(n1_dict[k]+n2_dict[k] ) /2
print(n1_dict['line1.weight'])
print(n2_dict['line1.weight'])
print(n3_dict['line1.weight'])

# %%
import numpy as np
x = np.arange(24)
x2 = np.arange(5)
x = np.setdiff1d(x,x2)
# %%
import torch
preds = torch.tensor([1,2,0])
labels =torch.tensor([1,2,5])
# (preds==labels).sum().item()
preds.copy_(labels)
labels[1]=66
preds

# %%
x = torch.tensor([1.0],requires_grad=True)
x1 = x.detach()
x2 = x.clone()
x3 = x.data
x4 = torch.tensor([1.0])
x4.copy_(x)
y = x**2
y.backward()
x1,x2,x3,x4,
# %%
x4.add_(1)
z= x4**3
z.backward()
x.grad
# %%
[1]*3

