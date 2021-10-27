# %%
import torch
from networks.MLP import MLP
net = MLP(200)

gobal_parameters = net.state_dict()
gobal_parameters
for k,v in gobal_parameters.items():
    print(v)
# %%
