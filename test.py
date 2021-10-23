# %%
import torch
outputs = torch.tensor([[1,2],[3,4]])
ret,pred = torch.max(outputs,1)
targets =torch.tensor([0,1]) 
(pred==targets).sum().item()