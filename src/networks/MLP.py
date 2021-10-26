import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict

class MLP(nn.Module):
    def __init__(self,n_hidden,n_outputs=10):
        super(MLP,self).__init__()
        self.act = OrderedDict()
        self.line1= nn.Linear(784,n_hidden,bias=False)
        self.line2 = nn.Linear(n_hidden,n_hidden,bias=False)
        self.fc1 = nn.Linear(n_hidden,n_outputs,bias=False)
    
    def forward(self,x):
        self.act['Lin1']=x
        x = self.line1(x)
        x = F.relu(x)
        self.act['lin2']=x
        x = self.line2(x)
        x = F.relu(x)
        self.act['fc1']=x
        x = self.fc1(x)
        return x