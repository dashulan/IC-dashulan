import torch
import torch.nn as nn
import torch.nn.functional as F


        
class MLPFF(nn.Module):
    def __init__(self,n_hidden):
        super(MLPFF,self).__init__()
        self.line1= nn.Linear(64,n_hidden,bias=False)
        self.line2 = nn.Linear(n_hidden,n_hidden,bias=False)
        self.fc1 = nn.Linear(n_hidden,64,bias=False)
    
    def forward(self,x):
        out = self.line1(x)
        out = F.relu(out)
        out = self.line2(out)
        out = F.relu(out)
        out = self.fc1(out)
        return out
