import torch
import numpy as np
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
    
    def get_representation_matrix(self,device, train_dataset): 
        r = np.arange(len(train_dataset)) 
        np.random.shuffle(r)
        r = torch.LongTensor(r).to(device)
        size = min(len(train_dataset),300)
        b = r[:size]
        example_data = train_dataset[b][0].view(-1,28*28)
        example_data = example_data.to(device)
        example_out = self.forward(example_data)
        
        batch_list=[size]*3
        mat_list = []
        act_key = list(self.act.keys())
        
        for i in range(len(act_key)):
            bsz = batch_list[i]
            act = self.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)
            
        return mat_list