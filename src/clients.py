from typing_extensions import TypeGuard
import torch
from torch.utils.data import DataLoader, dataset
from typing import DefaultDict
from torch.utils.data import TensorDataset
import numpy as np

from torch.utils.data.sampler import BatchSampler


class client(object):
    def __init__(self,trainDataset,device):

        self.train_ds = trainDataset
        self.train_dl = None
        self.local_parameters = None
        self.device = None

    def localUpdate(self,localEpoch,localBatchSize,model,
        lossFun,optim,global_parameters):
        model.load_state_dict(global_parameters,strict=True)
        self.train_dl = DataLoader(self.train_ds,batch_size=localBatchSize,shuffle=True)

        for epoch in range(localEpoch):
            for data,label in self.train_dl:
                data,label = data.to(self.device),label.to(self.device)
                preds =model(data)
                loss = lossFun(preds,label)
                loss.backward()
                optim.step()
                optim.zero_grad()
        return model.state_dict()
    
class ClientsGroup(object):
    def __init__(self,datasetName,numOfClients):
        self.dataset_name = datasetName
        self.clientsNum = numOfClients
        self.clients_set = {}
        self.device = None

    def datasetBalanceAllocation(self):
        mnistDataset = None

        train_data = None
        train_label = None

        shared_size = None
        shared_id = np.random.permutation(dataszie//shared_size)

        for i in range(self.clientsNum):
            shared_id1 = shared_id[i*2]
            shared_id2 = shared_id[i*2+1]
            data_shareds1 = train_data[shared_id1*shared_size:shared_id1*shared_size+shared_size]
            data_shareds2 = train_data[shared_id2*shared_size:shared_id2*shared_size+shared_size]
            label_shareds1 = train_label[shared_id1*shared_size:shared_id1*shared_size+shared_size]
            label_shareds2 = train_label[shared_id2*shared_size:shared_id2*shared_size+shared_size]

            local_data,local_label = np.vstack((data_shareds1,data_shareds2)),np.vstack((label_shareds1,label_shareds2))
            someone = client(TensorDataset(torch.tensor(local_data),torch.tensor(local_label)),self.device)
            self.clients_set[f'client{i}'] = someone
        
