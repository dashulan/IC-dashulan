import re
import torch
import tqdm
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from networks.MLP import MLP
from utils import pmnist_dataset
from typing import List
from torch.optim import SGD


class ServerBase:
    def __init__(self) -> None:
        self.datasets = None
        self.clients: List[ClientBase] = []
        self.TaskNum = 10
        self.alldata = None
        self.model:MLP = MLP(200)
        self.global_parameters ={}
        self.loss_func= torch.nn.CrossEntropyLoss()
        self.opti = SGD(self.model.parameters(),lr=0.1)
        self.sum_parameters = None
        self.testDataset = None
        self.currentTask = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def run(self):
        self.init_clients()
        self.init_data()
        self.init_global_weights()
        self.train_clients()

    def train_clients(self):
        commu = 10
        for tid in range(self.TaskNum):
            self.splitTaskData2Client(tid)
            self.currentTask = tid
            for curr_round in range(commu):
                self.sum_parameters = None
                for client in self.clients:

                    local_parameters = client.localUpdate(5,10,self.model,self.loss_func,self.opti,self.global_parameters)

                    if self.sum_parameters is None:
                        self.sum_parameters = {}
                        for key,var in local_parameters.items():
                            self.sum_parameters[key] = var.clone()
                    else:
                        for var in self.sum_parameters:
                            self.sum_parameters[var] = self.sum_parameters[var] +local_parameters[var]

                for var in self.global_parameters:
                    self.global_parameters[var] = (self.sum_parameters[var]/len(self.clients))
                self.serverEvaluate(tid)
            
            print("-"*40)
            print("after Task {tid}")
            for ttid in range(0,tid):
                acc = self.serverEvaluate(ttid)
                print(f"evaluate at task {ttid} | acc = {acc} ")
            print("-"*40)

    
    def serverEvaluate(self,tid):
        test_datasets = self.loadTesTaskData(tid)
        testloader = DataLoader(test_datasets,batch_size=20)
        total_acc ,total_loss=0,0
        total_num=0
        model = self.model
        model.load_state_dict(self.get_global_parameters())
        with torch.no_grad():
            for data,label in testloader:
                data,label = data.to(self.device),label.to(self.device)
                data=data.view(-1,28*28)
                preds = model(data)
                loss  = self.loss_func(preds,label)
                preds = torch.argmax(preds,dim=1)
                total_acc +=(preds==label).sum().item()
                total_loss+=loss
                total_num+=len(label)
        print(f"Task : {tid} | Test : avg_loss = {total_loss/total_num} | acc = {total_acc/total_num}")        
        return total_acc/total_num
    

    def init_data(self):
        self.splitData2Task()

    def init_global_weights(self):
        for key,var in self.model.state_dict().items():
            self.global_parameters[key] =var.clone()
    

    def get_global_parameters(self):
        return self.global_parameters

    def init_clients(self):
        for i in range(3):
            self.clients.append(ClientBase(cid=i))

    def datasetsAllocation(self):
        dataPerTask = []

        pass

    def splitData2Task(self):
        self.alldata, taskcla, input_shape = pmnist_dataset.get(42, pc_valid=0.1)

    def splitTaskData2Client(self, tid):
        # 采样
        datalen = len(self.alldata[tid]["train"]["x"])
        cNum = len(self.clients)
        sizePerClient = datalen // cNum
        indices = np.arange(datalen)
        for client in self.clients:
            client_indices = np.random.choice(
                indices, min(sizePerClient, len(indices)), replace=False
            )
            cTrainDataset = TensorDataset(
                self.alldata[tid]["train"]["x"][client_indices],
                self.alldata[tid]["train"]["y"][client_indices],
            )
            # cTestDataset = TensorDataset(self.alldata[tid]['test']['x'][client_indices],self.alldata[tid]['test']['y'][client_indices])
            client.load_data(cTrainDataset)
            indices = np.setdiff1d(indices, client_indices)

    def get_weights(self):
        pass

    def set_weights(self):
        pass

    def loadTesTaskData(self,tid):
        return TensorDataset(self.alldata[tid]['test']['x'],self.alldata[tid]['test']['y'])


class ClientBase:
    def __init__(self, cid) -> None:
        self.cid = cid
        self.train_dataset = None
        self.train_dl: DataLoader = None
        self.state = {}
        self.tasks = []
        self.local_args = {
            "epoch": 10,
            "batchsize": 50,
        }
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load_data(self, trainDataset):
        self.train_dataset = trainDataset

    def localUpdate(
        self,
        localEpoch,
        localBatchSize,
        model: MLP,
        lossFun,
        optim: optim.Optimizer,
        global_parameters,
    ):
        model.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(
            self.train_dataset, batch_size=localBatchSize, shuffle=True
        )

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data=data.view(-1,28*28)
                data, label = data.to(self.device), label.to(self.device)
                preds = model(data)
                loss = lossFun(preds, label)
                loss.backward()
                optim.step()
                optim.zero_grad()
        return model.state_dict()


if __name__ == "__main__":
    s = ServerBase()
    s.run()
