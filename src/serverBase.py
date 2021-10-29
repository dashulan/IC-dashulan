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
        self.model.to(self.device)
        self.proj = None
        self.feature_list =[]
        self.threshold = np.array([0.95,0.99,0.99])


    def run(self):
        self.init_clients()
        self.init_data()
        self.init_global_weights()
        self.train_clients()

    def train_clients(self):
        commu = 1
        for tid in range(self.TaskNum):
            self.splitTaskData2Client(tid)
            self.currentTask = tid
            for curr_round in range(commu):
                self.sum_parameters = None
                for client in self.clients:

                    local_parameters = client.localUpdate(5,10,self.model,self.loss_func,self.opti,self.global_parameters,self.proj)

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
            
            
            self.getProj()
            
            print("-"*40)
            print(f"after Task {tid}")
            for ttid in range(0,tid+1):
                self.serverEvaluate(ttid)
                # print(f"evaluate at task {ttid} | acc = {acc} ")
            print("-"*40)

    def getProj(self):
        reps_mat = self.getReprMatrix()
        
        self.feature_list = self.update_GPM(self.threshold,reps_mat,self.feature_list,)
        proj_mat = self.calculateProj()
        self.proj = proj_mat
    
    def calculateProj(self):
        projection_mat = []
        for i in range(len(self.model.act)):
            Up = torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).to(self.device)
            projection_mat.append(Up)
        return projection_mat

    def getReprMatrix(self):
        temp_matrix = []
        for idx,client in enumerate(self.clients):
            temp_matrix.append(client.get_representation_matrix(self.model,self.global_parameters))
        repr_matrix = []
        for i in range(3):
            temp = []
            for j in range(len(temp_matrix)):
                temp.append(temp_matrix[j][i])
            repr_matrix.append(np.concatenate(temp,axis=1))
        return repr_matrix
        
    def update_GPM(self,threshold,mat_list,feature_list=[]):

        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,vh = np.linalg.svd(activation,full_matrices=False)
                sval_total = (S**2).sum()
                sval_ration = (S**2)/sval_total
                r= np.sum(np.cumsum(sval_ration)<threshold[i])
                feature_list.append(U[:,0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,vh1 = np.linalg.svd(activation,full_matrices=False)
                sval_total = (S1**2).sum()


                act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat,full_matrices=False)
                
                sval_hat = (S**2).sum()
                sval_ration = (S**2)/sval_total
                accumulated_sval = (sval_total-sval_hat)/sval_total
                
                r = 0

                for ii in range(sval_ration.shape[0]):
                    if accumulated_sval<threshold[i]:
                        accumulated_sval+=sval_ration[ii]
                        r+=1
                    else:
                        break
                if r==0:
                    print(f"Skip Updating GPM for layer: {i+1}")
                    continue
                
                Ui = np.hstack((feature_list[i],U[:,0:r]))
                if Ui.shape[1]>Ui.shape[0]:
                    feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    feature_list[i] = Ui
        return feature_list
                


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
        proj_mat:List
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
                if proj_mat is None:
                    optim.step()
                else:
                    for  k,(m,params) in enumerate(model.named_parameters()):
                        sz = params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),proj_mat[k]).view(params.size())
                optim.zero_grad()

        return model.state_dict()
    
    def get_representation_matrix(self,model,global_parameters):
        model.load_state_dict(global_parameters, strict=True)
        r = np.arange(len(self.train_dl.dataset))
        np.random.shuffle(r)
        r = torch.LongTensor(r).to(self.device)
        size = min(len(self.train_dataset),300)
        b = r[:size]
        example_data = self.train_dl.dataset[b][0].view(-1,28*28)
        example_data = example_data.to(self.device)
        example_out = model(example_data)
        
        batch_list=[size]*3
        mat_list = []
        act_key = list(model.act.keys())
        
        for i in range(len(act_key)):
            bsz = batch_list[i]
            act = model.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)
            
        return mat_list



if __name__ == "__main__":
    s = ServerBase()
    s.run()
