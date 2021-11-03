import re
import torch
import tqdm
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from networks.MLP import MLP
from networks.resnet18 import ResNet, ResNet18
from networks.resnet import Renset32,resenet3232
from utils import pmnist_dataset
from typing import List
from torch.optim import SGD
from datasets import cifar100


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
        self.feature_list = []
        self.proj = []
        self.threshold = np.array([0.965] * 33)

    def load_data(self, trainDataset):
        self.train_dataset = trainDataset

    def localUpdate(
        self,
        localEpoch,
        localBatchSize,
        model: ResNet,
        lossFun,
        optim: optim.Optimizer,
        global_parameters,
        taskid
    ):
        model.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(
            self.train_dataset, batch_size=localBatchSize, shuffle=True
        )

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # data=data.view(-1,28*28)
                data, label = data.to(self.device), label.to(self.device)
                preds = model(data)
                loss = lossFun(preds[taskid], label)
                loss.backward()
                if self.proj:
                    # for  k,(m,params) in enumerate(model.named_parameters()):
                    #     sz = params.grad.data.size(0)
                    #     params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),proj_mat[k]).view(params.size())
                    kk=0
                    for k, (m,params) in enumerate(model.named_parameters()):
                        if k<15 and len(params.size())!=1:
                            sz =  params.grad.data.size(0)
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                    self.proj[kk]).view(params.size())
                            kk +=1
                        elif (k<15 and len(params.size())==1) :
                            params.grad.data.fill_(0)
                optim.step()
                optim.zero_grad()

            valid_loss, valid_acc = self.evaluate(taskid,model.state_dict(),model,lossFun)

            # Adapt lr
            if valid_loss < self.bestLoss:
                self.bestLoss = valid_loss
                self.bestModel = model.state_dict()
                patience = 5

        return model.state_dict()
    
    
    def getProj(self,model,global_parameters):
        repr_mat = self.get_representation_matrix(model,global_parameters)
        self.feature_list = self.updateGPM(self.threshold,repr_mat,self.feature_list)
        self.proj = self.calculateProj()

    def calculateProj(self):
        projection_mat = []
        # for i in range(len(self.model.act)):
        #     Up = torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).to(self.device)
        #     projection_mat.append(Up)
                    # # Projection Matrix Precomputation
        for i in range(len(self.feature_list)):
            Uf=torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).to(self.device)
            print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            projection_mat.append(Uf)
        return projection_mat

    def get_representation_matrix(self,model:ResNet,global_parameters):
        model.load_state_dict(global_parameters, strict=True)
        mat_list= model.get_representation_matrix_ResNet18(self.device,self.train_dataset)
        self.repr_matrix = mat_list
        return mat_list
    

    def updateGPM(self,threshold,mat_list,feature_list=[]):

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
                
