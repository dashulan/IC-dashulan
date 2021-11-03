import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from networks.MLP import MLP
from networks.resnet18 import ResNet, ResNet18
from utils import pmnist_dataset
from typing import List
from torch.optim import SGD
from datasets import cifar100
from clientBase import  ClientBase


class ServerBase:
    def __init__(self) -> None:
        self.datasets = None
        self.clients: List[ClientBase] = []
        self.TaskNum = 10
        self.alldata = None
        self.taskCla = []
        # self.model:MLP = MLP(200)
        self.model:ResNet18 = None

        
        self.global_parameters ={}
        self.loss_func= torch.nn.CrossEntropyLoss()
        self.opti = None
        self.sum_parameters = None

        self.testDataset = None
        self.currentTask = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.proj = None
        self.feature_list =[]

        # self.threshold = np.array([0.95,0.99,0.99])

        self.threshold = np.array([0.965] * 20)


    def run(self):
        self.init_clients()
        self.init_data()
        self.initModel()
        self.init_global_weights()
        self.train_clients()

    def train_clients(self):
        commu = 160
        for tid in range(self.TaskNum):
            self.splitTaskData2Client(tid)
            self.currentTask = tid
            self.opti = SGD(self.model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
            lr = 0.1
            for curr_round in range(commu):
                self.sum_parameters = None
                # print("-" * 20)
                for client in self.clients:

                    local_parameters = client.localUpdate(1,128,self.model,self.loss_func,self.opti,self.global_parameters,tid,self.proj)

                    if self.sum_parameters is None:
                        self.sum_parameters = {}
                        for key,var in local_parameters.items():
                            self.sum_parameters[key] = var.clone()
                    else:
                        for var in self.sum_parameters:
                            self.sum_parameters[var] = self.sum_parameters[var] +local_parameters[var]
                            # self.sum_parameters[var] =  local_parameters[var]

                    # for var in self.global_parameters:
                    #     self.global_parameters[var] = local_parameters[var]

                    #
                    # print(f'model in client {client.cid} test')
                    # self.serverEvaluate(tid,parameters=local_parameters)
                # print("-"*20)

                for var in self.global_parameters:
                    self.global_parameters[var] = (self.sum_parameters[var]/len(self.clients))
                    # self.global_parameters[var] = self.sum_parameters[var]
                # self.serverEvaluate(tid)
                if curr_round==80 or curr_round==120:
                    lr /=10
                    self.opti.param_groups[0]['lr'] = lr

            # for client in self.clients:
            #     client.getProj(self.model,self.global_parameters)
            self.getProj()
            
            print("-"*40)
            print(f"after Task {tid}")
            for ttid in range(0,tid+1):
                self.serverEvaluate(ttid)
                # print(f"evaluate at task {ttid} | acc = {acc} ")
            print("-"*40)

    # GPM based method
    def getProj(self):
        reps_mat = self.getReprMatrix()
        
        self.feature_list = self.update_GPM(self.threshold,reps_mat,self.feature_list,)
        proj_mat = self.calculateProj()
        self.proj = proj_mat
    
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

    def getReprMatrix(self):
        temp_matrix = []
        for idx,client in enumerate(self.clients):
            temp_matrix.append(client.get_representation_matrix(self.model,self.global_parameters))
        repr_matrix = []
        for j in range(len(temp_matrix[0])):
            temp = np.zeros_like(temp_matrix[0][j])
            for i in range(1):
                temp+=temp_matrix[i][j]
                # temp.append(temp_matrix[i][j])
            repr_matrix.append(temp/1)
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


    # evaluate
    def serverEvaluate(self,tid,parameters=None,model=None):
        test_datasets = self.loadTesTaskData(tid)
        testloader = DataLoader(test_datasets,batch_size=20)
        total_acc ,total_loss=0,0
        total_num=0
        model = self.model
        if parameters:
            model.load_state_dict(parameters)
        else:
            model.load_state_dict(self.get_global_parameters())
        with torch.no_grad():
            for data,label in testloader:
                data,label = data.to(self.device),label.to(self.device)
                # data=data.view(-1,28*28)
                preds = model(data)
                loss  = self.loss_func(preds[tid],label)
                preds = torch.argmax(preds[tid],dim=1)
                total_acc +=(preds==label).sum().item()
                total_loss+=loss
                total_num+=len(label)
        print(f"Task : {tid} | Test : avg_loss = {total_loss/total_num} | acc = {total_acc/total_num}")        
        return total_acc/total_num

    # init method
    def initModel(self):
        self.model = ResNet18(self.taskcla,20)
        self.opti = SGD(self.model.parameters(),lr=0.1)
        self.model.to(self.device)

    def init_clients(self):
        for i in range(1):
            self.clients.append(ClientBase(cid=i))

    def init_global_weights(self):
        for key,var in self.model.state_dict().items():
            self.global_parameters[key] =var.clone()

    # dataBased Methods
    def init_data(self):
        self.splitData2Task()

    def splitData2Task(self):
        # self.alldata, taskcla, input_shape = pmnist_dataset.get(42, pc_valid=0.1)
        self.alldata, self.taskcla, input_shape = cifar100.get(42, pc_valid=0.1)

    def splitTaskData2Client(self, tid):

        cNum = len(self.clients)

        datalen = len(self.alldata[tid]["train"]["x"])
        vdatalen = len(self.alldata[tid]['valid']['x'])

        sizePerClient = datalen // cNum
        vsizePerClient = vdatalen // cNum

        indices = np.arange(datalen)
        vindices = np.arange(vdatalen)

        for client in self.clients:
            client_indices = np.random.choice(
                indices, min(sizePerClient, len(indices)), replace=False
            )
            vclient_indices = np.random.choice(
                vindices,min(vsizePerClient,len(vindices)),replace=False
            )

            cTrainDataset = TensorDataset(
                self.alldata[tid]["train"]["x"][client_indices],
                self.alldata[tid]["train"]["y"][client_indices],
            )
            cValidDataset = TensorDataset(
                self.alldata[tid]['valid']['x'][vclient_indices],
                self.alldata[tid]['valid']['y'][vclient_indices],
            )
            # cTestDataset = TensorDataset(self.alldata[tid]['test']['x'][client_indices],self.alldata[tid]['test']['y'][client_indices])
            client.load_data(cTrainDataset,cValidDataset)
            indices = np.setdiff1d(indices, client_indices)
            vindices = np.setdiff1d(vindices,vclient_indices)

    def loadTesTaskData(self,tid):
        return TensorDataset(self.alldata[tid]['test']['x'],self.alldata[tid]['test']['y'])

    # utils methods
    def adjust_learning_rate(self,optimizer, round):
        """Sets the learning rate to the initial LR decayed by 10 every 10 round"""
        lr = 0.1 * (0.1 ** (round // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_global_parameters(self):
        return self.global_parameters

if __name__ == "__main__":
    s = ServerBase()
    s.run()
