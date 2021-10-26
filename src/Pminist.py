import argparse
import time
import numpy as np
from typing import OrderedDict
import torch
from torch import cuda
from  typing import  OrderedDict
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader
from copy import  deepcopy
from networks.MLP import MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
old_model = None

# tempture
def cross_entropy(outputs,targets,exp=1.0,size_average=True,eps=1e-5):
    out = torch.nn.functional.softmax(outputs,dim=1)
    tar = torch.nn.functional.softmax(targets,dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out/out.sum(1).view(-1,1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar/tar.sum(1).view(-1,1).expand_as(tar)
    out = out+eps/out.size(1)
    out = out/out.sum(1).view(-1,1).expand_as(out)
    ce = -(tar*out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce

def criterion_my(outputs,targets,outputs_old=None):
    T=2
    loss=0
    loss+=1*cross_entropy(outputs,outputs_old,exp=1.0/T)
    return loss+torch.nn.functional.cross_entropy(outputs,targets)

def train(model,trn_loader,optim,criterion):
    model.train()
    
    for batch,(data,target) in enumerate(trn_loader):
        data=data.view(-1,28*28)
        data,target = data.to(device),target.to(device)
        outputs = model(data)
        loss  =criterion(outputs,target)

        optim.zero_grad()
        loss.backward()
        optim.step()
def train_projected(model,trn_loader,optim,criterion,proj_mat,old_model):
    model.train()
    old_model.eval()
    
    for batch,(data,target) in enumerate(trn_loader):
        data=data.view(-1,28*28)
        data,target = data.to(device),target.to(device)
        outputs_old= old_model(data)
        outputs = model(data)
        # loss  = criterion_my(outputs,target,outputs_old)
        loss = criterion(outputs,target)

        optim.zero_grad()
        loss.backward()
        for  k,(m,params) in enumerate(model.named_parameters()):
            sz = params.grad.data.size(0)
            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),proj_mat[k]).view(params.size())
        optim.step()

def eval(model,test_loader,optim,criterion):
    model.eval()
    size = len(test_loader.dataset)
    test_loss=0
    correct=0
    num_batches = len(test_loader)
    with torch.no_grad() :
        for data,target in test_loader:
            data=data.view(-1,28*28)
            data,target = data.to(device),target.to(device)
            outputs = model(data)
            loss  =criterion(outputs,target)

            ret,preds = torch.max(outputs,1)
            correct += (preds==target).sum().item()
            test_loss+=loss

    test_loss /=num_batches
    correct /=size 
    # print(f"Test Loss:\n Acc：{100*correct:>0.1f},Avg loss: {test_loss:>8f}")
    return test_loss,correct

def get_representation_matrix(model,trn_loader):
    r = np.arange(len(trn_loader.dataset))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[:300]
    example_data = trn_loader.dataset[b][0].view(-1,28*28)
    example_data = example_data.to(device)
    example_out = model(example_data)
    
    batch_list=[300,300,300]
    mat_list = []
    act_key = list(model.act.keys())
    
    for i in range(len(act_key)):
        bsz = batch_list[i]
        act = model.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)
        
    return mat_list

def update_GPM(threshold,mat_list,feature_list=[]):

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
            



def main(args):
    
    from utils import pmnist_dataset
    Pminst,taskcla,input_shape  = pmnist_dataset.get(42,pc_valid=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model = MLP(100,10).to(device)
    optim = torch.optim.SGD(model.parameters(),lr=0.01)
    print(f"device={device}")
    taskid=0
    task_list= []
    testloaders=[]
    tasklist=[]

    
    feature_list = []
    acc_matrix=np.zeros((10,10))

    for k,ncla in taskcla:
        trn_dataset = dataset.TensorDataset(Pminst[k]['train']['x'],Pminst[k]['train']['y'])
        test_dataset = dataset.TensorDataset(Pminst[k]['test']['x'],Pminst[k]['test']['y'])
        val_dataset = dataset.TensorDataset(Pminst[k]['valid']['x'],Pminst[k]['valid']['y'])

        trn_loader = DataLoader(trn_dataset,batch_size=10)
        val_loader = DataLoader(val_dataset,batch_size=10)
        test_loader = DataLoader(test_dataset,batch_size=64)
        testloaders.append(test_loader)
        tasklist.append(k)
        
        threshold = np.array([0.95,0.99,0.99])

        if k==0: 
            print('-'*40)
            for epoch in range(1,args.n_epochs+1):
                clock0=time.time()
                train(model,trn_loader,optim,criterion)
                clock1 = time.time()
                # tr_loss,tr_acc = eval(model,trn_loader,optim,criterion)
                print(f"Epoch {epoch} | train:loss={0:.3f}, acc={0:.5f}% | time={1000*(clock1-clock0):5.1f} |",end='')

                valid_loss,valid_acc = eval(model,val_loader,optim,criterion)
                print(f"valid:loss={valid_loss:.3f}, acc={valid_acc:.5f}% |",end='')
                print()

            print('-'*40) 
            test_loss,test_acc = eval(model,test_loader,optim,criterion)
            print(f"Test:loss={test_loss:.3f},acc={test_acc:.5f}%")

            old_model = deepcopy(model)
            mat_list = get_representation_matrix(model,trn_loader)
            feature_list = update_GPM(threshold,mat_list,feature_list)
        else:
            projection_mat = []
            for i in range(len(model.act)):
                Up = torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                projection_mat.append(Up)
            print('-'*40)
            for epoch in range(1,args.n_epochs+1):
                clock0=time.time()
                # train(model,trn_loader,optim,criterion)
                train_projected(model,trn_loader,optim,criterion,projection_mat,old_model)
                clock1 = time.time()
                # tr_loss,tr_acc = eval(model,trn_loader,optim,criterion)
                print(f"Epoch {epoch} | train:loss={0:.3f}, acc={0:.5f}% | time={1000*(clock1-clock0):5.1f} |",end='')

                valid_loss,valid_acc = eval(model,val_loader,optim,criterion)
                print(f"valid:loss={valid_loss:.3f}, acc={valid_acc:.5f}% |",end='')
                print()

            print('-'*40) 
            test_loss,test_acc = eval(model,test_loader,optim,criterion)
            print(f"Test:loss={test_loss:.3f},acc={test_acc:.5f}%")

            old_model = deepcopy(model)
            mat_list = get_representation_matrix(model,trn_loader)
            feature_list = update_GPM(threshold,mat_list,feature_list)
            


        for tk,tl in zip(tasklist,testloaders):
            _,acc_matrix[k][tk] = eval(model,tl,optim,criterion)
        print(acc_matrix)
            


        

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=10, metavar='N',
                        help='number of training epochs/task (default: 5)')
    args = parser.parse_args()
    main(args)
