import itertools
import re
from numpy import mod
from torch._C import device
from torch.optim.optimizer import Optimizer

from torch.optim.sgd import SGD
from Pminist import train
from networks.resnet import resenet3232
from networks.resnet18 import ResNet18
from datasets import cifar100
from torch.utils.data import DataLoader,TensorDataset
import torch
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter, writer
import numpy as np





def cross_entropy( outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """Calculates cross-entropy with temperature scaling"""
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce

def criterion_base( t, outputs, targets):
    """Returns the loss value"""
    return torch.nn.functional.cross_entropy(outputs[t],targets)

def criterion_lwf( t, outputs, targets, outputs_old=None):
    """Returns the loss value"""
    loss = 0
    if t > 0:
        # Knowledge distillation loss for all previous tasks
        loss += 1* cross_entropy(torch.cat(outputs[:t],dim=1),torch.cat(outputs_old[:t],dim=1), exp=1.0 / 2)
    # # Current cross-entropy loss -- with exemplars use all heads
    # if len(self.exemplars_dataset) > 0:
    #     return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
    return loss + torch.nn.functional.cross_entropy(outputs[t],targets)

def criterion_features( t, outputs, targets, old_features,new_features):
    """Returns the loss value"""
    loss = 0
    if t > 0:
        # Knowledge distillation loss for all previous tasks
        loss += 1* cross_entropy(old_features,new_features, exp=1.0 / 2)
        # pass
        # pass
    # # Current cross-entropy loss -- with exemplars use all heads
    # if len(self.exemplars_dataset) > 0:
    #     return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
    return loss + torch.nn.functional.cross_entropy(outputs[t],targets)

def criterion_ewc(t,outputs,targets,fisher_mat,model,old_params):
    loss = 0
    if t>0:
        loss_reg = 0
        for n,p in model.named_parameters():
            if n in fisher_mat.keys():
                loss_reg +=torch.sum(fisher_mat[n]*(p-old_params[n]).pow(2))/2
        loss+=100000*loss_reg
    return loss+F.cross_entropy(outputs[t],targets)


def train_epoch(model,t, trn_loader,device,optim,fisher,old_params):
    """Runs a single epoch"""
    model.train()
    allloss = 0
    allnum=0
    for images, targets in trn_loader:
        # Forward old model

        # targets_old = None
        # if t > 0:
        #     targets_old = model_old(images.to(device))

        # Forward current model
        outputs = model(images.to(device))

        # loss = criterion_base(t, outputs, targets.to(device))
        # loss = criterion_features(t, outputs, targets.to(device), model_old.features,model.features)
        loss = criterion_ewc(t,outputs,targets.to(device),fisher,model,old_params)
        allloss+=loss
        allnum+=len(targets)

        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        optim.step()

    return allloss/allnum

def eval(model,tst_loader,t,device):

    model.eval()
    total_loss = 0
    total_acc=0
    total_num = 0
    for images,targets in tst_loader:
        images,targets = images.to(device),targets.to(device)
        # data=data.view(-1,28*28)
        preds = model(images)
        # loss  = torch.nn.function(preds[t],label)
        loss = torch.nn.functional.cross_entropy(preds[t],targets)
        preds = torch.argmax(preds[t],dim=1)
        total_acc +=(preds==targets).sum().item()
        total_loss+=loss
        total_num+=len(targets)
    return total_loss/total_num,total_acc/total_num


def compute_fisher(trn_loader,t,model,device,optim):
    current_fisher = {n:torch.zeros(p.shape).to(device) for n,p in model.named_parameters() if p.requires_grad}

    n_samples_batches = []
    model.train()
    for images,targets in trn_loader:
        outputs = model(images.to(device))
        preds = targets.to(device)

        loss = F.cross_entropy(outputs[t],preds)
        optim.zero_grad()
        loss.backward()

        for n,p in model.named_parameters():
            if p.grad is not None:
                current_fisher[n]+=p.grad.pow(2)*len(targets)
    
    nsamples = len(trn_loader.dataset)
    ffisher  = {n:(p/nsamples) for n,p in current_fisher.items()}
    return ffisher

def post_train(t,trn_loader,fisher,model,device,optim):
    old_params = {n:p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}

    current_fisher = compute_fisher(trn_loader,t,model,device,optim)

    if t>0:
        for n in fisher.keys():
            fisher[n] = 0.5 * fisher[n] +0.5*current_fisher[n]
        return fisher,old_params

    return current_fisher,old_params

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_config = args

    alldata, taskcla, input_shape = cifar100.get(42, pc_valid=0.1)

    model = resenet3232(taskcla)
    # model = ResNet18(taskcla,64)

    model.to(device)
    model_state = {}
    # model_old = ResNet18(taskcla,64)
    model_old = resenet3232(taskcla)

    old_params = {}
    fisher = {}

    tstes = []
    writer = SummaryWriter(f"./log/{args['exp_name']}/")
        
    for t in range(len(taskcla)):
        trn_datasets = TensorDataset(alldata[t]['train']['x'],alldata[t]['train']['y'])
        valid_datasets = TensorDataset(alldata[t]['valid']['x'],alldata[t]['valid']['y'])
        tst_datasets = TensorDataset(alldata[t]['test']['x'],alldata[t]['test']['y'])
        trn_loader = DataLoader(trn_datasets,batch_size=128,shuffle=True)
        tst_loader = DataLoader(tst_datasets,batch_size=64,shuffle=True)
        valid_loader = DataLoader(valid_datasets,batch_size=64,shuffle=True)

        
        tstes.append(tst_loader)
        optim = SGD(model.parameters(),lr=train_config['lr'],momentum=train_config['momentum'],weight_decay=train_config['wd'])
        lr = 0.1
        best_loss = np.inf
        best_model = {}
        for e in range(train_config['nepochs']):
            l = train_epoch(model,t,trn_loader,device,optim,fisher,old_params)
            print(f"epoch : {e} | loss : {l} | ",end='')
            validloss,acc = eval(model,valid_loader,t,device)
            print(f"valid loss : {validloss} | acc : {acc} | ")
            if validloss<best_loss:
                best_loss = validloss
                best_model = model.state_dict()

            if e==40 or e==60:
                lr /=10
                optim.param_groups[0]['lr']=lr
            writer.add_scalar(f"task:{t}/Loss/valid",validloss,e)
            writer.add_scalar(f"task:{t}/Accuracy/valid",acc,e)

        fisher ,oldp=  post_train(t,trn_loader,fisher,model,device,optim)
        old_params = oldp

        avgacc = 0
        model.load_state_dict(best_model)
        for tt in range(0,t+1):
            testloss,acc = eval(model,tstes[tt],tt,device)
            avgacc +=acc
            print(f"task : {tt} | test loss : {testloss} | acc : {acc} | ")

        writer.add_scalar(f"avgtask/Acc/",avgacc/(t+1),t)
        
        # eval(model,tst_loader,t)
        # model_state = deepcopy(model.state_dict())
        # model_old.load_state_dict(model_state)
        # model_old.to(device)
        # model_old.eval()

if __name__ =='__main__':
    args = {
        'lr':0.1,
        'nepochs':80,
        'momentum':0.9,
        'wd':5e-4,
        'exp_name':'ewc_fix10w'
    }

    main(args)