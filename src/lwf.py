import re
from networks.resnet import resenet3232
from datasets import cifar100
from torch.utils.data import DataLoader,TensorDataset
import torch
from copy import deepcopy



alldata, taskcla, input_shape = cifar100.get(42, pc_valid=0.1)

model = resenet3232(taskcla)
optim = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_old = deepcopy(model)



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

def criterion( t, outputs, targets, outputs_old=None):
    """Returns the loss value"""
    loss = 0
    if t > 0:
        # Knowledge distillation loss for all previous tasks
        loss += 1* cross_entropy(outputs[t], outputs_old[t], exp=1.0 / 2)
    # # Current cross-entropy loss -- with exemplars use all heads
    # if len(self.exemplars_dataset) > 0:
    #     return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
    return loss + torch.nn.functional.cross_entropy(outputs[t],targets)


# def train_epoch(trn_loader,loss_fn):

#     for data,label in trn_loader:
#         data,label = data.to(device),label.to(device)
#         # data=data.view(-1,28*28)
#         preds = model(data)
#         loss  = loss_fn(preds[tid],label)
#         preds = torch.argmax(preds[tid],dim=1)
#         total_acc +=(preds==label).sum().item()
#         total_loss+=loss
#         total_num+=len(label)

def train_epoch(model,model_old,t, trn_loader):
    """Runs a single epoch"""
    model.train()
    allloss = 0
    allnum=0
    for images, targets in trn_loader:
        # Forward old model
        targets_old = None
        if t > 0:
            targets_old = model_old(images.to(device))
        # Forward current model
        outputs = model(images.to(device))

        loss = criterion(t, outputs, targets.to(device), targets_old)
        allloss+=loss
        allnum+=len(targets)
        # print(f"loss : {loss}")
        # Backward
        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        optim.step()
    # print(f"epoch : {}")/
    return allloss/allnum

def eval(model,tst_loader,t):

    model.eval()
    total_loss = 0
    total_acc=0
    total_num = 0
    for data,label in tst_loader:
        data,label = data.to(device),label.to(device)
        # data=data.view(-1,28*28)
        preds = model(data)
        # loss  = torch.nn.function(preds[t],label)
        loss = torch.nn.functional.cross_entropy(preds[t],label)
        preds = torch.argmax(preds[t],dim=1)
        total_acc +=(preds==label).sum().item()
        total_loss+=loss
        total_num+=len(label)
    return total_loss/total_num,total_acc/total_num


# def serverEvaluate(self,tid):
#     test_datasets = self.loadTesTaskData(tid)
#     testloader = DataLoader(test_datasets,batch_size=20)
#     total_acc ,total_loss=0,0
#     total_num=0
#     model = self.model
#     model.load_state_dict(self.get_global_parameters())
#     with torch.no_grad():
#         for data,label in testloader:
#             data,label = data.to(self.device),label.to(self.device)
#             # data=data.view(-1,28*28)
#             preds = model(data)
#             loss  = self.loss_func(preds[tid],label)
#             preds = torch.argmax(preds[tid],dim=1)
#             total_acc +=(preds==label).sum().item()
#             total_loss+=loss
#             total_num+=len(label)
#     print(f"Task : {tid} | Test : avg_loss = {total_loss/total_num} | acc = {total_acc/total_num}")        
#     return total_acc/total_num
    
for t in range(len(taskcla)):
    trn_datasets = TensorDataset(alldata[t]['train']['x'],alldata[t]['train']['y'])
    tst_datasets = TensorDataset(alldata[t]['train']['x'],alldata[t]['train']['y'])
    trn_loader = DataLoader(trn_datasets,batch_size=128,shuffle=True)
    tst_loader = DataLoader(tst_datasets,batch_size=64,shuffle=True)

    # for e in range(1):
    #     l = train_epoch(model,model_old,t,trn_loader)
    #     print(f"epoch : {e} | loss : {l}")
    
    eval(model,tst_loader,t)
    model_old = deepcopy(model)