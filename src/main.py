import torch

from torchvision import transforms
from networks.MLP_Fea  import MLPFF

from torch.optim.sgd import SGD
from Pminist import train
from networks.resnet import resenet3232
from networks.resnet18 import ResNet18
from datasets import cifar100_aug
from torch.utils.data import DataLoader,TensorDataset
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter, writer
import numpy as np
from datasets.memory_dataset import MemoryDataset
from ICAppBase import ICBase

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

alldata, taskcla, input_shape = cifar100_aug.get(42, pc_valid=0.1)
model = resenet3232(taskcla)
optim = SGD(model.parameters(),lr=0.03,momentum=0.9,weight_decay=1e-4)
app = ICBase(optim,model)

normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],std=[0.2009, 0.1984, 0.2023])
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

for t in range(len(taskcla)):
    trn_datasets = TensorDataset(alldata[t]['train']['x'],alldata[t]['train']['y'])
    vld_datasets = TensorDataset(alldata[t]['valid']['x'],alldata[t]['valid']['y'])
    tst_datasets = TensorDataset(alldata[t]['test']['x'],alldata[t]['test']['y'])

    # train_datasets = MemoryDataset({'x':alldata[t]['train']['x'],'y':alldata[t]['train']['y']},augmentation)
    # valid_datasets = MemoryDataset({'x':alldata[t]['valid']['x'],'y':alldata[t]['valid']['y']},augmentation)
    # test_datasets = MemoryDataset({'x':alldata[t]['test']['x'],'y':alldata[t]['test']['y']},augmentation)
    # vld_datasets = TensorDataset(alldata[t]['valid']['x'],alldata[t]['valid']['y'])
    # tst_datasets = TensorDataset(alldata[t]['test']['x'],alldata[t]['test']['y'])

    # train_datasets = MemoryDataset(trn_datasets,augmentation)
    # valid_datasets = MemoryDataset(vld_datasets,augmentation)
    # test_datasets = MemoryDataset(tst_datasets,augmentation)
    


    trn_loader = DataLoader(trn_datasets,batch_size=128,shuffle=True,drop_last=True)
    tst_loader = DataLoader(tst_datasets,batch_size=64,shuffle=True)
    valid_loader = DataLoader(vld_datasets,batch_size=64,shuffle=True)

    lr = 0.1
    for e in range(80):
        train_loss  = app.train_eopch(trn_loader,t)
        valid_loss,valid_acc = app.eval(tst_loader,t)
        print(f"Task : {t} | train loss : {train_loss} | valid loss : {valid_loss} | valid acc : {valid_acc} |")
        if e==40 or e==60:
            lr/=10
            app.adjist_lr(lr)
    app.adjust_lr()
    
    for tt in range(t+1):
        test_loss,avg_acc = app.eval(tst_loader,t)
        print(f"Task : {t} | test loss : {test_loss} | avg_acc : {avg_acc}")

    
    
