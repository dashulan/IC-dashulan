from copy import deepcopy
import torch
import  math
from torch.optim.lr_scheduler import  LambdaLR
from torch.optim.sgd import SGD
from networks.resnet import resenet3232,resnet1818
from datasets import memory_dataset
from datasets.memory_dataset import MemoryDataset, get
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ICAppBase import ICBase
from lwf2 import App_lwf_feat
from GPM import App_gpm
from LwfGpm import App_lwfGpm
from lwfISDA import App_lwfIsda
from FT import App_FT
from approach.lwf import  App_lwf
import argparse
from networks import allmodels
from torchvision import transforms
import random
import os
import time
from torch.utils.data import dataset
from networks.resnetGpm import ResNet18


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)


def main(args):
    
    seed_everything(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    alldata, taskcla = memory_dataset.get()

    # model = resenet3232(taskcla).to(device)
    model = resnet1818(taskcla,64).to(device)


    optim = SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.wd)

    # app = ICBase(optim,model)
    # app = App_lwf(optim,model)
    app = App_lwf_feat(optim,model)
    # app = App_gpm(optim,model)
    # app = App_lwfGpm(optim,model)
    # app = App_lwfIsda(optim,model)
    # app = App_FT(optim,model)

    expName = f"{args.exp_name}_{args.network}_lr{args.lr}_wd{args.wd:.1e}_sz{args.batch_size}"
    write = SummaryWriter(f'./log/{expName}')


    tstes = []
    acc_matrix=np.zeros((10,10))
    acc_matrix_temp = np.zeros((10,10))

    for t in range(len(taskcla)):

        Train_transform,test_transfrom = get_transforms(None,4,32,True,((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),None)
        _,tempt = get_transforms(None,None,None,None,((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),None)
       
        trn_datasets = MemoryDataset(alldata[t]['train']['x'],alldata[t]['train']['y'],Train_transform)
        vld_datasets = MemoryDataset(alldata[t]['valid']['x'],alldata[t]['valid']['y'],tempt)
        tst_datasets = MemoryDataset(alldata[t]['test']['x'],alldata[t]['test']['y'],tempt)

        trn_loader = DataLoader(trn_datasets,batch_size=args.batch_size,shuffle=True,
                                drop_last=True,num_workers=4,pin_memory=True)
        valid_loader = DataLoader(vld_datasets,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)
        tst_loader = DataLoader(tst_datasets,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)


        tstes.append(tst_loader)


        train_loop(args, app, write, t, trn_loader, valid_loader)
        # avgaccb = testPreTasks(app, tstes, acc_matrix_temp, t)
        # write.add_scalar(f"avgtask/Acc/before",avgaccb/(t+1),t)

        # app.adjust_lr()
        avgacca = testPreTasks(app, tstes, acc_matrix, t,write)

        app.train_post(trn_loader=trn_loader,model=model)
        bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 

        print(f"avg_acc : {100*avgacca/(t+1):5.2f}%")
        # print (f'Backward transfer: {100*bwt:5.2f}%')
        write.add_scalar(f"avgtask/Acc/after",avgacca/(t+1),t)
    write.close()

def testPreTasks(app, tstes, acc_matrix, t,write):
    print("*"*40)
    avgacc = 0
    for tt in range(t+1):
        test_loss,test_acc = app.eval(tstes[tt],tt)
        avgacc+=test_acc
        acc_matrix[t,tt] = test_acc
        write.add_scalar(f"task:{t}/acc_avg",test_acc,tt)

    print('Accuracies =')
    for i_a in range(t+1):
        print('\t',end='')
        for j_a in range(acc_matrix.shape[1]):
            print(f'{100*acc_matrix[i_a,j_a]:5.1f}% ',end='')
        print()
    print("*"*40)
    return avgacc

def train_loop(args, app:ICBase, write, t, trn_loader, valid_loader):
    best_loss = np.inf
    lr = 0.1
    warm_up_iter = 5
    T_max = 150
    lr_max = 0.1
    lr_min = 1e-4
    lambda0 = lambda cur_iter: 200 * cur_iter if cur_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
                1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / (1e-4)
    #if t>4:
        #lr_max = 0.01
      #  lr_min = 1e-5
      #  lambda0 = lambda cur_iter: 20*cur_iter if cur_iter < warm_up_iter else \
      #      (lr_min + 0.5 * (lr_max - lr_min) * (
      #                  1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / (0.01)

    scheduler = LambdaLR(app.optim, lr_lambda=lambda0)
    for e in range(args.nepochs):
        clock0 = time.time()
        app.train_eopch(trn_loader,t)
        clock1 = time.time()
        valid_loss,valid_acc = app.eval(valid_loader,t)
        print(f"epoch {e:3d}, time = {clock1-clock0:5.1f}s | valid loss : {valid_loss:.5f} | valid acc : {valid_acc*100:.1f}% |",end='')
        write.add_scalar(f"task:{t}/Loss/valid",valid_loss,e)
        write.add_scalar(f"task:{t}/Accuracy/valid",valid_acc,e)
        write.add_scalar(f"task:{t}/lr",app.optim.param_groups[0]['lr'],e)
        # if e<=5:
        scheduler.step()

        if valid_loss< best_loss:
            best_loss=valid_loss
            best_model=app.get_model()
            patience=6
            print(' *',end='')
        print()
            
        # if e==80 or e== 120:
        #     lr/=10
        #     app.adjust_lr(lr)
    app.set_model_(best_model)
    
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--exp_name', default='base', type=str,
                        help='Experiment name (default=%(default)s)')

    # dataset args
   
    parser.add_argument('--batch_size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet18', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=['finetuning','lwf','gpm'],
                    help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=80, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')

    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')

    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--wd', default=1e-4, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')

    # Args -- Incremental Learning Framework
    args = parser.parse_args()
    main(args)