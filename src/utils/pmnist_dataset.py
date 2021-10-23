import os, sys
import re
import numpy as np
from numpy.core.records import array
from numpy.random import RandomState
from scipy.sparse.construct import random
import torch
from torch._C import TracingState, _is_tracing
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

mnist_dir = './data'
pmnist_dir = './data/binary_pmnist'

def get(seed=0,fixed_order=False,pc_valid=0.1):
    data={}
    data_shape = [1,28,28]

    nperm = 10
    seeds = np.array(list(range(nperm)),dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds,random_state=seed)

    if not os.path.isdir(pmnist_dir):
        os.mkdir(pmnist_dir)
        
        mean = (0.1307,)
        std = (0.3081,)
        dat={}
        dat['train'] = datasets.MNIST(mnist_dir,train=True,download=True,transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize(mean,std)]
        ))
        dat['test'] = datasets.MNIST(mnist_dir,train=False,download=True,transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize(mean,std)]
        ))
        
        for i,r in enumerate(seeds):
            print(i,end=',')
            sys.stdout.flush()
            data[i] ={}
            data[i]['name'] = f"pmnist--{i}"        
            data[i]['ncla']=10
            
            for s in ['train','test']:
                loader = DataLoader(dat[s],batch_size=1,shuffle=False)
                data[i][s]={'x':[],'y':[]}
                for img,target in loader:
                    aux = img.view(-1).numpy()
                    aux= shuffle(aux,random_state=r*100+i)
                    img = torch.FloatTensor(aux).view(data_shape)
                    data[i][s]['x'].append(img)
                    data[i][s]['y'].append(target.numpy()[0])
            
                    
            for s in ['train','test']:
                data[i][s]['x']=torch.stack(data[i][s]['x']).view(-1,data_shape[0],data_shape[1],data_shape[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'],dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser(pmnist_dir),f"data{r}{s}x.bin"))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser(pmnist_dir),f"data{r}{s}y.bin"))
            print()
    else:
        for i,r in enumerate(seeds):
            data[i] = dict.fromkeys(['name','ncla','train','test'])
            data[i]['ncla']=10
            data[i]['name'] = f"pmnist--{i}"
            
            for s in ['train','test']:
                data[i][s] ={'x':[],'y':[]}
                data[i][s]['x'] =torch.load(os.path.join(os.path.expanduser(pmnist_dir),f"data{r}{s}x.bin"))
                data[i][s]['y'] =torch.load(os.path.join(os.path.expanduser(pmnist_dir),f"data{r}{s}y.bin"))

    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(r,dtype=int)
        nvalid = int(pc_valid*len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']= data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']= data[t]['train']['y'][itrain].clone()

    n=0
    taskcla= []
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n +=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,data_shape