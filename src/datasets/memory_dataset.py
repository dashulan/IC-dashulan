from torch.utils.data import Dataset
from PIL import Image
import time

from torchvision import transforms


class MemoryDataset(Dataset):

    def __init__(self,x,y,transfrom=None) :
        self.images = x
        self.labels = y
        self.transfrom = transfrom
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) :
        x = Image.fromarray(self.images[index])
        x = self.transfrom(x)
        y = self.labels[index]
        return x,y

import numpy as np
from torchvision.datasets import CIFAR100 
from pathlib import Path
import random

def get():
    path =Path('./data')
    train_data=CIFAR100(path, train=True, download=True)
    test_data=CIFAR100(path, train=False, download=True)
    taskcla = []
    num_tasks=10
    validation = 0.1
    class_order = np.arange(100).tolist()
    # class_order = [
    #     62, 54, 84, 20, 94, 22, 40, 29, 78, 27, 26, 79, 17, 76, 68, 88, 3, 19, 31, 21, 33, 60, 24, 14, 6, 10,
    #     16, 82, 70, 92, 25, 5, 28, 9, 61, 36, 50, 90, 8, 48, 47, 56, 11, 98, 35, 93, 44, 64, 75, 66, 15, 38, 97,
    #     42, 43, 12, 37, 55, 72, 95, 18, 7, 23, 71, 49, 53, 57, 86, 39, 87, 34, 63, 81, 89, 69, 46, 2, 1, 73, 32,
    #     67, 91, 0, 51, 83, 13, 58, 80, 74, 65, 4, 30, 45, 77, 99, 85, 41, 96, 59, 52
    # ]
    
    num_classes = len(class_order)
    cperTask = np.array([num_classes//num_tasks]*num_tasks)
    cperTask_sum = np.cumsum(cperTask)
    data ={}
    first_classNum_in_task = np.concatenate(([0],cperTask_sum[:-1]))
    for t in range(num_tasks):
        data[t] = {}
        data[t]['name'] = ''
        data[t]['train'] = {'x':[],'y':[]}
        data[t]['valid'] = {'x':[],'y':[]}
        data[t]['test'] = {'x':[],'y':[]}

    for image,label in zip(train_data.data,train_data.targets):
        this_label  = class_order.index(label)

        task_id = (this_label >= cperTask_sum).sum()  
        data[task_id]['train']['x'].append(image)
        data[task_id]['train']['y'].append(this_label-first_classNum_in_task[task_id])

    for image,label in zip(test_data.data,test_data.targets):
        this_label  = class_order.index(label)

        task_id = (this_label >= cperTask_sum).sum()  
        data[task_id]['test']['x'].append(image)
        data[task_id]['test']['y'].append(this_label-first_classNum_in_task[task_id])

    for tt in range(num_tasks):
        data[tt]['ncla'] =len(np.unique(data[tt]['train']['y']))
        assert data[tt]['ncla'] == cperTask[tt],"something went wrong splitting classes"

    if validation:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['train']['y'])==cc)[0])
                rnd_img = random.sample(cls_idx,int(np.round(len(cls_idx)*validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['valid']['x'].append(data[tt]['train']['x'][rnd_img[ii]])
                    data[tt]['valid']['y'].append(data[tt]['train']['y'][rnd_img[ii]])
                    data[tt]['train']['x'].pop(rnd_img[ii])
                    data[tt]['train']['y'].pop(rnd_img[ii])

    for tt in data.keys():
        for split in ['train','valid','test']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    return data,taskcla
