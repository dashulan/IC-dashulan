import numpy as np
import torch.nn as nn
import torch 
from typing import OrderedDict
from torch.nn.functional import relu,avg_pool2d
import torch.nn.functional as F


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()

        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y

    def get_representation_matrix_ResNet18 (self, device, train_dataset): 
        # Collect activations by forward pass
        self.eval()
        r=np.arange(len(train_dataset))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(device)
        b=r[0:100] # ns=100 examples 
        # example_data = train_dataset[b][0].view(-1,28*28)
        example_data = train_dataset[b][0]
        example_data = example_data.to(device)
        example_out  = self(example_data)
        
        act_list =[]
        act_list.extend([self.act['conv_in'], 
            self.layer1[0].act['conv_0'], self.layer1[0].act['conv_1'], self.layer1[1].act['conv_0'], self.layer1[1].act['conv_1'],
            self.layer2[0].act['conv_0'], self.layer2[0].act['conv_1'], self.layer2[1].act['conv_0'], self.layer2[1].act['conv_1'],
            self.layer3[0].act['conv_0'], self.layer3[0].act['conv_1'], self.layer3[1].act['conv_0'], self.layer3[1].act['conv_1'],
            self.layer4[0].act['conv_0'], self.layer4[0].act['conv_1'], self.layer4[1].act['conv_0'], self.layer4[1].act['conv_1']])

        batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
        # network arch 
        stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
        map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
        in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 

        pad = 1
        sc_list=[5,9,13]
        p1d = (1, 1, 1, 1)
        mat_final=[] # list containing GPM Matrices 
        mat_list=[]
        mat_sc_list=[]
        for i in range(len(stride_list)):
            if i==0:
                ksz = 3
            else:
                ksz = 3 
            bsz=batch_list[i]
            st = stride_list[i]     
            k=0
            s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
            mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
            act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                        k +=1
            mat_list.append(mat)
            # For Shortcut Connection
            if i in sc_list:
                k=0
                s=compute_conv_output_size(map_list[i],1,stride_list[i])
                mat = np.zeros((1*1*in_channel[i],s*s*bsz))
                act = act_list[i].detach().cpu().numpy()
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                            k +=1
                mat_sc_list.append(mat) 

        ik=0
        for i in range (len(mat_list)):
            mat_final.append(mat_list[i])
            if i in [6,10,14]:
                mat_final.append(mat_sc_list[ik])
                ik+=1

        # print('-'*30)
        # print('Representation Matrix')
        # print('-'*30)
        # for i in range(len(mat_final)):
        #     print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        # print('-'*30)
        return mat_final    
    
    

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

