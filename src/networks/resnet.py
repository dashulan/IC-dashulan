# %%
from typing import ForwardRef, OrderedDict
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.functional import layer_norm
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
import torch
import torch.nn.functional as F

# from networks.resnet18 import ResNet

# __all__ = ['resnet32']

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count%2
        self.act[f"conv_{self.count}"] = x
        self.count +=1
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        
        self.act[f"conv_{self.count}"] =out
        self.count = self.count%2
        self.count+=1
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class Renset32(nn.Module):

    def __init__(self,block,layers,taskcla,num_class=10):

        self.inplanes = 16
        super(Renset32,self).__init__()
        # self.conv1 = conv3x3(3,16,1)
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(16)
        self.layer1 = self._make_layer(block,16,layers[0])
        self.layer2 = self._make_layer(block,32,layers[1],stride=2)
        self.layer3 = self._make_layer(block,64,layers[2],stride=2)
        
        self.avgpool = nn.AvgPool2d(8,stride=1)

        self.features = None
        
        # self.fc = nn.Linear(64*block.expansion,num_class)
        
        # self.head_var = 'fc'
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in taskcla:
            self.linear.append(nn.Linear(64 * block.expansion, n, bias=False))

        self.act = OrderedDict()
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    
    def _make_layer(self,block:BasicBlock,planes,blocks,stride=1):
        downsample = None
        if stride!=1 or  self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        pass
        layers = []
        layers.append(block(self.inplanes,planes*block.expansion,stride,downsample))
        self.inplanes = planes*block.expansion

        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz,3,32,32)
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        self.features = out
        tempout = out/out.norm(dim=1).view(-1,1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](tempout))
        # x = self.fc(x)
        return y

    def linear_head(self,x):
        y = []
        for t,i in self.taskcla:
            y.append(self.linear[t](x))
        return y

    def getActList(self):
        act_list =[]
        act_list.append(self.act['conv_in'])
        for l in range(1,4):
            layer = f"layer{l}"
            layer = getattr(self,layer)
            for b in range(0,5):
                block = layer[b]
                act_list.append(block.act['conv_0'])
                act_list.append(block.act['conv_1'])
        return act_list

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
        act_list.append(self.act['conv_in'])
        for l in range(1,4):
            layer = f"layer{l}"
            layer = getattr(self,layer)
            for b in range(0,5):
                block = layer[b]
                act_list.append(block.act['conv_0'])
                act_list.append(block.act['conv_1'])
                

        # act_list.extend([self.act['conv_in'], 
        #     self.layer1[0].act['conv_0'], self.layer1[0].act['conv_1'], self.layer1[1].act['conv_0'], self.layer1[1].act['conv_1'],
        #     self.layer2[0].act['conv_0'], self.layer2[0].act['conv_1'], self.layer2[1].act['conv_0'], self.layer2[1].act['conv_1'],
        #     self.layer3[0].act['conv_0'], self.layer3[0].act['conv_1'], self.layer3[1].act['conv_0'], self.layer3[1].act['conv_1']])
        



        batch_list =[]
        batch_list.extend([10]*31)
        # batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
        # network arch 
        # stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
        # map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
        # in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 
        in_channel = []
        in_channel.extend([3])
        in_channel.extend([16]*11)
        in_channel.extend([32]*10)
        in_channel.extend([64]*9)
        
        stride_list = []
        stride_list.extend([1])
        stride_list.extend([1]*10)
        stride_list.extend([2])
        stride_list.extend([1]*9)
        stride_list.extend([2])
        stride_list.extend([1]*9)
        
        map_list = []
        map_list.extend([32]*12)
        map_list.extend([16]*10)
        map_list.extend([8]*9)


        pad = 1
        # sc_list=[5,9,13]
        sc_list=[11,21]
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
            if i in [12,22]:
                mat_final.append(mat_sc_list[ik])
                ik+=1

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_final)):
            print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print('-'*30)
        return mat_final    

def resenet3232(taskcla):
    return Renset32(BasicBlock,[5,5,5],taskcla)



# x = torch.arange(64*32*32*3,dtype=torch.float32).reshape(64,3,32,32)
# y = torch.arange(64)
# from torch.utils.data import TensorDataset
# net = resenet3232()
# datasets = TensorDataset(x,y)
# x = torch.arange(1*32*32*3,dtype=torch.float32).reshape(1,3,32,32)
# net.get_representation_matrix_ResNet18('cpu',datasets)
# net(x)
# len(net.getActList())
