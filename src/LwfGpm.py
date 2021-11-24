from copy import deepcopy
import time
import torch
from torch.optim.sgd import SGD
from ICAppBase import ICBase
from Pminist import update_GPM
import torch.nn.functional as F
import numpy as np

from networks import resnetGpm
from networks.MLP import MLP

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


class App_lwfGpm(ICBase):
    def __init__(self, optim, model):
        super().__init__(optim, model)
        self.temp_model = None
        self.feature_list=[]
        self.proj = []
        self.threshold = np.array([0.965] * 20)


        self.lamba = torch.nn.parameter.Parameter(torch.tensor([0.5]*20,device=self.device).reshape(-1,1))
        self.model.register_parameter('lamba',self.lamba)
        self.model_old = deepcopy(model).to(self.device)
        self.optim = SGD(self.model.parameters(),lr=0.1)

    
    def setTempModel(self,model_state):
        self.temp_model = deepcopy(model_state)

    def getTempModel(self):
        return self.temp_model

    def train_post(self,trn_loader=None,**kwargs):
        self.adjust_lr()
        model_state = deepcopy(self.model.state_dict())
        self.model_old.load_state_dict(model_state)
        self.model_old.to(self.device)
        self.model_old.eval()

        model  = kwargs['model']
        if self.temp_model:
            model.load_state_dict(self.temp_model)
        self.lamba.data.fill_(0.5)
        act_mat = self.get_representation_matrix_ResNet18(model,trn_loader)
        self.update_GPM(act_mat)
        self.caculateProj()


    def eval(self,tst_loader,t):
        self.model.eval()
        acc_all,num_all,loss_all =0,0,0
        with torch.no_grad():
            for images,targets in tst_loader:
                images,targets = images.to(self.device),targets.to(self.device)
                outputs = self.model(images)
                loss = F.cross_entropy(outputs[t],targets)
                preds = torch.argmax(outputs[t],dim=1)
                acc_all +=(preds==targets).sum().item()
                loss_all+=loss
                num_all+=len(targets)
        return loss_all/num_all, acc_all/num_all


    def train_eopch(self, trn_loader, t):
        self.model.train()
        allloss = 0
        allnum=0
        for images, targets in trn_loader:
            # Forward old model

            # targets_old = None
            # if t > 0:
            #     targets_old = self.model_old(images.to(self.device))

            # Forward current model
            outputs = self.model(images.to(self.device))

            # loss = self.criterion(t,outputs,targets.to(self.device),targets_old)
            loss = self.criterion(t,outputs,targets.to(self.device))
         
            allloss+=loss
            allnum+=len(targets)

            self.optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            if self.proj:
                kk = 0 
                for k, (m,params) in enumerate(self.model.named_parameters()):
                    if len(params.size())==4:
                        sz =  params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            self.proj[kk]).view(params.size())
                        kk+=1
                    elif m=='lamba':
                        continue
                    elif len(params.size())==1 and t !=0:
                        params.grad.data.fill_(0)
            self.optim.step()
    
    def later_train_epoch(self,trn_loader,t):
        self.model.train()
        allloss = 0
        allnum=0
        for images, targets in trn_loader:
            outputs = self.model(images.to(self.device))

            loss = self.criterion_later(t,outputs,targets.to(self.device))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            allloss += loss
            allnum += len(targets)


    def cross_entropy(self,outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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

    def criterion(self, t,outputs, targets,outputs_old =None):
        # loss = 0
        # loss_reg=0
        # if t > 0:
        #     loss += 1* self.cross_entropy(torch.cat(outputs[:t],dim=1),torch.cat(outputs_old[:t],dim=1), exp=1.0 / 2)
        # return loss + loss_reg + torch.nn.functional.cross_entropy(outputs[t],targets)
        return torch.nn.functional.cross_entropy(outputs[t],targets)
    
    def criterion_later(self, t,outputs, targets):
        loss=0
        loss_reg=[]
        # loss_reg=0
        # loss_reg=0
        for n,p in self.model.named_parameters():
            if len(p.size())==4:
            # loss_reg += (1-torch.cosine_similarity(p.clone().reshape(-1,1),self.temp_model[n].reshape(-1,1),0))
                loss_reg.append((torch.sum((p-self.temp_model[n]).pow(2))/2).reshape(-1,1))
                # loss_reg += torch.sum((p-self.temp_model[n]).pow(2))/2
                
        loss_reg = torch.cat(loss_reg)
        loss_temp = torch.mm(loss_reg.t(),torch.max(torch.tensor([0.5]*20,device=self.device).reshape(-1,1),torch.exp(self.lamba)))

        # loss_temp  = self.model.lamba*loss_reg
        loss = torch.nn.functional.cross_entropy(outputs[t],targets)
        # maxL = loss+loss_reg
        # aph = loss_reg/maxL

        return loss_temp+loss
    
    def get_representation_matrix_ResNet18 (self,net, trn_loader): 
       
        net.eval()
        r=np.arange(len(trn_loader.dataset))
        np.random.shuffle(r)
        b=r[0:100] # ns=100 examples 
        datas = []
        for idx in b:
            a = trn_loader.dataset[idx]
            datas.append(a[0].reshape(1,3,32,32))
        example_data1 = np.concatenate(datas)
        example_data1 = torch.tensor(example_data1)
        example_data = example_data1.to(self.device)
        example_out  = net(example_data)

        
        act_list =[]
        act_list.extend([net.act['conv_in'], 
            net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
            net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
            net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
            net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

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

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_final)):
            print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print('-'*30)
        return mat_final    
        
    def update_GPM (self, mat_list):
        print ('Threshold: ', self.threshold) 
        if not self.feature_list:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<self.threshold[i]) #+1  
                self.feature_list.append(U[:,0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
                
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < self.threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                # update GPM
                Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    self.feature_list[i]=Ui
        
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            print ('Layer {} : {}/{}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-'*40)

    def caculateProj(self):
        feature_mat = []
            # Projection Matrix Precomputation
        for i in range(len(self.feature_list)):
            Uf=torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).to(self.device)
            print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            feature_mat.append(Uf)
        self.proj = feature_mat


    def train_loop_2(self,trn_loader,valid_loader,t,write=None):
        if t>0:
            self.adjust_lr()
            self.setTempModel(self.model.state_dict())
            self.model.load_state_dict(deepcopy(self.model_old.state_dict()))
            best_loss = np.inf
            lr = 0.1
            patience = 6
            for e in range(80):
                clock0 = time.time()
                self.later_train_epoch(trn_loader,t)
                clock1 = time.time()
                valid_loss,valid_acc = self.eval(valid_loader,t)
                    
                print(f"epoch {e:3d}, time = {clock1-clock0:5.1f}s | valid loss : {valid_loss:.5f} | valid acc : {valid_acc*100:.1f}% |",end='')
                write.add_scalar(f"task:{t}/Loss/valid",valid_loss,e)
                write.add_scalar(f"task:{t}/Accuracy/valid",valid_acc,e)
                    
                if valid_loss< best_loss:
                    best_loss=valid_loss
                    best_model=self.get_model()
                    patience=6
                    print(' *',end='')
                # else:
                #     patience-=1
                #     if patience<=0:
                #         lr/=3
                #         print(' lr={:.1e}'.format(lr),end='')
                #         if lr<args.lr_min:
                #             print()
                #             break
                #         patience=6
                #         self.adjust_lr(lr)
                print()
                    
                if e==40 or e==60:
                    lr/=10
                    self.adjust_lr(lr)

            self.set_model_(best_model)