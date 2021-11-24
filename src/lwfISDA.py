from copy import deepcopy
import torch
from torch.optim.sgd import SGD
from ICAppBase import ICBase
import torch.nn.functional as F
from ISDA import ISDALoss
import torch.nn as nn

class App_lwfIsda(ICBase):
    def __init__(self, optim, model):
        super().__init__(optim, model)
        self.model_old = deepcopy(model).to(self.device)
        self.isda_criterion = ISDALoss(self.model.features_num,100)


    
    
    def train_eopch(self, trn_loader, t,e=0):

        self.model.train()

        for i, (images, targets) in enumerate(trn_loader):
            images,targets = images.to(self.device),targets.to(self.device)


            loss, output = self.isda_criterion(self.model, self.model.linear[t], images, targets,t)
            loss = F.cross_entropy(outputs,targets)
            # fea = self.model(images,True)
            # outputs = self.fc[t](fea)
            # loss = F.cross_entropy(outputs,targets)

            self.optim.zero_grad()
            loss.backward()

            self.optim.step()


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
        loss = 0
        if t > 0:
            loss += 1* self.cross_entropy(torch.cat(outputs[:t],dim=1),torch.cat(outputs_old[:t],dim=1), exp=1.0 / 2)
        return loss + torch.nn.functional.cross_entropy(outputs[t],targets)
    
    def train_post(self,trn_loader,**kwargs):
        model_state = deepcopy(self.model.state_dict())
        self.model_old.load_state_dict(model_state)
        self.model_old.to(self.device)
        self.model_old.eval()

        self.adjust_lr()
        # self.isda_criterion = ISDALoss(self.model.features_num,10)
        # self.freeze_model()
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False


          
    def eval(self,tst_loader,t):
        self.model.eval()
        acc_all,num_all,loss_all =0,0,0
        with torch.no_grad():
            for data,label in tst_loader:
                data,label = data.to(self.device),label.to(self.device)
                outputs = self.model(data,True)
                outputs = self.fc[t](outputs)
                loss = self.criterion(outputs,label)
                preds = torch.argmax(outputs,dim=1)
                acc_all +=(preds==label).sum().item()
                loss_all+=loss
                num_all+=len(label)
            return loss_all/num_all, acc_all/num_all