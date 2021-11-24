from copy import deepcopy
import torch
from ICAppBase import ICBase
import torch.nn.functional as F
from loss import tripLoss
import  torch.profiler

class App_FT(ICBase):
    def __init__(self, optim, model):
        super().__init__(optim, model)
        self.triploss = tripLoss.TripletLoss()
        self.model_old = deepcopy(model).to(self.device)

    
    def train_eopch(self, trn_loader, t):
        self.model.train()
        for step,(images, targets) in enumerate(trn_loader):
            images,targets = images.to(self.device),targets.to(self.device)

            targets_old = None
            if t > 0:
                _,targets_old = self.model_old(images)

            features,outputs = self.model(images)

            # loss_trip = self.triploss(features,targets)
            # loss_trip = 0
            loss =  self.criterion(t,outputs,targets,targets_old)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def train_eopch2(self, trn_loader, t):
        self.model.train()
        for step,(images, targets) in enumerate(trn_loader):
            images,targets = images.to(self.device),targets.to(self.device)

            targets_old = None
            if t > 0:
                _,targets_old = self.model_old(images)

            features,outputs = self.model(images)

            loss_trip = self.triploss(features,targets)
            # loss_trip = 0
            loss = loss_trip + self.criterion(t,outputs,targets,targets_old)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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

    def criterion(self, t, outputs, targets, outputs_old=None):
        loss = 0
        if t > 0:
            loss += 1 * self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1),
                                           exp=1.0 / 2)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets)
    
    def train_post(self,trn_loader,**kwargs):
        model_state = deepcopy(self.model.state_dict())
        self.model_old.load_state_dict(model_state)
        self.model_old.to(self.device)
        self.model_old.eval()

        self.adjust_lr()


    def eval(self,tst_loader,t):
        self.model.eval()
        acc_all,num_all,loss_all =0,0,0
        with torch.no_grad():
            for images,targets in tst_loader:
                images,targets = images.to(self.device),targets.to(self.device)
                _,outputs=self.model(images)

                loss = F.cross_entropy(outputs[t],targets)

                preds = torch.argmax(outputs[t],dim=1)
                acc_all +=(preds==targets).sum().item()
                loss_all+=loss
                num_all+=len(targets)
            return loss_all/num_all, acc_all/num_all
  