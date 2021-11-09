import torch
import torch.nn as nn
import torch.nn.functional as F
class ICBase():
    def __init__(self,optim,model):
        super(ICBase).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.optim = optim
        self.model = model
    
    def train_eopch(self,trn_loader,t):
        
        self.model.train()
        for data,label in trn_loader:
            data,label = data.to(self.device),label.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs[t],label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
    def eval(self,tst_loader,t):
        self.model.eval()
        acc_all,num_all,loss_all =0,0,0
        with torch.no_grad():
            for data,label in tst_loader:
                data,label = data.to(self.device),label.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs,label)
                preds = torch.argmax(outputs[t],dim=1)
                acc_all +=(preds==label).sum().item()
                loss_all+=loss
                num_all+=len(label)
            return loss_all/num_all, acc_all/num_all
                

    def adjust_lr(self,lr=0.1):
        self.optim.param_groups[0]['lr']=lr

    def criterion(self,ouputs,targets):
        return F.cross_entropy(ouputs,targets)