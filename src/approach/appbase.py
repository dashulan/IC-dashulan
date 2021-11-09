import torch
import torch.nn.functional as F


class AppBase:
    def __init__(self) -> None:
        self.optim = None
        self.train_config = None
        self.device = None
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        pass

    def train_epoch(self, t, model, trn_loader):
        """Runs a single epoch"""
        model.train()
        allloss, allnum = 0, 0
        for images, targets in trn_loader:
            outputs = model(images.to(self.device))

            loss = self.criterion(t, outputs, targets.to(self.device))
            self.optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optim.step()

            allloss += loss
            allnum += len(targets)
        return allloss / allnum

    def criterion(self, t, outputs, targets):
        return F.cross_entropy(outputs[t], targets)
