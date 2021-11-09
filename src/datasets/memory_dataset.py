from torch.utils.data import Dataset
from PIL import Image

class MemoryDataset(Dataset):

    def __init__(self,data,transfrom) :
        self.images = data['x']
        self.labels = data['y']
        self.transfrom = transfrom
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) :
        # x = Image.fromarray(self.images[index])
        x = self.images[index]
        x = self.transfrom(x)
        y = self.labels[index]
        return x,y

