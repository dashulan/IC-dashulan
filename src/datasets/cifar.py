from torchvision.datasets import CIFAR100 
from pathlib import Path
import matplotlib.pyplot as plt
path =Path('./data')
import ssl
 
# ssl._create_default_https_context = ssl._create_unverified_context


tvcifar_trn = CIFAR100(path, train=True, download=True)
tvcifar_tst = CIFAR100(path, train=False, download=True)
trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}

img =trn_data['x'][0]
print(tvcifar_trn.data.shape)
plt.imshow(img)
plt.show()