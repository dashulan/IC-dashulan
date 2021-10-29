import torch
from torch._C import Decl
from torch.cuda import init

from clients import ClientsGroup
from networks.MLP import MLP
import torch.nn.functional as F
from torch import  optim
import tqdm

model = MLP(200)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_func = F.cross_entropy

opti = optim.SGD(model.parameters(),lr=0.01)

myClients = ClientsGroup('mnist',10,)
testDataLoader =None 


num_in_comm = int(1)

global_parameters = {}

for key,var in model.state_dict().items():
    global_parameters[key] =var.clone()

sum_parameters = None


class Server:
    def __init__(self) -> None:
        pass

    def train_clients(self):
        tasks = 10
        rounds = 10
        for t in range(tasks):
            for i in range(rounds):

                clients_in_comm = []
                for client in tqdm(clients_in_comm):

                    local_parameters = myClients.clients_set[client].localUpdate(5,10,model,loss_func,opti,global_parameters)

                    if sum_parameters is None:
                        sum_parameters = {}    
                        for key,var in local_parameters.item():
                            sum_parameters[key] = var.clone()
                    else:
                        for var in sum_parameters:
                            sum_parameters[var] 

                for var in global_parameters:
                    global_parameters[var] = (sum_parameters[var]/num_in_comm)
    
    
                    
