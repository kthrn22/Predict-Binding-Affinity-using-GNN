from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch.nn.functional import softmax, sigmoid, relu
from torch.nn import Parameter, ModuleList, BatchNorm1d
from torch import sigmoid, exp
from torch_geometric.nn import inits
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import torch

class Gate_Augmented_Graph_Attention_Layer(MessagePassing):    
    def __init__(self, in_channels, out_channels, add_self_loops):
        super(Gate_Augmented_Graph_Attention_Layer, self).__init__(aggr = "add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.Linear_W = Linear(in_channels, out_channels, bias = False)
        self.Linear_E = Linear(out_channels, out_channels, bias = False)
        self.Linear_U = Linear(2 * out_channels, 1)
        self.alpha = Parameter(torch.FloatTensor(1))
        self.beta = Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        inits.uniform(self.alpha.size(0), self.alpha)
        inits.uniform(self.beta.size(0), self.beta)
        self.Linear_W.reset_parameters()
        self.Linear_E.reset_parameters()
        self.Linear_U.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight = None):
        x = self.Linear_W(x)
        
        if self.add_self_loops:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight)

        alpha = self.edge_updater(edge_index, x = x)
        x_out = self.propagate(edge_index, x = x, alpha = alpha, edge_weight = edge_weight)
        z = self.gate(x, x_out)

        return z * x + (1 - z) * x_out
        #return x_out

    def edge_update(self, x_i, x_j):
        e =  torch.sum(x_i * self.Linear_E(x_j), dim = -1) + torch.sum(x_j * self.Linear_E(x_i), dim = -1)
        alpha = softmax(e, dim = -1) 
        return alpha

    def gate(self, x, x_out):
        concat = torch.cat([x, x_out], dim = 1)
        z = self.Linear_U(concat)
        return sigmoid(z)

    def message(self, x_j, alpha, edge_weight):

        if edge_weight is not None:
            weight = torch.Tensor([tensor if tensor == 1 else exp(-((tensor - self.alpha) ** 2) / self.beta) for tensor in edge_weight])
            weight = weight.view(edge_weight.shape)
            return weight.unsqueeze(-1) * alpha.unsqueeze(-1) * x_j
            
        return alpha.unsqueeze(-1) * x_j