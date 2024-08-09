import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, GINConv as BaseGINConv

class GIN(nn.Module):
    def __init__(self,args,input_dim):
        super(GIN,self).__init__()

        self.layers = args.conv_layers
        self.dropout = args.dropout
        self.pooling = self.load_pool(args.pooling_type)
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.convs = torch.nn.ModuleList()

        for layer in range(self.layers):
            if layer == 0:
                nn = Sequential(Linear(input_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            elif layer != (self.layers-1):
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.hidden_dim, self.out_dim), ReLU(), Linear(self.out_dim, self.out_dim))

            conv = GINConv(nn)
            self.convs.append(conv)

    def forward(self,x,edge_idx,graph_idx,edge_atten=None):  #graph_idx: [0,0,0,...,127,127]  which features in x belong to which graphs

        for layer in range(self.layers):
            x = F.relu(self.convs[layer](x, edge_idx,edge_atten=edge_atten))
            x = F.dropout(x,p=self.dropout,training=self.training)

        return self.pooling(x,graph_idx), x

    def load_pool(self,type):
        if type == 'max':
             return global_max_pool
        elif type == 'sum':
            return global_add_pool
        elif type == 'avg':
            return global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(type))

class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j
















