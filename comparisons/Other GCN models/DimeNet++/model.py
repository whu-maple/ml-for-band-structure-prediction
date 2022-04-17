import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
from modules.dimenet_pp import DimeNetPP


class Envelope(nn.Module):
    """
    Envelope function to scale the Bessel Basis functions
    more details can be found in https://arxiv.org/abs/2003.03123v1
    """
    def __init__(self, exponent):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):

        x_p_0 = x.pow(self.p - 1)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = 1 / x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2

        return env_val


class BesselBasisLayer(nn.Module):
    """
    Bessel Basis functions to expand the distances to feature vectors 
    """
    def __init__(self,
                 num_radial,
                 cutoff = 20.,
                 envelope_exponent = 6,):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    def reset_params(self):

        self.frequencies.data = torch.arange(1., torch.numel(self.frequencies) + 1.,).mul_(np.pi)

    def forward(self, d):

        d_scaled = d / self.cutoff
        d_cutoff = self.envelope(d_scaled)

        return d_cutoff * torch.sin(self.frequencies * d_scaled)
    

class MLP(nn.Module):
    """
    General multilayer perceptron model
    """
    def __init__(self, dim_list, activation,):
        super(MLP, self).__init__()
        
        self.num_layers = len(dim_list) - 1
        self.activation = activation
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(dim_list[i], dim_list[i+1]))

    def forward(self, x):

        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))

        return self.layers[-1](x)


class OnsiteNN(nn.Module):
    """
    The MLP used to obtain the on-site terms
    """
    def __init__(self, dim_list, activation = F.relu):
        super(OnsiteNN, self).__init__()

        self.onsite_mlp = MLP(dim_list, activation)

    def forward(self, feat):

        return self.onsite_mlp(feat)


class HoppingNN(nn.Module):
    """
    The MLPs used to obtain the hopping terms
    """
    def __init__(self,
                 dim_list1,
                 dim_list2,
                 activation = F.relu,):
        super(HoppingNN, self).__init__()

        self.hopping_mlp1 = MLP(dim_list1, activation)
        self.hopping_mlp2 = MLP(dim_list2, activation)

    def forward(self, feat, hopping_index, d, ex_d):
        
        hopping_feat = torch.sum(self.hopping_mlp1(feat)[hopping_index], dim = 1)
        hopping_feat = torch.cat([hopping_feat / (d ** 2), ex_d], 1)

        return self.hopping_mlp2(hopping_feat)
    
    
class WHOLEMODEL(nn.Module):
    
    def __init__(self,  
                  gnn_emb_size,
                  gnn_out_emb_size,
                  gnn_int_emb_size,
                  gnn_basis_emb_size,
                  gnn_num_blocks,
                  gnn_num_spherical,
                  gnn_num_radial,
                  onsite_dim_list,
                  hopping_dim_list1,
                  hopping_dim_list2,
                  expander_bessel_dim,
                  expander_bessel_cutoff,
                  expander_bessel_exponent = 6,
                  onsite_activation = F.relu, 
                  hopping_activation = F.relu,): 
        
        super(WHOLEMODEL, self).__init__()
        
        self.atomic_init_dim = 40
        self.atomic_feat = nn.Embedding(120, self.atomic_init_dim) 
            
        self.gnn=DimeNetPP(
                emb_size = gnn_emb_size,   
                out_emb_size = gnn_out_emb_size,  
                int_emb_size = gnn_int_emb_size,  
                basis_emb_size = gnn_basis_emb_size,  
                num_blocks = gnn_num_blocks, 
                num_spherical = gnn_num_spherical,   
                num_radial = gnn_num_radial,)
        
        self.onn = OnsiteNN(onsite_dim_list, onsite_activation)
        self.hnn = HoppingNN(hopping_dim_list1, hopping_dim_list2, hopping_activation)
        self.expander = BesselBasisLayer(expander_bessel_dim, expander_bessel_cutoff, expander_bessel_exponent)
        self.cutoff = expander_bessel_cutoff
        
    def forward(self, bg, transform, d):
        
        feat = self.atomic_feat(bg.ndata["species"]).reshape([-1,self.atomic_init_dim])
        
        l_bg = dgl.line_graph(bg, backtracking=False)
        feat = self.gnn(bg, l_bg)
        
        carbon_feat = feat[torch.where(bg.ndata["species"]==6)]
        o = self.onn(carbon_feat)
        h = self.hnn(carbon_feat, transform, d, self.expander(d))
     
        return o, torch.where(d <= self.cutoff, h, 0.)
        
