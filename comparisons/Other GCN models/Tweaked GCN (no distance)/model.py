import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity


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
    
    
class GraphConvLayer(nn.Module):
    """
    The graph convolution layer used to implement message passing
    it can distinguish the edges with different types and distances
    and assign them to different message passing weights
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bessel_cutoff = 4.,
                 bessel_exponent = 6,
                 feat_drop = 0.,
                 attn_drop = 0.,
                 activation = None,
                 bias=True):
        super(GraphConvLayer, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias = False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias = False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias = False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size = (1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size = (1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.prelu = nn.PReLU(num_heads)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size = (num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)

        if self._in_dst_feats != out_feats:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias = False)
        else:
            self.res_fc = Identity()

        self.bessel = BesselBasisLayer(num_heads, bessel_cutoff, bessel_exponent)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain = gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain = gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain = gain)
            
        nn.init.xavier_normal_(self.attn_l, gain = gain)
        nn.init.xavier_normal_(self.attn_r, gain = gain)
        nn.init.constant_(self.bias, 0)
        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain = gain)

    def forward(self, graph, feat, get_weight = False):

        with graph.local_scope():
            
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                    
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            # coeff = self.bessel(graph.edata['distance'].reshape([-1, 1])).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim = -1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim = -1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.prelu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
                
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
                
            if self.activation:
                rst = self.activation(rst)
                
            if get_weight:
                
                return rst, graph.edata['a']
            
            else:
                
                return rst


class GraphNN(nn.Module):
    """
    The graph convolution neural network used to perform message passing
    """
    def __init__(self,
                 dim_list,
                 head_list,
                 bessel_cutoff = 4.,
                 bessel_exponent = 6,
                 feat_drop = 0.,
                 attn_drop = 0.,
                 activation = None,
                 bias = True):  
        super(GraphNN, self).__init__()
        
        self.num_layers = len(dim_list) - 1
        self.layers = nn.ModuleList()
        head_list = [1] + head_list

        for i in range(self.num_layers - 1):
           self.layers.append(GraphConvLayer(dim_list[i]*head_list[i], dim_list[i+1], 
                                             head_list[i+1], bessel_cutoff, bessel_exponent,
                                             feat_drop, attn_drop, activation, bias,))
           
        self.layers.append(GraphConvLayer(dim_list[-2]*head_list[-2], dim_list[-1], 
                           head_list[-1], bessel_cutoff, bessel_exponent,
                           feat_drop, attn_drop, None, bias,))

    def forward(self, g, inputs):
        
        for i in range(self.num_layers-1) :
            inputs = self.layers[i](g, inputs).flatten(1)
            
        return self.layers[-1](g, inputs).mean(1)


class WHOLEMODEL(nn.Module):
    """
    The implement of the whole NN framework presented in the manuscript
    """
    def __init__(self,  
                  gnn_dim_list,
                  gnn_head_list,
                  onsite_dim_list,
                  hopping_dim_list1,
                  hopping_dim_list2,
                  expander_bessel_dim,
                  expander_bessel_cutoff,
                  expander_bessel_exponent = 6,
                  gnn_bessel_cutoff = 4., 
                  gnn_bessel_exponent = 6, 
                  gnn_feat_drop = 0., 
                  gnn_attn_drop = 0., 
                  gnn_bias = True, 
                  gnn_activation = F.relu, 
                  onsite_activation = F.relu, 
                  hopping_activation = F.relu,): 
        
        super(WHOLEMODEL, self).__init__()
        
        self.atomic_init_dim = gnn_dim_list[0]
        self.atomic_feat = nn.Embedding(120, self.atomic_init_dim) 
        
        self.gnn = GraphNN(gnn_dim_list, 
                           gnn_head_list,
                           gnn_bessel_cutoff,
                           gnn_bessel_exponent,
                           gnn_feat_drop, 
                           gnn_attn_drop, 
                           gnn_activation,
                           gnn_bias)
        
        self.onn = OnsiteNN(onsite_dim_list, onsite_activation)
        self.hnn = HoppingNN(hopping_dim_list1, hopping_dim_list2, hopping_activation)
        self.expander = BesselBasisLayer(expander_bessel_dim, expander_bessel_cutoff, expander_bessel_exponent)
        self.cutoff = expander_bessel_cutoff
        
    def forward(self, bg, hopping_index, d):
   
        feat = self.atomic_feat(bg.ndata["species"]).reshape([-1,self.atomic_init_dim])
        feat = self.gnn(bg, feat)    
        carbon_feat = feat[torch.where(bg.ndata["species"] == 6)]
        o = self.onn(carbon_feat)
        h = self.hnn(carbon_feat, hopping_index, d, self.expander(d))
          
        return o, torch.where(d <= self.cutoff, h, 0.)
        
