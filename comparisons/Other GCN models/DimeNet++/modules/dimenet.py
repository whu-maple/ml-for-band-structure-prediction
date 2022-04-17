import torch
import torch.nn as nn

from modules.activations import swish
from modules.bessel_basis_layer import BesselBasisLayer
from modules.spherical_basis_layer import SphericalBasisLayer
from modules.embedding_block import EmbeddingBlock
from modules.output_block import OutputBlock
from modules.interaction_block import InteractionBlock

class DimeNet(nn.Module):
    """
    DimeNet model.

    Parameters
    ----------
    emb_size
        Embedding size used throughout the model
    num_blocks
        Number of building blocks to be stacked
    num_bilinear
        Third dimension of the bilinear layer tensor
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    cutoff
        Cutoff distance for interatomic interactions
    envelope_exponent
        Shape of the smooth cutoff
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    output_init
        Initial function in output block
    """
    def __init__(self,
                 emb_size,
                 num_blocks,
                 num_bilinear,
                 num_spherical,
                 num_radial,
                 cutoff=5.0,
                 envelope_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2,
                 num_dense_output=3,
                 num_targets=12,
                 activation=swish,
                 output_init=nn.init.zeros_):
        super(DimeNet, self).__init__()

        self.num_blocks = num_blocks
        self.num_radial = num_radial

        # cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(num_radial=num_radial,
                                          cutoff=cutoff,
                                          envelope_exponent=envelope_exponent)

        self.sbf_layer = SphericalBasisLayer(num_spherical=num_spherical,
                                             num_radial=num_radial,
                                             cutoff=cutoff,
                                             envelope_exponent=envelope_exponent)
        
        # embedding block
        self.emb_block = EmbeddingBlock(emb_size=emb_size,
                                        num_radial=num_radial,
                                        bessel_funcs=self.sbf_layer.get_bessel_funcs(),
                                        cutoff=cutoff,
                                        envelope_exponent=envelope_exponent,
                                        activation=activation)
        
        # output block
        self.output_blocks = nn.ModuleList({
            OutputBlock(emb_size=emb_size,
                        num_radial=num_radial,
                        num_dense=num_dense_output,
                        num_targets=num_targets,
                        activation=activation,
                        output_init=output_init) for _ in range(num_blocks + 1)
        })

        # interaction block
        self.interaction_blocks = nn.ModuleList({
            InteractionBlock(emb_size=emb_size,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             num_bilinear=num_bilinear,
                             num_before_skip=num_before_skip,
                             num_after_skip=num_after_skip,
                             activation=activation) for _ in range(num_blocks)
        })
    
    def edge_init(self, edges):
        # Calculate angles k -> j -> i
        R1, R2 = edges.src['o'], edges.dst['o']
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # Transform via angles
        cbf = [f(angle) for f in self.sbf_layer.get_sph_funcs()]
        cbf = torch.stack(cbf, dim=1)  # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [None, 42]
        sbf = edges.src['rbf_env'] * cbf  # [None, 42]
        return {'sbf': sbf}
    
    def forward(self, g, l_g):
        # add rbf features for each edge in one batch graph, [num_radial,]
        g = self.rbf_layer(g)
        # Embedding block
        g = self.emb_block(g)
        # Output block
        P = self.output_blocks[0](g)  # [batch_size, num_targets]
        # Prepare sbf feature before the following blocks
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.apply_edges(self.edge_init)
        # Interaction blocks
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g, l_g)
            P += self.output_blocks[i + 1](g)
        
        return P