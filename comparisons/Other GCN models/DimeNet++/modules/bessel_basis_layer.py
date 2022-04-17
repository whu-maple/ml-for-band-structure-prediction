import numpy as np
import torch
import torch.nn as nn

from modules.envelope import Envelope

class BesselBasisLayer(nn.Module):
    def __init__(self,
                 num_radial,
                 cutoff,
                 envelope_exponent,):
        super(BesselBasisLayer, self).__init__()
        
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    def reset_params(self):
        self.frequencies.data = torch.arange(1., torch.numel(self.frequencies) + 1.,).mul_(np.pi)

    def forward(self, g):
        d_scaled = g.edata['d'] / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        g.edata['rbf'] = d_cutoff * torch.sin(self.frequencies * d_scaled)
        return g
