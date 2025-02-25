import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import *

class naiveMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config  = config
        self.mconfig = config.MLP
        self.inkeys  = self.mconfig.inkeys
        self.outkeys = self.mconfig.outkeys
        
        self.indim   = {
            key: get_dim(config, key) for key in self.inkeys
        }
        self.outdim  = {
            key: get_dim(config, key) for key in self.outkeys
        }
        self.outdim_offset = np.cumsum([v[1] for k, v in self.outdim.items()])
        
        units  = self.mconfig.units
        layers = []
        indim  = sum([v[1] for k, v in self.indim.items()])
        for unit in units:
            layers.append(nn.Linear(indim, unit))
            layers.append(acti_dict[self.mconfig.acti]())
            layers.append(norm_dict[self.mconfig.norm](unit))
            indim = unit
        
        layers.append(nn.Linear(indim, self.outdim_offset[-1]))
        self.net = nn.Sequential(*layers)
        self.post = nn.ModuleDict({
            k: get_postprocess(k) for k in self.outkeys
        })

    def forward(self, batch):
        x = []
        for key in self.inkeys:
            x.append(batch[key].flatten(1))
        x = torch.cat(x, dim=-1)
        x = self.net(x)
        output = {}
        offset = 0
        for i, key in enumerate(self.outkeys):
            output[key] = self.post[key](x[:, offset:offset + self.outdim[key][1]].view(x.shape[0], *self.outdim[key][0]))
            offset = self.outdim_offset[i]
        return output