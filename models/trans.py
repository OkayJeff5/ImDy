import torch
import torch.nn as nn
import numpy as np
from .utils import *

class naiveTrans(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config  = config
        self.mconfig = config.transformer
        self.inkeys  = self.mconfig.inkeys
        self.outkeys = self.mconfig.outkeys
        
        self.indim   = {
            key: get_dim(config, key) for key in self.inkeys
        }
        self.outdim  = {
            key: get_dim(config, key) for key in self.outkeys
        }
        self.outoffset = np.cumsum([self.outdim[k][0][0] for k in self.outkeys])
        
        self.in_emb   = nn.ParameterDict({
            k: nn.Parameter(torch.randn(1, self.indim[k][0][0], self.mconfig.dim).float()) for k in self.inkeys
        })
        self.out_emb  = nn.ParameterDict({
            k: nn.Parameter(torch.randn(1, self.outdim[k][0][0], self.mconfig.dim).float()) for k in self.outkeys
        })
        self.in_proj  = nn.ModuleDict({
            k: nn.Linear(self.indim[k][0][1] * self.indim[k][0][2], self.mconfig.dim) for k in self.inkeys
        })
        self.out_proj = nn.ModuleDict({
            k: nn.Linear(self.mconfig.dim, self.outdim[k][0][1] * self.outdim[k][0][2]) for k in self.outkeys
        })
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.mconfig.dim, nhead=self.mconfig.num_head)
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_layers=self.mconfig.num_layers, norm=norm_dict[self.mconfig.norm](self.mconfig.dim))
        self.post = nn.ModuleDict({
            k: get_postprocess(k) for k in self.outkeys
        })

    def forward(self, batch):
        x = []
        for key in self.inkeys:
            x.append(self.in_proj[key](batch[key].flatten(2)) + self.in_emb[key].expand(batch[key].shape[0], -1, -1))
        x = torch.cat(x, dim=1).permute(1, 0, 2) # L, B, C
        q = torch.cat([self.out_emb[k].expand(x.shape[1], -1, -1) for k in self.outkeys], dim=1).permute(1, 0, 2)
        y = self.decoder(q, x)
        offset = 0
        output = {}
        for i, key in enumerate(self.outkeys):
            xx = self.out_proj[key](y[offset:offset + self.outdim[key][0][0]]).permute(1, 0, 2)
            output[key] = self.post[key](xx.view(y.shape[1], *self.outdim[key][0]))
            offset = self.outoffset[i]
        return output