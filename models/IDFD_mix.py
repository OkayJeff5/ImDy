import torch
import torch.nn as nn
import numpy as np
from .utils import *
from dataset import normalize, denormalize

class mixIDFD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config     = config
        self.mconfig    = config.transformer
        self.inkeys_id  = self.mconfig.get('inkeys_id', ['mkr', 'mvel'])
        self.outkeys_id = self.mconfig.get('outkeys_id', ['tornorm', 'torvec', 'grf'])
        self.inkeys_fd  = self.mconfig.get('inkeys_fd', ['mkr_pre', 'torque', 'grf_pre'])
        self.outkeys_fd = self.mconfig.get('outkeys_fd', ['mkr_post'])
        self.outkeys_copy = self.mconfig.get('outkeys_copy', [])
        self.with_cls   = self.mconfig.get('cls', False)
        
        
        self.ID_inProj  = nn.Linear(self.mconfig.indim, self.mconfig.dim)
        self.ID_query   = nn.Parameter(torch.randn(1, 1, self.mconfig.dim).float())
        decoder_layer   = nn.TransformerDecoderLayer(d_model=self.mconfig.dim, nhead=self.mconfig.num_head, dim_feedforward=self.mconfig.dim * 2)
        self.ID_net     = nn.TransformerDecoder(decoder_layer, num_layers=self.mconfig.num_layers, norm=norm_dict[self.mconfig.norm](self.mconfig.dim))
        self.ID_adb     = nn.Linear(self.mconfig.dim, 29)
        self.ID_imdy    = nn.Linear(self.mconfig.dim, 81)
        
        self.FD_in_adb  = nn.Linear(23, self.mconfig.dim)
        self.FD_in_imdy = nn.Linear(75, self.mconfig.dim)
        # self.FD_net = nn.MultiheadAttention(self.mconfig.dim, self.mconfig.num_head, dropout=0.1)
        
        layers  = []
        indim   = self.mconfig.dim * 2
        for unit in self.mconfig.units:
            layers.append(nn.Linear(indim, unit))
            layers.append(acti_dict[self.mconfig.acti]())
            layers.append(norm_dict[self.mconfig.norm](unit))
            indim = unit
        self.FD_net = nn.Sequential(*layers)
        self.FD_adb = nn.Linear(indim, 23)
        self.FD_imdy = nn.Linear(indim, 72)
    
    def forward(self, batch):
        output = {}
        if len(batch['mkr_imdy']) > 0:
            id_in_imdy = torch.cat((batch['mkr_imdy'].flatten(2), batch['mvel_imdy'].flatten(2)), dim=-1).permute(1, 0, 2)
            id_in_imdy = self.ID_inProj(id_in_imdy)
            id_q_imdy  = self.ID_query.expand(-1, id_in_imdy.shape[1], -1)
            id_f_imdy  = self.ID_net(id_q_imdy, id_in_imdy)[0]
            id_imdy    = self.ID_imdy(id_f_imdy)
            output['torque_imdy'] = id_imdy[..., :69].view(-1, 1, 23, 3)
            output['grf_imdy']    = id_imdy[..., 69:].view(-1, 2, 2, 3)
        
        if len(batch['mkr_adb']) > 0:
            id_in_adb = torch.cat((batch['mkr_adb'].flatten(2), batch['mvel_adb'].flatten(2)), dim=-1).permute(1, 0, 2)
            id_in_adb = self.ID_inProj(id_in_adb)
            id_q_adb  = self.ID_query.expand(-1, id_in_adb.shape[1], -1)
            id_f_adb  = self.ID_net(id_q_adb, id_in_adb)[0]
            id_adb    = self.ID_adb(id_f_adb)
            output['torque_adb'] = id_adb[..., :17]
            output['grf_adb']    = id_adb[..., 17:].view(-1, 2, 2, 3)
        
        if self.training:
            if len(batch['mkr_imdy']) > 0:
                fd_in_imdy = torch.cat((self.FD_in_imdy(id_imdy[..., :75]), id_f_imdy), dim=-1)
                fd_f_imdy  = self.FD_net(fd_in_imdy)
                fd_imdy    = self.FD_imdy(fd_f_imdy)
                output['fd_imdy'] = fd_imdy
            if len(batch['mkr_adb']) > 0:
                fd_in_adb  = torch.cat((self.FD_in_adb(id_adb[..., :23]), id_f_adb), dim=-1)
                fd_f_adb   = self.FD_net(fd_in_adb)
                fd_adb     = self.FD_adb(fd_f_adb)
                output['fd_adb']  = fd_adb
        
        return output