import torch
import torch.nn as nn
import numpy as np
from .utils import *
from dataset import normalize, denormalize

class mkrIDFD_FD(nn.Module):
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
        
        self.indim_id = {
            key: get_dim(config, key) for key in self.inkeys_id
        }
        self.outdim_id = {
            key: get_dim(config, key) for key in self.outkeys_id
        }
        self.outdim_offset_id = np.cumsum([v[1] for k, v in self.outdim_id.items()])
        
        self.indim_fd = {
            key: get_dim(config, key) for key in self.inkeys_fd
        }
        self.outdim_fd = {
            key: get_dim(config, key) for key in self.outkeys_fd
        }
        
        self.ID_inProj  = nn.Linear(sum([v[0][1] * v[0][2] for k, v in self.indim_id.items()]), self.mconfig.dim)
        self.ID_query   = nn.Parameter(torch.randn(1, 1, self.mconfig.dim).float())
        decoder_layer   = nn.TransformerDecoderLayer(d_model=self.mconfig.dim, nhead=self.mconfig.num_head, dim_feedforward=self.mconfig.dim * 2)
        self.ID_net     = nn.TransformerDecoder(decoder_layer, num_layers=self.mconfig.num_layers, norm=norm_dict[self.mconfig.norm](self.mconfig.dim))
        # self.ID_net     = nn.MultiheadAttention(self.mconfig.dim, self.mconfig.num_head, dropout=0.1)
        self.ID_outProj = nn.Linear(self.mconfig.dim, self.outdim_offset_id[-1])
        self.ID_post = nn.ModuleDict({
            k: get_postprocess(k) for k in self.outkeys_id
        })
        
        if self.with_cls:
            self.cls = nn.Linear(self.mconfig.dim, 1)
        
        self.FD_inProj  = nn.ModuleDict({
            k: self.get_inProj(k, v) for k, v in self.indim_fd.items()
        })
        # self.FD_net = nn.MultiheadAttention(self.mconfig.dim, self.mconfig.num_head, dropout=0.1)
        self.FD_net = nn.TransformerDecoder(decoder_layer, num_layers=self.mconfig.num_layers, norm=norm_dict[self.mconfig.norm](self.mconfig.dim))
        self.FD_outProj = nn.ModuleDict({
            k: self.get_outProj(k, v) for k, v in self.outdim_fd.items()
        })
    
    def get_inProj(self, k, v):
        if k in ['mkr_pre', 'mkr']:
            return nn.Linear(v[0][1] * v[0][2], self.mconfig.dim)
        elif k in ['torque', 'grf_pre']:
            return nn.Linear(v[1], self.mconfig.dim)
        else:
            print(k)
            raise NotImplementedError
    
    def get_outProj(self, k, v):
        if k == 'mkr_post':
            return nn.Linear(self.mconfig.dim, v[0][1] * v[0][2])
        else:
            raise NotImplementedError
    
    def forward(self, batch):
        output = {}
        id_input = []
        for key in self.inkeys_id:
            id_input.append(batch[key].flatten(2))
        id_input = torch.cat(id_input, dim=-1).permute(1, 0, 2) # J, N, L*(3+3)
        id_input = self.ID_inProj(id_input)
        id_query = self.ID_query.expand(-1, id_input.shape[1], -1)
        id_feat  = self.ID_net(id_query, id_input)
        
        if self.with_cls:
            output['indicator'] = self.cls(id_feat).flatten()

            b       = batch['indicator'].shape[0] // 2
            id_feat = id_feat[:b]
            for key in self.inkeys_id:
                if key != 'indicator' and batch[key].shape[0] > b:
                    batch[key] = batch[key][:b]
        
        id_output = self.ID_outProj(id_feat)[0]
        offset = 0
        for key, v in self.outdim_id.items():
            output[key] = self.ID_post[key](id_output[:, offset:offset + v[1]].view(id_output.shape[0], *v[0]))
            offset      = offset + v[1]
        
        for key in self.outkeys_copy:
            if key == 'tornorm':
                std, mean = batch['torque_stdmean']
                output['tornorm'] = torch.linalg.vector_norm(denormalize(batch['torque'], mean, std), dim=-1)[..., None]
            elif key == 'grf':
                output['grf']     = batch['grf']
            elif key == 'torque':
                output['torque']  = batch['torque']

        if 'torque' not in output:
            output['torque'] = output['tornorm'] * output['torvec']
            std, mean = batch['torque_stdmean']
            output['torque'] = normalize(output['torque'], mean, std)
        if 'grf' not in output:
            output['grf'] = output['grfnorm'] * output['grfvec']
            std, mean = batch['grf_stdmean']
            output['grf'] = normalize(output['grf'], mean, std)
        fd_input = {}
        bz       = id_feat.shape[0]
        num_good = batch['torque'].shape[0]
        if batch.get('gt_ratio', 0.8) < 1:
            idx = torch.randperm(num_good)[int(num_good * batch.get('gt_ratio', 0.8)):]

        for key in self.inkeys_fd:

            fd_input[key] = batch[key].flatten(2) # N, J, L*3
            if key == 'torque':
                if batch.get('gt_ratio', 0.8) < 1:
                    fd_input[key][idx] = output['torque'][idx].flatten(2)
                fd_input[key] = torch.cat([fd_input[key], output['torque'][num_good:].flatten(2)])
            elif key == 'grf_pre':
                if batch.get('gt_ratio', 0.8) < 1:
                    fd_input[key][idx] = output['grf'][idx, :1].flatten(2)
                fd_input[key] = torch.cat([fd_input[key], output['grf'][num_good:, :1].flatten(2)])
            fd_input[key] = self.FD_inProj[key](fd_input[key])
        fd_feat            = torch.cat([v for v in fd_input.values()], dim=1).permute(1, 0, 2)
        fd_output          = self.FD_net(fd_input['mkr_pre'].permute(1, 0, 2), fd_feat)
        output['mkr_post'] = self.FD_outProj['mkr_post'](fd_output).permute(1, 0, 2)[:, None]
        
        return output