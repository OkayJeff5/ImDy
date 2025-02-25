import torch
import torch.nn as nn
import numpy as np
from .utils import *
from dataset import normalize, denormalize

class IDFD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config     = config
        self.mconfig    = config.MLP
        self.inkeys_id  = self.mconfig.get('inkeys_id', ['rot', 'pos'])
        self.outkeys_id = self.mconfig.get('outkeys_id', ['torque', 'grf'])
        self.inkeys_fd  = self.mconfig.get('inkeys_fd', ['rot_pre', 'pos_pre', 'torque', 'grf_pre'])
        self.outkeys_fd = self.mconfig.get('outkeys_fd', ['rot_post', 'pos_post'])
        self.outkeys_copy = self.mconfig.get('outkeys_copy', [])
        self.with_intermediate = self.mconfig.get('intermediate', False)
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
        self.outdim_offset_fd = np.cumsum([v[1] for k, v in self.outdim_fd.items()])
        
        
        units  = self.mconfig.units
        layers = []
        indim  = sum([v[1] for k, v in self.indim_id.items()])
        for unit in units:
            layers.append(nn.Linear(indim, unit))
            layers.append(acti_dict[self.mconfig.acti]())
            layers.append(norm_dict[self.mconfig.norm](unit))
            indim = unit
        self.ID      = nn.Sequential(*layers)
        self.ID_head = nn.Linear(indim, self.outdim_offset_id[-1])
        self.post = nn.ModuleDict({
            k: get_postprocess(k) for k in self.outkeys_id
        })
        
        if self.with_cls:
            self.cls = nn.Linear(indim, 1)
        
        layers  = []
        indim  = sum([v[1] for k, v in self.indim_fd.items()])
        if self.with_intermediate:
            indim += units[-1]
        for unit in units:
            layers.append(nn.Linear(indim, unit))
            layers.append(acti_dict[self.mconfig.acti]())
            layers.append(norm_dict[self.mconfig.norm](unit))
            indim = unit
        self.FD = nn.Sequential(*layers)
        self.FD_head = nn.Linear(indim, self.outdim_offset_fd[-1])

    def forward(self, batch):
        output = {}
        id_input = []
        for key in self.inkeys_id:
            id_input.append(batch[key].flatten(1))
        id_input  = torch.cat(id_input, dim=-1)
        id_feat   = self.ID(id_input)
        
        if self.with_cls:
            output['indicator'] = self.cls(id_feat).flatten()
            if self.training:
                b       = batch['indicator'].shape[0] // 2
                id_feat = id_feat[:b]
                for key in self.inkeys_id:
                    if key != 'indicator' and batch[key].shape[0] > b:
                        batch[key] = batch[key][:b]
        
        id_output = self.ID_head(id_feat)
        offset = 0
        for i, key in enumerate(self.outkeys_id):
            output[key] = self.post[key](id_output[:, offset:offset + self.outdim_id[key][1]].view(id_feat.shape[0], *self.outdim_id[key][0]))
            offset      = self.outdim_offset_id[i]
        
        for key in self.outkeys_copy:
            if key == 'tornorm':
                std, mean = batch['torque_stdmean']
                output['tornorm'] = torch.linalg.vector_norm(denormalize(batch['torque'], mean, std), dim=-1)[..., None]
            elif key == 'grf':
                output['grf']     = batch['grf']
            elif key == 'torque':
                output['torque']  = batch['torque']
        if self.training:
            if 'torque' not in output:
                output['torque'] = output['tornorm'] * output['torvec']
                std, mean = batch['torque_stdmean']
                output['torque'] = normalize(output['torque'], mean, std)
            if 'grf' not in output:
                output['grf'] = output['grfnorm'] * output['grfvec']
                std, mean = batch['grf_stdmean']
                output['grf'] = normalize(output['grf'], mean, std)
            fd_input = []
            bz       = id_feat.shape[0]
            num_good = batch['torque'].shape[0]
            if batch.get('gt_ratio', 0.8) < 1:
                idx = torch.randperm(num_good)[int(num_good * batch.get('gt_ratio', 0.8)):]
            for key in self.inkeys_fd:
                fd_input.append(batch[key].flatten(1))
                if key == 'torque':
                    if batch.get('gt_ratio', 0.8) < 1:
                        fd_input[-1][idx] = output['torque'][idx].flatten(1)
                    fd_input[-1] = torch.cat([fd_input[-1], output['torque'][num_good:].flatten(1)])
                if key == 'grf_pre':
                    if batch.get('gt_ratio', 0.8) < 1:
                        fd_input[-1][idx] = output['grf'][idx, :1].flatten(1)
                    fd_input[-1] = torch.cat([fd_input[-1], output['grf'][num_good:, :1].flatten(1)])
            if self.with_intermediate:
                fd_input.append(id_feat)
            fd_input  = torch.cat(fd_input, dim=-1)
            fd_output = self.FD(fd_input)
            fd_output = self.FD_head(fd_output)
            offset = 0
            for i, key in enumerate(self.outkeys_fd):
                output[key] = fd_output[:, offset:offset + self.outdim_fd[key][1]].view(fd_input.shape[0], *self.outdim_fd[key][0])
                offset = self.outdim_offset_fd[i]
        
        return output