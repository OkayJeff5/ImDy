from collections import OrderedDict, defaultdict
import pickle
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import logging
import torch.nn.functional as F
import joblib
import tqdm

from utils.geometry import *
from utils.marker_vids import part2marker, part2num

def normalize(data, mean, std):
    epsilon = 1e-8
    mask    = std.abs() < epsilon
    normed  = (data - mean) / std
    normed[..., mask] = 0.0
    return normed

def denormalize(data, mean, std):
    return data * std + mean

class rawTorqueDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG     = config
        self.SPLIT      = split
        self.dpath      = config.dpath
        self.rot_rep    = config.get('rot_rep', 'quat')
        self.dkeys      = config.get('dkeys', ['rot', 'pos', 'torque', 'grf'])
        self.past_kf    = config.get('PAST_KF', 2)
        self.fut_kf     = config.get('FUT_KF', 2)
        self.length     = config.get('length', 500)
        self.batch_size = config.get('batch_size', 3072)
        self.parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        # load data
        self.db = {
            key: joblib.load(os.path.join(self.dpath, f'{key}.pkl')) for key in config.dkeys
        }
        for key in self.db.keys():
            for i in tqdm.trange(len(self.db[key])):
                self.db[key][i] = torch.from_numpy(self.db[key][i]).float()
            
        self.num_data = len(self.db['rot'])

    def __len__(self):
        return self.length * self.batch_size
    
    def __getitem__(self, i):
        data = {}
        idx = np.random.randint(self.num_data)
        fid = np.random.randint(self.past_kf, self.db['rot'][idx].shape[0] - self.fut_kf - 2)
        rot = self.db['rot'][idx][fid - self.past_kf:fid + self.fut_kf + 2]
        pos = self.db['pos'][idx][fid - self.past_kf:fid + self.fut_kf + 2, :1]
        data = {
            'pos': pos[:, :1], # (l, 1, 3)
            'rot': rot, # (l, 24, 4)
            'torque': self.db['torque'][idx][fid:fid + 1], # (1, 24, 3)
            'grf': self.db['grf'][idx][fid:fid + 2], # (2, 24, 3)
        }
        return data


class determinedTorqueDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG  = config
        self.SPLIT   = split
        self.dpath   = config.dpath
        self.dkeys   = config.get('dkeys', ['rot', 'pos', 'torque', 'grf'])
        self.past_kf = config.get('PAST_KF', 2)
        self.fut_kf  = config.get('FUTURE_KF', 2)
        # load data
        self.db = {
            key: joblib.load(os.path.join(self.dpath, f'{key}.pkl')) for key in config.dkeys
        }
        for key in self.db.keys():
            for i in tqdm.trange(len(self.db[key])):
                self.db[key][i] = torch.from_numpy(self.db[key][i][:, :24]).float()
        self.dataid   = np.cumsum([0] + [self.db['rot'][i].shape[0] - self.past_kf - self.fut_kf - 2 for i in range(len(self.db['rot']))])

    def __len__(self):
        return self.dataid[-1]
    
    def get_data(self, key, idx):
        if key in ['rot', 'pos', 'mkr']:
            return {key: self.db[key][idx][fid - self.past_kf:fid + self.fut_kf + 2]}
        elif key == 'torque':
            return {key: self.db[key][idx][fid:fid + 1] if key in self.db and idx < len(self.db[key]) else None}
        elif key == 'grf':
            return {key: self.db[key][idx][fid:fid + 2] if key in self.db and idx < len(self.db[key]) else None}
        else:
            raise NotImplementedError
    
    def __getitem__(self, i):
        data = {}
        idx  = np.where(self.dataid <= i)[0][-1]
        fid  = i - self.dataid[idx] + self.past_kf
        data = {key: self.get_data(key, idx) for key in self.db.keys()}
        return data


class cartisianTorqueDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG     = config
        self.SPLIT      = split
        self.dpath      = config.dpath
        self.rot_rep    = config.get('rot_rep', 'quat')
        self.dkeys      = config.get('dkeys', ['rot', 'pos', 'torque', 'grf', 'mkr'])
        self.past_kf    = config.get('PAST_KF', 2)
        self.fut_kf     = config.get('FUTURE_KF', 2)
        self.parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        # load data
        self.db = {
            key: joblib.load(os.path.join(self.dpath, f'{key}.pkl')) for key in config.dkeys
        }
        for key in self.db.keys():
            if key == 'mkr':
                for i in tqdm.trange(len(self.db[key])):
                    self.db[key][i] = torch.from_numpy(self.db[key][i]).float()
            else:
                for i in tqdm.trange(len(self.db[key])):
                    if isinstance(self.db[key][i], torch.Tensor):
                        self.db[key][i] = self.db[key][i][:, :24].float()
                    else:
                        self.db[key][i] = torch.from_numpy(self.db[key][i][:, :24]).float()
        self.dataid   = np.cumsum([0] + [self.db['rot'][i].shape[0] - self.past_kf - self.fut_kf - 2 for i in range(len(self.db['rot']))])

    def __len__(self):
        return self.dataid[-1]
    
    def __getitem__(self, i):
        data = {}
        idx = np.where(self.dataid <= i)[0][-1]
        fid = i - self.dataid[idx] + self.past_kf
        rot = self.db['rot'][idx][fid - self.past_kf:fid + self.fut_kf + 2]
        pos = self.db['pos'][idx][fid - self.past_kf:fid + self.fut_kf + 2]
        mkr = self.db['mkr'][idx][fid - self.past_kf:fid + self.fut_kf + 2]
        data = {
            'mkr': mkr,
            'pos': pos, # (l, 24, 3)
            'rot': rot, # (l, 24, 4)
            'torque': self.db['torque'][idx][fid:fid + 1] if 'torque' in self.db and idx < len(self.db['torque']) else None, # (1, 24, 3)
            'grf': self.db['grf'][idx][fid:fid + 2] if 'grf' in self.db and idx < len(self.db['torque']) else None, # (2, 24, 3)
        }
        return data
        

class adbTorqueDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG     = config
        self.SPLIT      = split
        self.dpath      = config.dpath
        self.rot_rep    = config.get('rot_rep', 'quat')
        self.dkeys      = config.get('dkeys', ['torque', 'grf', 'mkr', 'weight'])
        self.past_kf    = config.get('PAST_KF', 2)
        self.fut_kf     = config.get('FUTURE_KF', 2)
        self.parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        # load data
        self.db = {
            key: joblib.load(os.path.join(self.dpath, f'{key}.pkl')) for key in config.dkeys
        }
        for key in self.db.keys():
            for i in tqdm.trange(len(self.db[key])):
                self.db[key][i] = torch.from_numpy(self.db[key][i]).float()
        self.dataid   = np.cumsum([0] + [self.db['mkr'][i].shape[0] - self.past_kf - self.fut_kf - 2 for i in range(len(self.db['mkr']))])

    def __len__(self):
        return self.dataid[-1]
    
    def __getitem__(self, i):
        data = {}
        idx = np.where(self.dataid <= i)[0][-1]
        fid = i - self.dataid[idx] + self.past_kf
        mkr = self.db['mkr'][idx][fid - self.past_kf:fid + self.fut_kf + 2]
        data = {
            'mkr': mkr,
            'torque': self.db['torque'][idx][fid:fid + 1] if 'torque' in self.db and idx < len(self.db['torque']) else None, # (1, 24, 3)
            'grf': self.db['grf'][idx][fid:fid + 2] if 'grf' in self.db and idx < len(self.db['torque']) else None, # (2, 24, 3)
            'weight': self.db['weight'][idx] if 'weight' in self.db else None,
        }
        return data

class onDiskDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG     = config
        self.SPLIT      = split
        self.dpath      = config.dpath
        self.num_mini   = config.get('num_mini', 100)
        self.past_kf    = config.get('PAST_KF', 2)
        self.fut_kf     = config.get('FUTURE_KF', 2)
        self.dkeys      = config.get('dkeys', ['torque', 'grf', 'mkr', 'weight'])
        self.keys       = joblib.load(os.path.join(self.dpath, 'cand.pkl'))
        self.data_ids   = []
        for key in tqdm.tqdm(self.keys):
            nf      = key[1]
            self.data_ids += [(key[0], i) for i in range(self.past_kf, nf - self.fut_kf - 2, self.num_mini) if nf - i > 10]
        print(len(self.data_ids))
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, i):
        key, fid = self.data_ids[i]
        data    = joblib.load(os.path.join(self.dpath, f'{key}'))
        batch = {
            'mkr':    torch.tensor(data['mpos'][fid:fid + self.num_mini]),       # (L, xx, 3)
            'root':   torch.tensor(data['jpos'][fid:fid + self.num_mini, :1]),  # (L, 1, 3)
            'torque': torch.tensor(data['torque'][fid:fid + self.num_mini, 1:]) if len(data['torque'].shape) == 3 else torch.from_numpy(data['torque'][fid:fid + self.num_mini]), # (L, 24, 3)
            'weight': torch.tensor(data['weight']) if 'weight' in data else 76.3284,
        }
        num = batch['mkr'].shape[0]
        batch.update({
            'grf': torch.tensor(data['grf'][fid:fid + num].reshape(num, -1, 3)),  # (L, 2, 3)
        })
        return batch
        

class mixedDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG     = config
        self.SPLIT      = split
        self.dpath      = config.dpath
        self.num_mini   = config.get('num_mini', 100)
        self.past_kf    = config.get('PAST_KF', 2)
        self.fut_kf     = config.get('FUTURE_KF', 2)
        self.keys       = joblib.load(os.path.join(self.dpath, config.fname))
        self.data_ids   = []
        for key in tqdm.tqdm(self.keys):
            nf      = key[1]
            self.data_ids += [(key[0], i) for i in range(self.past_kf, nf - self.fut_kf - 2, self.num_mini) if nf - i > 10]
        print(len(self.data_ids))
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, i):
        key, fid = self.data_ids[i]
        data    = joblib.load(os.path.join(self.dpath, f'{key}'))
        batch = {
            'mkr':    torch.tensor(data['mpos'][fid:fid + self.num_mini]),       # (L, xx, 3)
            'root':   torch.tensor(data['jpos'][fid:fid + self.num_mini, :1]),  # (L, 1, 3)
            'torque': torch.tensor(data['torque'][fid:fid + self.num_mini, 1:]).flatten(1) if len(data['torque'].shape) == 3 else torch.from_numpy(data['torque'][fid:fid + self.num_mini]), # (L, 24, 3)
            'weight': torch.tensor(data['weight']) if 'weight' in data else 76.3284,
            'fd':     torch.tensor(data['qpos'][fid:fid + self.num_mini]).flatten(1) if 'qpos' in data else torch.tensor(data['bpos'][fid:fid + self.num_mini])
        }
        num = batch['mkr'].shape[0]
        batch.update({
            'grf': torch.tensor(data['grf'][fid:fid + num].reshape(num, -1, 3)),  # (L, 2, 3)
        })
        return batch

def cls_augment(batch):
    rng_state = np.random.randn() 
    b = batch['rot'].shape[0]
    if rng_state > 0:
        perm = torch.randperm(batch['rot'].shape[1])
        rot_perm = batch['rot'][:, perm]
        batch['rot'] = torch.cat([batch['rot'], rot_perm])
        if 'pos' in batch:
            pos_perm = batch['pos'][:, perm]
            batch['pos'] = torch.cat([batch['pos'], pos_perm])
        elif 'fpos' in batch:
            fpos_perm = batch['fpos'][:, perm]
            batch['fpos'] = torch.cat([batch['fpos'], fpos_perm])
        batch['indicator'] = torch.cat([torch.ones(b), torch.zeros(b)]).to(batch['rot'].device)
    else:
        rot_perm = standardize_quaternion(batch['rot'] + torch.rand_like(batch['rot']) - 0.5)
        batch['rot'] = torch.cat([batch['rot'], rot_perm])
        if 'pos' in batch:
            pos_perm = batch['pos'] + torch.rand_like(batch['pos']) - 0.5
            batch['pos'] = torch.cat([batch['pos'], pos_perm])
        elif 'fpos' in batch:
            fpos_perm = batch['fpos'] + torch.rand_like(batch['fpos']) - 0.5
            batch['fpos'] = torch.cat([batch['fpos'], fpos_perm])
        batch['indicator'] = torch.cat([torch.ones(b), torch.zeros(b)]).to(batch['rot'].device)
    if 'angvel' in batch:
        angvel_perm = estimate_angular_velocity(rot_perm, 1, 'quat')
        batch['angvel'] = torch.cat([batch['angvel'], angvel_perm])
    if 'vel' in batch:
        vel_perm  = estimate_linear_velocity(pos_perm, 1)
        batch['vel'] = torch.cat([batch['vel'], vel_perm])
    if 'fvel' in batch:
        fvel_perm = estimate_linear_velocity(fpos_perm, 1)
        batch['fvel'] = torch.cat([batch['fvel'], fvel_perm])
    return batch

def get_preprocess_fn(config, device):
    if config.get('MODE', 'raw') in ['raw']:
        rot_rep    = config.get('rot_rep', 'quat')
        out_dkeys  = config.get('out_dkeys', ['rot', 'pos', 'torque', 'grf'])
        past_kf    = config.get('PAST_KF', 2)
        fut_kf     = config.get('FUTURE_KF', 2)
        joint_tor  = config.get('joint_tor', False)
        adb_tor    = config.get('adb_tor', False)
        cls_aug    = config.get('cls_aug', False)
        treadmill  = config.get('treadmill', False)
        mkr_sample = config.get('mkr_sample', 'random')
        parents    = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        if config.get('rm_prerot', True):
            pre_rot    = torch.Tensor([[[[.5, .5, .5, .5]]]]).expand(-1, past_kf + fut_kf + 2, 24, -1).to(device)
        else:
            pre_rot    = torch.Tensor([[[[1., .0, .0, .0]]]]).expand(-1, past_kf + fut_kf + 2, 24, -1).to(device)
        def preprocess_fn(batch):
            for key in batch.keys():
                if batch[key] is not None and not key == 'mkr':
                    batch[key] = batch[key].to(device)
            
            poses_global = batch.get('rot', None)     # (b, l, 24, 4)
            trans        = batch.get('pos', None)     # (b, l, 24, 3)
            grf          = batch.get('grf', None)     # (b, 2, 24, 3)
            torque       = batch.get('torque', None)  # (b, 1, 24, 3)
            data = {}
            heading_inv  = None
            if poses_global is not None:
                heading_inv = calculate_heading_quaternion_inverse(poses_global[:, past_kf:past_kf + 1, :1]).expand(-1, past_kf + fut_kf + 2, -1, -1) # (b, l, 1, 4)
            if heading_inv is not None:
                trans_inv       = heading_inv.expand(-1, -1, 24, -1)
                trans           = quaternion_apply(trans_inv, trans) # (b, l, 24, 3)
                origin          = trans[:, past_kf:past_kf + 1, :1, :2] * 1
                trans[..., :2] -= trans[:, past_kf:past_kf + 1, :1, :2] # not changing floor height
            else:
                origin = 0
            if 'pos' in out_dkeys:
                data['pos']      = trans[:, :, 0]
            if 'pos_pre' in out_dkeys:
                data['pos_pre']  = trans[:, :past_kf + 1, 0]
            if 'pos_post' in out_dkeys:
                data['pos_post'] = trans[:, past_kf + 1:past_kf + 2, 0]
            if 'fpos' in out_dkeys:
                data['fpos']     = trans
            if 'fpos_pre' in out_dkeys:
                data['fpos_pre']  = trans[:, :past_kf + 1]
            if 'fpos_post' in out_dkeys:
                data['fpos_post'] = trans[:, past_kf + 1:past_kf + 2]    
            if 'vel' in out_dkeys:
                data['vel']  = estimate_linear_velocity(trans[:, :, 0], 1)
            if 'fvel' in out_dkeys:
                data['fvel'] = estimate_linear_velocity(trans, 1)
            
            if 'mkr' in out_dkeys or 'mkr_pre' in out_dkeys or 'mkr_post' in out_dkeys or 'mvel' in out_dkeys:
                mkr_num = torch.unique(torch.Tensor([item.shape[1] for item in batch['mkr']])).int().tolist() # raw data might have different number of markers
                if mkr_sample == 'random':
                    mkr_sel = {
                        mnum: torch.randperm(mnum)[:torch.randint(low=22, high=min(96, mnum), size=[1])[0]] for mnum in mkr_num # for each different markerset, sample different number of markers
                    }
                    mnum = max([int(item.shape[0]) for item in mkr_sel.values()])
                    mkr  = torch.cat([
                        torch.cat([m[:, :mkr_sel[int(m.shape[1])]], torch.zeros(m.shape[0], mnum - mkr_sel[int(m.shape[1])].shape[0], 3)], dim=1)[None] for m in batch['mkr']
                    ]).to(device)
                    if heading_inv is not None:
                        mkr_inv = heading_inv.expand(-1, -1, mkr_num, -1)
                        mkr     = quaternion_apply(mkr_inv, batch['mkr'][:, :, mkr_sel[:mkr_num]])
                    else:
                        mkr     = batch['mkr'][:, :, mkr_sel[:mkr_num]]
                elif mkr_sample == 'perpart':
                    part_num = torch.randint(low=0, high=4, size=[24])
                    mkr_sel  = torch.cat([part2marker[i][torch.randperm(part2num[i])[:part_num[i]]] for i in range(24)]).long()
                    mkr      = torch.cat(batch['mkr']).to(device)
                    if heading_inv is not None:
                        mkr_inv  = heading_inv.expand(-1, -1, len(mkr_sel), -1)
                        mkr      = quaternion_apply(mkr_inv, mkr[:, :, mkr_sel])
                    else:
                        mkr      = batch['mkr'][:, :, mkr_sel]
                else:
                    mnum = max(mkr_num)
                    mkr  = torch.cat([
                        torch.cat([m, torch.zeros(m.shape[0], mnum - m.shape[1], 3)], dim=1)[None] for m in batch['mkr']
                    ]).to(device)
                    if heading_inv is not None:
                        mkr_inv  = heading_inv.expand(-1, -1, mkr.shape[2], -1)
                        mkr      = quaternion_apply(mkr_inv, mkr)
                if treadmill:
                    mkr[..., :2] -= trans[..., :1, :2]
                else:
                    mkr[..., :2] -= origin
                if 'mkr' in out_dkeys:
                    data['mkr'] = mkr.permute(0, 2, 1, 3)
                if 'mkr_pre' in out_dkeys:
                    data['mkr_pre']  = mkr[:, :past_kf + 1].permute(0, 2, 1, 3)
                if 'mkr_post' in out_dkeys:
                    data['mkr_post'] = mkr[:, past_kf + 1:past_kf + 2].permute(0, 2, 1, 3)
                if 'mvel' in out_dkeys:
                    data['mvel'] = estimate_linear_velocity(mkr, 1).permute(0, 2, 1, 3)
            
            if heading_inv is not None:
                heading_inv   = heading_inv.expand(-1, -1, 24, -1)
            if 'rot' in out_dkeys or 'rot_pre' in out_dkeys or 'rot_post' in out_dkeys or 'angvel' in out_dkeys:
                poses_global  = quaternion_multiply(heading_inv, poses_global)
                poses_global  = quaternion_multiply(poses_global, pre_rot)
                poses_inverse = quaternion_invert(poses_global)
                poses_local   = quaternion_multiply(poses_inverse[:, :, parents], poses_global)
                poses         = torch.cat([poses_global[:, :, :1], poses_local[:, :, 1:]], dim=2)
                rot           = random_augment_quaternion(rot_from_to(poses, 'quat', rot_rep)) # (b, l, 24, 4)
                if 'rot' in out_dkeys:
                    data['rot']      = rot
                if 'rot_pre' in out_dkeys:
                    data['rot_pre']  = rot[:, :past_kf + 1]
                if 'rot_post' in out_dkeys:
                    data['rot_post'] = rot[:, past_kf + 1:past_kf + 2]
                if 'angvel' in out_dkeys:
                    data['angvel'] = estimate_angular_velocity(rot, 1, 'quat')
            
            if 'torque' in out_dkeys:
                num_good = torque.shape[0]
                if adb_tor:
                    data['torque'] = torque[:, :, 6:]
                elif joint_tor:
                    data['torque'] = torque[:, :, 1:]
                else:
                    data['torque'] = quaternion_apply(heading_inv[:num_good, :1], torque)[:, :, 1:]
            
            if 'grf' in out_dkeys or 'grf_pre' in out_dkeys:
                num_good = grf.shape[0]
                if adb_tor:
                    grf = grf.view(-1, 2, 2, 3)
                if heading_inv is not None:
                    grf = quaternion_apply(heading_inv[:num_good, :2, :grf.shape[2]], grf)
                if 'grf' in out_dkeys:
                    data['grf']     = grf    # (2, 24, 3)
                if 'grf_pre' in out_dkeys:
                    data['grf_pre'] = grf[:, :1]
            
            if 'weight' in out_dkeys:
                data['weight'] = batch['weight']
            
            if cls_aug:
                data = cls_augment(data)
            return data
        return preprocess_fn
    elif config.get('MODE', 'raw') in ['adb', 'mkr']:
        rot_rep    = config.get('rot_rep', 'quat')
        out_dkeys  = config.get('out_dkeys', ['rot', 'pos', 'torque', 'grf'])
        past_kf    = config.get('PAST_KF', 2)
        fut_kf     = config.get('FUTURE_KF', 2)
        mkr_sample = config.get('mkr_sample', 'random')
        treadmill  = config.get('treadmill', False)
        def preprocess_fn(batch):
            
            data = defaultdict(list)
            if 'mkr' in out_dkeys or 'mkr_pre' in out_dkeys or 'mkr_post' in out_dkeys or 'mvel' in out_dkeys:
                mkr = []
                max_mnum = 0
                for item in batch['mkr']:
                    if mkr_sample == 'random': 
                        mkr_num = item.shape[1]
                        max_mnum = max(mkr_num, max_mnum)
                        mkr_sel = torch.randperm(mkr_num)
                        mkr_sel = mkr_sel[:torch.randint(low=22, high=min(96, mkr_num), size=[1])[0]]
                        mkr.append(item[:, mkr_sel].to(device))
                    else:
                        mkr_num = item.shape[1]
                        mkr.append(item.to(device))
                        max_mnum = max(mkr_num, max_mnum)
                for i in range(len(mkr)):
                    bidx   = torch.arange(mkr[i].shape[0] - past_kf - fut_kf - 2)
                    mkr[i] = torch.cat([mkr[i], torch.zeros(mkr[i].shape[0], max_mnum - mkr[i].shape[1], 3, device=device)], dim=1)
                    if treadmill:
                        root = batch['root'][i].to(device)
                        tmp = []
                        for j in range(past_kf + fut_kf + 2):
                            tmp.append(mkr[i][bidx + j, None])
                            tmp[-1][..., :2] -= root[bidx + past_kf, None, :, :2]
                        mkr[i] = torch.cat(tmp, dim=1)
                    else:
                        mkr[i] = torch.cat([
                            mkr[i][bidx + j, None] for j in range(past_kf + fut_kf + 2)
                        ], dim=1)
                mkr = torch.cat(mkr, dim=0).float()
                if 'mkr' in out_dkeys:
                    data['mkr'] = mkr.permute(0, 2, 1, 3)
                if 'mkr_pre' in out_dkeys:
                    data['mkr_pre']  = mkr[:, :past_kf + 1].permute(0, 2, 1, 3)
                if 'mkr_post' in out_dkeys:
                    data['mkr_post'] = mkr[:, past_kf + 1:past_kf + 2].permute(0, 2, 1, 3)
                if 'mvel' in out_dkeys:
                    data['mvel'] = estimate_linear_velocity(mkr, 1).permute(0, 2, 1, 3)

            if 'torque' in out_dkeys:
                torque = batch['torque']
                for i in range(len(torque)):
                    torque[i] = torque[i].to(device)
                    bidx   = torch.arange(torque[i].shape[0] - past_kf - fut_kf - 2) + past_kf
                    data['torque'].append(torque[i][bidx, None])
                data['torque'] = torch.cat(data['torque']).float()

            if 'grf' in out_dkeys or 'grf_pre' in out_dkeys:
                grf = batch['grf']
                for i in range(len(grf)):
                    grf[i] = grf[i].to(device)
                    bidx = torch.arange(grf[i].shape[0] - past_kf - fut_kf - 2) + past_kf
                    data['grf'].append(torch.cat([
                        grf[i][bidx, None], grf[i][bidx + 1, None],
                    ], dim=1))
                data['grf'] = torch.cat(data['grf']).float()
                if 'grf_pre' in out_dkeys:
                    data['grf_pre'] = data['grf'][:, :1]
            if 'weight' in out_dkeys:
                data['weight'] = torch.Tensor(batch['weight']).to(device).float()
            return data
        return preprocess_fn
    elif config.get('MODE', 'raw') in ['mix']:
        rot_rep    = config.get('rot_rep', 'quat')
        out_dkeys  = config.get('out_dkeys', ['rot', 'pos', 'torque', 'grf'])
        past_kf    = config.get('PAST_KF', 2)
        fut_kf     = config.get('FUTURE_KF', 2)
        mkr_sample = config.get('mkr_sample', 'random')
        treadmill  = config.get('treadmill', False)
        def preprocess_fn(batch):
            
            data = defaultdict(list)
            min_mnum = 999
            for i in range(len(batch['mkr'])):
                min_mnum = min(batch['mkr'][i].shape[1], min_mnum)
            for i in range(len(batch['mkr'])):
                mkr_sel = torch.randperm(min_mnum)
                bidx    = torch.arange(batch['mkr'][i].shape[0] - past_kf - fut_kf - 2)
                batch['mkr'][i] = batch['mkr'][i].to(device)
                if treadmill:
                    root = batch['root'][i].to(device)
                    tmp = []
                    for j in range(past_kf + fut_kf + 2):
                        tmp.append(batch['mkr'][i][bidx + j, None][:, :, mkr_sel])
                        tmp[-1][..., :2] -= root[bidx + past_kf, None, :, :2]
                    if batch['torque'][i].shape[-1] == 69:
                        data['mkr_imdy'].append(torch.cat(tmp, dim=1))
                    else:
                        data['mkr_adb'].append(torch.cat(tmp, dim=1))
                else:
                    if batch['torque'][i].shape[-1] == 69:
                        data['mkr_imdy'],append(torch.cat([
                            batch['mkr'][i][bidx + j, None][:, :, mkr_sel] for j in range(past_kf + fut_kf + 2)
                        ], dim=1))
                    else:
                        data['mkr_adb'].append(torch.cat([
                            batch['mkr'][i][bidx + j, None][:, :, mkr_sel] for j in range(past_kf + fut_kf + 2)
                        ], dim=1))
                
                batch['torque'][i] = batch['torque'][i].to(device)
                bidx   = torch.arange(batch['torque'][i].shape[0] - past_kf - fut_kf - 2) + past_kf
                if batch['torque'][i].shape[-1] == 69:
                    data['torque_imdy'].append(batch['torque'][i][bidx, None])
                    data['fd_imdy'].append(batch['fd'][i][bidx + 1])
                else:
                    data['torque_adb'].append(batch['torque'][i][bidx, None])
                    data['fd_adb'].append(batch['fd'][i][bidx + 1])
                        
                batch['grf'][i] = batch['grf'][i].to(device)
                bidx = torch.arange(batch['grf'][i].shape[0] - past_kf - fut_kf - 2) + past_kf
                
                if batch['torque'][i].shape[-1] == 69:
                    # print(batch['grf'][i].shape)
                    # assert 0
                    data['grf_imdy'].append(torch.cat([
                        batch['grf'][i][bidx, None][:, :, [8, 7]] + batch['grf'][i][bidx, None][:, :, [11, 10]], batch['grf'][i][bidx + 1, None][:, :, [8, 7]] + batch['grf'][i][bidx + 1, None][:, :, [11, 10]],
                    ], dim=1))
                    data['weight_imdy'] += [batch['weight'][i]] * len(bidx)
                else:
                    data['grf_adb'].append(torch.cat([
                        batch['grf'][i][bidx, None], batch['grf'][i][bidx + 1, None],
                    ], dim=1))
                    data['weight_adb'] += [batch['weight'][i]] * len(bidx)
                
            if len(data['mkr_imdy']) > 0:
                data['mkr_imdy']   = torch.cat(data['mkr_imdy'], dim=0).float()
                data['mvel_imdy']  = estimate_linear_velocity(data['mkr_imdy'], 1)
                data['mkr_imdy']   = data['mkr_imdy'].permute(0, 2, 1, 3)
                data['mvel_imdy']  = data['mvel_imdy'].permute(0, 2, 1, 3)
                data['torque_imdy']   = torch.cat(data['torque_imdy']).float().view(-1, 23, 3)
                data['fd_imdy']       = torch.cat(data['fd_imdy']).float().to(device)
                data['grf_imdy']      = torch.cat(data['grf_imdy']).float().to(device)
                data['grf_pre_imdy']  = data['grf_imdy'][:, :1]
                data['weight_imdy']   = torch.Tensor(data['weight_imdy']).float().to(device)
            if len(data['mkr_adb']) > 0:
                data['mkr_adb']     = torch.cat(data['mkr_adb'], dim=0).float()
                data['mvel_adb']    = estimate_linear_velocity(data['mkr_adb'], 1)
                data['mkr_adb']     = data['mkr_adb'].permute(0, 2, 1, 3)
                data['mvel_adb']    = data['mvel_adb'].permute(0, 2, 1, 3)
                data['torque_adb']  = torch.cat(data['torque_adb']).float()
                data['fd_adb']      = torch.cat(data['fd_adb']).float().to(device)
                data['grf_adb']     = torch.cat(data['grf_adb']).float().to(device)
                data['grf_pre_adb'] = data['grf_adb'][:, :1]
                data['weight_adb']  = torch.Tensor(data['weight_adb']).float().to(device)
            return data
        return preprocess_fn
        
    else: 
        def preprocess_fn(batch):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            return batch
        return preprocess_fn


def get_collate_fn(config):
    past_kf    = config.get('PAST_KF', 2)
    fut_kf     = config.get('FUTURE_KF', 2)
    if config.get('MODE', 'raw') in ['raw']:
        def collate_fn(data):
            batch_good = defaultdict(list)
            batch_bad  = defaultdict(list)
            b = len(data)
            for i in range(b):
                if data[i]['torque'] is None:
                    batch_bad['rot'].append(data[i]['rot'][None])
                    batch_bad['pos'].append(data[i]['pos'][None])
                    if 'mkr' in data[i]:
                        batch_bad['mkr'].append(data[i]['mkr'][None])
                    if 'weight' in data[i]:
                        batch_bad['weight'].append(data[i]['weight'])
                else:
                    if 'rot' in data[i]:
                        batch_good['rot'].append(data[i]['rot'][None])
                    if 'pos' in data[i]:
                        batch_good['pos'].append(data[i]['pos'][None])
                    if 'torque' in data[i]:
                        batch_good['torque'].append(data[i]['torque'][None])
                    if 'grf' in data[i]:
                        batch_good['grf'].append(data[i]['grf'][None])
                    if 'mkr' in data[i]:
                        batch_good['mkr'].append(data[i]['mkr'][None])
                    if 'weight' in data[i]:
                        batch_good['weight'].append(data[i]['weight'])
            batch = {}
            if 'rot' in batch_good:
                batch['rot'] = torch.cat(batch_good['rot'] + batch_bad['rot'])
            if 'pos' in batch_good:
                batch['pos'] = torch.cat(batch_good['pos'] + batch_bad['pos'])
            batch.update({
                'torque': torch.cat(batch_good['torque']) if len(batch_good['torque']) > 0 else None,
                'grf': torch.cat(batch_good['grf']) if len(batch_good['torque']) > 0 else None,
            })
            if 'mkr' in batch_good:
                batch.update({
                    'mkr': batch_good['mkr'] + batch_bad['mkr'],
            })
            if 'weight' in batch_good:
                batch.update({
                    'weight': torch.cat(batch_good['weight'] + batch_bad['weight']),
            })
            return batch
        return collate_fn
    elif config.get('MODE', 'raw') in ['adb', 'mkr', 'mix']:
        def collate_fn(data):
            b = len(data)
            batch = defaultdict(list)
            for i in range(b):
                batch['mkr'].append(data[i]['mkr'])
                batch['grf'].append(data[i]['grf'])
                batch['torque'].append(data[i]['torque'])
                batch['weight'] += [data[i]['weight'] for _ in range(data[i]['mkr'].shape[0] - past_kf - fut_kf - 2)]
                batch['root'].append(data[i]['root'])
                batch['fd'].append(data[i]['fd'])
            return batch
        return collate_fn
    else:
        raise NotImplementedError

def load_data_norm(config, device='cpu'):
    mean_std = joblib.load(os.path.join(config.DATASET.TRAIN.norm_path, 'mean_std.pkl'))

    # Convert to tensors
    for key in mean_std.keys():
        mean, std = mean_std[key]
        mean_std[key] = (
            torch.from_numpy(mean).float().to(device), 
            torch.from_numpy(std).float().to(device)
        )

    return mean_std

def normalize_input_batch(batch, mean_std):
    """The input tensor is not flattened"""
    for key in batch.keys():
        if key in ['contact', 'tornorm', 'torvec', 'grfnorm', 'grfvec', 'identifier', 'rot', 'rot_pre', 'rot_post', 'indicator', 'mkr_pre', 'mkr_post', 'mkr', 'mvel', 'weight']:
            pass
        elif key in mean_std.keys() and batch[key] is not None:
            std, mean = mean_std[key]
            batch[key] = normalize(batch[key], mean, std)
        else:
            raise ValueError(f"Key {key} not found in mean_std")
    return batch
    
def denormalize_input_batch(batch, mean_std):
    """The input tensor is not flattened"""
    for key in batch.keys():
        if key in ['contact', 'tornorm', 'torvec', 'grfnorm', 'grfvec', 'identifier', 'rot', 'rot_pre', 'rot_post', 'indicator', 'mkr_pre', 'mkr_post', 'mkr', 'mvel', 'weight']:
            pass
        elif key in mean_std.keys() and batch[key] is not None:
            std, mean = mean_std[key]
            batch[key] = denormalize(batch[key], mean, std)
        else:
            raise ValueError(f"Key {key} not found in mean_std")
    return batch

def denormalize_output_batch(batch, mean_std):
    """The output tensor is flattened as [bs, dim]"""
    for key in batch.keys():
        if key in ['contact', 'tornorm', 'torvec', 'grfnorm', 'grfvec', 'identifier', 'rot', 'rot_pre', 'rot_post', 'indicator', 'mkr_pre', 'mkr_post', 'mkr', 'mvel', 'weight']:
            pass
        elif key in mean_std.keys() and batch[key] is not None:
            std, mean = mean_std[key]
            batch[key] = denormalize(batch[key], mean, std)
        else:
            raise ValueError(f"Key {key} not found in mean_std")
    return batch