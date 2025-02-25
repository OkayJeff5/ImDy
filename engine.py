# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import tqdm
import joblib
from collections import defaultdict
from easydict import EasyDict as edict
import torch
import torch.utils.tensorboard
import torch.nn.functional as F
from dataset import denormalize_output_batch, normalize_input_batch, denormalize_input_batch
from utils.geometry import *
import utils.misc as utils

from torch.profiler import profile, record_function, ProfilerActivity

def debug_hook(name):
    def hook(grad):
        if grad.isnan().sum() > 0:
            print(name, grad.isnan().sum())
    return hook

def adv_augment(batch):
    idx           = torch.randperm(batch['torque'].shape[0])[::2]
    tornorm       = torch.linalg.vector_norm(batch['torque'], dim=-1)[..., None]
    norm_noise    = torch.rand_like(tornorm[idx]) * 60. - 30.
    tornorm[idx] += norm_noise 
    torvec        = F.normalize(batch['torque'], dim=-1)
    vec_noise     = F.normalize(torch.rand_like(torvec[idx]), dim=-1)
    ang_noise     = (torch.rand_like(tornorm[idx]) * 1.4 - 0.7) * torch.pi
    rot_noise     = vec_noise * ang_noise
    quat_noise    = rot_from_to(rot_noise, 'aa', 'quat')
    torvec[idx]   = quaternion_apply(quat_noise, torvec[idx])
    batch['torque'] = tornorm * torvec
    
    
    grfnorm       = torch.linalg.vector_norm(batch['grf'], dim=-1)[..., None]
    norm_noise    = torch.rand_like(grfnorm[idx]) * 40. - 20.
    grfnorm[idx] += norm_noise 
    grfvec        = F.normalize(batch['grf'], dim=-1)
    vec_noise     = F.normalize(torch.rand_like(grfvec[idx]), dim=-1)
    ang_noise     = (torch.rand_like(grfnorm[idx]) * 0.2 - 0.1) * torch.pi
    rot_noise     = vec_noise * ang_noise
    quat_noise    = rot_from_to(rot_noise, 'aa', 'quat')
    grfvec[idx]   = quaternion_apply(quat_noise, grfvec[idx])
    batch['grf'] = grfnorm * grfvec
    
    batch['identifier'] = torch.ones(tornorm.shape[0], device=tornorm.device)
    batch['identifier'][idx] = 0
    return batch

def calc_norm(data_loader, preprocess_fn, device, config):
    
    rot_rep  = config.get('rot_rep', 'quat')
    nkeys    = '_'.join(config.get('nkeys', ['rot', 'pos', 'torque', 'grf']))
    past_kf  = config.get('PAST_KF', 2)
    fut_kf   = config.get('FUTURE_KF', 2)
    config_identifier = f'{rot_rep}_{nkeys}_{past_kf}_{fut_kf}'
    if not config.get('rm_prerot', True):
        config_identifier += '_wpre'
    if config.get('joint_tor', False):
        config_identifier += '_jointtor'
    elif config.get('adb_tor', False):
        config_identifier += '_adb'
    if config.dpath == 'data/full_train':
        config_identifier += '_full'
    config_identifier = config.get('norm_name', config_identifier)
    
    mean_std = {}
    if os.path.exists(f'data/norm_{config_identifier}.pkl'):
        mean_std  = {k: (v[0].float().to(device), v[1].float().to(device)) for k, v in joblib.load(f'data/norm_{config_identifier}.pkl').items()}
        out_dkeys = config.get('out_dkeys', [])
        past_kf   = config.get('PAST_KF', 2)
        for k in out_dkeys:
            if k in mean_std or k in ['mkr', 'mvel', 'mkr_pre', 'mkr_post', 'weight']: continue
            orikey, suffix = k.split('_')
            if k == 'grf_pre':
                mean_std[k] = mean_std[orikey][0][:1], mean_std[orikey][1][:1]
            elif suffix == 'post' and orikey in mean_std:
                mean_std[k] = mean_std[orikey][0][past_kf + 1:past_kf + 2], mean_std[orikey][1][past_kf + 1:past_kf + 2]
            elif suffix == 'pre' and orikey in mean_std:
                mean_std[k] = mean_std[orikey][0][:past_kf + 1], mean_std[orikey][1][:past_kf + 1]
    else:
        print(f'Creating norm for {config_identifier}')
        data = defaultdict(list)
        cnt = 0
        for batch in tqdm.tqdm(data_loader):
            batch = preprocess_fn(batch)
            for key in batch.keys():
                data[key].append(batch[key].cpu())
            cnt += 1
        for key in data.keys():
            data[key]     = torch.cat(data[key])
            mean_std[key] = torch.std_mean(data[key], dim=0)# (data[key].mean(dim=0).to(device).float(), data[key].std(dim=0).to(device).float())
            mean_std[key] = (mean_std[key][0].to(device), mean_std[key][1].to(device))
        joblib.dump({
            k: (v[0].cpu(), v[1].cpu()) for k, v in mean_std.items()
        }, f'data/norm_{config_identifier}.pkl')
    return mean_std

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, preprocess_fn, data_norm, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, 
                    device: torch.device, epoch: int, 
                    max_norm: float = 0, global_step: int = 0,
                    gt_ratio: float = 0.8,
                    adversarial_assets: dict = None,
                    debug: bool = False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch  = preprocess_fn(batch)
        batch  = normalize_input_batch(batch, data_norm)
        if adversarial_assets is not None:
            adversarial_assets['optimizer'].zero_grad()
            adv_output_T = adversarial_assets['model'](batch)
            adv_loss     = adversarial_assets['criterion']({'identifier': torch.ones_like(adv_output_T['identifier'])}, adv_output_T)['L_adv']
            
            batch['torque_stdmean'] = data_norm['torque']
            batch['grf_stdmean']    = data_norm['grf']
            batch['gt_ratio']       = gt_ratio
            output = model(batch)
            _ = batch.pop('torque_stdmean')
            _ = batch.pop('grf_stdmean')
            _ = batch.pop('gt_ratio')
            adv_batch = {
                'pos': batch['pos'],
                'rot': batch['rot'],
                'grf': output['grf'],
                'torque': output['torque'],
            }
            adv_output_F = adversarial_assets['model'](adv_batch)
            adv_loss    += adversarial_assets['criterion']({'identifier': torch.zeros_like(adv_output_F['identifier'])}, adv_output_F)['L_adv']
            adv_loss_value = adv_loss.item()
            if not math.isfinite(adv_loss_value):
                print("Loss is {}, stopping training".format(adv_loss_value))
                for k, v in output.items():
                    print(k, torch.isnan(v).int().sum(), torch.max(v))
                for k, v in batch.items():
                    print(k, torch.max(v))
                sys.exit(1)
            adv_loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(adversarial_assets['model'].parameters(), max_norm)
            adversarial_assets['optimizer'].step()
            metric_logger.update(L_adv_d=adv_loss_value)
            
        optimizer.zero_grad()
        batch['torque_stdmean'] = data_norm['torque']
        batch['grf_stdmean']    = data_norm['grf']
        batch['gt_ratio']       = gt_ratio
        output = model(batch)
        _ = batch.pop('torque_stdmean')
        _ = batch.pop('grf_stdmean')
        _ = batch.pop('gt_ratio')
        loss_dict_raw = criterion(batch, output, False)
        if adversarial_assets is not None:
            adv_batch = {
                'pos': batch['pos'],
                'rot': batch['rot'],
                'grf': output['grf'],
                'torque': output['torque'],
            }
            adv_output = adversarial_assets['model'](adv_batch)
            loss_dict_raw['L_adv'] = adversarial_assets['criterion']({'identifier': torch.ones_like(adv_output['identifier'])}, adv_output)['L_adv']
        batch  = denormalize_input_batch(batch, data_norm)
        output = denormalize_output_batch(output, data_norm)
        loss_dict_raw.update(criterion(batch, output, True))
        weight_dict   = criterion.weight_dict
        loss_dict     = {k + '_scaled': v.mean() * weight_dict.get(k, 1.) for k, v in loss_dict_raw.items()}
        losses        = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value    = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k, v in output.items():
                print(k, torch.isnan(v).int().sum(), torch.max(v))
            for k, v in batch.items():
                print(k, torch.max(v))
            print(loss_dict)
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict, **loss_dict_raw)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        writer.add_scalars('losses_scaled', loss_dict, global_step)
        global_step += 1
        if debug: break
    optimizer.zero_grad()
    if adversarial_assets is not None:
        adversarial_assets['optimizer'].zero_grad()
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step
    
    
def train_one_epoch_mix(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, preprocess_fn, data_norm, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, 
                    device: torch.device, epoch: int, 
                    max_norm: float = 0, global_step: int = 0,
                    gt_ratio: float = 0.8,
                    adversarial_assets: dict = None,
                    debug: bool = False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch  = preprocess_fn(batch)
            
        optimizer.zero_grad()
        output = model(batch)
        loss_dict_raw = criterion(batch, output, False)
        weight_dict   = criterion.weight_dict
        loss_dict     = {k + '_scaled': v.mean() * weight_dict.get(k, 1.) for k, v in loss_dict_raw.items()}
        losses        = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value    = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k, v in output.items():
                print(k, torch.isnan(v).int().sum(), torch.max(v))
            for k, v in batch.items():
                print(k, torch.max(v))
            print(loss_dict)
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict, **loss_dict_raw)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        writer.add_scalars('losses_scaled', loss_dict, global_step)
        global_step += 1
        if debug: break
    optimizer.zero_grad()
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

def evaluate(model: torch.nn.Module, metric: torch.nn.Module, data_loader: Iterable, preprocess_fn, data_norm, writer: torch.utils.tensorboard.SummaryWriter, device: torch.device, epoch: int, dumpres = False):
    
    model.eval()
    metric.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    res = {}
    if dumpres:
        res = {
            'torque': [],
            'grf': [],
        }
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            batch  = preprocess_fn(batch)
            nf = batch[list(batch.keys())[0]].shape[0]
            if data_norm is not None:
                batch  = normalize_input_batch(batch, data_norm)
                batch['torque_stdmean'] = data_norm['torque']
                batch['grf_stdmean']    = data_norm['grf']
            output = model(batch)
            if data_norm is not None:
                _ = batch.pop('torque_stdmean')
                _ = batch.pop('grf_stdmean')
                output = denormalize_output_batch(output, data_norm)
                batch  = denormalize_input_batch(batch, data_norm)
            metric_dict = metric(batch, output)
            metric_logger.update(**metric_dict, cnt=nf)
            if dumpres:
                res['torque'].append(output['torque'].cpu().numpy())
                res['grf'].append(output['grf'].cpu().numpy())
    print("Averaged stats:", metric_logger)
    for k, meter in metric_logger.meters.items():
        res[k] = meter.global_avg
    writer.add_scalars('metrics', {k: v for k, v in res.items() if k not in ['torque', 'grf']}, epoch)
    return res

def evaluate_glink(model: torch.nn.Module, metric: torch.nn.Module, data_loader: Iterable, preprocess_fn, data_norm, writer: torch.utils.tensorboard.SummaryWriter, device: torch.device, epoch: int, dumpres = False):
    
    model.eval()
    metric.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    res = {}
    if dumpres:
        res = {
            'torque': [],
            'grf': [],
        }
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            batch  = preprocess_fn(batch)
            nf  = batch[list(batch.keys())[0]].shape[0]
            grf = batch['grf'][:, :, :2] * 1
            batch  = normalize_input_batch(batch, data_norm)
            output = model(batch)
            output = denormalize_output_batch(output, data_norm)
            batch  = denormalize_input_batch(batch, data_norm)
            batch['grf'] = grf
            metric_dict = metric(batch, output)
            metric_logger.update(**metric_dict, cnt=nf)
            if dumpres:
                res['torque'].append(output['torque'].cpu().numpy())
                res['grf'].append(output['grf'].cpu().numpy())
    print("Averaged stats:", metric_logger)
    for k, meter in metric_logger.meters.items():
        res[k] = meter.global_avg
    writer.add_scalars('metrics', {k: v for k, v in res.items() if k not in ['torque', 'grf']}, epoch)
    return res
    
def evaluate_orig(model: torch.nn.Module, data_loader: Iterable, preprocess_fn, data_norm, device: torch.device, dumpres = False):
    
    model.eval()
    
    print_freq = 100
    res = {}
    if dumpres:
        res = {
            'torque': [],
            'grf': [],
        }
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            batch  = preprocess_fn(batch)
            batch  = normalize_input_batch(batch, data_norm)
            output = model(batch)
            output = denormalize_output_batch(output, data_norm)
            output['torque'] = output['tornorm'] * output['torvec']
            output['grf'] = output['grfnorm'] * output['grfvec']
            if dumpres:
                res['torque'].append(output['torque'].cpu().numpy())
                res['grf'].append(output['grf'].cpu().numpy())
    return res
