import argparse
import random
import shutil
from collections import OrderedDict
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torch import optim
import os
import tqdm
import warnings
import json
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils.utils import backup_config_file, create_logger, set_seed
from utils.optimizer import build_optimizer, build_lr_scheduler
from dataset import load_data_norm, get_preprocess_fn, rawTorqueDataset, get_collate_fn, determinedTorqueDataset, onDiskDataset, cartisianTorqueDataset
from engine import train_one_epoch, evaluate, calc_norm
from models import get_model, get_loss, get_metric


def train(config, network, criterion, metric, optimizer, scheduler, train_loader, train_preprocess_fn, test_loader, test_preprocess_fn, data_norm,
          logger, writer, device, global_step, max_norm, start_ep, adversarial_assets):
    logger.info(f"Length of train loader: {len(train_loader)}")
    use_wandb = config.get('USE_WANDB', False)
    for epoch in range(start_ep, config.TRAIN.EPOCHS):
        
        train_stats, global_step = train_one_epoch(network, criterion, train_loader, train_preprocess_fn, data_norm, 
                                                   optimizer, writer, device, epoch, max_norm, global_step, 1 if epoch < config.TRAIN.EPOCHS / 10 else .8,
                                                   adversarial_assets)
        if scheduler is not None:
            scheduler.step()
        
        if use_wandb:
            wandb_log = {}
        
        if (epoch + 1) % config.log_interval == 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch + 1,}
            logger.info(json.dumps(log_stats))
            if use_wandb:
                wandb_log.update({k: v for k, v in log_stats.items() if v != []})
        
        if (epoch + 1) % config.eval_interval == 0:
            eval_stats = evaluate(network, metric, test_loader, test_preprocess_fn, data_norm, writer, device, epoch)
            log_stats = {**{f'test_{k}': v for k, v in eval_stats.items()},
                        'epoch': epoch + 1,}
            logger.info(json.dumps(log_stats))
            if use_wandb:
                wandb_log.update({k: v for k, v in log_stats.items() if v != []})

        if use_wandb:
            wandb.log(wandb_log)
        
        if (epoch + 1) % config.save_interval == 0:
            logger.info("Saving checkpoint...")
            checkpoint = {
                'epoch': epoch, # from 0
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
                'global_step': global_step,
                'adv_model_state': adversarial_assets['model'].state_dict if adversarial_assets is not None else None,
                'adv_optimizer_state': adversarial_assets['optimizer'].state_dict if adversarial_assets is not None else None,
            }
            if config.DEVICE_STR.startswith('cuda'):
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device=config.DEVICE_STR)
            ckpt_save_path = os.path.join(config.RUN_PATH, 'checkpoint.pth')
            torch.save(checkpoint, os.path.join(config.RUN_PATH, f'epoch_{epoch+1}.pt'))

def sync_config(config):
    config.TRAIN.PAST_KF   = config.get('PAST_KF', 2)
    config.TRAIN.FUTURE_KF = config.get('FUTURE_KF', 2)
    config.TRAIN.rot_rep   = config.get('rot_rep', 'quat')
    config.TRAIN.use_norm  = config.get('use_norm', False)
    
    config.TEST.PAST_KF   = config.get('PAST_KF', 2)
    config.TEST.FUTURE_KF = config.get('FUTURE_KF', 2)
    config.TEST.rot_rep   = config.get('rot_rep', 'quat')
    config.TEST.use_norm  = config.get('use_norm', False)
    
    config.MODEL.PAST_KF   = config.get('PAST_KF', 2)
    config.MODEL.FUTURE_KF = config.get('FUTURE_KF', 2)
    config.MODEL.rot_rep   = config.get('rot_rep', 'quat')
    config.MODEL.use_norm  = config.get('use_norm', False)
    
    config.DATASET.TRAIN.PAST_KF   = config.get('PAST_KF', 2)
    config.DATASET.TRAIN.FUTURE_KF = config.get('FUTURE_KF', 2)
    config.DATASET.TRAIN.rot_rep   = config.get('rot_rep', 'quat')
    config.DATASET.TRAIN.use_norm  = config.get('use_norm', False)
    
    config.DATASET.TEST.PAST_KF   = config.get('PAST_KF', 2)
    config.DATASET.TEST.FUTURE_KF = config.get('FUTURE_KF', 2)
    config.DATASET.TEST.rot_rep   = config.get('rot_rep', 'quat')
    config.DATASET.TEST.use_norm  = config.get('use_norm', False)
    
    return config

if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    if not hasattr(cli_conf, 'config_path'):
        cli_conf.config_path = 'configs/naive.yml'
    config   = OmegaConf.merge(OmegaConf.load(cli_conf.config_path), cli_conf)
    seed     = config.get('seed', 42)
    if config.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore")
    config.DEVICE_STR = f"cuda:{config.DEVICE}" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.DEVICE_STR)
    
    set_seed(seed)
    config = sync_config(config)
    logger = create_logger(config)
    writer = SummaryWriter(config.RUN_PATH)
    backup_config_file(config)

    if config.get('USE_WANDB', False):
        config_dict = {
            'model': config.MODEL.NAME,
            'optimizer': config.OPTIMIZER.TYPE,
            'learning_rate': config.OPTIMIZER.LR.base,
            'batch_size': config.TRAIN.BATCH_SIZE,
            'max_norm': config.TRAIN.max_norm
        }
        if config.MODEL.NAME == 'naiveMLP':
            config_dict['mlp_units'] = config.MODEL.MLP.units
        wandb.init(
            project='MotionEfforts',
            name=config.RUN_NAME,
            config=config_dict
        )

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"RUN name: {config.RUN_NAME}")
    logger.info(f"Using device: {config.DEVICE_STR}")
    logger.info("Initializing dataset...")
    
    
    # train_dataset = rawTorqueDataset(config.DATASET.TRAIN, split='train')
    if config.DATASET.TRAIN.get('MODE', 'raw') == 'adb':
        train_dataset = onDiskDataset(config.DATASET.TRAIN, split='train')
    elif config.DATASET.TRAIN.get('MODE', 'raw') == 'mkr':
        train_dataset = onDiskDataset(config.DATASET.TRAIN, split='train')
    else:
        train_dataset = determinedTorqueDataset(config.DATASET.TRAIN, split='train')
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True if not config.TRAIN.DEBUG else False,
        drop_last=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=config.TRAIN.PREFETCH,
        collate_fn=get_collate_fn(config.DATASET.TRAIN),
        pin_memory=True,
    )
    train_preprocess_fn = get_preprocess_fn(config.DATASET.TRAIN, device)
    if config.USE_NORM:
        data_norm = calc_norm(train_loader, train_preprocess_fn, device, config.DATASET.TRAIN)
    if config.DATASET.TEST.get('MODE', 'raw') == 'adb':
        test_dataset = onDiskDataset(config.DATASET.TEST, split='test')
    elif config.DATASET.TEST.get('MODE', 'raw') == 'mkr':
        test_dataset = onDiskDataset(config.DATASET.TEST, split='test')
    else:
        test_dataset = determinedTorqueDataset(config.DATASET.TEST, split='test')
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.TEST.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=config.TEST.PREFETCH,
        collate_fn=get_collate_fn(config.DATASET.TEST),
        pin_memory=True,
    )
    test_preprocess_fn  = get_preprocess_fn(config.DATASET.TEST, device)
    
    
    logger.info("Initializing network...")
    network   = get_model(config).to(device)
    criterion = get_loss(config.LOSS).to(device)
    metric    = get_metric(config.METRIC).to(device)

   #freeze encoder weight
    for key, param in network.named_parameters():
        if key.startswith('ID') and not key.startswith('ID_outProj'):
            param.requires_grad = False
   

    logger.info("Initializing optimizer...")
    optimizer = build_optimizer(network, config)
    scheduler = build_lr_scheduler(config, optimizer)
    
    adversarial_assets = None
    if config.get('adversarial', False):
        adversarial_assets = {}
        adversarial_assets['model'] = get_model(config.adversarial).to(device)
        adversarial_assets['criterion'] = get_loss(config.adversarial.LOSS).to(device)
        adversarial_assets['optimizer'] = build_optimizer(adversarial_assets['model'], config.adversarial)
        adversarial_assets['simultaneous'] = config.adversarial.get('simultaneous', False)

    if config.get('RESUME', False):
        checkpoint = torch.load(config.RESUME.RESUME_CKPT, map_location='cpu')
        network.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None and checkpoint['scheduler_state'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        if config.DEVICE_STR.startswith('cuda'):
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=config.DEVICE_STR)
        if adversarial_assets is not None:
            adversarial_assets['model'].load_state_dict(checkpoint['adv_model_state'])
            adversarial_assets['optimizer'].load_state_dict(checkpoint['adv_optimizer_state'])
            
        logger.info(f"Resuming from epoch {start_epoch+1}")
    else:
        start_epoch = 0
        global_step = 0
    
    if config.get('profile', False):
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
            with record_function("model_inference"):
                train_stats, global_step = train_one_epoch(network, criterion, train_loader, train_preprocess_fn, data_norm, 
                                                           optimizer, writer, device, 0, config.TRAIN.max_norm, global_step, 1,
                                                           adversarial_assets, True)
        prof.export_stacks(os.path.join(config.RUN_PATH, "profiler_stacks_cpu.txt"), "self_cpu_time_total")
        prof.export_stacks(os.path.join(config.RUN_PATH, "profiler_stacks_cuda.txt"), "self_cuda_time_total")

    
    logger.info("Start training...")
    train(config, network, criterion, metric, optimizer, scheduler, train_loader, train_preprocess_fn, test_loader, test_preprocess_fn, data_norm, 
          logger, writer, device, global_step, config.TRAIN.max_norm, start_epoch, adversarial_assets)
    logger.info("Training finished.")