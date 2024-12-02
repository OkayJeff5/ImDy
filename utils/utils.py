import argparse
from collections import OrderedDict
import shutil
import zipfile
from omegaconf import OmegaConf
import os
import logging
import time
import torch
import numpy as np
import random

def create_logger(config) -> logging.Logger:
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if hasattr(config, 'RUN_NAME'):
        if config.get('timestamp', True):
            config.RUN_NAME = current_time + '_' + config.get('RUN_NAME', '')
    else:
        config.RUN_NAME = current_time

    config.RUN_PATH = os.path.join('exp', config.RUN_NAME)
    os.makedirs(config.RUN_PATH, exist_ok=True)
    
    log_format  = '%(asctime)s %(message)s'
    time_format = '%Y-%m-%d %H:%M:%S'
    logger      = logging.getLogger()
    logger.setLevel(eval(f'logging.{config.LOGGING_LEVEL.upper()}'))
    formatter = logging.Formatter(log_format, datefmt=time_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file     = os.path.join(config.RUN_PATH, config.RUN_NAME + '.log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def backup_config_file(config):
    OmegaConf.save(config=config, f=os.path.join(config.RUN_PATH, 'config.yaml'))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)