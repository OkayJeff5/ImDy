import torch
from .MLP import naiveMLP
from .IDFD import IDFD
from .IDFD_mkr import mkrIDFD
from .IDFD_mix import mixIDFD
from .trans import naiveTrans
from .losses import lossLayer
from .metrics import metricLayer

model_dict = {
    'naiveMLP': naiveMLP,
    'IDFD': IDFD,
    'naiveTrans': naiveTrans,
    'mkrIDFD': mkrIDFD,
    'mixIDFD': mixIDFD,
}

def get_model(config):
    model = model_dict[config.MODEL.NAME](config.MODEL)
    if config.MODEL.get('ckpt', None):
        model.load_state_dict(torch.load(config.MODEL.ckpt, map_location='cpu')['model_state'])
    return model
    
def get_loss(config):
    FD_model = None
    if 'L_fd' in list(config.weight_dict.keys()):
        FD_model = model_dict[config.FD.NAME](config.FD)
    return lossLayer(config, FD_model)
    
def get_metric(config):
    return metricLayer(config)