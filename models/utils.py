import torch.nn as nn
import torch.nn.functional as F
import torch

def get_dim(config, key):
    if key == 'rot':
        rot_rep = config.get('rot_rep', 'quat')
        if rot_rep == 'quat':
            rotdim = 4
        elif rot_rep == 'aa':
            rotdim = 3
        elif rot_rep == 'rot6d':
            rotdim = 6
        return [config.PAST_KF + config.FUTURE_KF + 2, 24, rotdim], (config.PAST_KF + config.FUTURE_KF + 2) * 24 * rotdim
    elif key == 'rot_pre':
        rot_rep = config.get('rot_rep', 'quat')
        if rot_rep == 'quat':
            rotdim = 4
        elif rot_rep == 'aa':
            rotdim = 3
        elif rot_rep == 'rot6d':
            rotdim = 6
        return [config.PAST_KF + 1, 24, rotdim], (config.PAST_KF + 1) * 24 * rotdim
    elif key == 'rot_post':
        rot_rep = config.get('rot_rep', 'quat')
        if rot_rep == 'quat':
            rotdim = 4
        elif rot_rep == 'aa':
            rotdim = 3
        elif rot_rep == 'rot6d':
            rotdim = 6
        return [1, 24, rotdim], 24 * rotdim
    elif key == 'pos':
        return [config.PAST_KF + config.FUTURE_KF + 2, 1, 3], (config.PAST_KF + config.FUTURE_KF + 2) * 1 * 3
    elif key == 'fpos':
        return [config.PAST_KF + config.FUTURE_KF + 2, 24, 3], (config.PAST_KF + config.FUTURE_KF + 2) * 24 * 3
    elif key == 'vel':
        return [config.PAST_KF + config.FUTURE_KF + 2, 1, 3], (config.PAST_KF + config.FUTURE_KF + 2) * 1 * 3
    elif key == 'fvel':
        return [config.PAST_KF + config.FUTURE_KF + 2, 24, 3], (config.PAST_KF + config.FUTURE_KF + 2) * 24 * 3
    elif key == 'angvel':
        return [config.PAST_KF + config.FUTURE_KF + 2, 24, 3], (config.PAST_KF + config.FUTURE_KF + 2) * 24 * 3
    elif key == 'pos_pre':
        return [config.PAST_KF + 1, 1, 3], (config.PAST_KF + 1) * 1 * 3
    elif key == 'fpos_pre':
        return [config.PAST_KF + 1, 24, 3], (config.PAST_KF + 1) * 24 * 3
    elif key == 'pos_post':
        return [1, 1, 3], 3
    elif key == 'fpos_post':
        return [1, 24, 3], 72
    elif key == 'torque':
        if config.get('adb_tor', False):
            return [1, 17], 17
        return [1, 23, 3], 69
    elif key == 'torvec':
        return [1, 23, 3], 69
    elif key == 'tornorm':
        return [1, 23, 1], 23
    elif key == 'grf':
        if config.get('adb_tor', False):
            return [2, 2, 3], 12
        return [2, 24, 3], 144
    elif key == 'grfnorm':
        return [2, 24, 1], 48
    elif key == 'grfvec':
        return [2, 24, 3], 144
    elif key == 'grf_pre':
        if config.get('adb_tor', False):
            return [1, 2, 3], 6
        return [1, 24, 3], 72
    elif key == 'contact':
        if config.get('adb_tor', False):
            return [2, 2, 1], 4
        return [2, 24, 1], 48
    elif key == 'identifier':
        return [1, 1, 1], 1
    elif key == 'mkr':
        return [None, (config.PAST_KF + config.FUTURE_KF + 2), 3], None
    elif key == 'mkr_pre':
        return [None, (config.PAST_KF + 1), 3], None
    elif key == 'mkr_post':
        return [None, 1, 3], None
    elif key == 'mvel':
        return [None, (config.PAST_KF + config.FUTURE_KF + 2), 3], None

class vecPostLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.normalize(x, dim=-1)

class clampLayer(nn.Module):
    def __init__(self, thresh=None):
        super().__init__()
        self.thresh = thresh
    def forward(self, x):
        if self.thresh is not None:
            self.thresh = self.thresh.to(x.device)
        return torch.clamp(F.softplus(x), torch.zeros_like(x), self.thresh)


class sigmoidLayer(nn.Module):
    def __init__(self, thresh=None):
        super().__init__()
        self.thresh = thresh
    def forward(self, x):
        return F.sigmoid(x) * self.thresh


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye",
    "right_eye",
]
joint_torque_limits = [2500., 2500., 2500., 2600., 2600., 2600., 2500., 2500., 2500., 2000., 2000., 1500., 2000., 2000., 1500., 2000., 2000., 2000., 2000., 1500., 1500., 1200., 1200.,]

def get_postprocess(key, type=None):
    if key == 'torvec':
        return vecPostLayer()
    elif key == 'tornorm':
        if type is None or type == 'clamp':
            return clampLayer(torch.Tensor([[joint_torque_limits]])[..., None])
        elif type == 'sigmoid':
            return sigmoidLayer(torch.Tensor([[joint_torque_limits]])[..., None])
        else:
            return nn.Identity()
    elif key == 'grfvec':
        return vecPostLayer()
    elif key == 'grfnorm':
        return clampLayer(None)
    else:
        return nn.Identity()

acti_dict = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leakyrelu': nn.LeakyReLU,
}

norm_dict = {
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d,
}