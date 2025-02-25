import torch
import torch.nn as nn
import torch.nn.functional as F

class lossLayer(nn.Module):

    def __init__(self, config, FD_model=None):
        super().__init__()
        
        self.req         = list(config.weight_dict.keys())
        self.weight_dict = config.weight_dict
        self.past_kf = config.get('PAST_KF', 2)
        self.fut_kf  = config.get('FUTURE_KF', 2)
        if 'L_fd' in self.req:
            self.FD = FD_model
            self.FD.load_state_dict(torch.load(config.FD.ckpt)['model_state'])
        
    def loss_tor_adb(self, batch, output, denormed):
        if not denormed and len(batch['torque_adb']) > 0:
            loss = F.l1_loss(batch['torque_adb'], output['torque_adb'], reduction='mean')
            return {'L_tor_adb': loss}
        else:
            return {}

    def loss_tor_imdy(self, batch, output, denormed):
        if not denormed and len(batch['torque_imdy']) > 0:
            loss = F.l1_loss(batch['torque_imdy'], output['torque_imdy'], reduction='mean')
            return {'L_tor_imdy': loss}
        else:
            return {}
            
    def loss_grf_adb(self, batch, output, denormed):
        if not denormed and len(batch['grf_adb']) > 0:
            loss = F.l1_loss(batch['grf_adb'], output['grf_adb'], reduction='mean')
            return {'L_grf_adb': loss}
        else:
            return {}
    
    def loss_grf_imdy(self, batch, output, denormed):
        if not denormed and len(batch['grf_imdy']) > 0:
            loss = F.l1_loss(batch['grf_imdy'], output['grf_imdy'], reduction='mean')
            return {'L_grf_imdy': loss}
        else:
            return {}
    
    def loss_fd_adb(self, batch, output, denormed):
        if not denormed and len(batch['fd_adb']) > 0:
            loss = F.l1_loss(batch['fd_adb'], output['fd_adb'], reduction='mean')
            return {'L_fd_adb': loss}
        else:
            return {}
    
    def loss_fd_imdy(self, batch, output, denormed):
        if not denormed and len(batch['fd_imdy']) > 0:
            loss = F.l1_loss(batch['fd_imdy'], output['fd_imdy'], reduction='mean')
            return {'L_fd_imdy': loss}
        else:
            return {}

    def loss_torque_l1(self, batch, output, denormed):
        if not denormed:
            num_good = batch['torque'].shape[0]
            loss = F.l1_loss(batch['torque'][:num_good], output['torque'][:num_good], reduction='mean')
            return {'L_torque_l1': loss}
        else:
            return {}
    
    def loss_torque_l2(self, batch, output, denormed):
        if not denormed:
            num_good = batch['torque'].shape[0]
            loss = F.mse_loss(batch['torque'][:num_good], output['torque'][:num_good], reduction='mean')
            return {'L_torque_l2': loss}
        else:
            return {}
            
    def loss_grf_l1(self, batch, output, denormed):
        if not denormed:
            num_good = batch['torque'].shape[0]
            loss = F.l1_loss(batch['grf'][:num_good], output['grf'][:num_good], reduction='mean')
            return {'L_grf_l1': loss}
        else:
            return {}
    
    def loss_grf_l2(self, batch, output, denormed):
        if not denormed:
            num_good = batch['torque'].shape[0]
            loss = F.mse_loss(batch['grf'][:num_good], output['grf'][:num_good], reduction='mean')
            return {'L_grf_l2': loss}
        else:
            return {}
        
    def loss_torque_cos(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            ang_cos = F.cosine_similarity(batch['torvec'][:num_good], output['torvec'][:num_good], dim=-1)[..., None]
            mask    = (batch['tornorm'] > 0).float()
            loss = 1 - torch.mean(ang_cos * mask)
            return {'L_torque_cos': loss}
        else:
            return {}
        
    def loss_grf_cos(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            ang_cos = F.cosine_similarity(batch['grfvec'][:num_good], output['grfvec'][:num_good], dim=-1)[..., None]
            mask    = (batch['grfnorm'] > 0).float()
            loss = 1 - torch.mean(ang_cos * mask)
            return {'L_grf_cos': loss}
        else:
            return {}
    
    def L1_pos_post(self, batch, output, denormed):
        if not denormed:
            if 'pos_post' in batch:
                dis = F.l1_loss(batch['pos_post'].flatten(1), output['pos_post'].flatten(1), reduction='mean')
            else:
                dis = F.l1_loss(batch['fpos_post'].flatten(1), output['fpos_post'].flatten(1), reduction='mean')
            return {'L1_pos_post': dis}
        else:
            return {}
    
    def L1_mkr_post(self, batch, output, denormed):
        if not denormed:
            dis = F.l1_loss(batch['mkr_post'].flatten(1), output['mkr_post'].flatten(1), reduction='mean')
            return {'L1_mkr_post': dis}
        else:
            return {}
        
    def L1_rot_post(self, batch, output, denormed):
        if not denormed:
            dis = F.l1_loss(batch['rot_post'].flatten(1), output['rot_post'].flatten(1), reduction='mean')
            return {'L1_rot_post': dis}
        else:
            return {}
    
    def loss_torque_dis(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            idx      = torch.randperm(num_good)
            dis_pred = output['tornorm'][:num_good] - output['tornorm'][idx]
            dis_orig = batch['tornorm'] - batch['tornorm'][idx]
            loss = F.l1_loss(dis_pred, dis_orig, reduction='mean')
            return {'L_torque_dis': loss}
        else:
            return {}
    
    def loss_grf_dis(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            idx      = torch.randperm(num_good)
            dis_pred = output['grfnorm'][:num_good] - output['grfnorm'][idx]
            dis_orig = batch['grfnorm'] - batch['grfnorm'][idx]
            loss = F.l1_loss(dis_pred, dis_orig, reduction='mean')
            return {'L_grf_dis': loss}
        else:
            return {}
    
    def loss_torque_norm(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            loss = F.l1_loss(batch['tornorm'][:num_good], output['tornorm'][:num_good], reduction='mean')
            return {'L_torque_norm': loss}
        else:
            return {}
    
    def loss_grf_norm(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            loss = F.l1_loss(batch['grfnorm'][:num_good], output['grfnorm'][:num_good], reduction='mean')
            return {'L_grf_norm': loss}
        else:
            return {}
    
    def loss_fd(self, batch, output, denormed):
        if not denormed:
            inbatch = {
                'pos_pre': batch['pos'][:, :self.past_kf + 1],
                'rot_pre': batch['rot'][:, :self.past_kf + 1],
                'grf_pre': batch['grf'][:, :1],
                'torque': output['torque'],
                'pos_post': batch['pos'][:, self.past_kf + 1:self.past_kf + 2],
                'rot_post': batch['rot'][:, self.past_kf + 1:self.past_kf + 2],
            }
            oubatch = self.FD(inbatch)
            loss    = F.l1_loss(inbatch['pos_post'].flatten(1), oubatch['pos_post'], reduction='mean') + F.l1_loss(inbatch['rot_post'].flatten(1), oubatch['rot_post'], reduction='mean')
            return {'L_fd': loss}
        else:
            return {}
        
    def loss_contact(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            grf  = (batch['grfnorm'] > 1e-4).float().flatten(1) # B, L, J
            loss = F.binary_cross_entropy_with_logits(output['contact'][:num_good].flatten(1), grf)
            return {'L_contact': loss}
        else:
            return {}
        
    def loss_adv(self, batch, output, denormed):
        if not denormed:
            loss = F.binary_cross_entropy_with_logits(output['identifier'].flatten(), batch['identifier'].flatten())
            return {'L_adv': loss}
        else:
            return {}
            
    def loss_temporal(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            ori_tempo_dis = batch['grf'][:num_good, 1] - batch['grf'][:num_good, 0]
            rec_tempo_dis = output['grf'][:num_good, 1] - output['grf'][:num_good, 0]
            loss = F.l1_loss(ori_tempo_dis, rec_tempo_dis)
            return {'L_temp': loss}
        else:
            return {}
            
    def loss_cls(self, batch, output, denormed):
        if not denormed:
            loss = F.binary_cross_entropy_with_logits(output['indicator'], batch['indicator'])
            return {'L_cls': loss}
        else:
            return {}
    
    def loss_cls_dis(self, batch, output, denormed):
        if not denormed:
            num_good = batch['torque'].shape[0]
            b    = batch['indicator'].shape[0] // 2
            prob = torch.sigmoid(output['indicator'])
            loss = torch.clamp(torch.mean(prob[:num_good]) - torch.mean(prob[num_good:b]) + 0.1, 0, None) + torch.mean(1 + prob[b:] - prob[:b])
            return {'L_clsdis': loss}
        else:
            return {}
    
    def loss_work(self, batch, output, denormed):
        if denormed:
            num_good = batch['torque'].shape[0]
            ori_work = torch.sum(batch['angvel'][:num_good, self.past_kf, 1:] * batch['torque'][:num_good, 0], dim=-1) / 60
            rec_work = torch.sum(batch['angvel'][:num_good, self.past_kf, 1:] * output['torque'][:num_good, 0], dim=-1) / 60
            loss = F.l1_loss(ori_work, rec_work)
            return {'L_work': loss}
        else:
            return {}
            
    
    def get_loss(self, loss, batch, output, denormed):
        loss_map = {
            'L_torque_l1': self.loss_torque_l1,
            'L_torque_l2': self.loss_torque_l2,
            'L_grf_l1': self.loss_grf_l1,
            'L_grf_l2': self.loss_grf_l2,
            'L1_pos_post': self.L1_pos_post,
            'L1_rot_post': self.L1_rot_post,
            'L1_mkr_post': self.L1_mkr_post,
            'L_fd': self.loss_fd,
            'L_contact': self.loss_contact,
            'L_torque_cos': self.loss_torque_cos,
            'L_grf_cos': self.loss_grf_cos,
            'L_torque_dis': self.loss_torque_dis,
            'L_grf_dis': self.loss_grf_dis,
            'L_torque_norm': self.loss_torque_norm,
            'L_grf_norm': self.loss_grf_norm,
            'L_adv': self.loss_adv,
            'L_temp': self.loss_temporal,
            'L_cls': self.loss_cls,
            'L_work': self.loss_work,
            'L_clsdis': self.loss_cls_dis,
            'L_tor_adb': self.loss_tor_adb,
            'L_tor_imdy': self.loss_tor_imdy,
            'L_grf_adb': self.loss_grf_adb,
            'L_grf_imdy': self.loss_grf_imdy,
            'L_fd_adb': self.loss_fd_adb,
            'L_fd_imdy': self.loss_fd_imdy,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](batch, output, denormed)

    def forward(self, batch, output, denormed=False):
        # Compute all the requested losses
        losses = {}
        if denormed:
            if 'torque' in batch:
                batch['tornorm']  = torch.linalg.vector_norm(batch['torque'], dim=-1)[..., None]
                batch['torvec']   = F.normalize(batch['torque'], dim=-1)
            if 'torque' in output:
                if 'tornorm' not in output:
                    output['tornorm'] = torch.linalg.vector_norm(output['torque'], dim=-1)[..., None]
                if 'torvec' not in output:
                    output['torvec'] = F.normalize(output['torque'], dim=-1)
            if 'grf' in batch:
                batch['grfnorm']  = torch.linalg.vector_norm(batch['grf'], dim=-1)[..., None]
                batch['grfvec']   = F.normalize(batch['grf'])
            if 'grf' in output:
                if 'grfnorm' not in output:
                    output['grfnorm'] = torch.linalg.vector_norm(output['grf'], dim=-1)[..., None]
                if 'grfvec' not in output:
                    output['grfvec'] = F.normalize(output['grf'])
        for loss in self.req:
            losses.update(self.get_loss(loss, batch, output, denormed))
        return losses
