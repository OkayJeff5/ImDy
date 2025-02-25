import torch
import torch.nn as nn
import torch.nn.functional as F

class metricLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.req         = list(config.req)
        
    def L1_torque(self, batch, output):
        err = torch.abs(batch['torque'] - output['torque'])
        rel = err / batch['torque'].abs()
        return {
            'L1_torque': err.mean(), 
            'RelL1_torque': rel.mean(),
            'tau': torch.mean(err / batch['weight'][:, None, None]),
        }
    
    def L2_grf(self, batch, output):
        err     = torch.linalg.vector_norm(batch['torque'] - output['torque'], dim=-1)
        rel     = err / batch['torque'].abs()
        lambda_ = err / batch['weight'][:, None, None, None]
        return {
            'L2_grf': err.mean(), 
            'RelL2_grf': rel.mean(),
            'lambda': lambda_.mean(),
            'lambda_lf': lambda_[:, :, 1].mean(),
            'lambda_rf': lambda_[:, :, 0].mean(),
        }
        
    def mix_metric(self, batch, output):
        result = {}
        if len(batch['torque_adb']) > 0:
            result.update({
                'tau_adb': torch.mean(torch.abs(batch['torque_adb'].flatten(1) - output['torque_adb'].flatten(1)) / batch['weight_adb'][:, None]),
                'lambda_adb': torch.mean(torch.linalg.vector_norm(batch['grf_adb'] - output['grf_adb'], dim=-1) / batch['weight_adb'][:, None, None, None]),
            })
        if len(batch['torque_imdy']) > 0:
            result.update({
                'tau_imdy': torch.mean(torch.abs(batch['torque_imdy'].flatten(1) - output['torque_imdy'].flatten(1)) / batch['weight_imdy'][:, None]),
                'lambda_imdy': torch.mean(torch.linalg.vector_norm(batch['grf_imdy'] - output['grf_imdy'], dim=-1) / batch['weight_imdy'][:, None, None, None]),
            })
        return result
    
    def MSE_torque(self, batch, output):
        ang_cos = F.cosine_similarity(batch['torvec'], output['torvec'], dim=-1)[..., None]
        ang_cos[batch['tornorm'] * output['tornorm'] == 0] = 1.
        ang_err = torch.arccos(torch.clamp(ang_cos, -1, 1))
        mag_dis = torch.abs(batch['tornorm'] - output['tornorm'])
        mag_tor = torch.where(batch['tornorm'] == 0, mag_dis, batch['tornorm'])
        mag_rel = mag_dis / (mag_tor + 1e-8)
        dis = batch['torque'] - output['torque']
        dis = torch.linalg.vector_norm(dis, dim=-1)[..., None]
        tor = torch.where(batch['tornorm'] == 0, dis, batch['tornorm'])
        rel = dis / (tor + 1e-8)
        MSE_torque = dis.mean()
        return {
            'MSE_torque': MSE_torque, 
            'RelMSE_torque': rel.mean(), 
            'MagErr_torque': mag_dis.mean(), 
            'RelMagErr_torque': mag_rel.mean(),
            'Ang_torque': ang_err.mean(),
            'tau': MSE_torque / (76.3869 * 60),
        }
    
    def MSE_grf(self, batch, output):
        if 'contact' in output:
            contact = (output['contact'] > .5).float()
            output['grfnorm'] *= contact
            output['grfvec']  *= contact
            output['grf']     *= contact
        ang_cos = F.cosine_similarity(batch['grfvec'], output['grfvec'], dim=-1)[..., None]
        ang_cos[batch['grfnorm'] * output['grfnorm'] == 0] = 1.
        ang_err = torch.arccos(torch.clamp(ang_cos, -1, 1))
        mag_dis = torch.abs(batch['grfnorm'] - output['grfnorm'])
        mag_grf = torch.where(batch['grfnorm'] == 0, mag_dis, batch['grfnorm'])
        mag_rel = mag_dis / (mag_grf + 1e-8)
        mag_rel = torch.where(mag_rel > 1., 1., mag_rel)
        dis = batch['grf'] - output['grf']
        dis = torch.linalg.vector_norm(dis, dim=-1)[..., None]
        grf = torch.where(batch['grfnorm'] == 0, dis, batch['grfnorm'])
        rel = dis / (grf + 1e-8)
        rel = torch.where(rel > 1., 1., rel)
        dis_f = batch['grf'][:, :, [7, 8]] + batch['grf'][:, :, [10, 11]] - output['grf'][:, :, [7, 8]] - output['grf'][:, :, [10, 11]]
        dis_f = torch.linalg.vector_norm(dis_f, dim=-1)
        MSE_grf = dis.mean()
        MSE_grf_lf = dis_f[:, :, 0].mean()
        MSE_grf_rf = dis_f[:, :, 1].mean()
        return {
            'MSE_grf_lf': MSE_grf_lf,
            'MSE_grf_rf': MSE_grf_rf,
            'MSE_grf': MSE_grf, 
            'RelMSE_grf': rel.mean(), 
            'MagErr_grf': mag_dis.mean(), 
            'RelMagErr_grf': mag_rel.mean(),
            'Ang_grf': ang_err.mean(),
            'lamda': MSE_grf / 76.3869,
            'lamda_lf': MSE_grf_lf / 76.3869,
            'lamda_rf': MSE_grf_rf / 76.3869,
        }
    
    def L1_pos_post(self, batch, output):
        dis = F.l1_loss(batch['pos_post'].flatten(1), output['pos_post'])
        return {'L1_pos_post': dis}
        
    def L1_rot_post(self, batch, output):
        dis = F.l1_loss(batch['rot_post'].flatten(1), output['rot_post'])
        return {'L1_rot_post': dis}
    
    def get_loss(self, loss, batch, output):
        loss_map = {
            'MSE_torque': self.MSE_torque,
            'MSE_grf': self.MSE_grf,
            'L1_pos_post': self.L1_pos_post,
            'L1_rot_post': self.L1_rot_post,
            'L1_torque': self.L1_torque,
            'L2_grf':self.L2_grf,
            'mix_metric':self.mix_metric,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](batch, output)

    def forward(self, batch, output):
        # Compute all the requested losses
        if 'torque' in batch and 'MSE_torque' in self.req:
            batch['tornorm']  = torch.linalg.vector_norm(batch['torque'], dim=-1)[..., None]
            batch['torvec']   = F.normalize(batch['torque'], dim=-1)
        
        if 'torque' not in output and 'torque_imdy' not in output and 'torque_adb' not in output:
            output['torque'] = output['torvec'] * output['tornorm']
        elif 'MSE_torque' in self.req:
            if 'tornorm' not in output:
                output['tornorm'] = torch.linalg.vector_norm(output['torque'], dim=-1)[..., None]
            if 'torvec' not in output:
                output['torvec'] = F.normalize(output['torque'], dim=-1)
            
        if 'grf' in batch and 'MSE_grf' in self.req:
            batch['grfnorm']  = torch.linalg.vector_norm(batch['grf'], dim=-1)[..., None]
            batch['grfvec']   = F.normalize(batch['grf'], dim=-1)
        if 'grf' not in output and 'grf_imdy' not in output and 'grf_adb' not in output:
            output['grf'] = output['grfvec'] * output['grfnorm']
        elif 'MSE_grf' in self.req:
            if 'grfnorm' not in output:
                output['grfnorm'] = torch.linalg.vector_norm(output['grf'], dim=-1)[..., None]
            if 'grfvec' not in output:
                output['grfvec'] = F.normalize(output['grf'], dim=-1)

        metrics = {}
        for loss in self.req:
            metrics.update(self.get_loss(loss, batch, output))
        return metrics
