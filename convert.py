import numpy as np
import torch
import os
import tqdm
import smplx
import joblib
from utils.geometry import *
from utils.marker_vids import smpl_opensim_markerset, all_marker_vids

marker_vid = list(smpl_opensim_markerset.values())

body_model = smplx.create('models', model_type='smpl', gender='neutral', use_face_contour=True, num_betas=10, ext='pkl', use_pca=False, create_global_orient=False, create_body_pose=False, create_left_hand_pose=False, create_right_hand_pose=False, create_jaw_pose=False, create_leye_pose=False, create_reye_pose=False, create_betas=False, create_expression=False, create_transl=False,)

for suffix in ['train', 'test']:
    pos    = joblib.load(f'data/raw_{suffix}/pos.pkl')
    rot    = joblib.load(f'data/raw_{suffix}/rot.pkl')
    tor    = joblib.load(f'data/raw_{suffix}/torque.pkl')
    grf    = joblib.load(f'data/raw_{suffix}/grf.pkl')
    parents    = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    
    outdir = f'data/imdy_{suffix}'
    os.makedirs(outdir, exist_ok=True)
    
    for idx in tqdm.trange(len(pos)):
        nf            = pos[idx].shape[0]
        betas         = torch.zeros(nf, 10)
        trans         = torch.from_numpy(pos[idx][:, 0])
        poses_global  = torch.from_numpy(rot[idx])
        pre_rot       = torch.Tensor([[[.5, .5, .5, .5]]]).expand(nf, 24, -1)
        poses_global  = quaternion_multiply(poses_global, pre_rot)
        poses_inverse = quaternion_invert(poses_global)[:, parents]
        poses_local   = quaternion_multiply(poses_inverse, poses_global)
        poses         = quaternion_to_axis_angle(torch.cat([poses_global[:, :1], poses_local[:, 1:]], dim=1)).view(nf, -1)
        with torch.no_grad():
            smpl_output   = body_model(betas=betas, transl=trans, global_orient=poses[..., :3], body_pose=poses[..., 3:], return_verts=True, return_shaped=False)
        qpos  = poses.view(nf, -1, 3)
        jpos  = smpl_output.joints.cpu()
        verts = smpl_output.vertices.cpu()
        mpos  = verts[:, marker_vid]
            
        joblib.dump({
            'qpos': qpos.numpy(),
            'jpos': jpos.numpy(),
            'mpos': mpos.numpy(),
            'torque': tor[idx], 
            'grf': grf[idx],
        }, f'{outdir}/{idx}.pkl')
