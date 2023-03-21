import time
from matplotlib.pyplot import axis

import torch
import torch.nn as nn
import absl.flags as flags

FLAGS = flags.FLAGS

from network.fs_net_repo.PoseNet9D import PoseNet9D
from network.point_sample.pc_sample import PC_sample
from datasets.data_augmentation import defor_3D_pc
from datasets.data_augmentation import defor_3D_bb_in_batch, defor_3D_rt_in_batch, defor_3D_bc_in_batch
from losses.fs_net_loss import fs_net_loss
from losses.recon_loss import recon_6face_loss
from losses.geometry_loss import geo_transform_loss
from losses.prop_loss import prop_rot_loss
from engine.organize_loss import control_loss
from tools.training_utils import get_gt_v
from tools.lynne_lib.vision_utils import show_point_cloud


class HSPose(nn.Module):
    def __init__(self, train_stage):
        super(HSPose, self).__init__()
        self.posenet = PoseNet9D()
        self.train_stage = train_stage
        self.loss_recon = recon_6face_loss()
        self.loss_fs_net = fs_net_loss()
        self.loss_geo = geo_transform_loss()
        self.loss_prop = prop_rot_loss()
        self.name_fs_list, self.name_recon_list, \
            self.name_geo_list, self.name_prop_list = control_loss(self.train_stage)

    def forward(self, PC=None, depth=None, obj_id=None, camK=None,
                gt_R=None, gt_t=None, gt_s=None, mean_shape=None, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None, do_loss=False):
        output_dict = {}

        if PC is None:
            if self.train_stage == 'PoseNet_only':
                FLAGS.sample_method = 'basic'
                bs = depth.shape[0]
                H, W = depth.shape[2], depth.shape[3]
                sketch = torch.rand([bs, 6, H, W], device=depth.device).detach()
                PC = PC_sample(def_mask, depth, camK, gt_2D)
                if PC is None:
                    return output_dict, None
            else:
                raise NotImplementedError

        obj_mask = None
        sketch = None
        PC = PC.detach()
        if FLAGS.train:
            with torch.no_grad():
                PC_da, gt_R_da, gt_t_da, gt_s_da = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                                     aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)
                PC = PC_da
                gt_R = gt_R_da
                gt_t = gt_t_da
                gt_s = gt_s_da

        recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, \
        Pred_T, Pred_s = self.posenet(PC, obj_id)

        output_dict['mask'] = obj_mask
        output_dict['sketch'] = sketch
        output_dict['recon'] = recon
        output_dict['PC'] = PC
        output_dict['face_normal'] = face_normal
        output_dict['face_dis'] = face_dis
        output_dict['face_f'] = face_f
        output_dict['p_green_R'] = p_green_R
        output_dict['p_red_R'] = p_red_R
        output_dict['f_green_R'] = f_green_R
        output_dict['f_red_R'] = f_red_R
        output_dict['Pred_T'] = Pred_T
        output_dict['Pred_s'] = Pred_s
        output_dict['gt_R'] = gt_R
        output_dict['gt_t'] = gt_t
        output_dict['gt_s'] = gt_s

        if do_loss:
            p_recon = recon
            p_T = Pred_T
            p_s = Pred_s
            pred_fsnet_list = {
                'Rot1': p_green_R,
                'Rot1_f': f_green_R,
                'Rot2': p_red_R,
                'Rot2_f': f_red_R,
                'Recon': p_recon,
                'Tran': p_T,
                'Size': p_s,
            }

            if self.train_stage == 'Backbone_only':
                gt_green_v = None
                gt_red_v = None
            else:
                gt_green_v, gt_red_v = get_gt_v(gt_R)

            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Recon': PC,
                'Tran': gt_t,
                'Size': gt_s,
            }

            fsnet_loss = self.loss_fs_net(self.name_fs_list, pred_fsnet_list, gt_fsnet_list, sym)

            # prop loss
            pred_prop_list = {
                'Recon': p_recon,
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Scale': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_prop_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }

            prop_loss = self.loss_prop(self.name_prop_list, pred_prop_list, gt_prop_list, sym)

            pred_recon_list = {
                'F_n': face_normal,
                'F_d': face_dis,
                'F_c': face_f,
                'Rot1': p_green_R,
                'Rot1_f': f_green_R.detach(),
                'Rot2': p_red_R,
                'Rot2_f': f_red_R.detach(),
                'Tran': p_T,
                'Size': p_s,
            }

            gt_recon_list = {
                'R': gt_R,
                'T': gt_t,
                'Size': gt_s,
                'Mean_shape': mean_shape,
                'Points': PC,
            }

            recon_loss = self.loss_recon(self.name_recon_list, pred_recon_list, gt_recon_list, sym, obj_id)

            # geo loss
            pred_geo_list = {
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Size': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_geo_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }

            geo_loss = self.loss_geo(self.name_geo_list, pred_geo_list, gt_geo_list, sym)

            loss_dict = {}
            loss_dict['fsnet_loss'] = fsnet_loss
            loss_dict['recon_loss'] = recon_loss
            loss_dict['geo_loss'] = geo_loss
            loss_dict['prop_loss'] = prop_loss
        else:
            return output_dict

        return output_dict, loss_dict

    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids, check_points=False):
        """
        PC torch.Size([32, 1028, 3])
        gt_R torch.Size([32, 3, 3])
        gt_t torch.Size([32, 3])
        gt_s torch.Size([32, 3])
        mean_shape torch.Size([32, 3])
        sym torch.Size([32, 4])
        aug_bb torch.Size([32, 3])
        aug_rt_t torch.Size([32, 3])
        aug_rt_r torch.Size([32, 3, 3])
        model_point torch.Size([32, 1024, 3])
        nocs_scale torch.Size([32])
        obj_ids torch.Size([32])
        """

        def aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag):
            PC_new, gt_s_new, model_point_new = defor_3D_bb_in_batch(PC, model_point, gt_R, gt_t, gt_s + mean_shape, sym, aug_bb)
            gt_s_new = gt_s_new - mean_shape
            PC = torch.where(flag.unsqueeze(-1), PC_new, PC)
            gt_s = torch.where(flag, gt_s_new, gt_s)
            model_point_new = torch.where(flag.unsqueeze(-1), model_point_new, model_point)
            return PC, gt_s, model_point_new

        def aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag):
            PC_new, gt_R_new, gt_t_new = defor_3D_rt_in_batch(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
            PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
            gt_R_new = torch.where(flag.unsqueeze(-1), gt_R_new, gt_R)
            gt_t_new = torch.where(flag, gt_t_new, gt_t)
            return PC_new, gt_R_new, gt_t_new

        def aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag):
            pc_new, s_new, ey_up, ey_down = defor_3D_bc_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, model_point,
                                                                 nocs_scale)
            pc_new = torch.where(flag.unsqueeze(-1), pc_new, PC)
            s_new = torch.where(flag, s_new - mean_shape, gt_s)
            return pc_new, s_new, ey_up, ey_down

        def aug_pc_with_flag(PC, gt_t, flag, aug_pc_r):
            PC_new, defor = defor_3D_pc(PC, gt_t, aug_pc_r, return_defor=True)
            PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
            return PC_new, defor
        

        # augmentation
        bs = PC.shape[0]

        prob_bb = torch.rand((bs, 1), device=PC.device)
        flag = prob_bb < FLAGS.aug_bb_pro
        PC, gt_s, model_point = aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag)

        prob_rt = torch.rand((bs, 1), device=PC.device)
        flag = prob_rt < FLAGS.aug_rt_pro
        PC, gt_R, gt_t = aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag)

        # only do bc for mug and bowl
        prob_bc = torch.rand((bs, 1), device=PC.device)
        flag = torch.logical_and(prob_bc < FLAGS.aug_bc_pro, torch.logical_or(obj_ids== 5, obj_ids == 1).unsqueeze(-1))
        PC, gt_s, _, _ = aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag)

        prob_pc = torch.rand((bs, 1), device=PC.device)
        flag = prob_pc < FLAGS.aug_pc_pro
        PC, _ = aug_pc_with_flag(PC, gt_t, flag, FLAGS.aug_pc_r)

        if check_points:
            pc_reproj = torch.matmul(gt_R.transpose(-1, -2), (PC - gt_t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
            model_point *= nocs_scale.unsqueeze(-1).unsqueeze(-1)
            for i in range(len(pc_reproj)):
                show_point_cloud([pc_reproj[i].detach().cpu().numpy(), model_point[i].detach().cpu().numpy()], colors=[[0,0,1], [1,0,0]], axis_size=0.1)

        return PC, gt_R, gt_t, gt_s

    def build_params(self, training_stage_freeze=None):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        if 'pose' in training_stage_freeze:
            for param in zip(self.posenet.parameters()):
                with torch.no_grad():
                    param.requires_grad = False

        # pose
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, self.posenet.parameters()),
                "lr": float(FLAGS.lr) * FLAGS.lr_pose,
            }
        )

        return params_lr_list

