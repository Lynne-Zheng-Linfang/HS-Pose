import time

import torch
import torch.nn as nn
from tools.geom_utils import batch_dot
import absl.flags as flags
from absl import app
FLAGS = flags.FLAGS  # can control the weight of each term here


class fs_net_loss(nn.Module):
    def __init__(self):
        super(fs_net_loss, self).__init__()
        if FLAGS.fsnet_loss_type == 'l1':
            self.loss_func_t = nn.L1Loss()
            self.loss_func_s = nn.L1Loss()
            self.loss_func_Rot1 = nn.L1Loss()
            self.loss_func_Rot2 = nn.L1Loss()
            self.loss_func_r_con = nn.L1Loss()
            self.loss_func_Recon = nn.L1Loss()
        elif FLAGS.fsnet_loss_type == 'smoothl1':   # same as MSE
            self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot1 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot2 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_r_con = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Recon = nn.SmoothL1Loss(beta=0.3)
        else:
            raise NotImplementedError

    def forward(self, name_list, pred_list, gt_list, sym):
        loss_list = {}
        if "Rot1" in name_list:
            # rot1  0.0003094673156738281
            loss_list["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_list["Rot1"])

        if "Rot1_cos" in name_list:
            #rot1_cos 0.0010631084442138672
            loss_list["Rot1_cos"] = FLAGS.rot_1_w * self.cal_cosine_dis(pred_list["Rot1"], gt_list["Rot1"])

        if "Rot2" in name_list:
            # rot2  0.0013346672058105469 
            loss_list["Rot2"] = FLAGS.rot_2_w * self.cal_loss_Rot2(pred_list["Rot2"], gt_list["Rot2"], sym)

        if "Rot2_cos" in name_list:
            # rot2_cos 0.0018091201782226562 
            loss_list["Rot2_cos"] = FLAGS.rot_2_w * self.cal_cosine_dis_sym(pred_list["Rot2"], gt_list["Rot2"], sym)

        if "Rot_regular" in name_list:
            # rot_r_a 0.0016679763793945312 
            loss_list["Rot_r_a"] = FLAGS.rot_regular * self.cal_rot_regular_angle(pred_list["Rot1"],
                                  pred_list["Rot2"], sym)

        if "Recon" in name_list:
            # not used
            begin = time.time()
            loss_list["Recon"] = FLAGS.recon_w * self.cal_loss_Recon(
                                                    pred_list["Recon"], 
                                                    gt_list["Recon"])
            print('recon', time.time() - begin)
            exit()

        if "Tran" in name_list:
            # tran 0.00024175643920898438
            loss_list["Tran"] = FLAGS.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])

        if "Size" in name_list:
            # size 0.00023317337036132812
            loss_list["Size"] = FLAGS.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])

        if "R_con" in name_list:
            # r_con 0.0017731189727783203
            loss_list["R_con"] = FLAGS.r_con_w * self.cal_loss_R_con(
                                pred_list["Rot1"], pred_list["Rot2"],
                                gt_list["Rot1"], gt_list["Rot2"],
                                pred_list["Rot1_f"], pred_list["Rot2_f"], sym)
        return loss_list

    def cal_loss_R_con_old(self, p_rot_g, p_rot_r, g_rot_g, g_rot_r, p_g_con, p_r_con, sym):
        dis_g = p_rot_g - g_rot_g    # bs x 3
        dis_g_norm = torch.norm(dis_g, dim=-1)   # bs
        p_g_con_gt = torch.exp(-13.7 * dis_g_norm * dis_g_norm)  # bs
        res_g = self.loss_func_r_con(p_g_con_gt, p_g_con)
        res_r = 0.0
        bs = p_rot_g.shape[0]
        for i in range(bs):
            if sym[i, 0] == 0:
                dis_r = p_rot_r[i, ...] - g_rot_r[i, ...]
                dis_r_norm = torch.norm(dis_r)   # 1
                p_r_con_gt = torch.exp(-13.7 * dis_r_norm * dis_r_norm)
                res_r += self.loss_func_r_con(p_r_con_gt, p_r_con[i])
        res_r = res_r / bs
        return res_g + res_r

    def cal_loss_R_con(self, p_rot_g, p_rot_r, g_rot_g, g_rot_r, p_g_con, p_r_con, sym):
        dis_g = p_rot_g - g_rot_g  # bs x 3
        dis_g_norm = torch.norm(dis_g, dim=-1)  # bs
        p_g_con_gt = torch.exp(-13.7 * dis_g_norm * dis_g_norm)  # bs
        res_g = self.loss_func_r_con(p_g_con_gt, p_g_con)

        dis_r = p_rot_r - g_rot_r
        dis_r_norm = torch.norm(dis_r, dim=-1)
        p_r_con_gt = torch.exp(-13.7 * dis_r_norm * dis_r_norm)
        sym_flag = sym[:, 0] == 0
        new_p_r_con_gt = torch.where(sym_flag, p_r_con_gt, torch.zeros_like(p_r_con_gt))
        new_p_r_con = torch.where(sym_flag, p_r_con, torch.zeros_like(p_r_con))
        # might need to ajust the mean, since the sample number of sym==0 is not batch size
        res_r = self.loss_func_r_con(new_p_r_con_gt, new_p_r_con)
        res = res_r + res_g
        return res

    def cal_loss_Rot1_old(self, pred_v, gt_v):
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
        for i in range(bs):
            pred_v_now = pred_v[i, ...]
            gt_v_now = gt_v[i, ...]
            res[i] = self.loss_func_Rot1(pred_v_now, gt_v_now)
        res = torch.mean(res)
        return res

    def cal_loss_Rot1(self, pred_v, gt_v):

        # loss_func_Rot1 = nn.L1Loss()
        res = self.loss_func_Rot1(pred_v, gt_v)
        return res

    def cal_loss_Rot2_old(self, pred_v, gt_v, sym):
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += self.loss_func_Rot2(pred_v_now, gt_v_now)
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        flag = (sym[:, 0] == 0).unsqueeze(-1)
        valid_num = sum(flag)
        new_pred_v = torch.where(flag, pred_v, torch.zeros_like(pred_v))
        new_gt_v = torch.where(flag, gt_v, torch.zeros_like(gt_v))
        res = self.loss_func_Rot2(new_pred_v, new_gt_v)
        if valid_num > 0:
            res = res * pred_v.size(0) / sum(flag)
        return res

    def cal_cosine_dis_old(self, pred_v, gt_v):
        # pred_v  bs x 6, gt_v bs x 6
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32).to(pred_v.device)
        for i in range(bs):
            pred_v_now = pred_v[i, ...]
            gt_v_now = gt_v[i, ...]
            res[i] = (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
        res = torch.mean(res)
        return res

    def cal_cosine_dis(self, pred_v, gt_v):
        # pred_v  bs x 6, gt_v bs x 6
        res = (1.0 - batch_dot(pred_v, gt_v)) * 2.0
        res = torch.mean(res)
        return res

    def cal_cosine_dis_sym_old(self, pred_v, gt_v, sym):
        # pred_v  bs x 6, gt_v bs x 6
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_cosine_dis_sym(self, pred_v, gt_v, sym):
        # pred_v  bs x 6, gt_v bs x 6
        res = (1.0 - batch_dot(pred_v, gt_v)) * 2.0
        flag = sym[:, 0] == 0
        res = torch.where(flag, res, torch.zeros_like(res))
        valid_num = sum(flag)
        res = torch.mean(res)
        if valid_num > 0:
            res = res * pred_v.size(0) / valid_num
        return res

    def cal_rot_regular_angle_old(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i, ...]
            z_direction = pred_v2[i, ...]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):

        res = torch.abs(batch_dot(pred_v1, pred_v2).squeeze(-1))
        flag = sym[:, 0] == 0
        res = torch.where(flag, res, torch.zeros_like(res))
        valid_num = sum(flag)
        res = torch.mean(res)
        if valid_num > 0:
            res = res * pred_v1.size(0) / valid_num
        return res

    def cal_loss_Recon(self, pred_recon, gt_recon):
        return self.loss_func_Recon(pred_recon, gt_recon)

    def cal_loss_Tran(self, pred_trans, gt_trans):
        return self.loss_func_t(pred_trans, gt_trans)

    def cal_loss_Size(self, pred_size, gt_size):
        return self.loss_func_s(pred_size, gt_size)
