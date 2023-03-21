"""
@Author: Linfang Zheng 
@Contact: zhenglinfang@icloud.com
@Time: 2023/03/06
@Note: Modified from 3D-GCN: https://github.com/zhihao-lin/3dgcn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.fs_net_loss import FLAGS

def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor_new(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    bs, num_points, num_dims = tensor.size()
    idx_base = torch.arange(0, bs, device=tensor.device).view(-1, 1, 1) * num_points
    idx = index + idx_base
    idx = idx.view(-1)
    feature = tensor.reshape(bs * num_points, -1)[idx, :]
    _, out_num_points, n = index.size()
    feature = feature.view(bs, out_num_points, n, num_dims)
    return feature

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)", return_unnormed = False):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor_new(vertices, neighbor_index)  # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    if return_unnormed:
        return neighbor_direction_norm.float(), neighbor_direction
    else:
        return neighbor_direction_norm.float()

class HSlayer_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""

    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.feat_k = 8
        self.kernel_num = kernel_num
        self.support_num = support_num
        self.relu = nn.ReLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.STE_layer = nn.Conv1d(3, kernel_num, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(2*kernel_num, kernel_num, kernel_size=1, bias=False)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                neighbor_num: 'int'):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        f_STE = self.STE_layer(vertices.transpose(-1,-2)).transpose(-1,-2).contiguous()
        receptive_fields_norm, _ = get_receptive_fields(neighbor_num, vertices, mode='RF-P')
        feature = self.graph_conv(receptive_fields_norm, vertices, neighbor_num)
        feature = self.ORL_forward(feature, vertices, neighbor_num)

        return feature + f_STE 
    
    def graph_conv(self, receptive_fields_norm,
                   vertices: "(bs, vertice_num, 3)",
                   neighbor_num: 'int',):
        """ 3D graph convolution using receptive fields. More details please check 3D-GCN: https://github.com/zhihao-lin/3dgcn

        Return (bs, vertice_num, kernel_num): the extracted feature.
        """
        bs, vertice_num, _ = vertices.size()
        support_direction_norm = F.normalize(self.directions, dim=0)  # (3, s * k)
        theta = receptive_fields_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.reshape(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, support_num, kernel_num)
        feature = torch.mean(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature

    def ORL_forward(self, feature, vertices, neighbor_num):
        f_global = get_ORL_global(feature, vertices, neighbor_num) 
        feat = torch.cat([feature, f_global], dim=-1)
        feature = self.conv2(feat.transpose(-1,-2)).transpose(-1,-2).contiguous() + feature
        return feature

    
class HS_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1)* out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.feat_k = 8
        self.STE_layer = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=1, bias=False)

        self.conv2 = nn.Conv1d(2*out_channel, out_channel, kernel_size=1, bias=False)

        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)",
                neighbor_num: "int"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        f_STE = self.STE_layer(feature_map.transpose(-1,-2)).transpose(-1,-2).contiguous()
        receptive_fields_norm, neighbor_index = get_receptive_fields(neighbor_num, 
                                                                       vertices, 
                                                                       feature_map=feature_map, 
                                                                       mode='RF-F')
        feature = self.graph_conv(receptive_fields_norm, neighbor_index, feature_map, vertices, neighbor_num)
        feature_fuse = self.ORL_forward(feature, vertices, neighbor_num)
        return feature_fuse + f_STE 
    
    def graph_conv(self, receptive_fields_norm, neighbor_index, feature_map,
                   vertices: "(bs, vertice_num, 3)",
                   neighbor_num: 'int',):
        """ 3D graph convolution using receptive fields. More details please check 3D-GCN: https://github.com/zhihao-lin/3dgcn

        Return (bs, vertice_num, kernel_num): the extracted feature.
        """
        bs, vertice_num, _ = vertices.size()
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = receptive_fields_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.reshape(bs, vertice_num, neighbor_num, -1) # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_map = feature_map @ self.weights + self.bias  # (bs, vertice_num, neighbor_num, (support_num + 1) * out_channel)
        feature_center = feature_map[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_map[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        feature_support = indexing_neighbor_new(feature_support, neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.mean(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature

    def ORL_forward(self, feature_fuse, vertices, neighbor_num):
        f_global = get_ORL_global(feature_fuse, vertices, neighbor_num) 
        feat = torch.cat([feature_fuse, f_global], dim=-1)
        feature_fuse = self.conv2(feat.transpose(-1,-2)).transpose(-1,-2).contiguous() + feature_fuse
        return feature_fuse

def get_receptive_fields(neighbor_num: "int", 
                         vertices: "(bs, vertice_num, 3)", 
                         feature_map: "(bs, vertice_num, in_channel)" = None, 
                         mode: 'string' ='RF-F'):
    """ Form receptive fields amd norm the direction vectors according to the mode.
    
    Args:
        neighbor_num (int): neighbor number.
        vertices (tensor): The 3D point cloud for forming receptive fields 
        feature_map (tensor, optional): The features for finding neighbors and should be provided if 'RF-F' is used. Defaults to None. 
        mode (str, optional): The metrics for finding the neighbors. Should only use 'RF-F' or 'RF-P'. 'RF-F' means forming the receptive fields using feature-distance, and 'RF-P' means using point-distance. Defaults to 'RF-F'.
    """
    assert mode in ['RF-F', 'RF-P']
    if mode == 'RF-F':
        assert feature_map is not None, "The feature_map should be provided if 'RF-F' is used"
        feat = feature_map
    else:
        feat = vertices
    neighbor_index = get_neighbor_index(feat, neighbor_num)
    neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
    return neighbor_direction_norm, neighbor_index

def get_ORL_global(feature:'(bs, vertice_num, in_channel)', vertices: '(bs, vertice_num, 3)', 
                          neighbor_num:'int'):
    vertice_num = feature.size(1)
    neighbor_index = get_neighbor_index(vertices, neighbor_num)
    feature = indexing_neighbor_new(feature, neighbor_index) # batch_size, num_points, k, num_dims
    feature = torch.max(feature, dim=2)[0]
    f_global = torch.mean(feature, dim=1, keepdim=True).repeat(1, vertice_num, 1)
    return f_global

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        neighbor_feature = indexing_neighbor_new(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)

    s = 3
    conv_1 = HSlayer_surface(kernel_num=32, support_num=s, neighbor_num=n)
    conv_2 = HS_layer(in_channel=32, out_channel=64, support_num=s)
    pool = Pool_layer(pooling_rate=4, neighbor_num=4)

    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(vertices, f1, n)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()
