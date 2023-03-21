import os
import numpy as np
import glob
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def get_cmd_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser = config_parser_args(parser)
    return parser.parse_args()


def config_parser_args(parser):
    """
    modify the arguments here.
    """
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='models', help='output folder')
    parser.add_argument('--outclass', type=int, default=2, help='point class')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
    return parser


def read_cfg_file(cfg_file_path):
    import configparser
    args = configparser.ConfigParser()
    assert os.path.exists(os.path.join('.',cfg_file_path))
    args.read(cfg_file_path)
    return args


def get_current_time():
    import time
    localtime = (time.localtime(time.time()))
    year = localtime.tm_year
    month = localtime.tm_mon
    day = localtime.tm_mday
    hour = localtime.tm_hour
    return (year, month, day, hour)


def ensure_folder_exists(absolute_folder_path):
    """Create folder if it is not exist, otherwise do nothing. """
    os.system('mkdir -p {}'.format(absolute_folder_path))


def get_matched_path_list(folder_path, matching_file_format = '*', sort = True):
    """ Get the paths which satisfy the format with folder_path/matching_file_format.
        Example of matching_file_format: '*_color.png'
    """
    glob_path = os.path.join(folder_path, matching_file_format)
    path_list = glob.glob(glob_path)
    if sort:
        path_list.sort()
    return path_list


def read_json_file(file_path):
    import json
    with open(file_path) as f:
        data = json.load(f)
    return data


def write_yaml(data, path):
    import ruamel.yaml as yaml
    with open(path,'w') as f:
        yaml.dump(data, f)


def to_cuda(data_list):
    out_list = []
    for data in data_list:
        if data is not None:
            data = data.cuda()
        out_list.append(data)
    return out_list


def get_MSE_loss(cuda=False):
    import torch.nn as nn
    loss = nn.MSELoss()
    if cuda:
        loss.cuda()
    return loss


def get_cross_entropy_loss(cuda=False):
    import torch.nn as nn
    loss = nn.CrossEntropyLoss()
    if cuda:
        loss.cuda()
    return loss


def progressbar_init(show_message, maxval):
    import progressbar
    widgets = [show_message, progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % maxval,
         ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=maxval,widgets=widgets)
    return bar


def init_video_writer(video_path, H, W, fps=20):
    import cv2
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    return video_writer

def to_homo_poses(Rs, Ts):
    poses = []
    for i in range(len(Rs)):
        poses.append(to_homo_pose(Rs[i], Ts[i]))
    return np.array(poses)
        
    
def to_homo_pose(R, T):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = T.reshape(3,)
    return pose