import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ..rotation_conversions import matrix_to_axis_angle

class PPNetDataset(Dataset):
    def __init__(self, poses_file="", device = 'cpu'):
        self.poses_file = poses_file
        self.data = self.read_file(self.poses_file)
        self.scale = torch.arange(1, 5)
        self.device = device
    
    def read_file(self):
        data = []
        with open(self.poses_file, 'r') as f:
            for lines in f:
                p_data = lines.strip().split(" ")
                p_data = [float(x) for x in p_data]
                t_data = torch.Tensor(p_data[3::4])
                del p_data[3::4]
                c_data = torch.Tensor(p_data).reshape((1,3,3))
                angle= matrix_to_axis_angle(c_data).reshape((-1))
                pose = torch.hstack((angle , t_data))
                data.append(pose)
        return torch.stack(data)

    def __len__(self):
        return self.data.shape[0] // 20

    def __getitem__(self, idx):
        input_poses = self.data[idx:idx+20]
        output_pose = self.data[idx + 20]
        scale_augment = np.random.choice(self.scale)
        input_poses[:,3:] = input_poses[:,3:]*scale_augment
        output_pose[3:] = output_pose[3:]*scale_augment
        return input_poses.to(self.device), output_pose.to(self.device)