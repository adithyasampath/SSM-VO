import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys
sys.path.append("../")
from rotation_conversions import matrix_to_axis_angle
from ppnet_utils import PPNetUtils

class PPNetDataset(Dataset):
    def __init__(self, poses_file="", device = 'cpu'):
        self.poses_file = poses_file
        self.data = self.read_file()
        self.scale = torch.linspace(0.8, 1.2, steps=10)
        self.device = device
    
    def read_file(self):
        data = []
        with open(self.poses_file, 'r') as f:
            for lines in f:
                p_data = lines.strip().split(" ")
                p_data = [float(x) for x in p_data]
                p_data = torch.Tensor(p_data).reshape((3,4))
                pose = torch.vstack((p_data , torch.Tensor([[0, 0, 0, 1]])))
                data.append(pose)
        return torch.stack(data)

    def __len__(self):
        return self.data.shape[0] // 20

    def convert_pose(self, rel_poses):
        N = rel_poses.shape[0]
        SEs = torch.zeros(N, 6).to(self.device)
        for i in range(N):
            r = rel_poses[i,:3,:3].reshape((1,3,3))
            t = rel_poses[i,:3,3]
            angle= matrix_to_axis_angle(r).reshape((-1))
            SEs[i] = torch.hstack((angle , t))
        return SEs

    def center_poses(self, input_poses):
        pputil = PPNetUtils(self.device)
        N = input_poses.shape[0] 
        rel_poses = [] #b , n, 20
        for i in range(1,N):
            rel_pose = pputil.ses2SEs(input_poses[i].reshape((1,6))).inverse() @ pputil.ses2SEs(input_poses[i-1].reshape((1,6)))
            rel_poses.append(torch.squeeze(rel_pose))

        rel_poses = torch.stack(rel_poses)
        rel_poses = self.convert_pose(rel_poses)
        centered_poses = pputil.translate_poses(rel_poses.reshape((1,19,6)))
        return torch.squeeze(centered_poses)

    def __getitem__(self, idx):
        input_poses = self.data[idx:idx+20]
        output_pose = self.data[idx + 20]
        scale_augment = np.random.choice(self.scale)
        #scale_augment = 1
        input_poses[:,0:3, 3] = input_poses[:,0:3, 3]*scale_augment
        output_pose[0:3,3] = output_pose[0:3,3]*scale_augment
        # input_poses = self.center_poses(input_poses)
        return input_poses.to(self.device), output_pose.to(self.device)