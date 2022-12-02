import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from ..kitti_utils import read_calib_file

class PPNetDataset(Dataset):
    def __init__(self, poses_dir=""):
        self.poses_dir = poses_dir
        self.reqd_files = [os.path.join(self.poses_dir, "{:02d}.txt".format(idx)) for idx in range(1, 9)]
        self.data = self.read_files(self.reqd_files)
    
    def read_files(self):
        data = []
        for idx in range(1, 9):
            file_data = open(reqd_files[idx-1], 'r').readlines()