import torch
import numpy as np
import os
from ppnet_loss import PPNetLoss

class PPNetTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, batch_size, epochs=20, num_worker=5, num_samples=100, exp_num=0):
        self.model = model
        # self.

    def train_epoch(self):
        pass