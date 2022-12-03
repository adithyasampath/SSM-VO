import torch
import numpy as np

class MotionLoss:
    def __init__(self, gamma=0.1, k=0.5):
        self.gamma = gamma
        self.k = k

    def compute_loss(self, mean, log_variance, target):
        variance = torch.exp(log_variance)
        numerator = (target-mean)
        l2 = torch.sum((torch.square((target-mean)) / variance), dim=1)
        l1 = torch.sum(log_variance, dim= 1)
        loss = self.gamma*l1 + torch.pow(l2, self.k)
        return loss

