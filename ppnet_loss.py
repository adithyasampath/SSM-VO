import torch

def compute_ppnet_loss(mean, log_variance, target, gamma=0.1 , k=0.5):
    variance = torch.exp(log_variance)
    l2 = torch.sum((torch.square((target-mean)) / variance), dim=1)
    l1 = torch.sum(log_variance, dim= 1)
    loss = gamma*l1 + torch.pow(l2, k)
    return loss

