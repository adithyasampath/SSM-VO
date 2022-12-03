import torch
import numpy as np
import os
from ppnet_loss import PPNetLoss
from torch.utils.data import ConcatDataset
from ppnet_dataloader import PPNetDataset
import math
from torch.utils.tensorboard import SummaryWriter
from ppnet_loss import MotionLoss
from PPnet import PPnet

class PPNetTrainer:
    def __init__(self, args):
        # data_dir="/home/aditya/SSM-VO/dataset/poses", split=[0.8,0.1,0.1], batch_size=32, epochs=20, num_worker=5, num_samples=100, exp_num=0

        # train args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.num_samples = args.num_samples
        self.exp_num = args.exp_num
        self.save_freq = args.save_freq
        self.val_freq = args.val_freq
        self.log_freq = args.log_freq
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")
        self.num_validation_samples = args.num_validation_samples
        self.best_val_acc = 0
        self.writer = SummaryWriter('./logs')
        self.save_dir = args.save_dir
        self.model_type = args.model_type
        self.val_loss = 0

        # model
        # input_size=6, output_size=6, seq=20, hidden_size=8, num_layer=1, batch_first=True, model_type="lstm"
        self.model = PPnet(args.input_size, args.output_size, args.hidden_size, args.num_layer, args.batch_first, args.model_type)
        if self.cuda:
            self.model = self.model.cuda()

        # data loaders
        self.data_dir = args.data_dir
        self.split = args.split
        files = [os.path.join(self.data_dir, "{:02d}.txt".format(idx)) for idx in range(1, 9)]
        loaders = [PPNetDataset(file_path, self.device) for file_path in files]
        len_loader = len(loaders)
        assert sum(self.split)==1, "Incorrect train, val, test split ratio"
        splits = [math.ceil(self.split[0]*len_loader), math.ceil(self.split[1]*len_loader), math.ceil(self.split[2]*len_loader)]
        train_dataset, val_dataset, test_dataset = loaders[:splits[0]], loaders[splits[0]: splits[0]+splits[1]], loaders[splits[0]+splits[1]:]
        self.train_loader = ConcatDataset(train_dataset)
        self.val_loader = ConcatDataset(val_dataset)
        self.test_loader = ConcatDataset(test_dataset) 
        
        # optimizer, loss
        self.loss = MotionLoss(args.gamma, args.k)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)


    def train(self):
        # get number of train batches
        num_batches = len(self.train_loader)
        # start training
        for epoch in range(self.epochs):
            
            # set model in train mode
            self.model.train()
            
            # accumulate loss in every epoch
            loss_accum = []

            for batch_idx, batch_data in tqdm(enumerate(self.train_loader)):
                # get current step
                current_step = epoch * num_batches + batch_idx

                # get batch data
                X, target = batch_data

                # model forward
                mean, log_variance = self.model(X)

                # loss
                self.optimizer.zero_grad()
                loss = self.loss.compute_loss(mean, log_variance, target)
                loss_accum.append(loss.item())
                loss.backward()
                self.optimizer.step()

                if current_step % self.log_freq == 0:
                    print("Epoch: {}, Batch: {}/{}, Train Loss: {}".format(epoch, batch_id, num_batches, loss))
                    self.writer.add_scalar('Train Loss (steps)', loss.item(), current_step)

            train_loss =  torch.mean(loss_accum)   
            print("Epoch: {}, Train Loss: {}".format(epoch, train_loss))
            self.writer.add_scalar('Train loss (epoch)', train_loss, epoch)    

            if epoch % self.val_freq == 0:
                self.val_loss = self.validate(epoch)

            if epoch % self.save_freq:
                if not self.val_loss or self.val_loss > self.best_val_acc:
                    torch.save(self.model, os.path.join(self.save_dir, f'ppnet_{self.model_type}_best_model.pth'))
                torch.save(self.model, os.path.join(self.save_dir, f'ppnet_{self.model_type}_{self.epoch}.pth'))
        
        final_val_loss = self.validate(epoch)
        return final_val_loss


    def validate(self, epoch):
        # set model in eval mode
        self.model.eval()
        
        # store loss for each batch
        num_batches = len(self.val_loader)
        val_loss_accum = []
        
        for batch_idx, batch_data in tqdm(enumerate(self.val_loader)):
            # get current step
            current_step = epoch * num_batches + batch_idx
            
            # get batch data
            X, target = batch_data

            # model forward
            mean, log_variance = self.model(X)

            # val loss
            loss = self.loss.compute_loss(mean, log_variance, target)
            val_loss_accum.append(loss.item())
            self.writer.add_scalar('Val Loss (steps)', loss.item(), current_step)

        val_loss = torch.mean(val_loss_accum)
        print("Epoch: {}, Val Loss: {}".format(epoch, val_loss))
        self.writer.add_scalar('Val Loss (epoch)', val_loss, epoch)
        return val_loss