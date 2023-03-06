import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import normalize
import copy
import math
import os

import time
import lightly
import numpy as np
import torchvision
import pytorch_lightning as pl
from pytorch_optimizer import LARS
import os
batch_size = 256
lr_factor = batch_size / 25
class Projection(nn.Module):
    def __init__(self,input_dims,hidden_dims,out_dims):
        super(Projection, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims =  hidden_dims
        self.out_dims = out_dims
        self.model = nn.Sequential(
            nn.Linear(input_dims,hidden_dims),
            # nn.BatchNorm1d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims,out_dims)
        )
    def forward(self,x):
        outs = self.model(x)
        return outs
        
class SimCLRModel(pl.LightningModule):
    def __init__(self,learning_rate,temperature,decay,max_epochs=500,hidden_dims=2048,feature_dims=128):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.projection = Projection(self.resnet.fc.in_features,hidden_dims,feature_dims)
        self.resnet.fc = nn.Identity()
    def NTXtentloss(self,feats):
        # a_norm =  normalize(a,p=2,dim=1)
        # b_norm =  normalize(b,p=2,dim=1)
        # a_b_cat = torch.cat((a_norm,b_norm),dim=0) # 2N * d
        # a_b_cat_t = torch.transpose(a_b_cat, 0, 1) # d * 2N
        # sims = torch.matmul(a_b_cat,a_b_cat_t)
        # exp_sims = torch.exp(sims/self.hparams.temperature) 
        # extra = torch.diag(exp_sims)
        # denom = torch.sum(exp_sims,dim=1)
        # denom -= extra
        # exp_sims = torch.div(exp_sims,denom)
        # loss = -torch.log(exp_sims)
        # return torch.mean(loss)
        cos_sim = torch.nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        
        return nll
        

    def training_step(self, batch, batch_index):
        x,_,_ = batch
        x = torch.cat(x, dim=0) # (2*N,img)
        # print(x.shape)
        # h0 = self.resnet(x0)
        # h1 = self.resnet(x1)
        h = self.resnet(x)
        # z0 = self.projection(h0)
        z = self.projection(h) # (2*N,d)
        loss = self.NTXtentloss(z)
        self.log("train_loss_ssl", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # (x0, x1),_,_ = batch
        # h0 = self.resnet(x0)
        # h1 = self.resnet(x1)
        # z0 = self.projection(h0)
        # z1 = self.projection(h1)
        # loss = self.NTXtentloss(z0, z1)
        x,_,_ = batch
        x = torch.cat(x, dim=0) # (2*N,img)
        # h0 = self.resnet(x0)
        # h1 = self.resnet(x1)
        h = self.resnet(x)
        # z0 = self.projection(h0)
        z = self.projection(h) # (2*N,d)
        loss = self.NTXtentloss(z)
        self.log("val_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.AdamW(
        #     self.parameters(),lr=self.hparams.learning_rate,weight_decay=self.hparams.decay
        # )
        optim = LARS(self.parameters(),lr=self.hparams.learning_rate,weight_decay=self.hparams.decay)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=self.hparams.max_epochs,verbose=True,eta_min=0.05)
        return [optim], [cosine_scheduler]