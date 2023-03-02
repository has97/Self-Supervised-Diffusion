import torch
import torch.nn as nn
import torchvision
import copy
import math
import os

import time
import lightly
import numpy as np
import torchvision
import pytorch_lightning as pl
from lightly.utils import BenchmarkModule
from lightly.models.modules import heads
import os
batch_size = 256
lr_factor = batch_size / 25
class SimCLRModel(BenchmarkModule):
    def __init__(self,dataloader_kNN,num_classes):
        super().__init__(dataloader_kNN,num_classes)
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1),_,_ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000)
        return [optim], [cosine_scheduler]