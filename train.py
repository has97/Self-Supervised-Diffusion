from data.diff_aug import TrainAugmentation
from data.val_aug import ValAugmentation
from models.aug_simclr import SimCLRModel
import torchvision
import torch
import os
import lightly
import pytorch_lightning as pl
import time
from pytorch_lightning.loggers import WandbLogger
s = 1
seed=1
color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=128),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )
normalize_transform = torchvision.transforms.Normalize(
    mean=lightly.data.collate.imagenet_normalize["mean"],
    std=lightly.data.collate.imagenet_normalize["std"],
)
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        normalize_transform,
    ]
)
dataset_train_ssl = TrainAugmentation('../imagenette2/train',transform=train_transform)
dataset_train_knn = ValAugmentation('../imagenette2/train',transform=test_transforms)
dataset_test= ValAugmentation('../imagenette2/val',transform=test_transforms)
def get_data_loaders(batch_size=256,num_workers=12):
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_knn,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test
pl.seed_everything(seed)
dataloader_train_ssl, dataloader_train_knn, dataloader_test = get_data_loaders(batch_size=256)
model = SimCLRModel(dataloader_train_knn,num_classes=10)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_loss_ssl', mode='min')
wandb_logger = WandbLogger(project='simclr', log_model='all') # log all new checkpoints during training

trainer = pl.Trainer(
            max_epochs=1000,
            gpus=1,
            # default_root_dir=logs_root_dir,
            # strategy=distributed_backend,
            sync_batchnorm=False,
            logger=wandb_logger,
            # logger=logger,
            callbacks=[checkpoint_callback],
        )
start = time.time()
trainer.fit(
            model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test,
        )
end = time.time()
run = {
            "model": model_name,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "max_accuracy": model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(),
            "seed": seed,
        }
runs.append(run)
print(run)