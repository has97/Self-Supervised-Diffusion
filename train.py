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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

s = 1
seed=42
color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=224),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)
dataset_train_ssl = TrainAugmentation('../imagenet20',transform=train_transform)
dataset_test= ValAugmentation('../imagenet20_val',transform=test_transforms)
def get_data_loaders(batch_size=256,num_workers=12):
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl,dataloader_test
pl.seed_everything(seed)
dataloader_train_ssl,dataloader_test = get_data_loaders(batch_size=128)
model = SimCLRModel(learning_rate=0.2,decay=1e-4,temperature = 0.07)
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_weights_only=True,monitor='train_loss_ssl', mode='min')
wandb_logger = WandbLogger(project='simclr', log_model='all') # log all new checkpoints during training
early_stop_callback = EarlyStopping(monitor="val_loss_ssl", min_delta=0.00, patience=3, verbose=False, mode="min")
trainer = pl.Trainer(
            max_epochs=1000,
            accelerator="gpu",
            devices=[1],
            default_root_dir="self-sup/simclr",
            # strategy=distributed_backend,
            sync_batchnorm=False,
            logger=wandb_logger,
            # logger=logger,
            callbacks=[checkpoint_callback,early_stop_callback],
        )
start = time.time()
trainer.fit(
            model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test,
        )
end = time.time()
model_name="simclr"
run = {
            "model": model_name,
            "batch_size": batch_size,
            "epochs": 1000,
            "max_accuracy": model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(),
            "seed": seed,
        }
runs.append(run)
print(run)