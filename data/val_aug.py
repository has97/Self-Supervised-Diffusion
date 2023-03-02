import random
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
class ValAugmentation(Dataset):

    def __init__(self, root_train_dir, transform=None):
        self.class_names=[]
        self.samples=[]
        self.class_id_to_name={}
        self.class_name_to_id={}
        self.target=[]
        t=0
        for x in os.listdir(root_train_dir):
            self.class_names.append(x)
            self.class_id_to_name[x]=t
            self.class_name_to_id[t]=x
            t+=1
        for j in self.class_names:
            for k in os.listdir(root_train_dir+'/'+j):
                self.samples.append(root_train_dir+'/'+j+'/'+k)
                self.target.append(self.class_id_to_name[j])
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path =  self.samples[idx]
        target = self.target[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return img,torch.tensor(target, dtype=torch.long),torch.tensor(target, dtype=torch.long)