from typing import Any, Tuple, Dict
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from collections import defaultdict
import numpy as np
from itertools import product
from generate_augmentations import Augmentation
import tqdm
class ImageNetDataset():

    def __init__(self,  
                 synthetic_probability = 0.5,
                 max_classes = 100,
                 image_size= (256, 256)):
        self.class_names=[] 
        self.class_id_to_name={}
        self.class_name_to_id={}
        self.augmentations = Augmentation()
        t=0
        self.samples=[]
        self.target=[]
        for x in os.listdir('../imagenet20_30'):
            self.class_names.append(x)
            self.class_id_to_name[x]=t
            self.class_name_to_id[t]=x
            t+=1
        
        for j in self.class_names:
            for k in os.listdir('../imagenet20_30/'+j):
                self.samples.append('../imagenet20_30/'+j+'/'+k)
                os.makedirs('../diffimage/'+j+'/'+k, exist_ok=True)
                self.target.append(self.class_id_to_name[j])
                
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = train_transform

    def __len__(self):
        
        return len(self.samples)

    def get_image_by_idx(self, idx):

        return Image.open(self.samples[idx]).convert('RGB'),self.samples[idx]

    def get_label_by_idx(self, idx):

        return self.target[idx]
    def generate_augmentation(self):
        
        for idx in tqdm.tqdm(range(len(self.samples)), desc="Generating Augmentations"):
            image,path = self.get_image_by_idx(idx)
            # print(path)
            label = self.get_label_by_idx(idx)
            # print(label)
            # print(path)
            filename=path.split("/")[-1]
            class_names = self.class_name_to_id[label]
            aug_img = self.augmentations(image)
            for i in range(3):
                aug_img[i].save('../diffimage/'+class_names+'/'+filename+'/img'+str(i)+'.jpg')
        
    
#     def get_metadata_by_idx(self, idx: int) -> Dict:

#         return dict(name=self.class_names[self.all_labels[idx]])
