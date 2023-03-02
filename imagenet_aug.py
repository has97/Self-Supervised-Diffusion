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
        self.batch_size=2
        t=0
        self.samples=[]
        self.target=[]
        self.transforms = transforms.Compose([
                                            transforms.Resize((224,224)),
                                            transforms.ToTensor()
                                        ])
        for x in os.listdir('../imagenette2/train/'):
            self.class_names.append(x)
            self.class_id_to_name[x]=t
            self.class_name_to_id[t]=x
            t+=1
        
        for j in self.class_names:
            for k in os.listdir('../imagenette2/train/'+j):
                self.samples.append('../imagenette2/train/'+j+'/'+k)
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

    def get_image_by_idx(self, idxl,idxu):

        L= []
        # self.transforms(Image.open(self.samples[idxl]).convert('RGB'))
        for i  in self.samples[idxl:idxu]:
            L.append(self.transforms(Image.open(i).convert('RGB')).unsqueeze(0))
        return torch.cat(L,dim=0),self.samples[idxl:idxu]

    def get_label_by_idx(self, idxl,idxu):

        return self.target[idxl:idxu]
    def generate_augmentation(self):
        
        for idx in tqdm.tqdm(range(0,len(self.samples),self.batch_size), desc="Generating Augmentations"):
            image,path = self.get_image_by_idx(idx,idx+self.batch_size)
            # print(path)
            label = self.get_label_by_idx(idx,idx+self.batch_size)
            # print(image)
            # print(label)
            # print(path)
            # filename=path.split("/")[-1]
            class_names=[]
            for i in label:
                class_names.append(self.class_name_to_id[i])
            # print(class_names)
            # class_names = 
            aug_img = self.augmentations(image)
            t=0
            r=0
            for i in range(2):
                r=0
                for j in class_names:
                        filename=path[r].split("/")[-1]
                        # print('../diffimage/'+j+'/'+filename+'/img'+str(i)+'.jpg')
                        aug_img[t].save('../diffimage/'+j+'/'+filename+'/img'+str(i)+'.jpg')
                        t+=1
                        r+=1
                    # aug_img[t][i].save('../diffimage/'+class_names+'/'+filename+'/img'+str(i)+'.jpg')
        
    
#     def get_metadata_by_idx(self, idx: int) -> Dict:

#         return dict(name=self.class_names[self.all_labels[idx]])
