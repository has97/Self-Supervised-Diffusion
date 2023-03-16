import random
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import lightly
# normalize_transform = transforms.Normalize(
#     mean=lightly.data.collate.imagenet_normalize["mean"],
#     std=lightly.data.collate.imagenet_normalize["std"],
# )
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# transformer = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#          normalize,
#     ]
# )
class TrainAugmentation(Dataset):

    def __init__(self, root_train_dir, transform=None):
        self.class_names=[]
        self.samples=[]
        self.class_id_to_name={}
        self.class_name_to_id={}
        self.target=[]
        t=0
        for x in os.listdir(root_train_dir):
            self.class_names.append(x)
            # self.class_id_to_name[x]=t
            # self.class_name_to_id[t]=x
            # t+=1
        self.class_names.sort()
        for i in self.class_names:
            self.class_id_to_name[i]=t
            self.class_name_to_id[t]=i
            t+=1
        for j in self.class_names:
            for k in os.listdir(root_train_dir+'/'+j):
                self.samples.append(root_train_dir+'/'+j+'/'+k)
                self.target.append(self.class_id_to_name[j])
        # self.root_dir = root_dir
        self.transform = transform
        # self.transform1 = transformer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path =  self.samples[idx]
        target = self.target[idx]
        p = random.randint(0, 5)
        r = random.randint(0, 1)
        s = path.split('/')
        path1 = '../diffimage/'+s[-2]+'/'+s[-1]+'/img'+str(p)+'.jpg'
        img = Image.open(path).convert('RGB')
        img1 = Image.open(path1).convert('RGB')
        img_t = img
        if r==0:
            img1_t = img1
            # print("diffusion loaded!!:))")
        else:
            img1_t=  img
        # img1_t = img
        n_img=self.transform(img_t,img1_t)
        
        return n_img,torch.tensor(target, dtype=torch.long)