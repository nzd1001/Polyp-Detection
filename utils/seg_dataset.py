import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
class SegmentationDataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform):
        super(SegmentationDataset).__init__()
        img_list=os.listdir(img_dir)
        mask_list=os.listdir(mask_dir)
        self.img_list=[img_dir+img_name for img_name in img_list]
        self.mask_list=[mask_dir+mask_name for mask_name in mask_list]
        self.transform=transform
    def __getitem__(self,idx):
        img_name=self.img_list[idx]
        mask_name=self.mask_list[idx]
        img=Image.open(img_name)
        mask=Image.open(mask_name)
        #transform and normalize
        data=self.transform(img)/255
        label=self.transform(mask)/255
        #give red class 0, green class 1 and black class 2
        label=torch.where(label>0.8,1.0,0.0)
        label[2,:,:]=0.0001
        label=torch.argmax(label,dim=0).type(torch.int64)
         #re-map black to class 0, green to class 1 and red to class 2
        mapping={0:2,1:1,2:0}
        label.apply_(lambda x:mapping[x])
        return data,label
    def __len__(self):
        return len(self.img_list)