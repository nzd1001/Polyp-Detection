import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np

class TestDataset(Dataset):
    def __init__(self,test_dir,transform):
        super(TestDataset).__init__()
        img_list=os.listdir(test_dir)
        self.img_list=[test_dir+img_name for img_name in img_list]
        self.transform=transform
    def __getitem__(self,idx):
        img_path=self.img_list[idx]
        img=Image.open(img_path)
        h=img.size[1]
        w=img.size[0]
        data=self.transform(img)
        return data,img_path,h,w
    def __len__(self):
        return len(self.img_list)