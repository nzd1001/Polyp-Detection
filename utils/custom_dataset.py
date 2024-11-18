import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self,data,labels,aug):
        self.data=data
        self.labels=labels
        self.augmentation=aug
    def __getitem__(self,idx):
        img,label=self.data[idx],self.labels[idx]
        if self.augmentation:
            img=img.permute(1,2,0).numpy()
            label=label.numpy()
            augmented=self.augmentation(image=img,mask=label)
            img=augmented["image"]
            label=augmented["mask"]
            img=torch.from_numpy(np.transpose(img,(2,0,1))).float()
            label=torch.from_numpy(label).long()
        return img,label
    def __len__(self):
        return len(self.data)