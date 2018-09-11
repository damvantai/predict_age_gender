from __future__ import print_function
import os
import io
import sys
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class ListDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.fname = []
        self.age = []
        self.gender = []
        
        for filename in os.listdir(root):
            self.fname.append(filename)
            filename_split = filename.split("_")
            self.age.append(int(filename_split[0]))
            self.gender.append(int(filename_split[1]))
            
        self.num_imgs = len(self.fname)
    
    def __getitem__(self, idx):
        """Load image."""
        fname = self.fname[idx]
        age = self.age[idx]
        gender = self.gender[idx]
        
        img = Image.open(os.path.join(self.root, fname))
        img = self.transform(img)
        return img, age, gender
    
    def __len__(self):
        return self.num_imgs 