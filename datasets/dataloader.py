from pathlib import Path
from itertools import chain
import os
import random

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import copy
def listdir(dname):
    # 해당 경로 하위의 모든 image파일 경로 
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root,transform=None):
        self.samples=listdir(root)
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img=transforms.Grayscale(num_output_channels=1)(img)
        
        return img
    def __len__(self):
        return len(self.samples)

def get_train_loader(root,target_root,batchsize=8,num_workers=4,shuffle=True,size=[310,650],resize=False):
    
    if resize:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize([resize,resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    
    dataset=DefaultDataset(root,transform)
    target_dataset=DefaultDataset(target_root,transform)

    loader=data.DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           num_workers=num_workers,
                           shuffle=shuffle
                          )
    target_loader=data.DataLoader(dataset=target_dataset,
                                  batch_size=batchsize,
                                  num_workers=num_workers,
                                  shuffle=shuffle
                                  )
    
    return loader,target_loader

def get_test_loader(root,batch_size=8,size=[310,650],shuffle=False,resize=False):
    if resize:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize([resize,resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    dataset=DefaultDataset(root,transform)
    loader=data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle
                           )
    return loader
    

def get_eval_loader(root ,shuffle=False,batch_size=4,num_workers=4,drop_last=False):
    # evaluate dataloader 생성
    print('Preparing DataLoader for the evaluation phase...')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
        ])
    dataset=DefaultDataset(root,transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)