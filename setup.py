from __future__ import print_function
from __future__ import division
import subprocess
import numpy as np
import json, os, sys, random, pickle
import torchvision.datasets as dset
from torchvision import transforms 
import os
import skimage
#import IPython.display
#import matplotlib.pyplot as plt
from PIL import Image
import urllib
from collections import OrderedDict
import torchvision.datasets as dset
from torchvision import transforms 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
import skimage
import IPython.display
import urllib
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import clip
#from skimage import io
from pycocotools.coco import COCO
from sklearn.preprocessing import StandardScaler

import torch
print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')


import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data_loader import *

def setup_dirs(base_path: str, from_scratch: bool = False):
    """Setups up a directory for a single trial of a model

    Args:
        base_path: Path to all the model's trials, for example, if the
                   model - resnet50 is being trained on - coco, the base
                   path is: experiments/coco/resnet50/datestring
        from_scratch: 
    """
    os.mkdir(base_path+'/'+'metric_data', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'pca', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'pca/'+'correlation', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'pca/'+'cosine', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'pca/'+'euclidean', mode=0o777)

    os.mkdir(base_path+'/'+'metric_data/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'no_pca/'+'correlation', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'no_pca/'+'cosine', mode=0o777)
    os.mkdir(base_path+'/'+'metric_data/'+'no_pca/'+'euclidean', mode=0o777)

    os.mkdir(base_path+'/'+'cross_analysis', mode=0o777)
    os.mkdir(base_path+'/'+'cross_analysis/'+'openimages', mode=0o777)

    os.mkdir(base_path+'/'+'boxplots', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'pca', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'pca/'+'correlation', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'pca/'+'cosine', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'pca/'+'euclidean', mode=0o777)

    os.mkdir(base_path+'/'+'boxplots/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'no_pca/'+'correlation', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'no_pca/'+'cosine', mode=0o777)
    os.mkdir(base_path+'/'+'boxplots/'+'no_pca/'+'euclidean', mode=0o777)        
    
    os.mkdir(base_path+'/'+'features', mode=0o777)
    os.mkdir(base_path+'/'+'features/' + 'coco', mode=0o777)
    os.mkdir(base_path+'/'+'features/' + 'openimages', mode=0o777)

    os.mkdir(base_path+'/'+'features/'+'coco/'+'finetuned_features', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'coco/'+'finetuned_features/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'coco/'+'finetuned_features/'+'pca', mode=0o777)

    os.mkdir(base_path+'/'+'features/'+'coco/'+'pretrained_features', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'coco/'+'pretrained_features/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'coco/'+'pretrained_features/'+'pca', mode=0o777)

    os.mkdir(base_path+'/'+'features/'+'openimages/'+'finetuned_features', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'openimages/'+'finetuned_features/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'openimages/'+'finetuned_features/'+'pca', mode=0o777)

    os.mkdir(base_path+'/'+'features/'+'openimages/'+'pretrained_features', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'openimages/'+'pretrained_features/'+'no_pca', mode=0o777)
    os.mkdir(base_path+'/'+'features/'+'openimages/'+'pretrained_features/'+'pca', mode=0o777)
    
    os.mkdir(base_path+'/'+'model', mode=0o777)
    os.mkdir(base_path+'/'+'model'+'/'+'updates', mode=0o777)

    if from_scratch == True:
        os.mkdir(base_path + '/'+'model_scratch', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'metric_data', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'pca', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'pca/'+'cosine', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'pca/'+'correlation', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'pca/'+'euclidean', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'no_pca', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'no_pca/'+'cosine', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'no_pca/'+'correlation', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'boxplots/'+'no_pca/'+'euclidean', mode=0o777)   

        os.mkdir(base_path + '/'+'model_scratch/'+'pretrained_features', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'pretrained_features/'+'pca', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'pretrained_features/'+'no_pca', mode=0o777)

        os.mkdir(base_path + '/'+'model_scratch/'+'model', mode=0o777)
        os.mkdir(base_path + '/'+'model_scratch/'+'model/'+'model_updates', mode=0o777)

def setup_dataset(args):
    """Generates pytorch dataloaders with train and val split for finetuning dataset
    """
    dataloaders_dict = dict()
    if args.dataset == 'coco':
        if args.model_name == 'clip':
            crop_size = 224
        else:
            crop_size = 256
            
        train_preprocess = transforms.Compose([
                Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
                RandomCrop(crop_size),
                ToTensor()
        ])

        val_preprocess = transforms.Compose([
                Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
                CenterCrop(crop_size),
                ToTensor()
        ])
        train_imgs = args.dataset_path + "train2017/train2017"
        val_imgs = args.dataset_path + "val2017"
        train_anns = args.dataset_path + "annotations/instances_train2017.json"
        val_anns = args.dataset_path + "annotations/instances_val2017.json"
        coco_train = dset.CocoDetection(root=train_imgs, annFile=train_anns, 
                                        transform=train_preprocess)
        coco_val = dset.CocoDetection(root=val_imgs, annFile=val_anns,
                                        transform=val_preprocess)

        val_dataset = Coco(coco_val, split='val')
        train_dataset = Coco(coco_train, split='train')

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=10)

        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=1)
        
        dataloaders_dict['train'] = train_dataloader
        dataloaders_dict['val'] = val_dataloader
    elif args.dataset == 'openimages':

        preprocess = transforms.Compose([
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                ToTensor()
        ])
        train_dataset = OpenImages(Path(args.dataset_path), split="train", transform=preprocess)
        val_dataset = OpenImages(Path(args.dataset_path), split="val", transform=preprocess)


        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=6)

        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=1)
        dataloaders_dict['train'] = train_dataloader
        dataloaders_dict['val'] = val_dataloader
    else:
        print("Dataset: " + args.dataset + " not implemented")
    return dataloaders_dict