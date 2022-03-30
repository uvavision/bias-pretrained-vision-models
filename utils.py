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

def read_file(file_name: str):
    """Opens a file and returns a list of the contents
    
    Args:
        file_name: A path to the file that reads to be read
    
    Returns:
        urls: A list of strings in the file: fileName
    """
    try:
        fileObj = open(file_name, "r") 
        urls = fileObj.read().splitlines() 
        fileObj.close()
        return urls
    except:
        print(file_name + ": is not a valid file and could not be opened")

def analysis_data(config: str):
    """Reads a config file and returns the analysis dataset

    Args:
        config: A path to the config file for the analysis set
    
    Returns:
        all_analysis_data: A dictionary mapping class names to a list of images 
                           corresponding to that class. 
        analysis_data_names: A dictionary mapping class names to their abbreviations
                             to be used for plotting. 
    """
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    all_analysis_data = dict()
    analysis_data_names = dict()
    for category in conf['CLASSES']:
        all_analysis_data[category.lower()] = dict()
        for sub_category in conf['CLASSES'][category]:
            all_analysis_data[category.lower()][sub_category] = read_file(conf['CLASSES'][category][sub_category])
    for category in conf['CLASSES_NAMES']:
        analysis_data_names[category.lower()] = dict()
        for sub_category in conf['CLASSES_NAMES'][category]:
            analysis_data_names[category.lower()][sub_category] = conf['CLASSES_NAMES'][category][sub_category]
    return all_analysis_data, analysis_data_names