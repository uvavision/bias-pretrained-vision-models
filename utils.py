from __future__ import print_function
from __future__ import division
import subprocess
import numpy as np
import json, os, sys, random, pickle
import torchvision.datasets as dset
import os
from PIL import Image
import urllib
from collections import OrderedDict
import torchvision.datasets as dset
import urllib
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
import time
import copy
import torch
print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml

def read_file(file_name: str):
    """Opens a file and returns a list of the contents

    Args:
        file_name: A path to the file that reads to be read

    Returns:
        urls: A list of strings in the file: fileName
    """
    try:
        file_obj = open(file_name, "r")
        urls = file_obj.read().splitlines()
        file_obj.close()
        return urls
    except:
        print(file_name + ": is not a valid file and could not be opened")
        return []

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
    with open(config, 'r') as file:
        conf = yaml.safe_load(file)
    all_analysis_data = {}
    analysis_data_names = {}
    for category in conf['CLASSES']:
        all_analysis_data[category.lower()] = {}
        for sub_category in conf['CLASSES'][category]:
            all_analysis_data[category.lower()][sub_category] = read_file(conf['CLASSES'][category][sub_category])
    for category in conf['CLASSES_NAMES']:
        analysis_data_names[category.lower()] = {}
        for sub_category in conf['CLASSES_NAMES'][category]:
            analysis_data_names[category.lower()][sub_category] = conf['CLASSES_NAMES'][category][sub_category]
    return all_analysis_data, analysis_data_names



def load_features(folder: str, analysis_set: str, only_pretrained: bool = False):
    """Loads generated features for an analysis set from a trained model

    Args:
        folder: Path to model trial from which to load features
        analysis_set: Specifies which analysis set the features were computed on
        only_pretraied: When False, returns features generated from the pretrained and finetuned model
   
    Returns:
        features: A dictionary mapping class name (specified in config file of analysis_set) to a tensor
                  of features of size N x d where N is the number of examples in a class and is the dimension
                  size of the model
    """
    if only_pretrained == True:
        features = {}
        features_pt = os.listdir(folder+'/features/'+analysis_set + '/pretrained_features/')
        for file_name in features_pt:
            if file_name != '.ipynb_checkpoints':
                features[os.path.splitext(file_name)[0]] = np.load(folder + '/features/' + analysis_set + '/pretrained_features/' + file_name, allow_pickle=True)
    else:
        features = {}
        features_pt = os.listdir(folder+'/features/'+analysis_set +'/pretrained_features/')
        features_ft = os.listdir(folder+'/features/'+analysis_set +'/finetuned_features/')
        for file_name in features_pt:
            if file_name != '.ipynb_checkpoints':
                features[os.path.splitext(file_name)[0]] = np.load(folder +'/features/'+analysis_set + '/pretrained_features/' + file_name, allow_pickle=True)
        for file_name in features_ft:
            if file_name != '.ipynb_checkpoints':
                features[os.path.splitext(file_name)[0]] = np.load(folder +'/features/'+analysis_set + '/finetuned_features/' + file_name, allow_pickle=True)
    return features