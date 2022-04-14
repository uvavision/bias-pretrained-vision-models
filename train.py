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

import argparse
from model_init import *
from data_loader import *
from cosine_analysis.cosine_exp import *
from models_def.pytorch_models import *
from models_def.clip_model import *
from setup import * 
from utils import *
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

def lightning_setup(args):
    """Loads a finetuned model into memory and returns it for bias analysis
    
    Returns:
        model_ft: finetuned model from args.trial_path
    """
    model_ft = None
    models_implemented = ['moco_resnet50', 'simclr_resnet50', 'alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'bit_resnet50', 'virtex_resnet50']
    if args.model_name == 'clip': 
        model_setup = CLIP_model(args, args.trial_path)
        model_ft, _, _ = model_setup.setup_model()
        checkpoint = torch.load(args.trial_path+'/model/model.pt')
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        model_ft.eval()
    elif args.model_name in models_implemented:
        model_setup = LitPytorchModels(args, args.trial_path)
        base = args.trial_path + '/model/'+args.model_name +'/' +'version_0/checkpoints/'
        ckpt = os.listdir(base)[0]
        checkpoint = torch.load(base+ckpt)
        checkpoint['state_dict'] = dict(checkpoint['state_dict'])
        state_dict_mod = dict()
        for i in checkpoint['state_dict']:
            state_dict_mod[i[6:]] = checkpoint['state_dict'][i] 
        model_setup.model.load_state_dict(state_dict_mod)
        model_ft = model_setup.model.eval()
    else:
        print("Model not implemented")
    return model_ft

def lightning_train(args, dataloaders: dict, model_path: str, resume_training: bool = False):
    """Finetunes a model and saves the metadata in model_path

    Args:
        dataloaders: A dictionary with train and val pytorch dataloader objects
        model_path: Path to save the model's metadata for a trial
        resume_training: If true, uses saved checkpoint in model_path to resume training
    
    Returns:
        model_ft: finetuned model in eval mode 
    """
    seed_everything(args.seed, workers=True)
    model_setup = None
    model_ft = None
    criterion = None
    optimizer = None
    models_implemented = ['moco_resnet50', 'simclr_resnet50', 'alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'bit_resnet50', 'virtex_resnet50']
    if resume_training == True:
        if args.model_name == 'clip':
            model_setup = CLIP_model(args, model_path)
            model_ft, criterion, optimizer = model_setup.setup_model()
            checkpoint = torch.load(args.checkpoint+'/model/model.pt')
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model_ft, _, _ = model_setup.train_model(args.dataset, model_ft, dataloaders, criterion, optimizer, num_epochs=args.epochs)
            model_ft.eval()
            torch.save(model_ft.state_dict(), model_path+'/model/model_final.pt')
        elif args.model_name in models_implemented:
            model_setup = LitPytorchModels(args, model_path)
            logger = TensorBoardLogger(model_path+'/model', name=args.model_name)
            trainer = Trainer(gpus=1, default_root_dir=model_path+'/model', logger=logger, callbacks=[], max_epochs=args.epochs, resume_from_checkpoint=args.checkpoint)
            trainer.fit(model_setup, dataloaders['train'], dataloaders['val'])
            trainer.validate(dataloaders=dataloaders['val'])
            
            base = model_path + '/model/'+args.model_name +'/' +'version_0/checkpoints/'
            ckpt = os.listdir(base)[0]
            checkpoint = torch.load(base+ckpt)
            checkpoint['state_dict'] = dict(checkpoint['state_dict'])
            state_dict_mod = dict()
            for i in checkpoint['state_dict']:
                state_dict_mod[i[6:]] = checkpoint['state_dict'][i] 
            model_setup.model.load_state_dict(state_dict_mod)
            model_ft = model_setup.model.eval()

        else:
            print("Model not implemented")
    else:
        if args.model_name == 'clip': 
            model_setup = CLIP_model(args, model_path)
            model_ft, criterion, optimizer = model_setup.setup_model()
            model_ft, _, _ = model_setup.train_model(args.dataset, model_ft, dataloaders, criterion, optimizer, num_epochs=args.epochs)
            model_ft.eval()
        elif args.model_name in models_implemented:
            # call model's Lightning module trainer, train and return model_ft
            model_setup = LitPytorchModels(args, model_path)
            logger = TensorBoardLogger(model_path+'/model', name=args.model_name)
            trainer = Trainer(gpus=args.num_gpus, default_root_dir=model_path+'/model', logger=logger, callbacks=[], max_epochs=args.epochs)
            trainer.fit(model_setup, dataloaders['train'], dataloaders['val'])
            trainer.validate(dataloaders=dataloaders['val'])
            
            base = model_path + '/model/'+args.model_name +'/' +'version_0/checkpoints/'
            ckpt = os.listdir(base)[0]
            PATH = base+ckpt
            checkpoint = torch.load(PATH)
            checkpoint['state_dict'] = dict(checkpoint['state_dict'])
            state_dict_mod = dict()
            for i in checkpoint['state_dict']:
                state_dict_mod[i[6:]] = checkpoint['state_dict'][i] 
            model_setup.model.load_state_dict(state_dict_mod)
            model_ft = model_setup.model.eval()
        else:
            print("Model not implemented")
    return model_ft

def extract_features(args, model_path: str, only_pretrained: bool, model_ft=None):
    """Generates features for a model and saves them in model_path

    Args:
        model_path: Path to model trial from which to extract features
        only_pretraied: When False, returns features generated from the pretrained and finetuned model
        model_ft: If only_pretrained=False, model_ft is the finetuned version of the model and used to
                  extract features on the finetuned model
    
    Returns:
        features: A dictionary mapping class name (specified in config file of analysis_set) to a tensor 
                  of features of size N x d where N is the number of examples in a class and is the dimension
                  size of the model
    """
    if args.model_name == 'clip':
        feature_extractor = ClipViTFeatureExtractor(args, model_path)
    else:
        feature_extractor = PytorchFeatureExtractor(args, model_path)

    if only_pretrained == True:
        features = feature_extractor.extract_features(args, only_pretrained=True)
    else:
        features = feature_extractor.extract_features(args, model_ft=model_ft, only_pretrained=False)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int,
            help='number of classes in dataset', default=80)
    parser.add_argument('--batch_size', type=int, 
            help='batch size for training', default=32)
    parser.add_argument('--epochs', type=int,
            help='number of epochs to train for', default=15)
    parser.add_argument('--lr', type=float,
            help='learning rate', default=0.01)
    parser.add_argument('--lr_scheduler',
            help='learning rate scheduler: cosine or reduce', default='none') 
    parser.add_argument('--momentum', type=float,
            help='momentum value for sgd optimizer', default=0.9)
    parser.add_argument('--feature_extract',
            help='When false, finetune the whole model, when True, only update the reshaped layer parameters', action='store_true')
    parser.add_argument('--optimizer', type=str,
            help='optimizer for training, sgd or adam', default='sgd')
    parser.add_argument('--model_name', type=str,
            help='model type: clip, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, alexnet, vgg, densenet, googlenet', default='resnet18')
    parser.add_argument('--dataset', type=str,
            help='Options: coco and openimages', default='coco')
    # parser.add_argument('--coco_path', type=str,
    #         help='folder path to coco dataset', default="/localtmp/data/coco2017/coco_dataset/")
    parser.add_argument('--dataset_path', type=str,
            help='folder path to training dataset', default="/localtmp/data/coco2017/coco_dataset/")
    parser.add_argument('--analysis_set_path', type=str,
            help='folder path to analysis dataset', default="/localtmp/data/coco2017/coco_dataset/")
    # parser.add_argument('--openimages_path', type=str,
    #         help='folder path to openimages', default="/data/openimages/")
    parser.add_argument('--bias_analysis',
            help='If True, performs cosine self similarity experiments', action='store_true')
    parser.add_argument('--load_features',
            help='If model has been trained, and want to use saved features for bias analysis', action='store_true')
    parser.add_argument('--pretrained_features', 
            help='Extract features from pretrained model without finetuning', action='store_true')
    parser.add_argument('--resume_training',
            help='Resume training from a saved checkpoint', action='store_true')
    parser.add_argument('--config_file', type=str,
            help='path to config file', default="config/coco_ini")
    #parser.add_argument('--pca', type=float, 
            #help='Number of components for pca on features before bias analysis', default=0.0)
    #parser.add_argument('--bias_metric', type=str,
            #help='cosine or euclidean for bias analysis', default="cosine")
    #parser.add_argument('--trend_analysis',
            #help='analyze trends across finetuned, pretrained, or differences: ft, pt, diff', action='store_true') 
    parser.add_argument('--finetune',
            help='finetune or train model from scratch', action='store_true')
    parser.add_argument('--multiple_trials',
            help='plot bias analysis across multiple trials', action='store_true')
    parser.add_argument('--extract_cross_analysis_features',
            help='extract features for an analysis set different than data the model is trained on', action='store_true')
    # parser.add_argument('--cross_analysis',
    #         help='Bias analysis for model trained on a different dataset than the analysis set', action='store_true')
    parser.add_argument('--trial_path', type=str,
            help='path to training run', default='experiments/coco/bit_resnet50/2022-01-15 14:18:21')
    parser.add_argument('--checkpoint', type=str,
            help='path to training run', default='none')
    parser.add_argument('--analysis_set', type=str,
            help='which dataset to perform bias analysis on', default='coco')
    parser.add_argument('--seed', type=int,
            help='random seed', default=1234)
    parser.add_argument('--num_gpus', type=int,
            help='random seed', default=1)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    if args.bias_analysis == True and args.load_features == True and not args.pretrained_features and not args.extract_cross_analysis_features:
        # Don't finetune and just use saved features for full bias analysis with finetuned and pretrained features
        print("Using saved features for full bias analysis")
        if args.multiple_trials == True:
            run_experiment(args.model_name, args.trial_path, args.dataset, args.analysis_set, args.config_file, features=None, only_pretrained = False, multiple_trials=args.multiple_trials)
        else:
            features = load_features(args.trial_path, analysis_set=args.analysis_set, only_pretrained=False)
            run_experiment(args.model_name, args.trial_path, args.dataset, args.analysis_set, args.config_file, features=features, only_pretrained = False, multiple_trials=args.multiple_trials)
    elif args.extract_cross_analysis_features == True:
        # Extracts pretrained and finetuned features for a model on an analysis set regardless of what dataset the model was trained on
        print("Extracting features on pretrained and finetuned model for dataset: "+args.analysis_set)
        model_ft = lightning_setup(args)
        features = extract_features(args, args.trial_path, only_pretrained=False, model_ft=model_ft)
        if args.bias_analysis == True:
            run_experiment(args.model_name, args.trial_path, args.dataset, args.analysis_set, args.config_file, features=features, only_pretrained = False, multiple_trials=args.multiple_trials)
    elif args.pretrained_features == True:
        # Only extract pretrained features and perform bias analysis on pretrained features
        print("Extracting features on pretrained model for dataset: "+ args.analysis_set)
        features = extract_features(args, args.trial_path, only_pretrained=True, model_ft=None)
        if args.bias_analysis == True:
            run_experiment(args.model_name, args.trial_path, args.dataset, args.analysis_set, args.config_file, features=features, only_pretrained=True, multiple_trials=args.multiple_trials)
    else:
        # Finetune, train the model from scratch or resume training, extract both pretrained and finetuned features and run bias analysis on them
        datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if args.finetune:
            os.mkdir('experiments/'+ args.dataset + '/' + args.model_name+'/'+datestring)
            model_path = 'experiments/'+ args.dataset + '/' + args.model_name+'/'+datestring
            setup_dirs(model_path)
        else:
            os.mkdir('experiments/'+ args.dataset + '/' +args.model_name+'/model_scratch'+'/'+datestring)
            model_path = 'experiments/'+ args.dataset + '/' +args.model_name+'/model_scratch'+'/'+datestring
            setup_dirs(model_path, from_scratch=True)

        dataloaders_dict = setup_dataset(args)
        model_ft = lightning_train(args, dataloaders_dict, model_path, resume_training=args.resume_training)
        features = extract_features(args, model_path, only_pretrained=False, model_ft=model_ft)
        if args.bias_analysis == True:
            # Run full bias experiment on finetuned and pretrained features
            run_experiment(args.model_name, model_path, args.dataset, args.analysis_set, args.config_file, features=features, only_pretrained = False, multiple_trials=args.multiple_trials)

if __name__ == '__main__':
    main()
