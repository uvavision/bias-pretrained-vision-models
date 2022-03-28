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
from cosine_exp import *
from models_def.pytorch_models import *
from models_def.clip_model import *
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def readFile(fileName):
    fileObj = open(fileName, "r") 
    urls = fileObj.read().splitlines() 
    fileObj.close()
    return urls

def analysis_data(config):
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    all_analysis_data = dict()
    analysis_data_names = dict()
    for category in conf['CLASSES']:
        all_analysis_data[category.lower()] = dict()
        for sub_category in conf['CLASSES'][category]:
            all_analysis_data[category.lower()][sub_category] = readFile(conf['CLASSES'][category][sub_category])
    for category in conf['CLASSES_NAMES']:
        analysis_data_names[category.lower()] = dict()
        for sub_category in conf['CLASSES_NAMES'][category]:
            analysis_data_names[category.lower()][sub_category] = conf['CLASSES_NAMES'][category][sub_category]
    return all_analysis_data, analysis_data_names
    
def load_coco(dataType, dataDir):
    annFile='{}annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    annFile_caps = '{}annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile_caps)
    return coco, coco_caps

def load_features(folder, pca, analysis_set, only_pretrained=False):
    #features_pt = os.listdir(folder+'/pretrained_features/')
    if pca==0.0:
        pca_path = 'no_pca/'
    else:
        pca_path = 'pca/'
    if only_pretrained == True:
        features = dict()
        features_pt = os.listdir(folder+'/features/'+analysis_set + '/pretrained_features/'+pca_path)
        for file_name in features_pt:
            features[os.path.splitext(file_name)[0]] = np.load(folder + '/features/' + analysis_set + '/pretrained_features/'+pca_path + file_name, allow_pickle=True)
    else:
        features = dict()
        features_pt = os.listdir(folder+'/features/'+analysis_set +'/pretrained_features/'+pca_path)
        features_ft = os.listdir(folder+'/features/'+analysis_set +'/finetuned_features/'+pca_path)
        for file_name in features_pt:
            features[os.path.splitext(file_name)[0]] = np.load(folder +'/features/'+analysis_set + '/pretrained_features/' + pca_path + file_name, allow_pickle=True)
        for file_name in features_ft:
            features[os.path.splitext(file_name)[0]] = np.load(folder +'/features/'+analysis_set + '/finetuned_features/' + pca_path + file_name, allow_pickle=True)
    return features

def plot_acc(hist, title, save_path):
    plt.plot([i for i in range(len(hist))], hist)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy')
    plt.savefig(save_path)

def setup_dset(args):
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
        train_imgs = args.coco_path + "train2017/train2017"
        val_imgs = args.coco_path + "val2017"
        train_anns = args.coco_path + "annotations/instances_train2017.json"
        val_anns = args.coco_path + "annotations/instances_val2017.json"
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
        train_dataset = OpenImages(Path(args.openimages_path), split="train", transform=preprocess)
        val_dataset = OpenImages(Path(args.openimages_path), split="val", transform=preprocess)


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



def lightning_setup(args):
    model_ft = None

    models_implemented = ['moco_resnet50', 'simclr_resnet50', 'alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'bit_resnet50', 'virtex_resnet50']
    if args.model_name == 'clip': 
        model_setup = CLIP_model(args, args.trial_path)
        model_ft, criterion, optimizer = model_setup.setup_model()
        checkpoint = torch.load(args.trial_path+'/model/model.pt')
        model_ft.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name in models_implemented:
        model_setup = LitPytorchModels(args, args.trial_path)
        base = args.trial_path + '/model/'+args.model_name +'/' +'version_0/checkpoints/'
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

def lightning_train(args, dataloaders, model_path, resume_training=False, checkpoint=None):
    model_setup = None
    model_ft = None
    criterion = None
    optimizer = None
    models_implemented = ['moco_resnet50', 'simclr_resnet50', 'alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'bit_resnet50', 'virtex_resnet50']
    seed_everything(args.seed, workers=True)
    if resume_training == True:
        if args.model_name == 'clip':
            model_setup = CLIP_model(args, model_path)
            model_ft, criterion, optimizer = model_setup.setup_model()
            checkpoint = torch.load(model_path+'/model/model.pt')
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model_ft, hist, total_targets = model_setup.train_model(args.model_name, args.dataset, model_ft, dataloaders, criterion, optimizer, num_epochs=args.epochs, scheduler=args.lr_scheduler)
            torch.save(model_ft.state_dict(), model_path+'/model/model_final.pt')
        elif args.model_name in models_implemented:
            # call model's Lightning module trainer, train and return model_ft
            #base = model_path + '/model/'+args.model_name +'/' +'version_0/checkpoints/'
            #ckpt = os.listdir(base)[0]
            #PATH = base+ckpt

            model_setup = LitPytorchModels(args, model_path)
            logger = TensorBoardLogger(model_path+'/model', name=args.model_name)
            trainer = Trainer(gpus=1, default_root_dir=model_path+'/model', logger=logger, callbacks=[], max_epochs=args.epochs, resume_from_checkpoint=checkpoint)
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
    else:
        if args.model_name == 'clip': 
            model_setup = CLIP_model(args, model_path)
            model_ft, criterion, optimizer = model_setup.setup_model()
            model_ft, hist, total_targets = model_setup.train_model(args.model_name, args.dataset, model_ft, dataloaders, criterion, optimizer, num_epochs=args.epochs, scheduler=args.lr_scheduler)
        
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


def extract_features(model_name, args, model_path, only_pretrained, model_ft=None):
    if model_name == 'clip':
        feature_extractor = ClipViTFeatureExtractor(args.analysis_set, model_path, model_name, args.num_classes, args.openimages)
    else:
        feature_extractor = PytorchFeatureExtractor(args.analysis_set, model_path, model_name, args.num_classes, args.openimages)

    if only_pretrained == True:
        features = feature_extractor.extract_features(dataset_name=args.analysis_set, only_pretrained=True, finetune=args.finetune, config=args.config_file, pca=args.pca)
    else:
        features = feature_extractor.extract_features(model_ft=model_ft, dataset_name=args.analysis_set, only_pretrained=False, finetune=args.finetune, config=args.config_file, pca=args.pca)
    return features

def setup_dirs(base_path, from_scratch=False):
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
    parser.add_argument('--coco_path', type=str,
            help='folder path to coco dataset', default="/localtmp/data/coco2017/coco_dataset/")
    parser.add_argument('--openimages_path', type=str,
            help='folder path to openimages', default="/data/openimages/")
    parser.add_argument('--bias_analysis',
            help='If True, performs cosine self similarity experiments', action='store_false')
    parser.add_argument('--load_features',
            help='If model has been trained, and want to use saved features for bias analysis', action='store_true')
    parser.add_argument('--pretrained_features', 
            help='Extract features from pretrained model without finetuning', action='store_true')
    parser.add_argument('--resume_training',
            help='Resume training from a saved checkpoint', action='store_true')
    parser.add_argument('--config_file', type=str,
            help='path to config file', default="config/coco_ini")
    parser.add_argument('--pca', type=float, 
            help='Number of components for pca on features before bias analysis', default=0.0)
    parser.add_argument('--bias_metric', type=str,
            help='cosine or euclidean for bias analysis', default="cosine")
    parser.add_argument('--trend_analysis',
            help='analyze trends across finetuned, pretrained, or differences: ft, pt, diff', action='store_true') 
    parser.add_argument('--finetune',
            help='finetune or train model from scratch', action='store_true')
    parser.add_argument('--multiple_trials',
            help='plot bias analysis across multiple trials', action='store_true')
    parser.add_argument('--extract_cross_analysis_features',
            help='extract features for an analysis set different than data the model is trained on', action='store_true')
    parser.add_argument('--cross_analysis',
            help='Bias analysis for model trained on a different dataset than the analysis set', action='store_true')
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

    # Set up dataset
    torch.backends.cudnn.benchmark = True
    
    
    if not args.load_features and not args.extract_cross_analysis_features:
        datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if args.finetune:
            os.mkdir('experiments/'+ args.dataset + '/' + args.model_name+'/'+datestring)
            model_path = 'experiments/'+ args.dataset + '/' + args.model_name+'/'+datestring
            setup_dirs(model_path)
            only_pretrained_features = False
        else:
            os.mkdir('experiments/'+ args.dataset + '/' +args.model_name+'/model_scratch'+'/'+datestring)
            model_path = 'experiments/'+ args.dataset + '/' +args.model_name+'/model_scratch'+'/'+datestring
            setup_dirs(model_path, from_scratch=True)
            only_pretrained_features = True

    if args.bias_analysis == True and args.load_features == True and not args.pretrained_features and not args.extract_cross_analysis_features:
        # Don't finetune and just use saved features for full bias analysis with finetuned and pretrained features
        print("Using saved features for full bias analysis")
        features = load_features(args.trial_path, pca=args.pca, analysis_set=args.analysis_set, only_pretrained=False)
        run_experiment(args.model_name, args.trial_path, args.dataset, args.analysis_set, features, args.finetune, args.config_file, args.bias_metric, args.pca, trend_analysis = args.trend_analysis, only_pretrained = False, multiple_trials=args.multiple_trials)
    elif args.extract_cross_analysis_features == True:
        print("Extracting features for dataset: "+args.analysis_set)
        model_ft = lightning_setup(args)
        features = extract_features(args.model_name, args, args.trial_path, only_pretrained=False, model_ft=model_ft)
    elif args.bias_analysis == True and args.pretrained_features == True:
        # Only extract pretrained features and perform bias analysis on pretrained features
        print("Extracting pretrained features for bias analysis")
        #model_ft, criterion, optimizer = lightning_setup(args)
        features = extract_features(args.model_name, args, model_path, only_pretrained=True, model_ft=None)
        run_experiment(args.model_name, model_path, args.dataset, args.analysis_set, features, args.finetune, args.config_file, args.bias_metric, args.pca, args.trend_analysis, only_pretrained=True, multiple_trials=args.multiple_trials)
    else:
        dataloaders_dict = setup_dset(args)
        # Finetune or train the model from scratch, or resume training, extract both pretrained and finetuned features and save them
        model_ft = lightning_train(args, dataloaders_dict, model_path, resume_training=args.resume_training, checkpoint=args.checkpoint)
        features = extract_features(args.model_name, args, model_path, only_pretrained=False, model_ft=model_ft)
        if args.bias_analysis == True:
            if args.pca != 0.0:
                features = load_features(model_path, pca=args.pca, analysis_set=args.analysis_set, only_pretrained=False)
            # Run full bias experiment on finetuned and pretrained features
            run_experiment(args.model_name, model_path, args.dataset, args.analysis_set, features, args.finetune, args.config_file, args.bias_metric, args.pca, args.trend_analysis, only_pretrained = only_pretrained_features, multiple_trials=args.multiple_trials)

if __name__ == '__main__':
    main()
