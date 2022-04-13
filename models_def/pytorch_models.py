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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
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
from sklearn.metrics import fbeta_score

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
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from model_init import *
from data_loader import *
from cosine_analysis.cosine_exp import *
from train import *
import torchmetrics 
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from utils import *

class LitPytorchModels(pl.LightningModule):
    """Pytorch Lightning module for training models from torchvision

    Attributes:
        model: The initialized pretrained model with the last layer modified to fit the finetuning objective
        criterion: Loss function to be used during finetuning: set to BCEWithLogitsLoss()
        optimizer: Pytorch optimizer to use during finetuning: sgd, adamax, lars, adam
        model_name: Name of model to be finetuned
        dataset: Dataset to finetune on: coco or openimages
        scheduler: Learning rate scheduler: cosine, reduce, or none
        model_path: Path to model trial: ex. 'experiments/coco/bit_resnet50/2022-01-15 14:18:21'

    """
    def __init__(self, args, model_path):
        super().__init__()
        model_ft, criterion, optimizer = initialize_model_pytorch(args.model_name, args.num_classes, args.feature_extract, args.lr, args.momentum, args.optimizer, use_pretrained=args.finetune)
        self.model = model_ft
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_name =  args.model_name
        self.dataset = args.dataset

        if args.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 100)
            self.lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif args.lr_scheduler == 'reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            self.lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "Training Loss",
            }
        else:
            self.scheduler = None
        

        self.model_path = model_path

    def forward(self, x):
        outputs = self.model(x.float())
        return outputs

    def configure_optimizers(self):
        if self.scheduler == None:
            return self.optimizer
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler_config}
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs.float())
        loss = self.criterion(outputs, labels)

        preds = torch.sigmoid(outputs)
        if self.dataset == 'openimages':
            idx = 50000
        else:
            idx = 1000
        if batch_idx % idx == 0:
            print("Saving preds ... ")
            torch.save(inputs[:10], self.model_path+"/model/updates/train_inputs_"+str(batch_idx)+".pt")
            torch.save(labels[:10], self.model_path+"/model/updates/train_labels_"+str(batch_idx)+".pt")
            torch.save(preds[:10], self.model_path+"/model/updates/train_preds_"+str(batch_idx)+".pt")
        
        logs = {"train_loss": loss}
        batch_dictionary = {
            'loss': loss,
            'log': logs,
            'preds': preds.data.cpu(),
            'labels': labels.data.cpu()
        }
        self.log('Training Loss', loss, on_step=True, on_epoch=False)
        return batch_dictionary

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        total_preds = torch.cat([x['preds'] for x in outputs], 0) 
        total_targets = torch.cat([x['labels'] for x in outputs], 0) 
        task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
        task_f1_score_micro = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'micro')
        fbeta = fbeta_score(total_targets.numpy(), (total_preds >= 0.3).long().numpy(), 1.0, average='samples')
        meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss,
                            'Train Macro F1': torch.tensor(task_f1_score),
                            'Train Micro F1': torch.tensor(task_f1_score_micro), 
                            'Train Fbeta': torch.tensor(fbeta),
                            'Train MeanAP': torch.tensor(meanAP)}
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs}
        print('F1: {:.4f} mAP: {:.4f} fbeta: {:.4f}  Micro f1: {:.4f} Loss: {:.4f}'.format(task_f1_score, meanAP, fbeta, task_f1_score_micro, avg_loss))
        
        self.log('Avg Training Loss', avg_loss, on_step=False, on_epoch=True)
        self.log('Train Macro F1 Score', torch.tensor(task_f1_score), on_step=False, on_epoch=True)
        self.log('Train Micro F1 Score', torch.tensor(task_f1_score_micro), on_step=False, on_epoch=True)
        self.log('Train FBeta Score', torch.tensor(fbeta), on_step=False, on_epoch=True)
        self.log('Train Mean AP', torch.tensor(meanAP), on_step=False, on_epoch=True)
        #return epoch_dictionary

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs.float())
        loss = self.criterion(outputs, labels)

        preds = torch.sigmoid(outputs)
        if batch_idx % 1000 == 0:
            print("Saving preds ... ")
            torch.save(inputs, self.model_path+"/model/updates/val_inputs_"+str(batch_idx)+".pt")
            torch.save(labels, self.model_path+"/model/updates/val_labels_"+str(batch_idx)+".pt")
            torch.save(preds, self.model_path+"/model/updates/val_preds_"+str(batch_idx)+".pt")
        
        logs = {"val_loss": loss}
        batch_dictionary = {
            'loss': loss,
            'log': logs,
            'preds': preds.data.cpu(),
            'labels': labels.data.cpu()
        }
        self.log('Validation Loss', loss, on_step=True, on_epoch=False)
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        total_preds = torch.cat([x['preds'] for x in outputs], 0) 
        total_targets = torch.cat([x['labels'] for x in outputs], 0) 
        task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
        task_f1_score_micro = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'micro')
        fbeta = fbeta_score(total_targets.numpy(), (total_preds >= 0.3).long().numpy(), 1.0, average='samples')
        meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
        # creating log dictionary
        tensorboard_logs = {'Val loss': avg_loss,
                            'Val Macro F1': torch.tensor(task_f1_score),
                            'Val Micro F1': torch.tensor(task_f1_score_micro), 
                            'Val Fbeta': torch.tensor(fbeta),
                            'Val MeanAP': torch.tensor(meanAP)}
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs}
        print('F1: {:.4f} mAP: {:.4f} fbeta: {:.4f}  Micro f1: {:.4f} Loss: {:.4f}'.format(task_f1_score, meanAP, fbeta, task_f1_score_micro, avg_loss))

        self.log('Avg Validation Loss', avg_loss, on_step=False, on_epoch=True)
        
        self.log('Val Macro F1 Score', torch.tensor(task_f1_score), on_step=False, on_epoch=True)
        self.log('Val Micro F1 Score', torch.tensor(task_f1_score_micro), on_step=False, on_epoch=True)
        self.log('Val FBeta Score', torch.tensor(fbeta), on_step=False, on_epoch=True)
        self.log('Val Mean AP', torch.tensor(meanAP), on_step=False, on_epoch=True)

        #return epoch_dictionary

class PytorchFeatureExtractor():
    """Defines preprocesssing of analysis set and feature extraction for models from torchvision

    Attributes:
        dataset: The dataset used to extract features on --> analysis set
        model_path: Path to model trial that defines where to save the features: ex. 'experiments/coco/bit_resnet50/2022-01-15 14:18:21'
        model_name: Name of model to perform feature extraction on 
        num_classes: Number of classes in dataset 
       
    """

    def __init__(self, args, model_path):
        self.dataset = args.analysis_set
        self.analysis_set_path = args.analysis_set_path
        self.model_path = model_path
        self.model_name = args.model_name
        self.num_classes = args.num_classes
        

    def process_imgs(self, directory: list):
        """Preprocesses a list of tensor images using standard transformations"""
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        images_stacked = []

        for filename in directory:
            if self.dataset=='coco':
                input_image = Image.open(urllib.request.urlopen(filename)).convert("RGB")
            elif self.dataset == 'openimages':
                input_image = Image.open(self.analysis_set_path+filename.strip()+'.jpg').convert("RGB")
            # if adding a new analysis set, specify how to open the images in the .txt files in analysis_sets/
            else:
                print("Dataset not implemented")
            input_tensor = preprocess(input_image)
            images_stacked.append(input_tensor)
        return torch.tensor(torch.stack(images_stacked,0))

    def extract_features_model(self, model, directory: list, norm="norm_one"):
        """Preprocesses batch of images and extracts features from the penultimate layer of the model"""
        features_dir = [] # batch x 16 x 2048
        images_processed = self.process_imgs(directory)
        imgs_split = torch.split(images_processed, 16)
        for input_batch in imgs_split:
            # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                output = torch.squeeze(output)
                if input_batch.shape[0] == 1:
                    output = torch.unsqueeze(output, 0)
                features_dir.append(output)
        features = torch.tensor(torch.cat(features_dir, 0))
        if norm == "norm_one":
            features /= features.norm(dim=-1, keepdim=True)
        else:
            scaler = StandardScaler() 
            features = scaler.fit_transform(features.cpu())
        return torch.tensor(features)


    def build_features_pt(self, images: list):
        """Extracts features for model using the pretrained off-the-shelf version from torchvision"""
        model_loaded = load_models_pytorch(self.model_name, self.num_classes, True)
        if self.model_name != 'bit_resnet50':
            modules=list(model_loaded.children())[:-1]
            resnet_model=nn.Sequential(*modules)
            for p in resnet_model.parameters():
                p.requires_grad = False
            features = self.extract_features_model(resnet_model, images)
        else:
            if self.model_name == 'bit_resnet50':
                model_loaded.head = model_loaded.head[:-1]
                features = self.extract_features_model(model_loaded, images)
        return features

    def build_features_ft(self, model_ft, images: list):
        """Extracts features for model after it has been finetuned"""
        if self.model_name != 'bit_resnet50':
            modules=list(model_ft.children())[:-1]
            resnet_model=nn.Sequential(*modules)
            for p in resnet_model.parameters():
                p.requires_grad = False
            features = self.extract_features_model(resnet_model, images)
        else:
            if self.model_name == 'bit_resnet50':
                temp_model = copy.deepcopy(model_ft)
                temp_model.head = temp_model.head[:-1]
                features = self.extract_features_model(temp_model, images)
        return features
    
    def pca_analysis(self, features, pca_comps: float, path: str):
        """Performs PCA analysis on the features after they have been extracted"""
        feature_list = list(features.values())
        feature_list = [i.cpu().numpy() for i in feature_list]
        all_images_sizes = []
        for feature in feature_list:
            all_images_sizes.append(feature.shape[0])
        all_images_sizes = np.cumsum(all_images_sizes)
                
        all_features = np.concatenate(feature_list)
        if pca_comps > 1.0:
            comps = min(all_features.shape[0], all_features.shape[1])
            if pca_comps > comps:
                pca_comps = comps
        pca = PCA(n_components=int(pca_comps)) 
        features_transformed = pca.fit_transform(all_features)
        print(pca.singular_values_)
        
        features_transformed_dict = dict()
        features_split = np.split(features_transformed, all_images_sizes)
        for key, feature_transformed in zip(features, features_split):
            features_transformed_dict[key] = feature_transformed
            np.save(path + key, feature_transformed) 
        return features_transformed_dict
         

    def extract_features(self, args, model_ft=None, only_pretrained: bool = False):
        """Extracts and saves the features in model_path"""
        save_path = self.model_path+'/'
        analysis_data_loaded, analysis_data_names = analysis_data(args.config_file) # returns a dictionary mapping category[subcategory] = list_imgs
        if only_pretrained == True:
            print("Building pretrained features for ... " + self.model_name + " on analysis set: " + args.analysis_set)
            pretrained_features = dict()
            for category in analysis_data_loaded:
                for subcategory in analysis_data_loaded[category]:
                    print("Building pretrained features for: ", analysis_data_names[category][subcategory])
                    features = self.build_features_pt(analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/'+self.dataset+"/pretrained_features/" + analysis_data_names[category][subcategory] + "_pt.npy", features.cpu().numpy())
                    pretrained_features[analysis_data_names[category][subcategory] + "_pt"] = features
            #if args.pca != 0.0:
                #print("Computing PCA Analysis with n_components = " + str(args.pca))
                #pretrained_features = self.pca_analysis(pretrained_features, args.pca, save_path+'features/'+self.dataset+"/pretrained_features/pca/")
            return pretrained_features

        elif args.finetune == False:
            print("Building features for ... " + self.model_name + " trained from scratch" + " on analysis set: " + args.analysis_set)
            trained_features = dict()
            for category in analysis_data_loaded:
                for subcategory in analysis_data_loaded[category]:
                    print("Building finetuned features for: ", analysis_data_names[category][subcategory])
                    features = self.build_features_ft(model_ft, analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/' + self.dataset+"/pretrained_features/" + analysis_data_names[category][subcategory] + "_pt.npy", features.cpu().numpy())
                    trained_features[analysis_data_names[category][subcategory] + "_pt"] = features
            #if args.pca != 0.0:
                #print("Computing PCA Analysis with n_components = " + str(args.pca))
                #trained_features = self.pca_analysis(trained_features, args.pca, save_path+self.dataset+"/pretrained_features/pca/")
            return trained_features
        else:
            print("Building finetuned and pretrained features for ... " + self.model_name + " on analysis set: " + args.analysis_set)
            all_features = dict()
            all_features_ft = dict()
            all_features_pt = dict()
            
            for category in analysis_data_loaded:
                for subcategory in analysis_data_loaded[category]:
                    print("Building features for: ", analysis_data_names[category][subcategory])
                    features_ft = self.build_features_ft(model_ft, analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/'+self.dataset+"/finetuned_features/" + analysis_data_names[category][subcategory] + "_ft.npy", features_ft.cpu().numpy())
                    all_features[analysis_data_names[category][subcategory] + "_ft"] = features_ft
                    all_features_ft[analysis_data_names[category][subcategory] + "_ft"] = features_ft
                    
                    features_pt = self.build_features_pt(analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/' +self.dataset+"/pretrained_features/" + analysis_data_names[category][subcategory] + "_pt.npy", features_pt.cpu().numpy())
                    all_features[analysis_data_names[category][subcategory] + "_pt"] = features_pt
                    all_features_pt[analysis_data_names[category][subcategory] + "_pt"] = features_pt
            #if args.pca != 0.0:
                #print("Computing PCA Analysis with n_components = " + str(args.pca))
                #all_features_ft = self.pca_analysis(all_features_ft, args.pca, save_path+self.dataset+"/finetuned_features/pca/")
                #all_features_pt = self.pca_analysis(all_features_pt, args.pca, save_path+self.dataset+"/pretrained_features/pca/")
                #all_features = all_features_ft.update(all_features_pt)
            
            return all_features
