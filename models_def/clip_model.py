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
from sklearn import metrics
from sklearn.metrics import accuracy_score
import clip
#from skimage import io
from pycocotools.coco import COCO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score

import torch
print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from train import *
from model_init import *
from data_loader import *
from cosine_analysis.cosine_exp import *
from utils import * 
import yaml
from sklearn.decomposition import PCA

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 420
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class CLIP_model():
    def __init__(self, args, model_path):
        model_ft, criterion, optimizer = initialize_model_clip(args.num_classes, args.lr, args.momentum, args.optimizer)
        self.model = model_ft
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = args.dataset
        self.model_path = model_path

    def setup_model(self):
        return self.model, self.criterion, self.optimizer

    def convert_models_to_fp32(self, model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
            
    def convert_weights(self, model):
        """Convert applicable model parameters to fp16"""

        def _convert_weights_to_fp16(l):
            if isinstance(l, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, torch.nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        model.apply(_convert_weights_to_fp16)

    def train_model(self, model_name,  dataset_name,  model, dataloaders, criterion, optimizer, num_epochs=25, scheduler='none'):
        since = time.time()

        scaler = torch.cuda.amp.GradScaler()
        val_acc_history = []

        # best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epoch_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                res = list() # results from model (targets, preds)
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                
                for i, (inputs, labels) in (enumerate(tqdm.tqdm(dataloaders[phase]))):
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        preds = model(inputs.type(torch.half).cuda())
                        loss = criterion(preds, labels.to(preds.dtype).cuda())
                        preds_sigmoid = torch.sigmoid(preds)
                        # loss.backward()
                        if phase == 'train':
                            res.append((preds_sigmoid.data.cpu(), labels.data.cpu()))
                            scaler.scale(loss).backward()
                            self.convert_models_to_fp32(model)
                            #optimizer.step()
                            scaler.step(optimizer)
                            scaler.update()
                            self.convert_weights(model)
                            #train_loss += loss.item()
                        else:
                            res.append((preds.data.cpu(), labels.data.cpu())) 
                        if dataset_name == 'openimages':
                            if i % 70000 == 0:
                                total_preds = torch.cat([entry[0] for entry in res], 0) 
                                total_targets = torch.cat([entry[1] for entry in res], 0)
                                task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
                                task_f1_score_micro = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'micro')

                                fbeta = fbeta_score(total_targets.numpy(), (total_preds >= 0.3).long().numpy(), 1.0, average='samples')
                                meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
                                epoch_acc = accuracy_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy())
                                res = list()
                                epoch_acc = accuracy_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy())
                                print('{} F1: {:.4f} mAP: {:.4f} fbeta: {:.4f}  Micro f1: {:.4f}'.format(phase, task_f1_score, meanAP, fbeta, task_f1_score_micro))
                                            
                                    # statistics
                                running_loss += loss.item() * inputs.size(0)

                if dataset_name != 'openimages':
                    total_preds = torch.cat([entry[0] for entry in res], 0) 
                    total_targets = torch.cat([entry[1] for entry in res], 0)
                    task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
                    task_f1_score_micro = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'micro')

                    fbeta = fbeta_score(total_targets.numpy(), (total_preds >= 0.3).long().numpy(), 1.0, average='samples')
                    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
                    epoch_acc = accuracy_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy())
                    #res = list()
                    epoch_acc = accuracy_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy())
                    print('{} F1: {:.4f} mAP: {:.4f} fbeta: {:.4f}  Micro f1: {:.4f}'.format(phase, task_f1_score, meanAP, fbeta, task_f1_score_micro))
                                
                        # statistics
                    running_loss += loss.item() * inputs.size(0)
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, self.model_path + "/model/model.pt")
                if phase == 'val':           
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        return model, val_acc_history, total_targets

class ClipViTFeatureExtractor():
    def __init__(self, args, model_path):
        self.dataset = args.analysis_set
        if self.dataset == 'openimages':
            self.openimages_path = args.openimages_path+"val/"
        self.model_path = model_path
        self.model_name = args.model_name
        self.num_classes = args.num_classes

    def process_images(self, list_imgs):
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor()
        ])
        images = []
        for filename in [filename for filename in list_imgs]:
            if self.dataset == 'coco':
                image = preprocess(Image.open(urllib.request.urlopen(filename)).convert("RGB"))
            elif self.dataset == 'openimages':
                image = preprocess(Image.open(self.openimages_path+filename.strip()+'.jpg').convert("RGB"))
            else:
                print("Dataset not implemented")
            images.append(image) 
        return images

    def extract_features_clip_ft(self, images, model_ft, normalize='norm_one'): 
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        new_classifier = torch.nn.Sequential(*list(model_ft.children())[:-1])
        new_classifier.eval()
        image_input = torch.tensor(np.stack(images)).cuda()
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features = new_classifier(image_input.type(torch.half))
        if normalize == 'norm_one':
            image_features /= image_features.norm(dim=-1, keepdim=True)
        else:
            scaler = StandardScaler() 
            image_features = scaler.fit_transform(image_features.cpu())
        return torch.tensor(image_features)

    def extract_features_clip_pt(self, images, model, normalize='norm_one'): 
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        image_input = torch.tensor(np.stack(images)).cuda()
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        if normalize == 'norm_one':
            image_features /= image_features.norm(dim=-1, keepdim=True)
        else:
            scaler = StandardScaler() 
            image_features = scaler.fit_transform(image_features.cpu())
        return torch.tensor(image_features)
    
    def build_features_pt(self, images):
        images = self.process_images(images)
        torch.backends.cudnn.benchmark = True 
        model, preprocess = clip.load("ViT-B/32", jit=False) #Must set jit=False for training
        model = model.cuda()
        features = self.extract_features_clip_pt(images, model)
        return features
    
    def build_features_ft(self, model_ft, images):
        images = self.process_images(images)
        features = self.extract_features_clip_ft(images, model_ft)
        return features
    
    def pca_analysis(self, features, pca_comps, path):
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
    
    def extract_features(self, args, model_ft=None, only_pretrained=False):
        
        save_path = self.model_path+"/"
        analysis_data_loaded, analysis_data_names = analysis_data(args.config_file) # returns a dictionary mapping category[subcategory] = list_imgs
        #surfboard, car, refrigerator, random_loaded, man_background, woman_background, stop_sign = analysis_data(dataset_name, config)
        if only_pretrained == True:
            print("Building pretrained ViT/Resnet50 features ... " + " on analysis set: " + args.analysis_set)

            pretrained_features = dict()
            for category in analysis_data_loaded:
                for subcategory in analysis_data_loaded[category]:
                    print("Building pretrained features for: ", analysis_data_names[category][subcategory])
                    features = self.build_features_pt(analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/'+self.dataset+"/pretrained_features/no_pca/" + analysis_data_names[category][subcategory] + "_pt.npy", features.cpu().numpy())
                    pretrained_features[analysis_data_names[category][subcategory] + "_pt"] = features
            if args.pca != 0.0:
                print("Computing PCA Analysis with n_components = " + str(args.pca))
                pretrained_features = self.pca_analysis(pretrained_features, args.pca, save_path+self.dataset+"/pretrained_features/pca/")
            return pretrained_features
        else:
            print("Building finetuned and pretrained features for ViT/Resnet50" + " on analysis set: " + args.analysis_set)
            all_features = dict()
            all_features_ft = dict()
            all_features_pt = dict()
            for category in analysis_data_loaded:
                for subcategory in analysis_data_loaded[category]:
                    print("Building finetuned features for: ", analysis_data_names[category][subcategory])
                    features_ft = self.build_features_ft(model_ft, analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/' +self.dataset+"/finetuned_features/no_pca/" + analysis_data_names[category][subcategory] + "_ft.npy", features_ft.cpu().numpy())
                    all_features[analysis_data_names[category][subcategory] + "_ft"] = features_ft
                    all_features_ft[analysis_data_names[category][subcategory] + "_ft"] = features_ft
                    print("Building pretrained features for: ", analysis_data_names[category][subcategory])
                    features_pt = self.build_features_pt(analysis_data_loaded[category][subcategory])
                    np.save(save_path+'features/'+self.dataset+"/pretrained_features/no_pca/" + analysis_data_names[category][subcategory] + "_pt.npy", features_pt.cpu().numpy())
                    all_features[analysis_data_names[category][subcategory] + "_pt"] = features_pt
                    all_features_pt[analysis_data_names[category][subcategory] + "_pt"] = features_pt
            if args.pca != 0.0:
                print("Computing PCA Analysis with n_components = " + str(args.pca))
                all_features_ft = self.pca_analysis(all_features_ft, args.pca, save_path+self.dataset+"/finetuned_features/pca/")
                all_features_pt = self.pca_analysis(all_features_pt, args.pca, save_path+self.dataset+"/pretrained_features/pca/")
                all_features = all_features_ft.update(all_features_pt)
            return all_features
