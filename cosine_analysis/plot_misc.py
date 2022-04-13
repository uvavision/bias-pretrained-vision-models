from __future__ import print_function
from __future__ import division
import random

import matplotlib
import matplotlib.lines as mlines
from data_loader import V

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
from scipy import stats
import torch
print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random
import scipy.stats as stats

import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(10, 8), dpi=1500, facecolor='w', edgecolor='k')
from matplotlib.font_manager import FontProperties
import yaml
import dcor 
import itertools
import matplotlib.cm as cm
import datetime
import plotly.graph_objects as go
import pandas as pd

fontP = FontProperties()
fontP.set_size('small')

def plot_misc(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_features, plot_type):
    """Plots bias analysis experiment results for different subset of classes
    'Indiv': single classes --> man, woman, surfboard, car, refrigerator, etc. (intra-class)
    'Pairs': multi-label --> man+surfboard, woman+car, etc. (intra-class)
    'Comps' compares two classes --> man vs. surfboard, woman+car vs woman etc. (inter-class)
    
    Args:
        model_name: Name of model to perform bias metric experiment on
        dataset_name: Analysis set
        save_path: Path to save averaged results, ex. 'experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'
        mins_maxes_pt: Dictionary mapping class name (V in config file) to a list of mins and maxes of 
                       bias metric iterations (result of intra_class_similarity_error_bars() and 
                       inter_class_similarity_error_bars()) for features extracted from pretrained model for intra-class
        mins_maxes_ft: Dictionary mapping class name (LABEL_NAMES in config file) to a list of mins and maxes of 
                       bias metric iterations (result of intra_class_similarity_error_bars() and 
                       inter_class_similarity_error_bars()) for features extracted from finetuned model for intra-class
        mins_maxes_comps_pt: Dictionary mapping class name (LABEL_NAMES in config file) to a list of mins and maxes of 
                             bias metric iterations (result of intra_class_similarity_error_bars() and 
                             inter_class_similarity_error_bars()) for features extracted from pretrained model for inter-class
        mins_maxes_comps_ft: Dictionary mapping class name (LABEL_NAMES in config file) to a list of mins and maxes of 
                             bias metric iterations (result of intra_class_similarity_error_bars() and 
                             inter_class_similarity_error_bars()) for features extracted from finetuned model for inter-class
        comps_stats: Dictionary mapping class comparisons (ex. man+woman vs. man) defined in COMPS in config to
                     a tuple of (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity() for features extracted 
                     from pretrained model
        comps_stats_ft: Dictionary mapping class comparisons (ex. man+woman vs. man) defined in COMPS in config to
                     a tuple of (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity() for features extracted 
                     from finetuned model
        self_similarities_stats: Dictionary mapping class names (LABEL_NAMES in config) to a tuple of 
                                 (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity() 
                                 for features extracted from pretrained model
        self_similarities_stats_ft: Dictionary mapping class names (LABEL_NAMES in config) to a tuple of 
                                 (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity() 
                                 for features extracted from finetuned model
        category_features: Classes to be plotted
        plot_type: Indiv, Pairs, Comps --> subset of classes in analysis set
        
    """
    y_dict = dict()
    y_dict_ft = dict()
    for i in category_features:
        if i in comps_stats:
            y_dict[i] = comps_stats[i][3]
            y_dict_ft[i] = comps_stats_ft[i][3]
        else:
            y_dict[i] = self_similarities_stats[i][3]
            y_dict_ft[i] = self_similarities_stats_ft[i][3]

    y_dict_comp = dict()

    #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
    y_sorted = y_dict
    y = np.asarray(list(y_sorted.values()))
    labels = list(y_sorted.keys())

    for label in y_sorted:
        y_dict_comp[label] = y_dict_ft[label]
    y_ft = np.asarray(list(y_dict_comp.values()))
    spearman_coeff = stats.spearmanr(y, y_ft)
    spearman_save = {model_name:spearman_coeff}
    np.save(save_path+'/metric_data/'+ dataset_name + '/'+'spearman_'+plot_type+'.npy', spearman_save)


    y_err = []
    y_err_ft = []
    for key in y_sorted:
        if key in mins_maxes_pt:
            y_err.append(mins_maxes_pt[key])
        else:
            y_err.append(mins_maxes_comps_pt[key])
        
    for key in y_dict_comp:
        if key in mins_maxes_ft:
            y_err_ft.append(mins_maxes_ft[key])
        else:
            y_err_ft.append(mins_maxes_comps_ft[key])  
    yerr_vals = np.asarray(y_err).T
    yerr_vals_ft = np.asarray(y_err_ft).T

    yerr = np.abs(yerr_vals-y)
    yerr_ft = np.abs(yerr_vals_ft-y_ft)

    fig = plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']

    colors_err = ['turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue',
            'gold', 'peru', 'palegreen']
    colors_err_ft = ['peachpuff', 'dodgerblue', 'tomato', 'g', 'b', 'c', 'm', 'lime', 'orange']


    for i in range(len(y_dict)):
        plt.errorbar(np.asarray([i]), np.asarray([y[i]]), yerr=yerr[:, i:i+1], fmt='o', color='b', ecolor='b')

    for i in range(len(y_dict)):
        plt.errorbar(np.asarray([i]), np.asarray([y_ft[i]]), yerr=yerr_ft[:, i:i+1], fmt="^", color='g', ecolor='g')
    plt.xticks(np.arange(len(y_dict)), labels = labels, rotation=45)


    if plot_type == 'pairs':
        title = "Model: "+ model_name + " Finetuned on: " + dataset_name + ", Paired Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name + ", Paired Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/'+'pairs.pdf'
        save_diff = save_path + '/boxplots/' + dataset_name + '/' +'pairs_diff.pdf'
        bbox = (0.5, -0.6)
        bottom = 0.5
    elif plot_type == 'comps':
        title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Comparison Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name +", Comparison Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/' + 'comps.pdf'
        save_diff = save_path + '/boxplots/'+ dataset_name + '/'+'comps_diff.pdf'
        bbox = (0.5, -0.7)
        bottom = 0.5
    elif plot_type == 'indiv':
        title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Individual Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/' +'indv.pdf'
        save_diff = save_path + '/boxplots/'+ dataset_name + '/' + 'indv_diff.pdf'  
        bbox = (0.5, -0.6)
        bottom = 0.5
    else:
        title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Class vs. Gender, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name +", Class vs. Gender (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/' + 'object_comp.pdf'
        save_diff = save_path + '/boxplots/' + dataset_name + '/' + 'object_comp_diffs.pdf'
        bbox = (0.5, -0.8)
        bottom = 0.5

    # plt.legend(labels, bbox_to_anchor=(1.1, 1.1), loc='upper left', prop=fontP)
    #l1 = plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
    
    triangle = matplotlib.lines.Line2D([], [], color='g', marker='^', linestyle='None',
                          markersize=10, label='Finetuned')
    circle = matplotlib.lines.Line2D([], [], color='b', marker='o', linestyle='None',
                          markersize=10, label='Pretrained')            
    plt.legend(handles=[circle, triangle], loc='lower center', bbox_to_anchor=(0.5, -0.8), prop=fontP, ncol=2)
    #plt.gca().add_artist(l1)

    plt.grid(b=True)
    fig.subplots_adjust(bottom=bottom)


    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("cosine score")
    plt.savefig(save, format='pdf')

    fig2 = plt.figure()
    for i in range(len(y_dict)):
        plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
    plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
    plt.grid(b=True)
    fig2.subplots_adjust(bottom=bottom)
    #plt.tight_layout()

    plt.title(title_diff)
    plt.xlabel("Classes")
    plt.ylabel("cosine score")
    plt.savefig(save_diff, format='pdf')

