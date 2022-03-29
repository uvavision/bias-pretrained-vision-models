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
from cosine_analysis.plot_intra_class import *
from cosine_analysis.plot_intra_class_avg import *
from cosine_analysis.plot_misc import *
from cosine_analysis.plot_misc_avg import *

fontP = FontProperties()
fontP.set_size('small')

def cosine_similarity_function(a, b):
    similarity = cosine_similarity(a, b)
    return np.mean(similarity), np.std(similarity)

def euclidean_function(a, b):
    distance = euclidean_distances(a, b)
    return np.mean(distance), np.std(distance)

def distance_correlation_function(a, b):
    if a.shape[0] != b.shape[0]:
        if a.shape[0] < b.shape[0]:
            b = b[1:]
        else:
            a = a[1:]
    distance = dcor.distance_correlation(a, b)
    return np.mean(distance), np.std(distance)

def split_features(features, bias_metric):
    try:
        np_features = features.cpu().numpy()
    except:
        np_features = features
    split = np_features[np.random.permutation(np.arange(np_features.shape[0]))]
    shape = split.shape[0]//2
    split1, split2 = split[:shape], split[shape:]
    if bias_metric == 'cosine':
        similarity, standard_deviation = cosine_similarity_function(split1, split2)
    elif bias_metric == 'euclidean':
        similarity, standard_deviation = euclidean_function(split1, split2)
    elif bias_metric == 'correlation':
        similarity, standard_deviation = distance_correlation_function(split1, split2)
    else:
        print("Bias metric not implemented")
    return similarity, standard_deviation

def random_split_two_features(features_one, features_two):
    try:
        np_features_one = features_one.cpu().numpy()
        np_features_two = features_two.cpu().numpy()
    except:
        np_features_one = features_one
        np_features_two = features_two
    num_images_1 = np.random.randint(low=5, high=np_features_one.shape[0], size=1)
    num_images_2 = np.random.randint(low=5, high=np_features_two.shape[0], size=1)

    random_one = np_features_one[np.random.choice(np_features_one.shape[0], num_images_1, replace=False), :]
    random_two = np_features_two[np.random.choice(np_features_two.shape[0], num_images_2, replace=False), :]

    similarity, standard_deviation = cosine_similarity_function(random_one, random_two)
    return similarity, standard_deviation 

def similarities_features(features_one, features_two, num_iter):
    # computes similarities between two features by randomly permuting them
    cosine_similarities = []
    standard_deviations = []
    for i in range(num_iter):
        similarity, std = random_split_two_features(features_one, features_two)
        
        cosine_similarities.append(similarity)
        standard_deviations.append(std)
    
    std_similarity = np.std(cosine_similarities)
    mean_similarity = np.mean(cosine_similarities)

    t_test_calc = stats.ttest_1samp(a=random.sample(cosine_similarities, 30), popmean=mean_similarity)
    if t_test_calc[1] > 0.05:
        t_test = 'fail to reject'
    else:
        t_test = 'reject'
    
    return cosine_similarities, standard_deviations, std_similarity, mean_similarity, t_test

def self_similarities(features, num_iter, bias_metric):
    # computes similarities between one set of features split in half
    cosine_similarities = []
    standard_deviations = []
    for i in range(num_iter):
        similarity, std = split_features(features, bias_metric)
        cosine_similarities.append(similarity)
        standard_deviations.append(std)
    std_similarity = np.std(cosine_similarities)
    mean_similarity = np.mean(cosine_similarities)

    t_test_calc = stats.ttest_1samp(a=random.sample(cosine_similarities, 30), popmean=mean_similarity)
    if t_test_calc[1] > 0.05:
        t_test = 'fail to reject'
    else:
        t_test = 'reject'

    return cosine_similarities, standard_deviations, std_similarity, mean_similarity, t_test


def self_similarities_error_bars(features, num_iter, bias_metric):
    # computes similarities between one set of features split in half
    cosine_similarities = []
    standard_deviations = []
    for i in range(num_iter):
        similarity, std = split_features(features, bias_metric)
        cosine_similarities.append(similarity)
        standard_deviations.append(std)
    std_similarity = np.std(cosine_similarities)
    mean_similarity = np.mean(cosine_similarities)
    return [min(cosine_similarities), max(cosine_similarities)]


def similarities_features_error_bars(features_one, features_two, num_iter):
    # computes similarities between two sets of features by randomly permuting them
    cosine_similarities = []
    standard_deviations = []
    for i in range(num_iter):
        similarity, std = random_split_two_features(features_one, features_two)
        
        cosine_similarities.append(similarity)
        standard_deviations.append(std)
    
    std_similarity = np.std(cosine_similarities)
    mean_similarity = np.mean(cosine_similarities)
    
    return [min(cosine_similarities), max(cosine_similarities)]


def load_features(folder, dataset_name, pca_comps, pretrained=False):
    if pca_comps == 'None':
        pca_path = 'no_pca/'
    else:
        pca_path = 'pca/'
    if pretrained == True:
        features = dict()
        features_pt = os.listdir(folder+'features/' + dataset_name + '/pretrained_features/'+pca_path)
        for file_name in features_pt:
            features[os.path.splitext(file_name)[0]] = np.load(folder + 'features/' + dataset_name +'/pretrained_features/'+pca_path + file_name, allow_pickle=True)
    else:
        features = dict()
        features_pt = os.listdir(folder+'features/' + dataset_name +'/pretrained_features/'+pca_path)
        features_ft = os.listdir(folder+'features/' + dataset_name +'/finetuned_features/'+pca_path)
        for file_name in features_pt:
            features[os.path.splitext(file_name)[0]] = np.load(folder + 'features/' + dataset_name +'/pretrained_features/'+ pca_path + file_name, allow_pickle=True)
        for file_name in features_ft:
            features[os.path.splitext(file_name)[0]] = np.load(folder + 'features/' + dataset_name +'/finetuned_features/'+ pca_path + file_name, allow_pickle=True)
    return features

def multiple_trials_exp(model_name, save_path, train_dataset, dataset_name, config_path, bias_metric, pca_comps):
    if pca_comps == 'None':
        pca_path = 'no_pca/'
    else:
        pca_path = 'pca/'

    base_path = 'experiments/'+ train_dataset+'/'+model_name
    contents = os.listdir(base_path) 
    all_self_sim_pt = []
    all_self_sim_ft = []
    all_comps_sim_pt = []
    all_comps_sim_ft = []
    for i in contents:
        if i != 'orig' and i != '.ipynb_checkpoints' and i != 'averaged':
            dir = os.listdir(base_path+'/'+i+'/features/' +dataset_name +'/finetuned_features'+'/'+pca_path)
            if len(dir) != 0:
                self_similarities_pt = np.load(base_path+'/'+i+'/metric_data/' + dataset_name + '/'+pca_path + bias_metric+'/self_similarities_pt.npy', allow_pickle=True)
                self_similarities_ft = np.load(base_path+'/'+i+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_ft.npy', allow_pickle=True)
                comps_similarities_pt = np.load(base_path+'/'+i+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_pt.npy', allow_pickle=True)
                comps_similarities_ft = np.load(base_path+'/'+i+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_ft.npy', allow_pickle=True)

                all_self_sim_pt.append(self_similarities_pt.item())
                all_self_sim_ft.append(self_similarities_ft.item())
                all_comps_sim_pt.append(comps_similarities_pt.item())
                all_comps_sim_ft.append(comps_similarities_ft.item())
    final_self_similarities_pt, mins_maxes_pt = get_multiple_trials_stats(all_self_sim_pt) # key: (list[cos], mean of means)
    final_self_similarities_ft, mins_maxes_ft = get_multiple_trials_stats(all_self_sim_ft)
    final_comps_similarities_pt, mins_maxes_comps_pt = get_multiple_trials_stats(all_comps_sim_pt)
    final_comps_similarities_ft, mins_maxes_comps_ft = get_multiple_trials_stats(all_comps_sim_ft)

    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_pt_averaged.npy', final_self_similarities_pt)
    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_pt_mins_maxes_averaged.npy', mins_maxes_pt)

    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_ft_averaged.npy', final_self_similarities_ft)
    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_ft_mins_maxes_averaged.npy', mins_maxes_ft)

    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_comps_pt_averaged.npy', final_comps_similarities_pt)
    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_comps_pt_mins_maxes_averaged.npy', mins_maxes_comps_pt)

    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_comps_ft_averaged.npy', final_comps_similarities_ft)
    np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_comps_ft_mins_maxes_averaged.npy', mins_maxes_comps_ft)


    # Plot results from multiple trials
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    individual_plots_cats = config['INDIVIDUAL_PLOTS']['category_list']
    # TODO in the following plots, fix the title and save paths 
    for cat in individual_plots_cats:
        plot_indiv_categories_mult_trials(model_name, dataset_name, save_path, config['INDIVIDUAL_PLOTS'][cat], final_comps_similarities_pt, final_self_similarities_pt, cat, bias_metric, pca_comps)
        plot_indiv_cats_comps_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, final_comps_similarities_pt, final_comps_similarities_ft, final_self_similarities_pt, final_self_similarities_ft, cat, config['INDIVIDUAL_PLOTS_COMPS'][cat], bias_metric, pca_comps)
    for misc in config['MISC_PLOTS']['misc_plots_names']:
        plot_misc_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, final_comps_similarities_pt, final_comps_similarities_ft, final_self_similarities_pt, final_self_similarities_ft, config['MISC_PLOTS'][misc], misc, bias_metric, pca_comps)


def run_experiment(model_name, save_path, train_dataset, dataset_name, features, config_path, bias_metric, pca_comps, only_pretrained=False, multiple_trials=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    labels = config['LABELS_NAMES']
    comps = config['COMPS']

    NUM_ITER = 50
    if pca_comps == 0.0:
        pca_comps = 'None'
        pca_path = 'no_pca/'
    else:
        pca_path = 'pca/'
    
    individual_plots_cats = config['INDIVIDUAL_PLOTS']['category_list']

    if multiple_trials == False:
        # self_similarities
        self_similarities_stats_pt = dict() # keys classes_names in config
        self_similarities_stats_ft = dict()

        self_similarities_stats_pt_significance = dict()
        self_similarities_stats_ft_significance = dict()

        comps_similarities_stats_pt = dict()
        comps_similarities_stats_ft = dict()

        comps_similarities_stats_pt_significance = dict()
        comps_similarities_stats_ft_significance = dict()

        mins_maxes_pt = dict()
        mins_maxes_comps_pt = dict()

        mins_maxes_ft = dict()
        mins_maxes_comps_ft = dict()  


        for feature in features:
            # generates pretrained and finetuned
            cos, std, sim_std, sim_mean, t_test = self_similarities(features[feature], NUM_ITER, bias_metric)
            if feature.endswith("_pt"):
                self_similarities_stats_pt[labels[feature[:-3]]] = (cos, std, sim_std, sim_mean)
                self_similarities_stats_pt_significance[labels[feature[:-3]]] = t_test
                mins_maxes_pt[labels[feature[:-3]]] = self_similarities_error_bars(features[feature], NUM_ITER, bias_metric) # keys are labels_names in config
            else:
                self_similarities_stats_ft[labels[feature[:-3]]] = (cos, std, sim_std, sim_mean)
                self_similarities_stats_ft_significance[labels[feature[:-3]]] = t_test
                mins_maxes_ft[labels[feature[:-3]]] = self_similarities_error_bars(features[feature], NUM_ITER, bias_metric) # keys are labels_names in config

        for comp in comps:
            # comps[comp] is a list of comparisons
            cos, std, sim_std, sim_mean, t_test = similarities_features(features[comps[comp][0]+"_pt"], features[comps[comp][1]+"_pt"], NUM_ITER)        
            comps_similarities_stats_pt[comp] = (cos, std, sim_std, sim_mean)
            comps_similarities_stats_pt_significance[comp] = t_test

            errorbars_pt = similarities_features_error_bars(features[comps[comp][0]+"_pt"], features[comps[comp][1]+"_pt"], NUM_ITER)
            mins_maxes_comps_pt[comp] = errorbars_pt
            if only_pretrained == False:
                cos, std, sim_std, sim_mean, t_test = similarities_features(features[comps[comp][0]+"_ft"], features[comps[comp][1]+"_ft"], NUM_ITER)
                comps_similarities_stats_ft_significance[comp] = t_test
                comps_similarities_stats_ft[comp] = (cos, std, sim_std, sim_mean)
                errorbars_ft = similarities_features_error_bars(features[comps[comp][0]+"_ft"], features[comps[comp][1]+"_ft"], NUM_ITER)
                mins_maxes_comps_ft[comp] = errorbars_ft

        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_pt.npy', self_similarities_stats_pt)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_ft.npy', self_similarities_stats_ft)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_pt.npy', comps_similarities_stats_pt)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_ft.npy', comps_similarities_stats_ft)
        
        np.save(save_path + '/boxplots/'+ dataset_name + '/' + pca_path + bias_metric + '/'+'self_similarities_pt_significance.npy', self_similarities_stats_pt_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'self_similarities_ft_significance.npy', self_similarities_stats_ft_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'comps_similarities_pt_significance.npy', comps_similarities_stats_pt_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'comps_similarities_ft_significance.npy', comps_similarities_stats_ft_significance)

    
    if only_pretrained == True:
        for cat in individual_plots_cats:
            plot_indiv_categories(model_name, dataset_name, save_path, config['INDIVIDUAL_PLOTS'][cat], comps_similarities_stats_pt, self_similarities_stats_pt, cat, bias_metric, pca_comps)
            #if trend_analysis==True:
                #trend_analysis_setup(only_pretrained, dataset_name, config, finetune, pca_comps, bias_metric)
    else:
        #if trend_analysis==True:
            #trend_analysis_setup(only_pretrained, dataset_name, config, finetune, pca_comps, bias_metric)
        if multiple_trials == True:
            if not os.path.isdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'):
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots', mode=0o777)

                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/' + 'openimages', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/'+'openimages'+'/'+'no_pca', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/'+'openimages'+'/'+'no_pca'+'/'+'cosine', mode=0o777)

                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/'+'coco', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/'+'coco'+'/'+'no_pca', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'metric_data/'+'coco'+'/'+'no_pca'+'/'+'cosine', mode=0o777)

                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/' + 'openimages', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/'+'openimages'+'/'+'no_pca', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/'+'openimages'+'/'+'no_pca'+'/'+'cosine', mode=0o777)

                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/'+'coco', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/'+'coco'+'/'+'no_pca', mode=0o777)
                os.mkdir('experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'+'/'+'boxplots/'+'coco'+'/'+'no_pca'+'/'+'cosine', mode=0o777)



            save_path_mult = 'experiments/'+train_dataset+'/' +model_name +'/'+ 'averaged'
            multiple_trials_exp(model_name, save_path_mult, train_dataset, dataset_name, config_path, bias_metric, pca_comps)
        else:
            for cat in individual_plots_cats:
                plot_indiv_categories(model_name, dataset_name, save_path, config['INDIVIDUAL_PLOTS'][cat], comps_similarities_stats_pt, self_similarities_stats_pt, cat, bias_metric, pca_comps)
                plot_indiv_cats_comps(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_similarities_stats_pt, comps_similarities_stats_ft, self_similarities_stats_pt, self_similarities_stats_ft, cat, config['INDIVIDUAL_PLOTS_COMPS'][cat], bias_metric, pca_comps)
            for misc in config['MISC_PLOTS']['misc_plots_names']:
                plot_misc(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_similarities_stats_pt, comps_similarities_stats_ft, self_similarities_stats_pt, self_similarities_stats_ft, config['MISC_PLOTS'][misc], misc, bias_metric, pca_comps)



