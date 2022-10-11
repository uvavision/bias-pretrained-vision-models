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
from plot_intra_class import *
from plot_intra_class_avg import *
from plot_misc import *
from plot_misc_avg import *

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


# def plot_indiv_categories(model_name, dataset_name, save_path, category_features, comps_stats, self_similarities_stats, category_name, bias_metric, pca_comps):
#     y = []
#     yerr_vals = []
#     labels = list(category_features)
#     #  (cos, std, sim_std, sim_mean)
#     for i in category_features:
#         if i in comps_stats:
#             y.append(comps_stats[i][3])
#             yerr_vals.append([min(comps_stats[i][0]), max(comps_stats[i][0])])
#         else:
#             y.append(self_similarities_stats[i][3])
#             yerr_vals.append([min(self_similarities_stats[i][0]), max(self_similarities_stats[i][0])])
#     y = np.asarray(y)
#     yerr_vals = np.asarray(yerr_vals).T
#     yerr = np.abs(yerr_vals-y)

#     fig = plt.figure()
#     colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'tomato', 'turquoise']
#     for i in range(y.shape[0]):
#         plt.errorbar(np.asarray([i]), np.asarray([y[i]]), yerr=yerr[:, i:i+1], fmt='o', color='black', ecolor='black')
#     plt.xticks(ticks = np.arange(y.shape[0]), labels = labels, rotation=45)
#     #plt.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.6), ncol=3)
#     fig.subplots_adjust(bottom=0.35)
#     plt.grid(b=True)
#     if str(pca_comps) != 'None':
#         plt.title("Model: "+model_name + ", Class: " + category_name + ", PCA: " + str(pca_comps))
#     else:
#         plt.title("Model: "+model_name + ", Class: " + category_name)

#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric+" score")
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     save = save_path + '/boxplots/'+ dataset_name + '/' + pca_path + bias_metric+'/'+category_name+'_filtered_'+bias_metric+'.pdf'
#     #plt.savefig(save_path + '/boxplots/' + category_name+'_filtered_'+ bias_metric+'.pdf')
#     plt.savefig(save, format='pdf')

# def plot_indiv_cats_comps(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_name, category_features, bias_metric, pca_comps):

#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     y_dict = dict()
#     y_dict_ft = dict()
#     for i in category_features:
#         if i in comps_stats:
#             y_dict[i] = comps_stats[i][3]
#             y_dict_ft[i] = comps_stats_ft[i][3]
#         else:
#             y_dict[i] = self_similarities_stats[i][3]
#             y_dict_ft[i] = self_similarities_stats_ft[i][3]

#     y_dict_comp = dict()

#     #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
#     y_sorted = y_dict
#     y = np.asarray(list(y_sorted.values()))

#     labels = list(y_sorted.keys())

#     for label in y_sorted:
#         y_dict_comp[label] = y_dict_ft[label]

#     y_ft = np.asarray(list(y_dict_comp.values()))
#     spearman_coeff = stats.spearmanr(y, y_ft)
#     spearman_save = {model_name:spearman_coeff}
#     np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/spearman_indiv.npy', spearman_save)



#     y_err = []
#     y_err_ft = []
#     for key in y_sorted:
#         if key in mins_maxes_pt:
#             y_err.append(mins_maxes_pt[key])
#         else:
#             y_err.append(mins_maxes_comps_pt[key])
        
#     for key in y_dict_comp:
#         if key in mins_maxes_ft:
#             y_err_ft.append(mins_maxes_ft[key])
#         else:
#             y_err_ft.append(mins_maxes_comps_ft[key])
         
#     yerr_vals = np.asarray(y_err).T
#     yerr_vals_ft = np.asarray(y_err_ft).T

#     yerr = np.abs(yerr_vals-y)
#     yerr_ft = np.abs(yerr_vals_ft-y_ft)

#     fig = plt.figure()

#     colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver']

#     colors_err = ['turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue',
#             'gold', 'peru', 'palegreen']
#     colors_err_ft = ['peachpuff', 'dodgerblue', 'tomato', 'g', 'b', 'c', 'm', 'lime', 'orange']


#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y[i]]), yerr=yerr[:, i:i+1], fmt='o', color='b', ecolor='b')

#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i]]), yerr=yerr_ft[:, i:i+1], fmt='^', color='g', ecolor='g')
#     plt.xticks(ticks = np.arange(len(y_dict)), labels = labels, rotation=45)

#     #l1 = plt.legend(labels, bbox_to_anchor=(0.5, -0.6), loc='lower center', prop=fontP, ncol=3)
#     #plt.gca().add_artist(l1)
#     triangle = matplotlib.lines.Line2D([], [], color='g', marker='^', linestyle='None',
#                           markersize=10, label='Finetuned')
#     circle = matplotlib.lines.Line2D([], [], color='b', marker='o', linestyle='None',
#                           markersize=10, label='Pretrained')            
#     plt.legend(handles=[circle, triangle], loc='lower center', bbox_to_anchor=(0.5, -0.8), prop=fontP, ncol=2)
#     #plt.gca().add_artist(l1)

#     plt.grid(b=True)
#     fig.subplots_adjust(bottom=0.5)
#     if str(pca_comps) != 'None':
#         plt.title("Model: "+ model_name + " Finetuned on: " + dataset_name+", Class: "+ category_name + "\n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps))
#     else:
#         plt.title("Model: "+model_name + " Finetuend on: " + dataset_name+", Class: "+ category_name + "\n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3)))


#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric + " score")


#     save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path + bias_metric+'/'+category_name+ '_comp_' + bias_metric+'.pdf'
#     #plt.savefig(save_path + '/boxplots/' + category_name+'_filtered_'+ bias_metric+'.pdf')
#     plt.savefig(save, format='pdf')

#     #plt.savefig(save_path + '/boxplots/' +category_name+ '_comp_' + bias_metric+'.pdf')

#     fig2 = plt.figure()
#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     fig2.subplots_adjust(bottom=0.35)
#     if str(pca_comps) != 'None':
#         plt.title("Model: " + model_name + " Finetuned on: " + dataset_name+ ", Class: "+ category_name +"\n (Finetuned - Pretrained)"+ ", PCA: " + str(pca_comps))
#     else:
#         plt.title("Model: " + model_name + " Finetuned on: " +dataset_name +", Class: "+ category_name +"\n (Finetuned - Pretrained)")

#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric + " score")

#     save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path + bias_metric+'/'+category_name+'_comp_diff_' + bias_metric+'.pdf'
#     #plt.savefig(save_path + '/boxplots/' + category_name+'_filtered_'+ bias_metric+'.pdf')
#     plt.savefig(save, format='pdf')

#     #plt.savefig(save_path + '/boxplots/' + category_name+'_comp_diff_' + bias_metric+'.pdf')

# def plot_misc(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_features, plot_type, bias_metric, pca_comps):
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'
#     y_dict = dict()
#     y_dict_ft = dict()
#     for i in category_features:
#         if i in comps_stats:
#             y_dict[i] = comps_stats[i][3]
#             y_dict_ft[i] = comps_stats_ft[i][3]
#         else:
#             y_dict[i] = self_similarities_stats[i][3]
#             y_dict_ft[i] = self_similarities_stats_ft[i][3]

#     y_dict_comp = dict()

#     #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
#     y_sorted = y_dict
#     y = np.asarray(list(y_sorted.values()))
#     labels = list(y_sorted.keys())

#     for label in y_sorted:
#         y_dict_comp[label] = y_dict_ft[label]
#     y_ft = np.asarray(list(y_dict_comp.values()))
#     spearman_coeff = stats.spearmanr(y, y_ft)
#     spearman_save = {model_name:spearman_coeff}
#     np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/spearman_'+plot_type+'.npy', spearman_save)


#     y_err = []
#     y_err_ft = []
#     for key in y_sorted:
#         if key in mins_maxes_pt:
#             y_err.append(mins_maxes_pt[key])
#         else:
#             y_err.append(mins_maxes_comps_pt[key])
        
#     for key in y_dict_comp:
#         if key in mins_maxes_ft:
#             y_err_ft.append(mins_maxes_ft[key])
#         else:
#             y_err_ft.append(mins_maxes_comps_ft[key])  
#     yerr_vals = np.asarray(y_err).T
#     yerr_vals_ft = np.asarray(y_err_ft).T

#     yerr = np.abs(yerr_vals-y)
#     yerr_ft = np.abs(yerr_vals_ft-y_ft)

#     fig = plt.figure()
#     colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']

#     colors_err = ['turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue',
#             'gold', 'peru', 'palegreen']
#     colors_err_ft = ['peachpuff', 'dodgerblue', 'tomato', 'g', 'b', 'c', 'm', 'lime', 'orange']


#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y[i]]), yerr=yerr[:, i:i+1], fmt='o', color='b', ecolor='b')

#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i]]), yerr=yerr_ft[:, i:i+1], fmt="^", color='g', ecolor='g')
#     plt.xticks(np.arange(len(y_dict)), labels = labels, rotation=45)


#     if plot_type == 'pairs':
#         if str(pca_comps) != 'None':
#             title = "Model: "+ model_name + " Finetuned on: " +dataset_name+", Paired Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name+ ", Paired Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name + ", Paired Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name + ", Paired Classes (Finetuned - Pretrained)"
#         save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path+bias_metric+'/pairs_'+bias_metric+ '.pdf'
#         save_diff = save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric+'/pairs_diff_' + bias_metric+ '.pdf'
#         bbox = (0.5, -0.6)
#         bottom = 0.5
#     elif plot_type == 'comps':
#         if str(pca_comps) != 'None':
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name + ", Comparison Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Model: "+model_name + " Finetuned, Comparison Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Comparison Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name +", Comparison Classes (Finetuned - Pretrained)"
#         save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path +bias_metric+ '/comps_' + bias_metric+ '.pdf'
#         save_diff = save_path + '/boxplots/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_diff_' + bias_metric+'.pdf'
#         bbox = (0.5, -0.7)
#         bottom = 0.5
#     elif plot_type == 'indiv':
#         if str(pca_comps) != 'None':
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Individual Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Individual Classes, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained)"
#         save = save_path + '/boxplots/'+ dataset_name + '/' +pca_path+bias_metric+'/indv_' + bias_metric +'.pdf'
#         save_diff = save_path + '/boxplots/'+ dataset_name + '/' + pca_path + bias_metric+ '/indv_diff_' + bias_metric+'.pdf'  
#         bbox = (0.5, -0.6)
#         bottom = 0.5
#     else:
#         if str(pca_comps) != 'None':
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Class vs. Gender, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Model: "+model_name +" Finetuned on: " + dataset_name + ", Class vs. Gender Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Model: "+ model_name + " Finetuned on: " + dataset_name +", Class vs. Gender, \n Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Model: "+model_name + " Finetuned on: " + dataset_name +", Class vs. Gender (Finetuned - Pretrained)"
#         save = save_path + '/boxplots/'+ dataset_name + '/' + pca_path +bias_metric + '/object_comp_'+ bias_metric+'.pdf'
#         save_diff = save_path + '/boxplots/' + dataset_name + '/'+pca_path + bias_metric + '/object_comp_diffs_' + bias_metric +'.pdf'
#         bbox = (0.5, -0.8)
#         bottom = 0.5

#     # plt.legend(labels, bbox_to_anchor=(1.1, 1.1), loc='upper left', prop=fontP)
#     #l1 = plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
    
#     triangle = matplotlib.lines.Line2D([], [], color='g', marker='^', linestyle='None',
#                           markersize=10, label='Finetuned')
#     circle = matplotlib.lines.Line2D([], [], color='b', marker='o', linestyle='None',
#                           markersize=10, label='Pretrained')            
#     plt.legend(handles=[circle, triangle], loc='lower center', bbox_to_anchor=(0.5, -0.8), prop=fontP, ncol=2)
#     #plt.gca().add_artist(l1)

#     plt.grid(b=True)
#     fig.subplots_adjust(bottom=bottom)


#     plt.title(title)
#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric+" score")
#     plt.savefig(save, format='pdf')

#     fig2 = plt.figure()
#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
#     plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     fig2.subplots_adjust(bottom=bottom)
#     #plt.tight_layout()

#     plt.title(title_diff)
#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric+" score")
#     plt.savefig(save_diff, format='pdf')


# def calc_trend_stats(config, dataset_name, model_name, finetune, pretrained, bias_metric, pca_comps, trial):
#     if finetune:
#         save_path = 'experiments/'+ dataset_name + '/' + model_name+'/'+trial
#     else:
#         save_path = 'experiments/'+ dataset_name + '/' + model_name +'/' + trial + '/model_scratch'

#     labels = config['LABELS_NAMES']
#     comps = config['COMPS']

#     NUM_ITER = 50
#     # self_similarities
#     self_similarities_stats_pt = dict() # keys classes_names in config
#     self_similarities_stats_ft = dict()

#     comps_similarities_stats_pt = dict()
#     comps_similarities_stats_ft = dict()

#     mins_maxes_pt = dict()
#     mins_maxes_comps_pt = dict()

#     mins_maxes_ft = dict()
#     mins_maxes_comps_ft = dict()
#     features = load_features(save_path, dataset_name, pca_comps)
#     for feature in features:
        
#         # generates pretrained and finetuned
#         cos, std, sim_std, sim_mean, _ = self_similarities(features[feature], NUM_ITER, bias_metric)
#         if feature.endswith("_pt"):
#             self_similarities_stats_pt[labels[feature[:-3]]] = (cos, std, sim_std, sim_mean)
#             mins_maxes_pt[labels[feature[:-3]]] = self_similarities_error_bars(features[feature], NUM_ITER, bias_metric) # keys are labels_names in config
#         else:
#             self_similarities_stats_ft[labels[feature[:-3]]] = (cos, std, sim_std, sim_mean)
#             mins_maxes_ft[labels[feature[:-3]]] = self_similarities_error_bars(features[feature], NUM_ITER, bias_metric) # keys are labels_names in config

#     for comp in comps:
#         # comps[comp] is a list of comparisons
#         #print("COMP: ", comp)
#         #print("FEATURES: ", list(features.keys()))
#         cos, std, sim_std, sim_mean, _ = similarities_features(features[comps[comp][0]+"_pt"], features[comps[comp][1]+"_pt"], NUM_ITER)        
#         comps_similarities_stats_pt[comp] = (cos, std, sim_std, sim_mean)
#         errorbars_pt = similarities_features_error_bars(features[comps[comp][0]+"_pt"], features[comps[comp][1]+"_pt"], NUM_ITER)
#         mins_maxes_comps_pt[comp] = errorbars_pt
#         if pretrained == False:
#             cos, std, sim_std, sim_mean, _ = similarities_features(features[comps[comp][0]+"_ft"], features[comps[comp][1]+"_ft"], NUM_ITER)
#             comps_similarities_stats_ft[comp] = (cos, std, sim_std, sim_mean)
#             errorbars_ft = similarities_features_error_bars(features[comps[comp][0]+"_ft"], features[comps[comp][1]+"_ft"], NUM_ITER)
#             mins_maxes_comps_ft[comp] = errorbars_ft
#     return (self_similarities_stats_pt, self_similarities_stats_ft, comps_similarities_stats_pt, comps_similarities_stats_ft)


# def trends_plot_info(plot_type, model_one, model_two, mode, spearman_coeff, bias_metric, dataset, pca_comps):
#     # plt.legend(labels, loc='lower left', ncol=12)
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     if plot_type == 'pairs':
#         title = mode +": " +dataset+ " - Paired Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+ '/'+pca_path+str(bias_metric)+ '/pairs_'+model_one+'_'+model_two+'_'+mode+'_'+bias_metric+'.pdf'
#     elif plot_type == 'comps':
#         title = mode +": " +dataset+ " - Comparisons Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3)) + " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+ '/'+pca_path+str(bias_metric)+ '/comps_'+model_one+'_'+model_two+'_'+mode+'.pdf'
#     elif plot_type == 'indiv':
#         title = mode +": " +dataset+" - Individual Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+ '/'+pca_path+str(bias_metric)+ '/indiv_'+model_one+'_'+model_two+'_'+mode+'_'+bias_metric+'.pdf'
#     else:
#         title = mode +": " +dataset+" - Class vs. Gender, Inter-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+ '/'+pca_path+str(bias_metric)+ '/object_comp_'+model_one+'_'+model_two+'_'+mode+'_'+bias_metric+'.pdf'
#     return title, save

# def trends_plot_info_single_plot(plot_type, model_name, mode, spearman_coeff, bias_metric, dataset, pca_comps):
#     # plt.legend(labels, loc='lower left', ncol=12)
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     if plot_type == 'pairs':
#         title = mode +": " +dataset+ " - Paired Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+'/'+pca_path+str(bias_metric)+  '/pairs_'+model_name+'_'+mode+'_'+bias_metric+'.pdf'
#     elif plot_type == 'comps':
#         title = mode +": " +dataset+ " - Comparisons Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3)) + " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+'/'+pca_path+str(bias_metric)+  '/comps_'+model_name+'_'+mode+'_'+bias_metric+'.pdf'
#     elif plot_type == 'indiv':
#         title = mode +": " +dataset+" - Individual Classes, Intra-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+'/'+pca_path+str(bias_metric)+  '/indiv_'+model_name+'_'+mode+'_'+bias_metric+'.pdf'
#     else:
#         title = mode +": " +dataset+" - Class vs. Gender, Inter-Feature Clustering \n Metric: "+bias_metric+", Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ " PCA: " + str(pca_comps)
#         save = 'experiments/trends/'+dataset+'/'+pca_path+str(bias_metric)+  '/object_comp_'+model_name+'_'+mode+'_'+bias_metric+'.pdf'
#     return title, save

# def calc_dataset_sensitivity(model_one, model_two, dataset, bias_metric, pca_comps, y_pt_one, y_ft_one, y_pt_two, y_ft_two, one_cats, two_cats):
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     # Proportion of differences
#     y_diff_one = [a_i - b_i for a_i, b_i in zip(y_ft_one, y_pt_one)]
#     y_diff_two = [a_i - b_i for a_i, b_i in zip(y_ft_two, y_pt_two)]
#     model_one_diff = np.array(y_diff_one)
#     model_two_diff = np.array(y_diff_two)

#     dataset_sensitivity_avg = np.mean(model_one_diff / model_two_diff)
#     one_min, one_max = one_cats[np.argmin(model_one_diff)], one_cats[np.argmax(model_one_diff)]
#     two_min, two_max = two_cats[np.argmin(model_two_diff)], two_cats[np.argmax(model_two_diff)]

#     # Difference in proportions
#     model_one_prop = np.abs(np.array(y_ft_one) / np.array(y_pt_one))
#     model_two_prop = np.abs(np.array(y_ft_two) / np.array(y_pt_two))
#     #model_one_prop = np.abs(y_ft_one / y_pt_one)
#     #model_two_prop = np.abs(y_ft_two / y_pt_two)
#     avg_prop = np.mean(np.array([a_i - b_i for a_i, b_i in zip(model_one_prop, model_two_prop)]))



#     all_results = {'avg_diff': dataset_sensitivity_avg,
#                     model_one: (one_min, one_max),
#                     model_two: (two_min, two_max),
#                     'avg_prop': avg_prop}
#     np.save('experiments/dataset_sensitivity/'+dataset+'/'+pca_path + str(bias_metric)+'/'+model_one+'_'+model_two+'_'+str(bias_metric)+'.npy', all_results)

# def plot_trends(model_one, model_two, dataset_name, model_one_stats, model_two_stats, category_features, plot_type, bias_metric, pca_comps):
#     comps_stats_one_pt = model_one_stats[2]
#     comps_stats_one_ft = model_one_stats[3]
#     self_stats_one_pt = model_one_stats[0]
#     self_stats_one_ft = model_one_stats[1]

#     comps_stats_two_pt = model_two_stats[2]
#     comps_stats_two_ft = model_two_stats[3]
#     self_stats_two_pt = model_two_stats[0]
#     self_stats_two_ft = model_two_stats[1]

#     y_dict_pt_one = dict()
#     y_dict_ft_one = dict()

#     y_dict_pt_two = dict()
#     y_dict_ft_two = dict()

#     for i in category_features:
#         if i in comps_stats_one_pt:
#             y_dict_pt_one[i] = comps_stats_one_pt[i][3]
#             y_dict_ft_one[i] = comps_stats_one_ft[i][3]
#         else:
#             y_dict_pt_one[i] = self_stats_one_pt[i][3]
#             y_dict_ft_one[i] = self_stats_one_ft[i][3]
#     for i in category_features:
#         if i in comps_stats_two_pt:
#             y_dict_pt_two[i] = comps_stats_two_pt[i][3]
#             y_dict_ft_two[i] = comps_stats_two_ft[i][3]
#         else:
#             y_dict_pt_two[i] = self_stats_two_pt[i][3]
#             y_dict_ft_two[i] = self_stats_two_ft[i][3]

    
#     y_pt_one = list(y_dict_pt_one.values())
#     y_pt_two = list(y_dict_pt_two.values())
    
#     y_ft_one = list(y_dict_ft_one.values())
#     y_ft_two = list(y_dict_ft_two.values())

#     y_diff_one = [a_i - b_i for a_i, b_i in zip(y_ft_one, y_pt_one)]
#     y_diff_two = [a_i - b_i for a_i, b_i in zip(y_ft_two, y_pt_two)]
#     calc_dataset_sensitivity(model_one, model_two, dataset_name, bias_metric, pca_comps, y_pt_one, y_ft_one, y_pt_two, y_ft_two, list(y_dict_ft_one.keys()), list(y_dict_ft_two.keys()))


#     fig = plt.figure()
#     colors = cm.rainbow(np.linspace(0, 1, len(y_pt_one)))
#     index = 0
#     spearman_coeff = stats.spearmanr(y_pt_one, y_pt_two)
#     labels = list(y_dict_pt_one.keys())
#     for y, c in zip(y_pt_two, colors):
#         plt.scatter(y_pt_one[index], y, color=c, label=labels[index])
#         index+=1
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     plt.xlabel(model_one)
#     plt.ylabel(model_two)
#     fig.subplots_adjust(bottom=0.35)
#     title, save_path = trends_plot_info(plot_type, model_one, model_two, "Pretrained", spearman_coeff, bias_metric, dataset_name, pca_comps)
#     plt.title(title)
#     plt.savefig(save_path, format='pdf')

#     fig2 = plt.figure()
#     colors = cm.rainbow(np.linspace(0, 1, len(y_ft_one)))
#     index = 0
#     spearman_coeff = stats.spearmanr(y_ft_one, y_ft_two)
#     labels = list(y_dict_ft_one.keys())
#     for y, c in zip(y_ft_two, colors):
#         plt.scatter(y_ft_one[index], y, color=c, label=labels[index])
#         index+=1
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     plt.xlabel(model_one)
#     plt.ylabel(model_two)
#     fig2.subplots_adjust(bottom=0.35)
#     title, save_path = trends_plot_info(plot_type, model_one, model_two, "Finetuned", spearman_coeff, bias_metric, dataset_name, pca_comps)
#     plt.title(title)
#     plt.savefig(save_path, format='pdf')

#     fig3 = plt.figure()
#     colors = cm.rainbow(np.linspace(0, 1, len(y_diff_one)))
#     index = 0
#     spearman_coeff = stats.spearmanr(y_diff_one, y_diff_two)
#     labels = list(y_dict_ft_one.keys())
#     for y, c in zip(y_diff_two, colors):
#         plt.scatter(y_diff_one[index], y, color=c, label=labels[index])
#         index+=1
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     plt.xlabel(model_one)
#     plt.ylabel(model_two)
#     fig3.subplots_adjust(bottom=0.35)
#     title, save_path = trends_plot_info(plot_type, model_one, model_two, "Difference", spearman_coeff, bias_metric, dataset_name, pca_comps)
#     plt.title(title)
#     plt.savefig(save_path, format='pdf')

#     fig4 = plt.figure()
#     colors = cm.rainbow(np.linspace(0, 1, len(y_pt_one)))
#     index = 0
#     spearman_coeff = stats.spearmanr(y_pt_one, y_ft_one)
#     labels = list(y_dict_ft_one.keys())
#     for y, c in zip(y_ft_one, colors):
#         plt.scatter(y_pt_one[index], y, color=c, label=labels[index])
#         index+=1
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     plt.xlabel(model_one+" Pretrained")
#     plt.ylabel(model_one+ " Finetuned")
#     fig4.subplots_adjust(bottom=0.35)
#     title, save_path = trends_plot_info_single_plot(plot_type, model_one, str(model_one), spearman_coeff, bias_metric, dataset_name, pca_comps)
#     plt.title(title)
#     plt.savefig(save_path, format='pdf')

#     fig5 = plt.figure()
#     colors = cm.rainbow(np.linspace(0, 1, len(y_pt_two)))
#     index = 0
#     spearman_coeff = stats.spearmanr(y_pt_two, y_ft_two)
#     labels = list(y_dict_ft_two.keys())
#     for y, c in zip(y_ft_two, colors):
#         plt.scatter(y_pt_two[index], y, color=c, label=labels[index])
#         index+=1
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     plt.xlabel(model_two+" Pretrained")
#     plt.ylabel(model_two+ " Finetuned")
#     fig5.subplots_adjust(bottom=0.35)
#     title, save_path = trends_plot_info_single_plot(plot_type, model_two, str(model_two), spearman_coeff, bias_metric, dataset_name, pca_comps)
#     plt.title(title)
#     plt.savefig(save_path, format='pdf')

# def trend_analysis_setup(pretrained, dataset_name, config, finetune, pca_comps, bias_metric):
#     model_list = os.listdir('experiments/'+dataset_name)
#     if '.ipynb_checkpoints' in model_list:
#         model_list.remove('.ipynb_checkpoints')
#     model_stats = dict()
#     for model in model_list:
#         #print("MODEL: ", model)
#         temp = os.listdir('experiments/'+dataset_name+'/'+model)
#         trial = 'None'
#         for trial_name in temp:
#             try:
#                 time.strptime(trial_name, "%Y-%m-%d %H:%M:%S")
#                 trial = trial_name
#             except:
#                 continue
        
#         model_stats[model] = calc_trend_stats(config, dataset_name, model, finetune, pretrained, bias_metric, pca_comps, trial)
#     model_combinations = list(itertools.combinations(model_list, 2))
#     print(model_combinations)

#     individual_plots_cats = config['INDIVIDUAL_PLOTS']['category_list']
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca'
#     save_path = 'experiments/trends/'+pca_path+str(bias_metric)+'/'

#     for comp in model_combinations:
#         stats_one = model_stats[comp[0]]
#         stats_two = model_stats[comp[1]]
#         for misc in config['MISC_PLOTS']['misc_plots_names']:
#             plot_trends(comp[0], comp[1], dataset_name, stats_one, stats_two, config['MISC_PLOTS'][misc], misc, bias_metric, pca_comps)





# def plot_indiv_categories_mult_trials(model_name, dataset_name, save_path, category_features, comps_stats, self_similarities_stats, category_name, bias_metric, pca_comps):
#     y = []
#     yerr_vals = []
#     labels = list(category_features)
#     #  (cos, std, sim_std, sim_mean)
#     for i in category_features:
#         if i in comps_stats:
#             y.append(comps_stats[i][1])
#             yerr_vals.append([min(comps_stats[i][0]), max(comps_stats[i][0])])
#         else:
#             y.append(self_similarities_stats[i][1])
#             yerr_vals.append([min(self_similarities_stats[i][0]), max(self_similarities_stats[i][0])])
#     y = np.asarray(y)
#     yerr_vals = np.asarray(yerr_vals).T
#     yerr = np.abs(yerr_vals-y)

#     temp_errs = zip(*yerr)
    
#     fig = go.Figure(data=go.Scatter(
#             x=labels,
#             y=y,
#             mode='markers',
#             name = 'Pretrained',
#             marker=dict(
#                 color='blue',
#                 size=10,
#             ),
#             marker_symbol = 'circle',
#             error_y=dict(
#                 type='data',
#                 symmetric=False,
#                 array=yerr[1],
#                 arrayminus=yerr[0])
#             ))
#     if str(pca_comps) != 'None':
#         title = "Averaged, Model: "+model_name + ", Class: " + category_name + ", PCA: " + str(pca_comps)
#     else:
#         title = "Averaged, Model: "+model_name + ", Class: " + category_name

#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric+" score")

#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'

#     fig.update_layout(
#         title={
#             'text': title,
#             'y':0.9,
#             'x':0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'},
#         height=600,
#         width=700,
#         plot_bgcolor="#FFF",  # Sets background color to white
#         xaxis=dict(
#             title="Classes",
#             linecolor="#BCCCDC",  # Sets color of X-axis line
#             showgrid=True  # Removes X-axis grid lines
#         ),
#         yaxis=dict(
#             title=bias_metric + " score",  
#             linecolor="#BCCCDC",  # Sets color of Y-axis line
#             showgrid=True,  # Removes Y-axis grid lines    
#         ),
#         legend=dict(
#             yanchor="bottom",
#             y=0.01,
#             xanchor="right",
#             x=0.99
#         ))
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')


#     save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path + bias_metric+'/'+category_name+'_filtered_'+bias_metric+'_averaged.pdf'
#     #plt.savefig(save_path + '/boxplots/' + category_name+'_filtered_'+ bias_metric+'.pdf')
#     fig.write_image(save)


# def plot_indiv_cats_comps_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_name, category_features, bias_metric, pca_comps):
#     y_dict = dict()
#     y_dict_ft = dict()
#     for i in category_features:
#         if i in comps_stats:
#             y_dict[i] = comps_stats[i][1]
#             y_dict_ft[i] = comps_stats_ft[i][1]
#         else:
#             y_dict[i] = self_similarities_stats[i][1]
#             y_dict_ft[i] = self_similarities_stats_ft[i][1]

#     y_dict_comp = dict()

#     #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
#     y_sorted = y_dict
#     y = np.asarray(list(y_sorted.values()))

#     labels = list(y_sorted.keys())

#     for label in y_sorted:
#         y_dict_comp[label] = y_dict_ft[label]

#     y_ft = np.asarray(list(y_dict_comp.values()))
#     spearman_coeff = stats.spearmanr(y, y_ft)

#     y_err = [] 
#     y_err_ft = []

#     for key in y_sorted:
#         if key in mins_maxes_pt:
#             y_err.append(mins_maxes_pt[key]) # appending a list of [min(vals), max(vals)] #centroid mean(vals)
#         else:
#             y_err.append(mins_maxes_comps_pt[key])
        
#     for key in y_dict_comp:
#         if key in mins_maxes_ft:
#             y_err_ft.append(mins_maxes_ft[key])
#         else:
#             y_err_ft.append(mins_maxes_comps_ft[key])
         
#     yerr_vals = np.asarray(y_err).T
#     yerr_vals_ft = np.asarray(y_err_ft).T

#     yerr = np.abs(yerr_vals-y)
#     yerr_ft = np.abs(yerr_vals_ft-y_ft)

#     temp_errs = zip(*yerr)
#     temp_errs_ft = zip(*yerr_ft)


#     fig = go.Figure(data=go.Scatter(
#             x=labels,
#             y=y,
#             mode='markers',
#             name = 'Pretrained',
#             marker=dict(
#                 color='blue',
#                 size=10,
#             ),
#             marker_symbol = 'circle',
#             error_y=dict(
#                 type='data',
#                 symmetric=False,
#                 array=yerr[1],
#                 arrayminus=yerr[0])
#             ))

#     trace = go.Scatter(
#             x=labels,
#             y=y_ft,
#             mode='markers',
#             name = 'Finetuned',
#             marker=dict(
#                 color='green',
#                 size=10,
#             ),
#             marker_symbol = 'triangle-up',
#             error_y=dict(
#                 type='data',
#                 symmetric=False,
#                 array=yerr_ft[1],
#                 arrayminus=yerr_ft[0])
#             )

#     fig.add_trace(trace)

#     if str(pca_comps) != 'None':
#         title = "Averaged, Model: "+ model_name + " Finetuned on: " + dataset_name+", Class: "+ category_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#     else:
#         title = "Averaged, Model: "+model_name + " Finetuend on: " + dataset_name+", Class: "+ category_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))

#     fig.update_layout(
#         title={
#             'text': title,
#             'y':0.9,
#             'x':0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'},
#         height=600,
#         width=700,
#         plot_bgcolor="#FFF",  # Sets background color to white
#         xaxis=dict(
#             title="Classes",
#             linecolor="#BCCCDC",  # Sets color of X-axis line
#             showgrid=True  # Removes X-axis grid lines
#         ),
#         yaxis=dict(
#             title=bias_metric + " score",  
#             linecolor="#BCCCDC",  # Sets color of Y-axis line
#             showgrid=True,  # Removes Y-axis grid lines    
#         ),
#         legend=dict(
#             yanchor="bottom",
#             y=0.01,
#             xanchor="right",
#             x=0.99
#         ))
#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'
#     colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

#     save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path + bias_metric+'/'+category_name+ '_comp_' + bias_metric+'_averaged.pdf'
#     fig.write_image(save)
#     fig2 = plt.figure()
#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
#     plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True)
#     fig2.subplots_adjust(bottom=0.35)
#     if str(pca_comps) != 'None':
#         plt.title("Averaged, Model: " + model_name + " Finetuned on: " + dataset_name+ ", Class: "+ category_name +"\n (Finetuned - Pretrained)"+ ", PCA: " + str(pca_comps))
#     else:
#         plt.title("Averaged, Model: " + model_name + " Finetuned on: " +dataset_name +", Class: "+ category_name +"\n (Finetuned - Pretrained)")


#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric + " score")

#     save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path + bias_metric+'/'+category_name+'_comp_diff_' + bias_metric+'_averaged.pdf'
#     plt.savefig(save, format='pdf')


# def plot_misc_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_features, plot_type, bias_metric, pca_comps):
#     y_dict = dict()
#     y_dict_ft = dict()
#     for i in category_features:
#         if i in comps_stats:
#             y_dict[i] = comps_stats[i][1]
#             y_dict_ft[i] = comps_stats_ft[i][1]
#         else:
#             y_dict[i] = self_similarities_stats[i][1]
#             y_dict_ft[i] = self_similarities_stats_ft[i][1]

#     y_dict_comp = dict()

#     #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
#     y_sorted = y_dict
#     y = np.asarray(list(y_sorted.values()))
#     labels = list(y_sorted.keys())

#     for label in y_sorted:
#         y_dict_comp[label] = y_dict_ft[label]
#     y_ft = np.asarray(list(y_dict_comp.values()))
#     spearman_coeff = stats.spearmanr(y, y_ft)

#     y_err = []
#     y_err_ft = []
#     for key in y_sorted:
#         if key in mins_maxes_pt:
#             y_err.append(mins_maxes_pt[key])
#         else:
#             y_err.append(mins_maxes_comps_pt[key])
        
#     for key in y_dict_comp:
#         if key in mins_maxes_ft:
#             y_err_ft.append(mins_maxes_ft[key])
#         else:
#             y_err_ft.append(mins_maxes_comps_ft[key])  
#     yerr_vals = np.asarray(y_err).T
#     yerr_vals_ft = np.asarray(y_err_ft).T

#     temp_errs_mins,  temp_errs_maxes = zip(*y_err)
#     temp_errs_ft_mins, temp_errs_ft_maxes = zip(*y_err_ft)

#     yerr = np.abs(yerr_vals-y)
#     yerr_ft = np.abs(yerr_vals_ft-y_ft)



#     fig = go.Figure(data=go.Scatter(
#         x=labels,
#         y=y,
#         mode='markers',
#         name = 'Pretrained',
#         marker=dict(
#             color='blue',
#             size=10,
#         ),
#         marker_symbol = 'circle',
#         error_y=dict(
#             type='data',
#             symmetric=False,
#             array=yerr[1],
#             arrayminus=yerr[0])
#         ))

#     trace = go.Scatter(
#             x=labels,
#             y=y_ft,
#             mode='markers',
#             name = 'Finetuned',
#             marker=dict(
#                 color='green',
#                 size=10,
#             ),
#             marker_symbol = 'triangle-up',
#             error_y=dict(
#                 type='data',
#                 symmetric=False,
#                 array=yerr_ft[1],
#                 arrayminus=yerr_ft[0])
#             )
#     fig.add_trace(trace)

#     if pca_comps == 'None':
#         pca_path = 'no_pca/'
#     else:
#         pca_path = 'pca/'


#     # plt.legend(labels, loc='lower left', ncol=12)
#     if plot_type == 'pairs':
#         if str(pca_comps) != 'None':
#             title = "Averaged Paired Classes, Model: "+ model_name + " Finetuned on: " +dataset_name+" <br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name+ ", Paired Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Averaged Paired Classes, Model: "+ model_name + " Finetuned on: " + dataset_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name + ", Paired Classes (Finetuned - Pretrained)"
#         #save = save_path + '/boxplots/pairs_'+bias_metric+ '.pdf'
#         save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path+bias_metric+'/pairs_'+bias_metric+ '_averaged.pdf'
#         #save_diff = save_path + '/boxplots/pairs_diff_' + bias_metric+ '.pdf'
#         save_diff = save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric+'/pairs_diff_' + bias_metric+ '_averaged.pdf'
#         bbox = (0.5, -0.65)
#         bottom_legend = -0.75
#         bottom = 0.5
#         #plot_save_info = pd.DataFrame([labels, y, temp_errs_mins, temp_errs_maxes, y_ft, temp_errs_ft_mins, temp_errs_ft_maxes], columns=['classes', 'pt_means', 'pt_mins', 'pt_maxes', 'ft_means', 'ft_mins', 'ft_maxes']).to_csv(save_path+'/metric_data/'+pca_path+bias_metric+'/pairs_plot.csv', index=False)
#         plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/pairs_plot.csv', index=False)
#     elif plot_type == 'comps':
#         if str(pca_comps) != 'None':
#             title = "Averaged Comparison Classes, Model: "+ model_name + " Finetuned on: " + dataset_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Averaged, Model: "+model_name + " Finetuned, Comparison Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Averaged Comparison Classes, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name +", Comparison Classes (Finetuned - Pretrained)"
#         #save = save_path + '/boxplots/comps_' + bias_metric+ '.pdf'
#         save = save_path + '/boxplots/'+ dataset_name + '/'+pca_path +bias_metric+ '/comps_' + bias_metric+ '_averaged.pdf'
#         #save_diff = save_path + '/boxplots/comps_diff_' + bias_metric+'.pdf'
#         save_diff = save_path + '/boxplots/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_diff_' + bias_metric+'_averaged.pdf'
#         bbox = (0.5, -0.75)
#         bottom_legend = -1.0
#         bottom = 0.5
#         plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_plot.csv', index=False)

#     elif plot_type == 'indiv':
#         if str(pca_comps) != 'None':
#             title = "Averaged Individual Classes, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Averaged, Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Averaged Individual Classes, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Averaged, Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained)"
#         #save = save_path + '/boxplots/indv_' + bias_metric +'.pdf'
#         #save_diff = save_path + '/boxplots/indv_diff_' + bias_metric+'.pdf'    
#         save = save_path + '/boxplots/' + dataset_name + '/'+pca_path+bias_metric+'/indv_' + bias_metric +'_averaged.pdf'
#         save_diff = save_path + '/boxplots/'+ dataset_name + '/' + pca_path + bias_metric+ '/indv_diff_' + bias_metric+'_averaged.pdf'  
#         bbox = (0.5, -0.65)
#         bottom_legend = -0.6
#         bottom = 0.5
#         #plot_save_info = pd.DataFrame([labels, y, temp_errs_mins, temp_errs_maxes, y_ft, temp_errs_ft_mins, temp_errs_ft_maxes], columns=['classes', 'pt_means', 'pt_mins', 'pt_maxes', 'ft_means', 'ft_mins', 'ft_maxes']).to_csv(save_path+'/metric_data/'+pca_path+bias_metric+'/indiv_plot.csv', index=False)
#         plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/indiv_plot.csv', index=False)

#     else:
#         if str(pca_comps) != 'None':
#             title = "Averaged Class vs. Gender, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))+ ", PCA: " + str(pca_comps)
#             title_diff = "Averaged, Model: "+model_name +" Finetuned on: " + dataset_name + ", Class vs. Gender Classes (Finetuned - Pretrained), PCA: " + str(pca_comps)
#         else:
#             title = "Averaged Class vs. Gender, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
#             title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name +", Class vs. Gender (Finetuned - Pretrained)"
#         #save = save_path + '/boxplots/object_comp_'+ bias_metric+'.pdf'
#         #save_diff = save_path + '/boxplots/object_comp_diffs_' + bias_metric +'.pdf'
#         save = save_path + '/boxplots/' + dataset_name + '/'+ pca_path +bias_metric + '/object_comp_'+ bias_metric+'_averaged.pdf'
#         save_diff = save_path + '/boxplots/' + dataset_name + '/'+pca_path + bias_metric + '/object_comp_diffs_' + bias_metric +'_averaged.pdf'
#         bbox = (0.5, -0.65)
#         bottom_legend = -0.9
#         bottom = 0.5
    
#     fig.update_layout(
#         title={
#             'text': title,
#             'y':0.9,
#             'x':0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'},
#         height=600,
#         width=700,
#         plot_bgcolor="#FFF",  # Sets background color to white
#         xaxis=dict(
#             title="Classes",
#             linecolor="#BCCCDC",  # Sets color of X-axis line
#             showgrid=True  # Removes X-axis grid lines
#         ),
#         yaxis=dict(
#             title=bias_metric + " score",  
#             linecolor="#BCCCDC",  # Sets color of Y-axis line
#             showgrid=True,  # Removes Y-axis grid lines    
#         ),
#         legend=dict(
#             yanchor="bottom",
#             y=0.01,
#             xanchor="right",
#             x=0.99
#         ))
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')


#     fig.write_image(save)

#     colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']

#     fig2 = plt.figure()
#     for i in range(len(y_dict)):
#         plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
#     plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
#     plt.grid(b=True, markevery=1)
#     fig2.subplots_adjust(bottom=bottom)
#     #plt.tight_layout()

#     plt.title(title_diff)
#     plt.xlabel("Classes")
#     plt.ylabel(bias_metric+" score")
#     plt.savefig(save_diff, format='pdf')

# def get_multiple_trials_stats(list_dicts):
#     final = dict()
#     mins_maxes = dict()
#     #print("PARAM: ", list_dicts[0])
    
#     for category in list_dicts[0]: # iterating over dictionary keys
#         # each category, trial[category] = cos, std, sim_std, sim_mean
#         final[category] = [[], 0.0]

#     for trial in list_dicts: # dictionary
#         for category in trial: # iterating over dictionary keys
#             # each category, trial[category] = cos, std, sim_std, sim_mean
#             final[category][0].extend(trial[category][0])
#             final[category][1] += trial[category][3]
#     for cat in final:
#         final[cat][1] /= len(list_dicts) # get mean of means

#     for category_name in final:
#         mins_maxes[category_name] = [min(final[category_name][0]), max(final[category_name][0])]

#     return final, mins_maxes

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
    if pca_comps == 0.0:
        pca_comps = 'None'

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

    if pca_comps == 'None':
        pca_path = 'no_pca/'
    else:
        pca_path = 'pca/'
    
    if multiple_trials == False:
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_pt.npy', self_similarities_stats_pt)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/self_similarities_ft.npy', self_similarities_stats_ft)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_pt.npy', comps_similarities_stats_pt)
        np.save(save_path+'/metric_data/'+ dataset_name + '/'+pca_path+bias_metric+'/comps_similarities_ft.npy', comps_similarities_stats_ft)
        
        np.save(save_path + '/boxplots/'+ dataset_name + '/' + pca_path + bias_metric + '/'+'self_similarities_pt_significance.npy', self_similarities_stats_pt_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'self_similarities_ft_significance.npy', self_similarities_stats_ft_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'comps_similarities_pt_significance.npy', comps_similarities_stats_pt_significance)
        np.save(save_path + '/boxplots/' + dataset_name + '/'+ pca_path + bias_metric + '/'+'comps_similarities_ft_significance.npy', comps_similarities_stats_ft_significance)

    individual_plots_cats = config['INDIVIDUAL_PLOTS']['category_list']
    
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



