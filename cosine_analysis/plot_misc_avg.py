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

def plot_misc_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_features, plot_type):
    """Plots bias analysis experiment results for different subset of classes averaging across all trials for a model
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
            y_dict[i] = comps_stats[i][1]
            y_dict_ft[i] = comps_stats_ft[i][1]
        else:
            y_dict[i] = self_similarities_stats[i][1]
            y_dict_ft[i] = self_similarities_stats_ft[i][1]

    y_dict_comp = dict()

    #y_sorted = {k: v for k, v in sorted(y_dict.items(), key=lambda item: item[1])}
    y_sorted = y_dict
    y = np.asarray(list(y_sorted.values()))
    labels = list(y_sorted.keys())

    for label in y_sorted:
        y_dict_comp[label] = y_dict_ft[label]
    y_ft = np.asarray(list(y_dict_comp.values()))
    spearman_coeff = stats.spearmanr(y, y_ft)

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

    temp_errs_mins,  temp_errs_maxes = zip(*y_err)
    temp_errs_ft_mins, temp_errs_ft_maxes = zip(*y_err_ft)

    yerr = np.abs(yerr_vals-y)
    yerr_ft = np.abs(yerr_vals_ft-y_ft)



    fig = go.Figure(data=go.Scatter(
        x=labels,
        y=y,
        mode='markers',
        name = 'Pretrained',
        marker=dict(
            color='blue',
            size=10,
        ),
        marker_symbol = 'circle',
        error_y=dict(
            type='data',
            symmetric=False,
            array=yerr[1],
            arrayminus=yerr[0])
        ))

    trace = go.Scatter(
            x=labels,
            y=y_ft,
            mode='markers',
            name = 'Finetuned',
            marker=dict(
                color='green',
                size=10,
            ),
            marker_symbol = 'triangle-up',
            error_y=dict(
                type='data',
                symmetric=False,
                array=yerr_ft[1],
                arrayminus=yerr_ft[0])
            )
    fig.add_trace(trace)

    # plt.legend(labels, loc='lower left', ncol=12)
    if plot_type == 'pairs':

        title = "Averaged Paired Classes, Model: "+ model_name + " Finetuned on: " + dataset_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name + ", Paired Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/'+'pairs_averaged.pdf'
        save_diff = save_path + '/boxplots/' + dataset_name + '/' +'pairs_diff_averaged.pdf'
        bbox = (0.5, -0.65)
        bottom_legend = -0.75
        bottom = 0.5
        plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+'pairs_plot.csv', index=False)
    elif plot_type == 'comps':
        title = "Averaged Comparison Classes, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name +", Comparison Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/'+ dataset_name + '/' + 'comps_averaged.pdf'
        save_diff = save_path + '/boxplots/'+ dataset_name + '/'+'comps_diff_averaged.pdf'
        bbox = (0.5, -0.75)
        bottom_legend = -1.0
        bottom = 0.5
        plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+'comps_plot.csv', index=False)

    elif plot_type == 'indiv':
        title = "Averaged Individual Classes, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Averaged, Model: "+model_name +" Finetuned on: " + dataset_name + ", Individual Classes (Finetuned - Pretrained)"
        save = save_path + '/boxplots/' + dataset_name + '/'+'indv_averaged.pdf'
        save_diff = save_path + '/boxplots/'+ dataset_name + '/' + 'indv_diff_averaged.pdf'  
        bbox = (0.5, -0.65)
        bottom_legend = -0.6
        bottom = 0.5
        plot_save_info = pd.DataFrame({'classes':labels, 'pt_means':y, 'pt_mins':temp_errs_mins, 'pt_maxes':temp_errs_maxes, 'ft_means':y_ft, 'ft_mins':temp_errs_ft_mins, 'ft_maxes':temp_errs_ft_maxes}).to_csv(save_path+'/metric_data/'+ dataset_name + '/'+'indiv_plot.csv', index=False)

    else:
        title = "Averaged Class vs. Gender, Model: "+ model_name + " Finetuned on: " + dataset_name +"<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))
        title_diff = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name +", Class vs. Gender (Finetuned - Pretrained)"
        save = save_path + '/boxplots/' + dataset_name + '/' + 'object_comp_averaged.pdf'
        save_diff = save_path + '/boxplots/' + dataset_name + '/' + 'object_comp_diffs_averaged.pdf'
        bbox = (0.5, -0.65)
        bottom_legend = -0.9
        bottom = 0.5
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=600,
        width=700,
        plot_bgcolor="#FFF",  # Sets background color to white
        xaxis=dict(
            title="Classes",
            linecolor="#BCCCDC",  # Sets color of X-axis line
            showgrid=True  # Removes X-axis grid lines
        ),
        yaxis=dict(
            title="cosine score",  
            linecolor="#BCCCDC",  # Sets color of Y-axis line
            showgrid=True,  # Removes Y-axis grid lines    
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')


    fig.write_image(save)

    colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']

    fig2 = plt.figure()
    for i in range(len(y_dict)):
        plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
    plt.legend(labels, bbox_to_anchor=bbox, loc='lower center', prop=fontP, ncol=3)
    plt.grid(b=True, markevery=1)
    fig2.subplots_adjust(bottom=bottom)
    #plt.tight_layout()

    plt.title(title_diff)
    plt.xlabel("Classes")
    plt.ylabel("cosine score")
    plt.savefig(save_diff, format='pdf')

def get_multiple_trials_stats(list_dicts):
    """Averages cosine and error bar stats across all trials for a model
    
    Args:
        list_dicts: List of dictionaries where each dictionary corresponds to a single trial for a model ex. [{class_name: (list[cos], mean of cosine similarities), ... } ...]
    Returns:
        final: A dictionary mapping class name to a list of cosine similarity scores for each trial and the mean of all cosine results across all trials
        mins_maxes: A dictionarry mapping class name to a list of mins_maxes defining the error bars for each class averaged across model trials

    """
    final = dict()
    mins_maxes = dict()
    
    for category in list_dicts[0]: # iterating over dictionary keys
        # each category, trial[category] = cos, std, sim_std, sim_mean
        final[category] = [[], 0.0]

    for trial in list_dicts: 
        for category in trial: # iterating over dictionary keys
            # each category, trial[category] = cos, std, sim_std, sim_mean
            final[category][0].extend(trial[category][0])
            final[category][1] += trial[category][3]
    for cat in final:
        final[cat][1] /= len(list_dicts) # get mean of means

    for category_name in final:
        mins_maxes[category_name] = [min(final[category_name][0]), max(final[category_name][0])]

    return final, mins_maxes