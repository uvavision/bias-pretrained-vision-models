from __future__ import print_function
from __future__ import division
import random

import matplotlib
import matplotlib.lines as mlines
from data_loader import V

import numpy as np
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
import itertools
import matplotlib.cm as cm
import datetime
import plotly.graph_objects as go
import pandas as pd

fontP = FontProperties()
fontP.set_size('small')

def plot_indiv_categories_mult_trials(model_name, dataset_name, save_path, category_features, comps_stats, self_similarities_stats, category_name):
    """Plots bias analysis experiment results for 'Indiv' subset of classes averaged across trials for a model:
    'Indiv': single classes --> man, woman, surfboard, car, refrigerator, etc. (intra-class)

    Args:
        model_name: Name of model to perform bias metric experiment on
        dataset_name: Analysis set
        save_path: Path to save bias analysis experiment results
        category_features: Classes to be plotted
        comps_stats: Dictionary mapping class comparisons (ex. man+woman vs. man) defined in COMPS in config to
                     a tuple of (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity() for features extracted
                     from pretrained model
        self_similarities_stats: Dictionary mapping class names (LABEL_NAMES in config) to a tuple of
                                 (cos, std, sim_std, sim_mean) --> the output of inter_class_similarity()
                                 for features extracted from pretrained model
        category_name: Name of single class: 'car, surfboard, refrigerator' etc. --> config['INDIVIDUAL_PLOTS']['category_list']  
    """
    y = []
    yerr_vals = []
    labels = list(category_features)
    #  (cos, std, sim_std, sim_mean)
    for i in category_features:
        if i in comps_stats:
            y.append(comps_stats[i][1])
            yerr_vals.append([min(comps_stats[i][0]), max(comps_stats[i][0])])
        else:
            y.append(self_similarities_stats[i][1])
            yerr_vals.append([min(self_similarities_stats[i][0]), max(self_similarities_stats[i][0])])
    y = np.asarray(y)
    yerr_vals = np.asarray(yerr_vals).T
    yerr = np.abs(yerr_vals-y)

    temp_errs = zip(*yerr)
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
    title = "Averaged, Model: "+model_name + ", Class: " + category_name

    plt.xlabel("Classes")
    plt.ylabel("cosine score")

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


    save = save_path + '/boxplots/'+ dataset_name + '/' +category_name+'_filtered_averaged.pdf'
    fig.write_image(save)


def plot_indiv_cats_comps_mult_trials(model_name, dataset_name, save_path, mins_maxes_pt, mins_maxes_ft, mins_maxes_comps_pt, mins_maxes_comps_ft, comps_stats, comps_stats_ft, self_similarities_stats, self_similarities_stats_ft, category_name, category_features):
    """Plots bias analysis experiment results for 'Comps' subset of classes averaged across trials for a model:
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
        category_name: Name of single class: 'car, surfboard, refrigerator' etc. --> config['INDIVIDUAL_PLOTS']['category_list']
        category_features: Classes to be plotted: config['INDIVIDUAL_PLOTS_COMPS'][category_name]
        
    """
    y_dict = {}
    y_dict_ft = {}
    for i in category_features:
        if i in comps_stats:
            y_dict[i] = comps_stats[i][1]
            y_dict_ft[i] = comps_stats_ft[i][1]
        else:
            y_dict[i] = self_similarities_stats[i][1]
            y_dict_ft[i] = self_similarities_stats_ft[i][1]

    y_dict_comp = {}

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
            y_err.append(mins_maxes_pt[key]) # appending a list of [min(vals), max(vals)] #centroid mean(vals)
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

    temp_errs = zip(*yerr)
    temp_errs_ft = zip(*yerr_ft)


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

    title = "Averaged, Model: "+model_name + " Finetuned on: " + dataset_name+", Class: "+ category_name + "<br> Spearman Coeff: "+ str(round(spearman_coeff[0], 3)) + " @p " + str(round(spearman_coeff[1], 3))

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
    colors = ['r', 'g', 'b', 'c', 'm', 'lime', 'orange', 'steelblue', 'silver', 'turquoise', 'violet', 'gold', 'lawngreen', 'pink', 'deepskyblue', 'palegreen', 'peachpuff', 'dodgerblue', 'peru', 'tomato']
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    save = save_path + '/boxplots/'+ dataset_name +'/'+ category_name+ '_comp_averaged.pdf'
    fig.write_image(save)
    fig2 = plt.figure()
    for i in range(len(y_dict)):
        plt.errorbar(np.asarray([i]), np.asarray([y_ft[i] - y[i]]), yerr=yerr_ft[:, i:i+1] - yerr[:, i:i+1], fmt='o', color=colors[i], ecolor=colors[i])
    plt.legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', prop=fontP, ncol=3)
    plt.grid(b=True)
    fig2.subplots_adjust(bottom=0.35)

    plt.title("Averaged, Model: " + model_name + " Finetuned on: " +dataset_name +", Class: "+ category_name +"\n (Finetuned - Pretrained)")


    plt.xlabel("Classes")
    plt.ylabel("cosine score")

    save = save_path + '/boxplots/'+ dataset_name + '/' +category_name+'_comp_diff_averaged.pdf'
    plt.savefig(save, format='pdf')
