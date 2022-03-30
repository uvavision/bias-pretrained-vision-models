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
import wget 
import torch
print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')
import os.path 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models_def.bit_model import *
import models_def.resnet_model as resnet_self_supervised
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """
    Layer-wise adaptive rate scaling
    - Converted from Tensorflow to Pytorch from:
    https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    - Based on:
    https://github.com/noahgolmant/pytorch-lars
    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        lr (int): Length / Number of layers we want to apply weight decay, else do not compute
        momentum (float, optional): momentum factor (default: 0.9)
        use_nesterov (bool, optional): flag to use nesterov momentum (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")
        eta (float, optional): LARS coefficient (default: 0.001)
    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    """

    def __init__(self, params, lr, len_reduced, momentum=0.9, use_nesterov=False, weight_decay=0.0, classic_momentum=True, eta=0.001):

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            classic_momentum=classic_momentum,
            eta=eta,
            len_reduced=len_reduced
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eta = eta
        self.len_reduced = len_reduced

    def step(self, epoch=None, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            learning_rate = group['lr']

            # TODO: Hacky
            counter = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: This really hacky way needs to be improved.
                # Note Excluded are passed at the end of the list to are ignored
                if counter < self.len_reduced:
                    grad += self.weight_decay * param

                # Create parameter for the momentum
                if "momentum_var" not in param_state:
                    next_v = param_state["momentum_var"] = torch.zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_var"]

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        g_norm.ge(0), (self.eta * w_norm / g_norm), torch.Tensor([1.0]).to(device)), torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    next_v.mul_(momentum).add_(scaled_lr, grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)

                # Not classic_momentum
                else:

                    next_v.mul_(momentum).add_(grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (grad)

                    else:
                        update = next_v

                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    v_norm = torch.norm(update)

                    device = v_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        v_norm.ge(0), (self.eta * w_norm / v_norm), torch.Tensor([1.0]).to(device)), torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    p.data.add_(-scaled_lr * update)

                counter += 1

        return loss


def get_lars_optimizer(models):
    '''Get the desired optimiser
    - Selects and initialises an optimiser with model params.
    - if 'LARS' is selected, the 'bn' and 'bias' parameters are removed from
     model optimisation, only passing the parameters we want.
    Args:
        models (tuple): models which we want to optmise, (e.g. encoder and projection head)
        mode (string): the mode of training, (i.e. 'pretrain', 'finetune')
        args (Dictionary): Program Arguments
    Returns:
        optimizer (torch.optim.optimizer):
    '''

    # Learning Rate
    #args.scaled_learning_rate = (args.learning_rate * (args.batch_size / 256))
    #args.scaled_finetune_learning_rate = (args.finetune_learning_rate * (args.batch_size / 256))
    scaled_learning_rate = (0.03 * (265 / 256))
    scaled_finetune_learning_rate = (0.0003 * (256 / 256))

    params_models = []
    reduced_params = []

    removed_params = []

    skip_lists = ['bn', 'bias']

    for m in models:

        m_skip = []
        m_noskip = []

        params_models += list(m.parameters())

        for name, param in m.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
    # Set hyperparams depending on mode

    lr = scaled_finetune_learning_rate
    #wd = args.finetune_weight_decay
    wd = 1e-5

    print("reduced_params len: {}".format(len(reduced_params)))
    print("removed_params len: {}".format(len(removed_params)))

    optimizer = LARS(reduced_params+removed_params, lr=0.01,
                        weight_decay=wd, eta=0.001, use_nesterov=True, len_reduced=len(reduced_params))

    return optimizer



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    init_weights(model)

def init_weights(m):
    '''Initialize weights with zeros
    '''
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

def load_moco(base_encoder, checkpoint_path):
    """ Loads the pre-trained MoCo model parameters.
        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.
    Args:
        base_encoder (model): Randomly Initialised base_encoder.
        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """

    # Load the pretrained model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    return base_encoder

def load_models_pytorch(model_name, num_classes, use_pretrained):
    model_ft = None
    model_init = dict()
    model_init['resnet18'] = models.resnet18
    model_init['resnet34'] = models.resnet34
    model_init['resnet50'] = models.resnet50
    model_init['resnet101'] = models.resnet101
    model_init['resnet152'] = models.resnet152
    model_init['resnext50_32x4d'] = models.resnext50_32x4d
    model_init['resnext101_32x8d'] = models.resnext101_32x8d
    model_init['wide_resnet50_2'] = models.wide_resnet50_2
    model_init['wide_resnet101_2'] = models.wide_resnet101_2
    model_init['alexnet'] = models.alexnet
    model_init['vgg'] = models.vgg11_bn
    model_init['densenet'] = models.densenet121
    model_init['googlenet'] = models.googlenet

    
    if model_name == 'bit_resnet50':
        if not os.path.isfile("BiT-M-R50x1.npz"):
            model_ft_file = wget.download('https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz')
        model_ft_bit = KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
        model_ft_bit.load_from(np.load("BiT-M-R50x1.npz"))
        model_init['bit_resnet50'] = model_ft_bit
        model_ft = model_init['bit_resnet50']
    elif model_name == 'moco_resnet50':
        base_encoder = resnet_self_supervised.resnet50x1(num_classes=num_classes)  # Encoder
        base_encoder = load_moco(base_encoder, 'models_def/moco_v1_200ep_pretrain.pth.tar')
        model_init['moco_resnet50'] = base_encoder
        model_ft = model_init['moco_resnet50']
    elif model_name == 'simclr_resnet50':
        base_encoder = resnet_self_supervised.resnet50x1(num_classes=1000)
        checkpoint = torch.load('models_def/ResNet50_1x/resnet50-1x.pth')
        base_encoder.load_state_dict(checkpoint['state_dict'])
        base_encoder.fc = nn.Linear(2048, num_classes, bias=True)
        model_init['simclr_resnet50'] = base_encoder
        model_ft = model_init['simclr_resnet50']
    elif model_name == 'virtex_resnet50':
        virtex = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)
        virtex.avgpool = nn.AdaptiveAvgPool2d((1,1))
        virtex.fc = nn.Linear(2048, num_classes)
        model_init['virtex_resnet50'] = virtex
        model_ft = model_init['virtex_resnet50']

    elif model_name in model_init:
        model_ft = model_init[model_name](pretrained=use_pretrained)
    else:
        print("Invalid model name, choose from moco_resnet50, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, alexnet, vgg, densenet, googlenet, bit_resnet50, virtex_resnet50")
    return model_ft

def initialize_model_pytorch(model_name, num_classes, feature_extract, lr, momentum, optimizer_name, use_pretrained):
    model_ft = load_models_pytorch(model_name, num_classes, use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    if model_name == 'alexnet' or model_name == 'vgg':
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name != 'bit_resnet50' and model_name != 'moco_resnet50'and model_name != 'simclr_resnet50':
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(model_ft)
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()

    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)


    # Observe that all parameters are being optimized
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=0.0001)
    elif optimizer_name=='adamax':
        optimizer = optim.Adamax(params_to_update, lr=lr, weight_decay=0.0005)
    elif optimizer_name == 'lars':
        optimizer = get_lars_optimizer((model_ft,))
    else:
        optimizer = torch.optim.Adam(params_to_update, lr=lr, betas=(0.9,0.98),eps=1e-6,weight_decay=1e-6) #Params from paper
    # Setup the loss fxn
    criterion = nn.BCEWithLogitsLoss()

    return model_ft, criterion, optimizer

def initialize_model_clip(num_classes, feature_extract, lr, momentum, optimizer_name, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model, preprocess = clip.load("ViT-B/32", jit=False) #Must set jit=False for training
    model = model.cuda()
    model_ft = torch.nn.Sequential(model.visual, torch.nn.Linear(512,num_classes)).cuda()
    criterion = torch.nn.BCEWithLogitsLoss().cuda() # --> already includes sigmoid 
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper
    print(model_ft)
    return model_ft, criterion, optimizer
