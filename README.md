# Uncovering the Effects of Biases in Pretrained Visual Recognition Models

[Jaspreet Ranjit](https://jr4fs.github.io/), [Tianlu Wang](https://tianlu-wang.github.io/), [Baishakhi Ray](https://www.rayb.info/), [Vicente Ordóñez](https://www.cs.rice.edu/~vo9/)

## Installation 
```bash
# ensure conda is installed
# gather the dependencies for running scripts in this repo
conda env create -f environment.yml
conda activate bias_vision
```

### Setup Datasets
We currently support finetuning on the following datasets: COCO 2017 and Open Images, please refer to section "Training on an additional dataset" for details on how to add an additional dataset for finetuning 

1. #### COCO 2017: Download and setup images and annotations [here](https://cocodataset.org/#download)

2. #### Open Images: Download and setup images and annotations [here](https://storage.googleapis.com/openimages/web/download.html)

### Setup Models
1. To use SimCLR ResNet50, download ResNet50 (1x) from [here](https://github.com/google-research/simclr) and place the foler in the `models_def/` directory
2. To use MoCo ResNet50, download MoCo v1 from [here](https://github.com/facebookresearch/moco) and place the .tar file in `models_def/` directory

## Usage
Currently, this repo supports the following six features:
1. Feature extraction for a finetuned model on a specified analysis set: extracting features from a loaded pretrained model (model with loaded weights) *and* that same model after it has been finetuned. The following command also runs bias analysis on the extracted features with the ```--bias_analysis``` flag.
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --dataset <name of dataset the model was finetuned on: e.g. 'coco'> \
    --num_classes <number of classes /in dataset the model was finetuned on: e.g. 80> \
    --extract_cross_analysis_features \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --trial_path < path to finetuned model: e.g. 'experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29'> \
    --bias_analysis \
    --finetune
```
2. Feature extraction for a pretrained model on an analysis set: extracting features from a loaded pretrained model. The following command also runs bias analysis on the extracted features with the ```--bias_analysis``` flag.
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --dataset <name of dataset the model was finetuned on: e.g. 'coco'> \
    --num_classes <number of classes /in dataset the model was finetuned on: e.g. 80> \
    --pretrained_features \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --trial_path < path to finetuned model: e.g. 'experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29'> \
    --bias_analysis \
    --finetune
```
3. Using saved features (both pretrained and finetuned) to perform bias analysis. 
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --dataset <name of dataset the model was finetuned on: e.g. 'coco'> \
    --num_classes <number of classes /in dataset the model was finetuned on: e.g. 80> \
    --load_features \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --trial_path < path to finetuned model: e.g. 'experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29'> \
    --bias_analysis \
    --finetune
```
4. For a given model, average across finetuning trial runs and perform bias analysis experiment
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --num_classes <number of classes in dataset the model was finetuned on> \
    --load_features \
    --multiple_trials \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --bias_analysis \
    --finetune
```
5. Finetune an available model, perform feature extraction on analysis set and bias analysis on extracted features
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --dataset <name of dataset to be finetunend on: e.g. 'coco'> \
    --dataset_path <path to dataset the model will be finetuned on> \
    --num_classes <number of classes /in dataset the model will be finetuned on: e.g. 80> \
    --batch_size <\#> \
    --epochs <\#> \
    --lr <learning rate: e.g. 0.001> \
    --lr_scheduler <e.g. 'reduce', 'cosine', 'none'>
    --momentum <\#> \
    --optimizer <e.g. 'sgd', 'adam', 'adamax', 'lars'> \
    --finetune \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --bias_analysis \
    --seed <\#>
```
6. Resume training for an available model, perform feature extraction on analysis set and bias analysis on extracted features
```bash
CUDA_VISIBLE_DEVICES=\# python train.py \
    --model_name <name of finetuned model: e.g. 'bit_resnet50'> \
    --dataset <name of dataset to be finetunend on: e.g. 'coco'> \
    --dataset_path <path to dataset the model will be finetuned on> \
    --num_classes <number of classes /in dataset the model will be finetuned on: e.g. 80> \
    --batch_size <\#> \
    --epochs <\#, must be higher than the number of epochs the model was originally trained for> \
    --lr <learning rate: e.g. 0.001> \
    --lr_scheduler <e.g. 'reduce', 'cosine', 'none'>
    --momentum <\#> \
    --optimizer <e.g. 'sgd', 'adam', 'adamax', 'lars'> \
    --resume_training \
    --analysis_set <name of analysis set: e.g. 'openimages'> \
    --analysis_set_path <path to analysis set dataset> \
    --config_file <analysis set config: e.g. 'config/openimages.yaml'> \ 
    --bias_analysis \
    --finetune \
    --checkpoint <path to checkpoint: e.g. 'experiments/coco/resnet50/2022-01-19\ 16\:43\:15/model/resnet50/version_0/checkpoints/epoch\=24-step\=46224.ckpt'> \
    --seed <\#>
```
## Replicating Results

## Customization

### Adding a Model
The following models have already been implemented: 
```bash
['clip', 'moco_resnet50', 'simclr_resnet50','bit_resnet50', 'resnet50', 'resnet18','alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet34', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'virtex_resnet50']
```

### Adding an Analysis Set

### Training on an additional dataset

## Contents
- `analysis_sets/`: coco and openimages analysis sets where each subfolder contains text files for each class in an analysis sets that details image ids or urls for that dataset
- `config/`: config files for each analysis set
- `cosine_analysis/`: functions to replicate results from paper and generate bias analysis results on additional trials
- `models_def/`: Definitions for model types, contains training and feature extraction details
- `data_loader.py`: Dataloaders for training datasets
- `model_init.py`: Initializes model for finetuning by reshaping the last layer and configures the optimizers, loss function and other hyperparameters
- `train.py` : contains generalized training details and cmd line functions