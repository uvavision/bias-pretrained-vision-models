# Uncovering the Effects of Biases in Pretrained Visual Recognition Models

[Jaspreet Ranjit](https://jr4fs.github.io/), [Tianlu Wang](https://tianlu-wang.github.io/), [Baishakhi Ray](https://www.rayb.info/), [Vicente Ordóñez](https://www.cs.rice.edu/~vo9/)

## Installation 
```bash
# ensure conda is installed
# gather the dependencies for running scripts in this repo
conda env create -f environment.yml
conda activate bias_vision
```
### Setup experiments/ folder
```bash
python custom.py \
    --model_list <list of models to test> \
    --training_datasets <list of training datasets to test>
```
For example, if I have two training datasets: ['coco', 'openimages'] and two models ['model_one', 'model_two']
```bash
python custom.py --initial_setup --model_list model_one, model_two --training_datasets coco openimages
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
1. To recreate Table 2, refer to the jupyter notebook: `experimental_work/ieat.ipynb`
2. To recreate Table 3, Figure 3, Table 4 and Figure 4 and all the subsequent plots in the Supplementary material, it is assumed that the `experiments/training_dataset/model_name/` folder contains multiple trial runs for a model, and the results are averaged across these runs. By changing the `--model_name, --num_classes, --analysis_set, --analysis_set_path --config_file` flags, you can generate different sets of results and plots using the saved features and finetuned models 
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
3. We release the metadata for the finetuned models in the `experiments/` directory. All the models from torchvision are saved using pytorch lightning. This directory is set up as follows: 

```bash
experiments
├── coco <training dataset>
│   ├── model_one
│       ├── trial_one
│            ├── boxplots
│                ├── analysis_set_one <coco>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│                ├── analysis_set_two <openimages>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│            ├── features
│                ├── analysis_set_one <coco>
│                    ├── pretrained_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                    ├── finetuned_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                ├── analysis_set_two <openimages>
│                    ├── pretrained_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                    ├── finetuned_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│            ├── metric_data
│                ├── analysis_set_one <coco>
│                    ├── metric_data.npy
│                ├── analysis_set_two <openimages>
│                    ├── metric_data.npy
│            ├── model <training metadata>
│        ├── trial_two
│        ...
│        ├── trial_x
│        ├── averaged <results from averaging across trials>
│            ├── boxplots
│                ├── analysis_set_one <coco>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│                ├── analysis_set_two <openimages>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│            ├── metric_data
│                ├── analysis_set_one <coco>
│                    ├── metric_data.npy
│                ├── analysis_set_two <openimages>
│                    ├── metric_data.npy
│   ...
│   └── model_n
├── openimages <training dataset>
│   ├── model_one
│       ├── trial_one
│            ├── boxplots
│                ├── analysis_set_one <coco>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│                ├── analysis_set_two <openimages>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│            ├── features
│                ├── analysis_set_one <coco>
│                    ├── pretrained_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                    ├── finetuned_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                ├── analysis_set_two <openimages>
│                    ├── pretrained_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│                    ├── finetuned_features
│                        ├── feature_one.npy
│                        ...
│                        ├── feature_n.npy
│            ├── metric_data
│                ├── analysis_set_one <coco>
│                    ├── metric_data.npy
│                ├── analysis_set_two <openimages>
│                    ├── metric_data.npy
│            ├── model <training metadata>
│        ├── trial_two
│        ...
│        ├── trial_x
│        ├── averaged <results from averaging across trials>
│            ├── boxplots
│                ├── analysis_set_one <coco>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│                ├── analysis_set_two <openimages>
│                    ├── boxplot_one.pdf
│                    ...
│                    ├── boxplots_n.pdf
│            ├── metric_data
│                ├── analysis_set_one <coco>
│                    ├── metric_data.npy
│                ├── analysis_set_two <openimages>
│                    ├── metric_data.npy
│   ...
│   └── model_n
```

## Customization

### Adding a Model
The following models have already been implemented: 
```bash
['clip', 'moco_resnet50', 'simclr_resnet50','bit_resnet50', 'resnet50', 'resnet18','alexnet', 'vgg', 'densenet', 'fasterrcnn', 'retinanet', 'googlenet', 'resnet34', 
                    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'virtex_resnet50']
```
1. In `train.py`, modify the `models_implemented` list in `lightning_setup()` and `lightning_train()` and add the name of the model 
2. In `model_init.py`, modify `load_models_pytorch()` to setup the model to be trained with pytorch lightning for multi-label classification on an available dataset 
3. Lastly, set up the directory as follows: (assumes the `experiments/` folder exists - see section above)
```bash
python custom.py \
    --root <path to experiments folder> \
    --model_name <name of model to add> \
    --add_model
```
For example, if I want to add a model named: resnet to all the training datasets in the `experiments/` folder:
```bash
python custom.py --root experiments --model_name resnet --add_model
```
4. *This is discouraged but included here if absolutely necessary* Note, if your model cannot be trained with pytorch lightning, you will need to define a separate function in `model_init.py` following the example for clip: `initialize_model_clip()`. Additionally, you will need to add a file such as `clip_model.py` in `models_def/` defining the training functions and feature extraction logic for such a model. This will also involve modifying `lightning_train(), lightning_setup(), main()` and `extract_features()` in `train.py` to include separate calls for this model. 

### Adding an Analysis Set
1. In `analysis_sets/`, create an additional directory with the name of your analysis set that specifies .txt files for each class in the set. The .txt file should contain image_ids or urls to the images in that class
2. In `config/`, create a .yaml file specifying the metadata for that analysis set. Follow `coco.yaml` for an example. Label names and classes names categories define abbreviations to be used during plotting
3. `pytorch_models.py` includes a `PytorchFeatureExtractor` class (`clip_model.py` includes one as well) which includes a `process_imgs()` function that specifies how to access the images in the .txt files in `analysis_sets/`. Add in an additional line for your analysis set and modify the class attributes accordingly if needed. 
4. Set up the directory as follows: (assumes the `experiments/` folder exists - see section above)
root, analysis_set, model_list

```bash
python custom.py \
    --root <path to experiments folder> \
    --model_list <list of models to test> \
    --analysis_set <name of analysis_set to add> \
    --add_analysis_set
```
For example, if I want to add an analysis set named: ieat to all the training datasets in the `experiments/` folder for each model in model_list:
```bash
python custom.py --root experiments --model_list model1 model2 --analysis_set ieat --add_analysis_set
```

### Training on an additional dataset

Setup up the directory as follows: 

```bash
python custom.py \
    --root <path to experiments folder> \
    --model_list <list of models to test> \
    --training_set_name <list of training datasets to test>
    --add_training_set
```
For example, if I have two training datasets: ['coco', 'openimages'] and two models ['model_one', 'model_two', and want to add an additional training dataset: imagenet to each of the models for each of the training datasets
```bash
python custom.py --root experiments --model_list model_one, model_two --training_set_name imagenet --add_training_set
```
 

## Contents
- `analysis_sets/`: coco and openimages analysis sets where each subfolder contains text files for each class in an analysis sets that details image ids or urls for that dataset
- `config/`: config files for each analysis set
- `cosine_analysis/`: functions to replicate results from paper and generate bias analysis results on additional trials
- `models_def/`: Definitions for model types, contains training and feature extraction details
- `data_loader.py`: Dataloaders for training datasets
- `model_init.py`: Initializes model for finetuning by reshaping the last layer and configures the optimizers, loss function and other hyperparameters
- `train.py` : contains generalized training details and cmd line functions
- `experiments/`: contains metadata for all trained models in the paper


## Notes
- The following features are not updated: trends experiment, check backups/cosine.py for source code for this experiment. This experiment plots models against each other on a single plot