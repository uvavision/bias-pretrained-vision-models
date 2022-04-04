# Uncovering the Effects of Biases in Pretrained Visual Recognition Models

[Jaspreet Ranjit](https://jr4fs.github.io/), [Tianlu Wang](https://tianlu-wang.github.io/), [Baishakhi Ray](https://www.rayb.info/), [Vicente Ordóñez](https://www.cs.rice.edu/~vo9/)

## Installation 
```bash
# ensure conda is installed
# gather the dependencies for running scripts in this repo
conda env create -f environment.yml
conda activate bias_vision
```
## Usage
Currently, this repo supports the following five features:
1. Feature extraction for a finetuned model on a specified analysis set: extracting features from a loaded pretrained model (model with loaded weights) *and* that same model after it has been finetuned
    a. 
2. Feature extraction for a pretrained model on an analysis set: extracting features from a loaded pretrained model 
3. Using saved features (both pretrained and finetuned) to perform bias analysis 
4. For a given model, average across finetuning trial runs and perform bias analysis experiment
5. Finetuning or resuming training for an available model, perform feature extraction on analysis set and bias analysis on extracted features
### Replicating Results

## Customization

### Adding a Model

### Adding an Analysis Set

## Contents