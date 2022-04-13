#!/bin/bash

# modify the variables below 
model_name=bit_resnet50
coco_path=/localtmp/data/coco2017/coco_dataset/
coco_analysis_path=/localtmp/data/coco2017/coco_dataset/
oi_analysis_path=../VisionResearch/finetuneClip/openimages_dataset/val/
#TODO: Add three variables --> openimages trial path, coco_trial path, checkpoint path 

# Feature 1
CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --extract_cross_analysis_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --extract_cross_analysis_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --extract_cross_analysis_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --extract_cross_analysis_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

# Feature 2
CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --pretrained_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --pretrained_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --pretrained_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --pretrained_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

# Feature 3
CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --load_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL
    
CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --load_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/coco/bit_resnet50/2022-01-21\ 19\:03\:29/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL  

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --load_features --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL
    
CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --load_features --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --trial_path experiments/openimages/bit_resnet50/2022-02-23\ 11\:44\:36/ --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL
    
# Feature 4

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --load_features --multiple_trials --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --num_classes 80 --load_features --multiple_trials --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --load_features --multiple_trials --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS || echo FAIL

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset openimages --num_classes 601 --load_features --multiple_trials --analysis_set openimages --analysis_set_path $oi_analysis_path --config_file config/openimages.yaml --finetune --bias_analysis > /dev/null 2>&1 && echo SUCCESS_DONE || echo FAIL

# Feature 5 --> supports testing finetuning on COCO

#CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --dataset_path $coco_path --batch_size 64 --epochs 1 --num_classes 80 --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --bias_analysis --finetune > /dev/null 2>&1 && echo SUCCESS || echo FAIL

# Feature 6 --> supports testing resuming training on COCO 

#CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model_name --dataset coco --dataset_path $coco_path --batch_size 64 --epochs 1 --num_classes 80 --analysis_set coco --analysis_set_path $coco_analysis_path --config_file config/coco.yaml --bias_analysis --finetune --resume_training --checkpoint experiments/coco/bit_resnet50/2022-04-04\ 16\:50\:19/model/bit_resnet50/version_0/checkpoints/epoch\=0-step\=1848.ckpt > /dev/null 2>&1 && echo SUCCESS || echo FAIL

