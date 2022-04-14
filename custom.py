import os
import argparse

def setup_experiments(model_list, training_datasets):
    os.mkdir('experiments')
    for train in training_datasets:
        os.mkdir('experiments/'+train)
    for train in training_datasets:
        for model in model_list:
            os.mkdir('experiments/'+train+'/'+model)

def add_training_dataset(root, training_dataset, model_list):
    os.mkdir(root+'/'+training_dataset)
    for model in model_list:
        os.mkdir(root+'/'+training_dataset+'/'+model)
    
def add_model(root, model_name):
    training_datasets = os.listdir(root)
    for dataset in training_datasets:
        if dataset != '.ipynb_checkpoints':
            os.mkdir(root+'/'+dataset+'/'+model_name)
    

def add_analysis_set(root, analysis_set, model_list):
    # By default, coco and openimages are set up as analysis sets 
    # Assumes setup_experiments has already been run and each model in model_list
    #     contains at least one model trial
    training_datasets = os.listdir(root)
    for training_dataset in training_datasets:
        if training_dataset != '.ipynb_checkpoints':
            for model in model_list:
                trials = os.listdir(root+'/'+training_dataset+'/'+model)
                for trial in trials:
                    if trial != '.ipynb_checkpoints':
                        os.mkdir(root+'/'+training_dataset+'/'+model+'/'+trial+'/boxplots/'+analysis_set)
                        os.mkdir(root+'/'+training_dataset+'/'+model+'/'+trial+'/metric_data/'+analysis_set)
                        if trial != 'averaged':
                            os.mkdir(root+'/'+training_dataset+'/'+model+'/'+trial+'/features/'+analysis_set)
                            os.mkdir(root+'/'+training_dataset+'/'+model+'/'+trial+'/features/'+analysis_set+'/pretrained_features')
                            os.mkdir(root+'/'+training_dataset+'/'+model+'/'+trial+'/features/'+analysis_set+'/finetuned_features')
                        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
            help='path to experiments_folder', default='experiments')
    parser.add_argument('--analysis_set', type=str,
            help='name of analysis_set', default='test_analysis_set')
    parser.add_argument('--model_name', type=str,
            help='name of model to add', default='test_model')     
    parser.add_argument('--training_set_name', type=str,
            help='name of training set to add', default='train_dataset')          
    parser.add_argument('--training_datasets', nargs='+', 
            help='List of training datasets')
    parser.add_argument('--model_list', nargs='+', 
            help='List of models')
    parser.add_argument('--initial_setup',
            help='sets up the experiments/ folder', action='store_true')
    parser.add_argument('--add_model',
            help='adds new model', action='store_true')
    parser.add_argument('--add_analysis_set',
            help='adds new analysis_set', action='store_true')
    parser.add_argument('--add_training_set',
            help='adds new training', action='store_true')
    args = parser.parse_args()


    if args.initial_setup == True:
        setup_experiments(args.model_list, args.training_datasets)
    elif args.add_model == True:
        add_model(args.root, args.model_name)
    elif args.add_analysis_set == True:
        add_analysis_set(args.root, args.analysis_set, args.model_list)
    elif args.add_training_set == True:
        add_training_dataset(args.root, args.training_set_name, args.model_list)
    
if __name__ == '__main__':
    main()
