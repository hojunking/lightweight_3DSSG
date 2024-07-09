#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    #os.sys.path.append('./pytorch_geometric/torch_geometric')
    os.sys.path.append('./src')

## select process (origin, KD, pruning : model.py)
from model.model import MMGNet
from src.utils.config import Config
from utils import util
import torch
import argparse

def main():
    config = load_config()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)
    
    if config.MODEL.use_pretrain:
        if os.path.exists(config.MODEL.use_pretrain):
            print(f'===   load pretrain model: {config.MODEL.use_pretrain}   ===')
        else:
            raise FileNotFoundError(f"The folder '{config.MODEL.use_pretrain}' does not exist.")
    model = MMGNet(config)


    save_path = os.path.join(config.PATH,'config', model.model_name, model.exp)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'config.json')
    config.DEVICE = 'cuda'
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump(config, f)
                
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")

    # just for test
    if config.MODE == 'eval':
        print('start validation...')

        model.load(best=True)
        model.validation()
        exit()
    
    if config.MODE == 'prune':
        print('===   Start Pruning   ===')
        print("Pruning method: ", config.pruning_method)
        
        """ Structured pruning"""
        if config.pruning_method == 'st':
            print("Pruning method: Structured pruning")
            if config.pruning_part == 'encoder':
                print("Structured pruning model part: Encoder")
                model.encoder_pruning()
            elif config.pruning_part == 'gcn':
                print("Structured Pruning model part: GCN")
                model.gcn_pruning()
            elif config.pruning_part == 'classifier':
                print("Structured Pruning model part: Classifier")
                model.classifier_pruning()
            elif config.pruning_part == 'all':
                print("Structured Pruning model part: Encoder,GCN, Classifier")
                model.encoder_pruning()
                model.gcn_pruning()
                model.classifier_pruning()
            else:
                print("Error: Unknown model part specified.")
                exit()
        elif config.pruning_method == 'unst':
            """ Unstructured pruning"""
            print("Pruning method: Unstructured pruning")
            if config.pruning_part == 'encoder':
                print("Unstructured pruning model part: Encoder")
                model.apply_pruning("encoder")
            elif config.pruning_part == 'gcn':
                print("Unstructured Pruning model part: GCN")
                model.apply_pruning("gnn")
            elif config.pruning_part == 'classifier':
                print("Unstructured Pruning model part: Classifier")
                model.apply_pruning("classifier")
            elif config.pruning_part == 'all':
                print("Unstructured Pruning model part: Encoder,GCN, Classifier")
                model.apply_pruning("encoder")
                model.apply_pruning("gnn")
                model.apply_pruning("classifier")
            else:
                print("Error: Unknown model part specified.")
                exit()
            pruning_result = config.exp +'.txt'
            model.calculate_sparsity(pruning_result)
        else :
            print("Error: Unknown pruning method specified.")
            exit()
        print('\n===  Pruning Done   ===\n')
        

        model.train()
        
        ## After retraining, we need to validate the model
        model.load()
        model.config.EVAL = True
        model.validation()

        exit()
    try:
        model.load()
    except:
        print('unable to load previous model.')
    
    ## WITHOUT PRUNING
    model.train()
    # we test the best model in the end
    model.config.EVAL = True
    print('start validation...')
    model.load()
    model.validation()
    #flops = flops / 1e9

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval','prune'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--part', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--ratio', type=str)
    parser.add_argument('--pretrained', type=str)


    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    if args.mode == 'prune' and not args.ratio:
        parser.error("--ratio is required when --mode is 'prune'")

    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.NAME = args.model
    config.exp = args.exp
    config.pruning_part = args.part
    config.MODEL.use_pretrain = args.pretrained
    if args.ratio:
        config.pruning_ratio = float(args.ratio)
    if args.method:
        config.pruning_method = args.method

    return config

if __name__ == '__main__':
    main()
