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
        print('===   Start Validation   ===')
        #model.load(best=True)
        
        print(f'===   Pruning Method: {config.pruning_method}   ===')
        print(f'ratio: {config.pruning.st_pruning_ratio} (Structured), {config.pruning.unst_pruning_ratio} (Unstructured)')
        if config.pruning_method == 'st':
            model.gcn_pruning()
        elif config.pruning_method == 'unst':
            model.apply_pruning("gnn")    

        model.config.EVAL = True
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
        
        elif config.pruning_method == 'st_unst':
            """ Unstructured + Structured pruning"""
            print("Pruning method: Structured + Unstructured pruning")
            if config.pruning_part == 'encoder':
                model.encoder_pruning()
                model.apply_pruning("encoder")
            elif config.pruning_part == 'gcn':
                model.gcn_pruning()
                model.apply_pruning("gnn")
            elif config.pruning_part == 'classifier':
                model.classifier_pruning()
                model.apply_pruning("classifier")
            elif config.pruning_part == 'all':
                model.classifier_pruning()
                model.gcn_pruning()
                model.encoder_pruning()
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
    flops = model.calc_FLOPs().total()
    flops = flops / 1e9
    print(f'\nTotal Flops: {flops:.4f} billion FLOPs')
    ## WITHOUT PRUNING
    model.train()
    # we test the best model in the end
    model.config.EVAL = True
    print('start validation...')
    model.load()
    model.validation()
    



def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval','prune'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--part', type=str)
    parser.add_argument('--st_ratio', type=str)
    parser.add_argument('--unst_ratio', type=str)
    parser.add_argument('--pretrained', type=str)


    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

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
    config.exp = args.exp
    config.pruning_part = args.part
    

    if args.pretrained:
        if os.path.exists(args.pretrained):
            print(f'===   load pretrain model: {args.pretrained}   ===')
            config.MODEL.use_pretrain = args.pretrained
        elif args.pretrained =='x':
            print('===   No pretrained weight start   ===')
        else:
            raise FileNotFoundError(f"The folder '{args.pretrained}' does not exist.")
    
    config.pruning.st_pruning_ratio, config.pruning.unst_pruning_ratio = 0, 0
    if args.st_ratio != '0' and args.unst_ratio != '0':
        config.pruning.st_pruning_ratio = float(args.st_ratio)
        config.pruning.unst_pruning_ratio = float(args.unst_ratio)
        config.pruning_method = "st_unst"
    elif args.st_ratio != '0':
        config.pruning.st_pruning_ratio = float(args.st_ratio)
        config.pruning_method = "st"
    elif args.unst_ratio != '0':
        config.pruning.unst_pruning_ratio = float(args.unst_ratio)
        config.pruning_method = "unst"
    else:
        config.pruning_method = 'none'
    return config

if __name__ == '__main__':
    main()
