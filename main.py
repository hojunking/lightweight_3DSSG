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
from fvcore.nn import FlopCountAnalysis

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
        print('start validation...')
        model.load(best=True)
        model.validation()
        exit()
    
    if config.MODE == 'prune':
        #model.load(best=True)
        flops = model.calc_FLOPs().total()
        origin_flops = flops / 1e9
        
        acc1_obj_cls_acc, acc5_obj_cls_acc, acc10_obj_cls_acc = 55.36, 78.44, 85.87
        acc1_rel_cls_acc, acc3_rel_cls_acc, acc5_rel_cls_acc = 89.83, 98.5, 99.51
        acc50_triplet_acc, acc100_triplet_acc = 89.36, 92.16
        print('start pruning...')

        # encoder pruning        
        model.encoder_pruning()
        #print(f'After encoder pruning paramters: {count_parameters(model.model)}')

        # gnn pruning
        #model.gnn_pruning()

        # classificer pruning
        #model.classifier_pruning()
        
        flops = model.calc_FLOPs().total()
        pruned_flops = flops / 1e9
        pruned_para = count_parameters(model.model)
        print(f"Pruning ratio: {config.pruning_pointnet.pruning_ratio}")
        print(f"Origin paramters: {27162021} || FLOPs: {origin_flops:.4f} billion FLOPs")
        print(f"After pruning paramters:  {pruned_para} || FLOPs: {pruned_flops:.4f} billion FLOPs")
        print(f"Diff parameter: {27162021 - pruned_para} || FLOPs: {origin_flops-pruned_flops:.4f} billion FLOPs ")
        print('\nstart training...\n')
        model.train()
        pruned_acc1_obj_cls_acc, pruned_acc5_obj_cls_acc, pruned_acc10_obj_cls_acc, pruned_acc1_rel_cls_acc, pruned_acc3_rel_cls_acc, pruned_acc5_rel_cls_acc, pruned_acc50_triplet_acc, pruned_acc100_triplet_acc, _ = model.validation()
        

        save_path = os.path.join(config.PATH, "results", config.NAME, config.exp)
        os.makedirs(save_path, exist_ok=True)
        f_in = open(os.path.join(save_path, 'result_pruned.txt'), 'w')
        
        print(f"Pruning ratio: {config.pruning_pointnet.pruning_ratio}", file=f_in)
        print(f"Origin paramters: 27162021 || FLOPs: {origin_flops:.4f} billion FLOPs", file=f_in)
        print(f"After pruning paramters:  {pruned_para} || FLOPs: {pruned_flops:.4f} billion FLOPs", file=f_in)
        print(f"Diff parameter: {27162021 - pruned_para} || FLOPs: {origin_flops-pruned_flops:.4f} billion FLOPs", file=f_in)
        print("Acc@1/obj_cls_acc: {:.4f} => {:.4f}".format(acc1_obj_cls_acc, pruned_acc1_obj_cls_acc), file=f_in)
        print("Acc@5/obj_cls_acc: {:.4f} => {:.4f}".format(acc5_obj_cls_acc, pruned_acc5_obj_cls_acc), file=f_in)
        print("Acc@10/obj_cls_acc: {:.4f} => {:.4f}".format(acc10_obj_cls_acc, pruned_acc10_obj_cls_acc), file=f_in)
        print("Acc@1/rel_cls_acc: {:.4f} => {:.4f}".format(acc1_rel_cls_acc, pruned_acc1_rel_cls_acc), file=f_in)
        print("Acc@3/rel_cls_acc: {:.4f} => {:.4f}".format(acc3_rel_cls_acc, pruned_acc3_rel_cls_acc), file=f_in)
        print("Acc@5/rel_cls_acc: {:.4f} => {:.4f}".format(acc5_rel_cls_acc, pruned_acc5_rel_cls_acc), file=f_in)
        print("Acc@50/triplet_acc: {:.4f} => {:.4f}".format(acc50_triplet_acc, pruned_acc50_triplet_acc), file=f_in)
        print("Acc@100/triplet_acc: {:.4f} => {:.4f}".format(acc100_triplet_acc, pruned_acc100_triplet_acc), file=f_in)
        
        f_in.close()
        exit()
    try:
        model.load()
    except:
        print('unable to load previous model.')
    print('\nstart training...\n')
    model.train()
    # we test the best model in the end
    model.config.EVAL = True
    print('start validation...')
    model.load()
    model.validation()

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

    return config

if __name__ == '__main__':
    main()
