from genericpath import isfile
import json
import os
if __name__ == '__main__':
    #os.sys.path.append('./pytorch_geometric/torch_geometric')
    os.sys.path.append('./src')
from src.model.model import MMGNet
from src.utils.config import Config
from utils import util
import torch
import argparse


def load_config():
    # load config file
    config = Config("./config/mmgnet.json")
    #print(config)
    if 'NAME' not in config:
        config_name = os.path.basename('./config/mmgnet.json')
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = ''
    config.MODE = 'train'
    config.exp = 'test'

    return config

config = load_config()
util.set_random_seed(config.SEED)

# if config.VERBOSE:
#     print(config)

model = MMGNet(config)
model

def get_pruner(self, model, example_inputs, num_classes):
    self.config.pruning_pointnet.sparsity_learning = False
    if self.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "l2":
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "fpgm":
        imp = tp.importance.FPGMImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "obdc":
        imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=self.num_obj_class)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "slim":
        self.config.pruning_pointnet.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "group_slim":
        self.config.pruning_pointnet.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning, group_lasso=True)
    elif self.prune_method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "group_sl":
        self.config.pruning_pointnet.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning)
    elif self.prune_method == "growing_reg":
        self.config.pruning_pointnet.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=self.prune_reg, delta_reg=self.prune_delta_reg, global_pruning=self.config.global_pruning)
    else:
        raise NotImplementedError
    print(f'prune method: {self.prune_method}')
    print(f'prune speed up: {self.prune_speed_up}')
    print(f'prune max pruning ratio: {self.prune_max_pruning_ratio}')
    print(f'prune soft keeping ratio: {self.prune_soft_keeping_ratio}')
    print(f'prune reg: {self.prune_reg}')
    print(f'prune delta reg: {self.prune_delta_reg}')
    print(f'prune weight decay: {self.prune_weight_decay}')
    print(f'prune global pruning: {self.prune_global_pruning}')
    print(f'prune sl lr decay milestones: {self.prune_sl_lr_decay_milestones}')
    print(f'prune sparsity learning: {self.config.pruning_pointnet.sparsity_learning}')
    print(f'prune iterative steps: {self.config.pruning_pointnet.iterative_steps}')

    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = [model.conv3]
    pruning_ratio_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=self.config.pruning_pointnet.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=self.config.pruning_pointnet.max_pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner