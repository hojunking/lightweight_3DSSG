if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import copy
import os, glob

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset
from src.model.SGFN_MMG.model import Mmgnet
from src.model.SGFN_MMG.baseline_sgfn import SGFN
from src.model.SGFN_MMG.baseline_sgpn import SGPN
from src.utils import op_utils
from src.utils.eva_utils_acc import get_mean_recall, get_zero_shot_recall
# pruning
import torch_pruning as tp
from functools import partial
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
import torch.nn.utils.prune as prune
from prettytable import PrettyTable

import time
class MMGNet():
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.NAME
        print(f'model name : {self.model_name}')
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        self.masks = {}

        ''' Pruning_pointnet config '''
        self.prune_method = config.pruning_pointnet.method
        self.prune_speed_up = config.pruning_pointnet.speed_up
        self.prune_max_pruning_ratio = config.pruning_pointnet.max_pruning_ratio
        self.pruning_ratio = config.pruning_ratio
        self.prune_soft_keeping_ratio = config.pruning_pointnet.soft_keeping_ratio
        self.prune_reg = config.pruning_pointnet.reg
        self.prune_delta_reg = config.pruning_pointnet.delta_reg
        self.prune_weight_decay = config.pruning_pointnet.weight_decay
        self.prune_global_pruning = config.pruning_pointnet.global_pruning
        self.prune_sl_lr_decay_milestones = config.pruning_pointnet.sl_lr_decay_milestones 

        ''' Build dataset '''
        dataset = None
        if config.MODE  == 'train' or config.MODE == 'prune':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)

            
                
        if config.MODE  == 'train' or config.MODE  == 'trace' or config.MODE  == 'eval' or config.MODE == 'prune':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL)
            dataset = self.dataset_valid

        num_obj_class = len(self.dataset_valid.classNames)   
        num_rel_class = len(self.dataset_valid.relationNames)
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class

        if config.MODE  == 'train':
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)
        elif config.MODE  == 'eval':
            self.total = self.config.total = len(self.dataset_valid) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_valid) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_valid) // self.config.Batch_Size)
        elif config.MODE  == 'prune':
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)

        ''' Build Model '''
        if self.model_name == 'Mmgnet':
            self.model = Mmgnet(self.config, num_obj_class, num_rel_class).to(config.DEVICE)
        elif self.model_name == 'sgfn':
            self.model = SGFN(self.config, num_obj_class, num_rel_class).to(config.DEVICE)
        elif self.model_name == 'sgpn':
            self.model = SGPN(self.config, num_obj_class, num_rel_class).to(config.DEVICE)
        else:
            print(f'Unknown model name: {self.model_name}')
            raise NotImplementedError
        
        ## load pre-trained weights
        if self.mconfig.use_pretrain != "":
            self.model.load_pretrain_model(self.mconfig.use_pretrain, is_freeze=False)


        self.samples_path = os.path.join(config.PATH, self.model_name, self.exp,  'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, self.exp, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, self.exp, 'traced')
        self.writter = None
        
        if not self.config.EVAL:
            pth_log = os.path.join(config.PATH, "logs", self.model_name, self.exp)
            self.writter = SummaryWriter(pth_log)
        
    def load(self, best=False):
        return self.model.load(best)
    
    def init_named(self):
        # 디렉토리 경로 설정
        root_dir = os.path.join(self.config.PATH, 'ckp', self.model_name, self.exp)

        # _best.pth 파일들의 목록을 가져오기
        best_files = glob.glob(os.path.join(root_dir, '*_best.pth'))
        print(best_files)
        for file_path in best_files:
            # 새 파일 이름 생성
            new_file_path = file_path.replace('_best.pth', '_pre.pth')
            # 파일 이름 변경
            os.rename(file_path, new_file_path)


    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
    
    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids

    def get_pruner(self, model, example_inputs, num_classes, ignored_layers=[]):
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
        #args.is_accum_importance = is_accum_importance
        unwrapped_parameters = []
        #ignored_layers = []
        pruning_ratio_dict = {}
        #print("before add ignored layers: ", ignored_layers)

        # ignore output layers
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
                ignored_layers.append(m)
            elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
                ignored_layers.append(m)
        
        # Here we fix iterative_steps=200 to prune the model progressively with small steps 
        # until the required speed up is achieved.
        print("ignored layers: ", ignored_layers)
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

    def train(self):
        self.init_named()
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=self.config.WORKERS,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )

        self.model.epoch = 1
        keep_training = True
        
        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it', 'Misc/lr'])
                
        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
                   
        # if self.mconfig.use_pretrain != "":
        #     self.model.load_pretrain_model(self.mconfig.use_pretrain, is_freeze=True)
        
        for k, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"Para {k} need grad")
        ''' Train '''
        while(keep_training):

            if self.model.epoch > self.config.MAX_EPOCHES:
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)
            
            for items in loader:

                self.model.train()
                ''' get data '''
                obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
                
                logs = self.model.process_train(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, with_log=True,
                                                weights_obj=self.dataset_train.w_cls_obj, 
                                                weights_rel=self.dataset_train.w_cls_rel,
                                                ignore_none_rel = False)
                iteration = self.model.iteration
                logs += [
                    ("Misc/epo", int(self.model.epoch)),
                    ("Misc/it", int(iteration)),
                    ("lr", self.model.lr_scheduler.get_last_lr()[0])
                ]
                
                progbar.add(1, values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, iteration)
                # if self.model.iteration >= self.max_iteration:
                #     break

            progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            self.save()

            if (self.model.epoch > 20 and 'VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                print('start validation...')
                _, _, _, _, _, _, _, _, rel_acc_val = self.validation()
                self.model.eva_res = rel_acc_val
                self.save()
            
            #self.track_pruned_weights()
            self.model.epoch += 1
            
            # if self.update_2d:
            #     print('load copy model from last epoch')
            #     # copy param from previous epoch
            #     model_pre = Mmgnet(self.config, self.num_obj_class, self.num_rel_class).to(self.config.DEVICE)
            #     for k, p in model_pre.named_parameters():
            #         p.data.copy_(self.model.state_dict()[k])
            #     model_pre.model_pre = None
            #     self.model.update_model_pre(model_pre)
                   
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]
    
    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None and not self.config.EVAL:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)
                    
    def save(self):
        self.model.save ()


    def calc_FLOPs(self):
        drop_last = True
        sample_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=16,
            num_workers=0,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )
        #print(self.dataset_train[0][0].unsqueeze(0).shape)
        loader = iter(sample_loader)
        item = next(loader)
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(item)
        inputs = (obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids, False)
        #kwargs = {'descriptor': descriptor, 'batch_ids':batch_ids,'istrain': False}
        return FlopCountAnalysis(self.model, inputs)
    
    def go_prune(self, prun_type, model_name ,pruner, example_inputs, base_ops, origin_param_count, idx=0):
        current_speed_up = 1
        pruned_ratio = 0
        if prun_type == "pointnet":
            print('prune pointnet', model_name)
            while pruned_ratio < self.pruning_ratio:
                
                pruner.step()
                pruned_ops, params_count = tp.utils.count_ops_and_params(getattr(self.model, model_name), example_inputs=example_inputs)
                
                pruned_ratio = (origin_param_count - params_count) / origin_param_count
                current_speed_up = float(base_ops) / pruned_ops
                if pruner.current_step == pruner.iterative_steps:
                    break
            
        elif prun_type == "gcn":
            print(f'prune :{model_name}[{idx}]')
            if self.model_name in ['sgfn', 'sgpn']:
                while pruned_ratio < self.pruning_ratio:
                    
                    pruner.step()
                    pruned_ops, params_count = tp.utils.count_ops_and_params(getattr(self.model.gcn, model_name)[idx], example_inputs=example_inputs)
                    
                    pruned_ratio = (origin_param_count - params_count) / origin_param_count
                    current_speed_up = float(base_ops) / pruned_ops
                    if pruner.current_step == pruner.iterative_steps:
                        break
            else:
                
                while pruned_ratio < self.pruning_ratio:
                
                    pruner.step()
                    pruned_ops, params_count = tp.utils.count_ops_and_params(getattr(self.model.mmg, model_name)[idx], example_inputs=example_inputs)
                    
                    pruned_ratio = (origin_param_count - params_count) / origin_param_count
                    current_speed_up = float(base_ops) / pruned_ops
                    if pruner.current_step == pruner.iterative_steps:
                        break
        else:
            raise NotImplementedError
        print(f'en_current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')

    def encoder_pruning(self, debug_mode = False):
        
            
        print(f'===  {self.model_name} (encoder) Pruning   ===')
        prun_type = "pointnet"
        ''' obj_encoder pruning'''
        # random input example
        obj_encoder_input_example = torch.randn(138, 3, 128).to(self.config.DEVICE)

        # get pruner 
        obj_encoder_pruner = self.get_pruner(self.model.obj_encoder, example_inputs=obj_encoder_input_example, num_classes=self.num_obj_class, ignored_layers=[self.model.obj_encoder.conv3])
        
        # get base ops and origin params count
        encoder_base_ops, encoder_origin_params_count = tp.utils.count_ops_and_params(self.model.obj_encoder, example_inputs=obj_encoder_input_example)
        
        # start pruning
        self.go_prune(prun_type,"obj_encoder",obj_encoder_pruner, obj_encoder_input_example, encoder_base_ops, encoder_origin_params_count)
        
        ''' rel_encoder 2d/3d pruning'''
        # rel_encoder_2d_example = torch.randn(1070, 11, 1).to(self.config.DEVICE)
        # rel_encoder_3d_example = torch.randn(1070, 11, 1).to(self.config.DEVICE)
        # rel_encoder_2d_prunner = self.get_pruner(self.model.rel_encoder_2d, example_inputs=rel_encoder_2d_example, num_classes=self.num_obj_class, ignored_layers=[self.model.rel_encoder_2d.conv3])
        # rel_encoder_3d_prunner = self.get_pruner(self.model.rel_encoder_3d, example_inputs=rel_encoder_3d_example, num_classes=self.num_obj_class, ignored_layers=[self.model.rel_encoder_3d.conv3])
        # rel_encoder_3d_base_ops, rel_encoder_3d_origin_params_count = tp.utils.count_ops_and_params(self.model.rel_encoder_3d, example_inputs=rel_encoder_3d_example)
        # rel_encoder_2d_base_ops, rel_encoder_2d_origin_params_count = tp.utils.count_ops_and_params(self.model.rel_encoder_2d, example_inputs=rel_encoder_2d_example)
        # self.go_prune(prun_type, "rel_encoder_2d",rel_encoder_2d_prunner, rel_encoder_2d_example, rel_encoder_2d_base_ops, rel_encoder_2d_origin_params_count)
        # self.go_prune(prun_type, "rel_encoder_3d",rel_encoder_3d_prunner, rel_encoder_3d_example, rel_encoder_3d_base_ops, rel_encoder_3d_origin_params_count)
    def gcn_pruning(self, debug_mode = False):
        print(f'=== {self.model_name} (GCN) Pruning   ===')
        prun_type = "gcn"
        
        if self.model_name == 'sgfn':
            num_nodes, num_edges =130, 964
            node_features_example = torch.randn(num_nodes, 512).to(self.config.DEVICE)
            edge_features_example = torch.randn(num_edges, 256).to(self.config.DEVICE)
            edge_indices_example = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).to(self.config.DEVICE)
            
            gcn_example_input = (node_features_example, edge_features_example, edge_indices_example)

            # gcn[depth] pruning
            for idx in range(0,self.mconfig.N_LAYERS):
                ignore_layers = [self.model.gcn.gconvs[idx].edgeatten.proj_edge, self.model.gcn.gconvs[idx].edgeatten.proj_value, 
                                self.model.gcn.gconvs[idx].edgeatten.nn_edge[2]]
                print(ignore_layers)
                gcn_3ds_base_ops, gcn_3ds_origin_params_count = tp.utils.count_ops_and_params(self.model.gcn.gconvs[idx], example_inputs=gcn_example_input)
                
                print(f'gcn_3d[{idx}]_base_ops: {gcn_3ds_base_ops}, gcn_3d[{idx}]_origin_params_count: {gcn_3ds_origin_params_count}')
                gcn_3ds_pruner = self.get_pruner(self.model.gcn.gconvs[idx], example_inputs=gcn_example_input, num_classes=512, ignored_layers=ignore_layers)
                self.go_prune(prun_type, "gconvs",gcn_3ds_pruner, gcn_example_input, gcn_3ds_base_ops, gcn_3ds_origin_params_count, idx)
        
        elif self.model_name == 'sgpn':
            num_nodes, num_edges = 136, 1048
            node_features_example = torch.randn(num_nodes, 512).to(self.config.DEVICE)
            edge_features_example = torch.randn(num_edges, 256).to(self.config.DEVICE)
            edge_indices_example = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).to(self.config.DEVICE)
            gcn_example_input = (node_features_example, edge_features_example, edge_indices_example)

            ignore_layers = []
            for idx in range(0,self.mconfig.N_LAYERS):
                for name, module in self.model.gcn.gconvs[idx].named_modules():
                    if name == 'nn1.2' and isinstance(module, torch.nn.Linear):
                        ignore_layers.append(module)
                    if name == 'nn2.2' and isinstance(module, torch.nn.Linear):
                        ignore_layers.append(module)

                gcn_3ds_base_ops, gcn_3ds_origin_params_count = tp.utils.count_ops_and_params(self.model.gcn.gconvs[idx], example_inputs=gcn_example_input)
                
                print(f'gcn_3d[{idx}]_base_ops: {gcn_3ds_base_ops}, gcn_3d[{idx}]_origin_params_count: {gcn_3ds_origin_params_count}')
                gcn_3ds_pruner = self.get_pruner(self.model.gcn.gconvs[idx], example_inputs=gcn_example_input, num_classes=1, ignored_layers=ignore_layers)
                self.go_prune(prun_type, "gconvs",gcn_3ds_pruner, gcn_example_input, gcn_3ds_base_ops, gcn_3ds_origin_params_count, idx)
        
        ## vl-sat mmg pruning
        else:
            
            num_nodes, num_edges =100, 150
            node_features_example = torch.randn(num_nodes, 512).to(self.config.DEVICE)
            edge_features_example = torch.randn(num_edges, 512).to(self.config.DEVICE)
            edge_indices_example = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).to(self.config.DEVICE)
            gcn_example_input = (node_features_example, edge_features_example, edge_indices_example)
            
            # gcn_2d[depth] pruning
            for idx in range(0,self.mconfig.N_LAYERS):
                ignore_layers = [self.model.mmg.gcn_2ds[idx].edgeatten.proj_edge, self.model.mmg.gcn_2ds[idx].edgeatten.proj_value]
                gcn_2ds_base_ops, gcn_2ds_origin_params_count = tp.utils.count_ops_and_params(self.model.mmg.gcn_2ds[idx], example_inputs=gcn_example_input)
                
                print(f'gcn_2d[{idx}]_base_ops: {gcn_2ds_base_ops}, gcn_2d[{idx}]_origin_params_count: {gcn_2ds_origin_params_count}')
                gcn_2ds_pruner = self.get_pruner(self.model.mmg.gcn_2ds[idx], example_inputs=gcn_example_input, num_classes=512, ignored_layers=ignore_layers)
                self.go_prune(prun_type, "gcn_2ds",gcn_2ds_pruner, gcn_example_input, gcn_2ds_base_ops, gcn_2ds_origin_params_count, idx)

            # gcn_3d[depth] pruning
            for idx in range(0,self.mconfig.N_LAYERS):
                ignore_layers = [self.model.mmg.gcn_3ds[idx].edgeatten.proj_edge, self.model.mmg.gcn_3ds[idx].edgeatten.proj_value]
                gcn_3ds_base_ops, gcn_3ds_origin_params_count = tp.utils.count_ops_and_params(self.model.mmg.gcn_3ds[idx], example_inputs=gcn_example_input)
                
                print(f'gcn_3d[{idx}]_base_ops: {gcn_3ds_base_ops}, gcn_3d[{idx}]_origin_params_count: {gcn_3ds_origin_params_count}')
                gcn_3ds_pruner = self.get_pruner(self.model.mmg.gcn_3ds[idx], example_inputs=gcn_example_input, num_classes=512, ignored_layers=ignore_layers)
                self.go_prune(prun_type, "gcn_3ds",gcn_3ds_pruner, gcn_example_input, gcn_3ds_base_ops, gcn_3ds_origin_params_count, idx)
    
        
    def classifier_pruning(self, debug_mode = False):
        print(f'===  {self.model_name} (classifier) Pruning   ===')
        prun_type = "pointnet"
        
        if self.model_name == 'sgfn' or 'sgpn':
            # random input example
            obj_pred_input_example = torch.randn(130, 512).to(self.config.DEVICE)
            rel_pred_input_example = torch.randn(964, 256).to(self.config.DEVICE)

            # get pruner 
            obj_pred_pruner = self.get_pruner(self.model.obj_predictor, example_inputs=obj_pred_input_example, num_classes=160, ignored_layers=[])
            rel_pred_pruner = self.get_pruner(self.model.rel_predictor, example_inputs=rel_pred_input_example, num_classes=26, ignored_layers=[])
            
            # get base ops and origin params count
            obj_pred_base_ops, obj_pred_origin_params_count = tp.utils.count_ops_and_params(self.model.obj_predictor, example_inputs=obj_pred_input_example)
            rel_pred_base_ops, rel_pred_origin_params_count = tp.utils.count_ops_and_params(self.model.rel_predictor, example_inputs=rel_pred_input_example)
            
            # start pruning
            self.go_prune(prun_type,"obj_predictor",obj_pred_pruner, obj_pred_input_example, obj_pred_base_ops, obj_pred_origin_params_count)
            self.go_prune(prun_type,"rel_predictor",rel_pred_pruner, rel_pred_input_example, rel_pred_base_ops, rel_pred_origin_params_count)
        ## VL-SAT classifier
        else:
            # random input example
            rel_pred_3d_input_example = torch.randn(1070, 512).to(self.config.DEVICE)
            rel_pred_2d_input_example = torch.randn(1070, 512).to(self.config.DEVICE)

            # get pruner 
            rel_pred_3d_pruner = self.get_pruner(self.model.rel_predictor_3d, example_inputs=rel_pred_3d_input_example, num_classes=26, ignored_layers=[])
            rel_pred_2d_pruner = self.get_pruner(self.model.rel_predictor_2d, example_inputs=rel_pred_2d_input_example, num_classes=26, ignored_layers=[])
            
            # get base ops and origin params count
            rel_pred_3d_base_ops, rel_pred_3d_origin_params_count = tp.utils.count_ops_and_params(self.model.rel_predictor_3d, example_inputs=rel_pred_3d_input_example)
            rel_pred_2d_base_ops, rel_pred_2d_origin_params_count = tp.utils.count_ops_and_params(self.model.rel_predictor_2d, example_inputs=rel_pred_2d_input_example)
            
            # start pruning
            self.go_prune(prun_type,"rel_predictor_3d",rel_pred_3d_pruner, rel_pred_3d_input_example, rel_pred_3d_base_ops, rel_pred_3d_origin_params_count)
            self.go_prune(prun_type,"rel_predictor_2d",rel_pred_2d_pruner, rel_pred_2d_input_example, rel_pred_2d_base_ops, rel_pred_2d_origin_params_count)
    
    
    def apply_pruning(self, apply_part):
        if apply_part == "encoder":
            encoders = ['obj_encoder', 'rel_encoder_2d', 'rel_encoder_3d']
            for encoder_name in encoders:
                print(f"encoder: {encoder_name} pruning:{self.pruning_ratio} start!")
                for name, module in getattr(self.model, encoder_name).named_modules():
                    if isinstance(module, torch.nn.Conv1d):
                        prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                        
                        prune.remove(module, 'weight')
        elif apply_part == "gnn":
            print("gnn pruning start")
            gnn_name = 'mmg'
            print(f"gnn: {gnn_name} pruning:{self.pruning_ratio} start!")
            for name, module in getattr(self.model, gnn_name).named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                    self.masks[name] = module.weight_mask.clone().detach()
                    prune.remove(module, 'weight')
            print(f'masks: {self.masks}')
        elif apply_part == "classifier":
            print(f"classifier: {apply_part} pruning:{self.pruning_ratio} start!")
            classifiers = ['obj_predictor_3d', 'rel_predictor_3d', 'obj_predictor_2d', 'rel_predictor_2d']
            for predicator in classifiers:
                for name, module in getattr(self.model, predicator).named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                        
                        prune.remove(module, 'weight')
        elif apply_part == 'all':
            print(f"ALL Pruning start!")
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                    
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Conv1d):
                    prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
                    
                    prune.remove(module, 'weight')
        else:
            print("pruning error!")
        print(f"{apply_part} pruning success!")


    def track_pruned_weights(self):
        for name, module in getattr(self.model, 'mmg').named_modules():
            if name in self.masks:
                mask = self.masks[name]
                weight = module.weight.data
                pruned_weights = weight[mask == 0]
                print(f"Module: {name} | Pruned weights mean: {pruned_weights.mean().item()} | std: {pruned_weights.std().item()}")
            # else:
            #     print("No pruned weights found in the model.")

    def calculate_sparsity(self, pruning_result):
        # visualize via table
        table = PrettyTable(["Layer", "Total Parameters", "Non-zero Parameters", "Sparsity (%)"])
        total_params = total_non_zero = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                if name.endswith('weight'):
                    non_zero_params = torch.count_nonzero(param).item()
                else:
                    non_zero_params = num_params
                sparsity = 100.0 * (1 - non_zero_params / num_params)
                table.add_row([name, num_params, non_zero_params, f"{sparsity:.2f}"])
                total_params += num_params
                total_non_zero += non_zero_params
        total_sparsity = 100.0 * (1 - total_non_zero / total_params)
        table.add_row(["Total", total_params, total_non_zero, f"{total_sparsity:.2f}"])
        # save
        save_path = os.path.join(self.config.PATH, 'pruning_ratio')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(self.config.PATH, pruning_result)

        with open(save_path, "w") as f:
            f.write(str(table))
        print(f"save_path: {save_path}")


    def validation(self, debug_mode = False):
        
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=16,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )

        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])

        total_inference_time = 0
        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_val(items)            
            
            # Start timing
            start_time = time.time()

            with torch.no_grad():
                # if self.model.config.EVAL:
                #     top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                #         = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices, use_triplet=True)
                # else:
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                    = self.model.process_val(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=True)
            
            # End timing
            end_time = time.time()
            total_inference_time += end_time - start_time


            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            if cls_matrix is not None:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)

            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                    ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),]

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])


        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        
        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
            f_in = open(os.path.join(save_path, 'result.txt'), 'w')
        else:
            f_in = None   
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)

        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)

     
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        #print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        #print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        #print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in) 
        #print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        #print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        #print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        #print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        #print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        #print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        #print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        #print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        #print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
        print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
        print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)
        

        average_inference_time_per_data = total_inference_time / len(val_loader)
        # Convert seconds to hour:minute:second
        hours, remainder = divmod(total_inference_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format and print the time
        formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
        print(f"Eval: Total inference time: {formatted_time}",file=f_in)
        print(f"Average_inference_time_per_data: {average_inference_time_per_data}",file=f_in)

        if self.model.config.EVAL:
            f_in.close()
        
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                ("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                ("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                ("mean_recall@50", mean_recall[0]),
                ("mean_2d_recall@50", mean_recall_2d[0]),
                ("mean_recall@100", mean_recall[1]),
                ("mean_2d_recall@100", mean_recall_2d[1]),
                ("zero_shot_recall@50", zero_shot_recall[0]),
                ("zero_shot_recall@100", zero_shot_recall[1]),
                ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                ("all_zero_shot_recall@100", all_zero_shot_recall[1])
                ]
        
        self.log(logs, self.model.iteration)
        #return mean_recall[0]
        #     print(f"Total inference time: {total_inference_time}")
        return obj_acc_1, obj_acc_5, obj_acc_10, rel_acc_1, rel_acc_3, rel_acc_5, triplet_acc_50, triplet_acc_100, mean_recall[0]
    
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100

    